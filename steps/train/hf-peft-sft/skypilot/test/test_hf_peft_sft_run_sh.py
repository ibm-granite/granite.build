# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for src/hf_peft_sft_run.sh.

`accelerate`, `nvidia-smi` and the venv python are stubs: the fake accelerate
logs its argv, cwd and env, then (like the trainer) writes both adapter files
into --output_dir, minus the one named in FAKE_OMIT. These pin the SHELL;
training is e2e's job.
"""

import json
import pathlib
import stat
import subprocess

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
_SCRIPT = _SRC / "hf_peft_sft_run.sh"

_ACCELERATE = """#!/usr/bin/env bash
python3 - "$@" <<'PY'
import json, os, sys
argv = sys.argv[1:]
keys = ("CLEARML_PROJECT", "CLEARML_TASK", "CLEARML_LOG_MODEL", "HF_DATASETS_CACHE", "WANDB_MODE")
with open(os.environ["ACC_LOG"], "w") as f:
    json.dump({"argv": argv, "cwd": os.getcwd(), "env": {k: os.environ.get(k) for k in keys}}, f)
if os.environ.get("FAKE_FAIL"):
    sys.exit(3)
out = argv[argv.index("--output_dir") + 1]
os.makedirs(out, exist_ok=True)
if os.environ.get("FAKE_OMIT") != "adapter_config.json":
    open(os.path.join(out, "adapter_config.json"), "w").write("{}")
if os.environ.get("FAKE_OMIT") != "adapter_model.safetensors":
    # A real safetensors header: one LoRA tensor, or none with FAKE_EMPTY_ADAPTER.
    import struct
    header = {"__metadata__": {"format": "pt"}}
    if not os.environ.get("FAKE_EMPTY_ADAPTER"):
        header["base_model.model.q_proj.lora_A.weight"] = {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
    raw = json.dumps(header).encode()
    body = b"" if os.environ.get("FAKE_EMPTY_ADAPTER") else b"\\0" * 4
    open(os.path.join(out, "adapter_model.safetensors"), "wb").write(struct.pack("<Q", len(raw)) + raw + body)
PY
"""

_PY = """#!/usr/bin/env bash
# venv python stub: answers the peft/trl version probe, and runs anything else
# (the saved-adapter check) on the host python with the step's src/ on the path.
case "$*" in
  *"import peft, trl"*) echo "0.21.2 1.17.1" ;;
  *) PYTHONPATH="__SRC__" exec python3 "$@" ;;
esac
"""


def _exe(path: pathlib.Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def node(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _exe(bin_dir / "accelerate", _ACCELERATE)
    _exe(
        bin_dir / "nvidia-smi",
        'echo "GPU 0: NVIDIA A10G (UUID: x)"\necho "GPU 1: NVIDIA A10G (UUID: y)"\n',
    )
    home = tmp_path / "home"
    (home / ".venv" / "bin").mkdir(parents=True)
    _exe(home / ".venv" / "bin" / "python", _PY.replace("__SRC__", str(_SRC)))
    (home / "accelerate").symlink_to(_SRC / "accelerate")
    (home / "hf_peft_sft_train.py").symlink_to(_SRC / "hf_peft_sft_train.py")
    work = tmp_path / "work"
    work.mkdir()
    return tmp_path, bin_dir, home, work


def _run(node, env=None, tail=("--dataset_path", "ds"), **opts):
    tmp, bin_dir, home, work = node
    o = {
        "base-model": "/staged/base",
        "output-id": "adapter",
        "output-path": "./adapter_output",
        "method": "lora",
        "distributed": "ddp",
        "mixed-precision": "bf16",
        "clearml-mode": "off",
        "clearml-project": "",
        "clearml-task": "",
    }
    o.update({k.replace("_", "-"): v for k, v in opts.items()})
    argv = ["bash", str(_SCRIPT)]
    for k, v in o.items():
        if v is not None:  # None = omit the option
            argv += [f"--{k}", v]
    argv += ["--", *tail]
    e = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp),
        "HF_PEFT_SFT_HOME": str(home),
        "ACC_LOG": str(tmp / "acc.json"),
        "SKYPILOT_NUM_GPUS_PER_NODE": "1",
    }
    e.update(env or {})
    e = {k: v for k, v in e.items() if v is not None}
    proc = subprocess.run(
        argv, capture_output=True, text=True, env=e, cwd=work, check=False
    )
    log = tmp / "acc.json"
    return proc, (json.loads(log.read_text()) if log.exists() else None)


def _opt(argv, name):
    return argv[argv.index(name) + 1]


def _launch(rec):
    return rec["argv"][
        : next(
            i for i, a in enumerate(rec["argv"]) if a.endswith("hf_peft_sft_train.py")
        )
    ]


def _train(rec):
    argv = rec["argv"]
    return argv[
        next(i for i, a in enumerate(argv) if a.endswith("hf_peft_sft_train.py")) + 1 :
    ]


def _markers(proc):
    return [ln for ln in proc.stdout.splitlines() if ln.startswith("GB_")]


class TestLaunch:
    def test_ddp_single_gpu(self, node):
        proc, rec = _run(node)
        assert proc.returncode == 0, proc.stderr
        assert _launch(rec) == [
            "launch",
            "--num_machines",
            "1",
            "--num_processes",
            "1",
            "--mixed_precision",
            "bf16",
        ]
        assert rec["cwd"] == str(node[3])

    def test_ddp_multi_gpu(self, node):
        # The stub must list as many GPUs as are allocated (the visibility check).
        _exe(
            node[1] / "nvidia-smi",
            "".join(f'echo "GPU {i}: NVIDIA A10G (UUID: g{i})"\n' for i in range(4)),
        )
        _, rec = _run(node, env={"SKYPILOT_NUM_GPUS_PER_NODE": "4"})
        launch = _launch(rec)
        assert "--multi_gpu" in launch
        assert _opt(launch, "--num_processes") == "4"
        assert "--config_file" not in launch

    @pytest.mark.parametrize(
        "method,dist,cfg",
        [
            ("lora", "fsdp", "fsdp.yaml"),
            ("alora", "fsdp", "fsdp.yaml"),
            ("qlora", "fsdp", "fsdp_qlora.yaml"),
            ("lora", "deepspeed", "deepspeed_z3.yaml"),
            ("qlora", "deepspeed", "deepspeed_z3.yaml"),
        ],
    )
    def test_config_file_per_method_and_distributed(self, node, method, dist, cfg):
        _, rec = _run(
            node,
            method=method,
            distributed=dist,
            env={"SKYPILOT_NUM_GPUS_PER_NODE": "2"},
        )
        launch = _launch(rec)
        assert _opt(launch, "--config_file") == str(node[2] / "accelerate" / cfg)
        assert "--multi_gpu" not in launch

    @pytest.mark.parametrize("dist", ["fsdp", "deepspeed"])
    def test_sharding_on_one_gpu_fails(self, node, dist):
        proc, rec = _run(node, distributed=dist)
        assert proc.returncode == 1 and rec is None
        assert "needs at least 2 GPUs" in proc.stderr

    def test_step_owned_flags_then_build_args(self, node):
        _, rec = _run(node, tail=("--dataset_path", "ds", "--max_steps", "3"))
        stage = str(node[3] / "adapter_output.partial")
        assert _train(rec) == [
            "--model_name_or_path",
            "/staged/base",
            "--output_dir",
            stage,
            "--method",
            "lora",
            "--distributed",
            "ddp",
            "--mixed_precision",
            "bf16",
            "--bf16",
            "true",
            "--fp16",
            "false",
            "--report_to",
            "none",
            "--save_strategy",
            "no",
            "--push_to_hub",
            "false",
            "--dataset_path",
            "ds",
            "--max_steps",
            "3",
        ]

    @pytest.mark.parametrize(
        "mp,bf16,fp16", [("fp16", "false", "true"), ("no", "false", "false")]
    )
    def test_precision_flags(self, node, mp, bf16, fp16):
        _, rec = _run(node, mixed_precision=mp)
        t = _train(rec)
        assert (_opt(t, "--bf16"), _opt(t, "--fp16")) == (bf16, fp16)

    def test_gpu_count_falls_back_to_nvidia_smi(self, node):
        _, rec = _run(node, env={"SKYPILOT_NUM_GPUS_PER_NODE": None})
        assert _opt(_launch(rec), "--num_processes") == "2"

    def test_no_gpus_fails_before_training(self, node):
        _exe(node[1] / "nvidia-smi", "exit 9\n")
        proc, rec = _run(node, env={"SKYPILOT_NUM_GPUS_PER_NODE": None})
        assert proc.returncode == 1 and rec is None
        assert "no GPUs" in proc.stderr

    def test_fewer_visible_gpus_than_allocated_fails_before_training(self, node):
        # SkyPilot says 4 GPUs were allocated, but only 2 are visible (the stub
        # nvidia-smi lists two): every rank past the visible count would bind an
        # already-used GPU, and NCCL fails later with "Multiple Ranks are using
        # the same GPU". Refuse up front with the real cause.
        proc, rec = _run(node, env={"SKYPILOT_NUM_GPUS_PER_NODE": "4"})
        assert proc.returncode == 1 and rec is None
        assert "hf-peft-sft: ERROR" in proc.stderr
        assert "4 GPU(s) were allocated" in proc.stderr
        assert "2 are visible" in proc.stderr

    def test_visible_gpus_matching_the_allocation_pass(self, node):
        proc, rec = _run(node, env={"SKYPILOT_NUM_GPUS_PER_NODE": "2"})
        assert proc.returncode == 0, proc.stderr
        assert _opt(_launch(rec), "--num_processes") == "2"

    def test_unreadable_visibility_does_not_block(self, node):
        # nvidia-smi missing or failing (e.g. a stripped container) must not turn
        # a working allocation into a refusal: only a positive undercount fails.
        _exe(node[1] / "nvidia-smi", "exit 9\n")
        proc, rec = _run(node, env={"SKYPILOT_NUM_GPUS_PER_NODE": "2"})
        assert proc.returncode == 0, proc.stderr
        assert _opt(_launch(rec), "--num_processes") == "2"

    def test_dataset_cache_in_the_workdir(self, node):
        _, rec = _run(node)
        assert rec["env"]["HF_DATASETS_CACHE"] == str(node[3] / "hf_cache" / "datasets")

    @pytest.mark.parametrize(
        "opt,bad",
        [("method", "dora"), ("distributed", "tp"), ("mixed_precision", "fp8")],
    )
    def test_bad_enum_refused(self, node, opt, bad):
        proc, rec = _run(node, **{opt: bad})
        assert proc.returncode == 1 and rec is None
        assert "hf-peft-sft: ERROR" in proc.stderr

    def test_missing_option_value_is_a_prefixed_error(self, node):
        tmp, bin_dir, home, work = node
        proc = subprocess.run(
            ["bash", str(_SCRIPT), "--base-model"],
            capture_output=True,
            text=True,
            env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HF_PEFT_SFT_HOME": str(home)},
            cwd=work,
            check=False,
        )
        assert proc.returncode == 1
        assert "hf-peft-sft: ERROR" in proc.stderr
        assert "unbound variable" not in proc.stderr

    def test_missing_required_option_is_a_prefixed_error(self, node):
        proc, rec = _run(node, output_id=None)
        assert proc.returncode == 1 and rec is None
        assert "hf-peft-sft: ERROR --output-id is required" in proc.stderr


class TestClearml:
    _CREDS = {
        "CLEARML_API_HOST": "https://h",
        "CLEARML_API_ACCESS_KEY": "a",
        "CLEARML_API_SECRET_KEY": "s",
    }

    def test_off(self, node):
        _, rec = _run(node, clearml_mode="off", env=self._CREDS)
        assert _opt(_train(rec), "--report_to") == "none"

    def test_on_with_names(self, node):
        _, rec = _run(node, clearml_mode="on", clearml_project="p q", clearml_task="t1")
        assert _opt(_train(rec), "--report_to") == "clearml"
        assert rec["env"]["CLEARML_PROJECT"] == "p q"
        assert rec["env"]["CLEARML_TASK"] == "t1"
        assert rec["env"]["CLEARML_LOG_MODEL"] == "false"

    def test_on_default_names(self, node):
        _, rec = _run(node, clearml_mode="on", method="qlora")
        assert rec["env"]["CLEARML_PROJECT"] == "hf-peft-sft"
        assert rec["env"]["CLEARML_TASK"] == "sft-qlora"

    def test_log_model_override_is_kept(self, node):
        _, rec = _run(node, clearml_mode="on", env={"CLEARML_LOG_MODEL": "true"})
        assert rec["env"]["CLEARML_LOG_MODEL"] == "true"

    def test_auto_needs_all_three_creds(self, node):
        _, rec = _run(node, clearml_mode="auto", env={"CLEARML_API_HOST": "https://h"})
        assert _opt(_train(rec), "--report_to") == "none"
        _, rec = _run(node, clearml_mode="auto", env=self._CREDS)
        assert _opt(_train(rec), "--report_to") == "clearml"

    def test_bad_mode_refused(self, node):
        proc, rec = _run(node, clearml_mode="maybe")
        assert proc.returncode == 1 and rec is None
        assert "hf-peft-sft: ERROR" in proc.stderr


class TestOutput:
    def test_markers_last_and_absolute(self, node):
        proc, _ = _run(node, method="alora")
        out = node[3] / "adapter_output"
        assert _markers(proc) == [
            "GB_STEP_METADATA_KEY:method GB_STEP_METADATA_VALUE:alora",
            "GB_STEP_METADATA_KEY:peft_version GB_STEP_METADATA_VALUE:0.21.2",
            "GB_STEP_METADATA_KEY:trl_version GB_STEP_METADATA_VALUE:1.17.1",
            f"GB_ARTIFACT_ID:adapter GB_ARTIFACT_PATH:{out}",
        ]
        assert proc.stdout.splitlines()[-1].startswith("GB_ARTIFACT_ID:adapter ")
        assert (out / "adapter_config.json").is_file()
        assert not (node[3] / "adapter_output.partial").exists()

    def test_absolute_output_path(self, node):
        target = node[0] / "shared" / "a"
        proc, _ = _run(node, output_path=str(target) + "/")
        assert f"GB_ARTIFACT_PATH:{target}" in proc.stdout

    def test_rerun_replaces_previous_adapter(self, node):
        _run(node)
        (node[3] / "adapter_output" / "old.bin").write_text("x")
        proc, _ = _run(node)
        assert proc.returncode == 0, proc.stderr
        assert not (node[3] / "adapter_output" / "old.bin").exists()

    def test_stale_partial_is_cleared(self, node):
        (node[3] / "adapter_output.partial" / "checkpoint-2").mkdir(parents=True)
        proc, _ = _run(node)
        assert proc.returncode == 0
        assert not (node[3] / "adapter_output" / "checkpoint-2").exists()

    def test_refuses_non_adapter_dir_before_training(self, node):
        d = node[3] / "adapter_output"
        d.mkdir()
        (d / "precious.txt").write_text("x")
        proc, rec = _run(node)
        assert proc.returncode == 1 and rec is None
        assert "not a previous adapter" in proc.stderr
        assert (d / "precious.txt").exists()

    @pytest.mark.parametrize(
        "where", ["same", "trailing-slash", "relative", "via-symlink", "inside"]
    )
    def test_refuses_output_overlapping_the_base_model(self, node, where):
        tmp, _, _, work = node
        base = work / "base"
        base.mkdir()
        (base / "config.json").write_text("{}")
        (base / "adapter_config.json").write_text("{}")
        (tmp / "link").symlink_to(base)
        out = {
            "same": str(base),
            "trailing-slash": f"{base}/",
            "relative": "./base",
            "via-symlink": str(tmp / "link"),
            "inside": str(base / "sub"),
        }[where]
        proc, rec = _run(node, base_model=str(base), output_path=out)
        assert proc.returncode == 1 and rec is None
        assert "overlaps the base_model" in proc.stderr
        assert (base / "config.json").is_file()

    def test_refuses_a_base_model_at_the_staging_dir(self, node):
        base = node[3] / "adapter_output.partial"
        base.mkdir()
        (base / "config.json").write_text("{}")
        proc, rec = _run(node, base_model=str(base))
        assert proc.returncode == 1 and rec is None
        assert "overlaps the base_model" in proc.stderr

    def test_sibling_with_a_shared_prefix_is_not_an_overlap(self, node):
        base = node[3] / "adapter"
        base.mkdir()
        (base / "config.json").write_text("{}")
        proc, _ = _run(
            node, base_model=str(base), output_path=str(node[3] / "adapter2")
        )
        assert proc.returncode == 0, proc.stderr

    @pytest.mark.parametrize(
        "missing", ["adapter_config.json", "adapter_model.safetensors"]
    )
    def test_incomplete_adapter_means_no_marker(self, node, missing):
        proc, _ = _run(node, env={"FAKE_OMIT": missing})
        assert proc.returncode == 1
        assert "no adapter" in proc.stderr
        assert _markers(proc) == []

    def test_empty_adapter_means_no_marker(self, node):
        # FSDP once saved a 40-byte adapter_model.safetensors with no tensors and
        # the step registered it. Both files exist here; the weights do not.
        proc, _ = _run(node, env={"FAKE_EMPTY_ADAPTER": "1"})
        assert proc.returncode == 1
        assert "hf-peft-sft: ERROR" in proc.stderr
        assert "has no weights" in proc.stderr
        assert _markers(proc) == []
        assert not (node[3] / "adapter_output").exists()

    def test_training_failure_propagates(self, node):
        proc, _ = _run(node, env={"FAKE_FAIL": "1"})
        assert proc.returncode == 3
        assert _markers(proc) == []

    @pytest.mark.parametrize("bad", ["my adapter", 'a"b'])
    def test_bad_output_id(self, node, bad):
        proc, rec = _run(node, output_id=bad)
        assert proc.returncode == 1 and rec is None
        assert "hf-peft-sft: ERROR" in proc.stderr

    @pytest.mark.parametrize("bad", ["/", ".", "..", "./", "sub/.."])
    def test_refuses_a_degenerate_output_path(self, node, bad):
        (node[3] / "sub").mkdir()
        proc, rec = _run(node, output_path=bad)
        assert proc.returncode == 1 and rec is None
        assert "hf-peft-sft: ERROR" in proc.stderr
        assert "does not name a directory to create" in proc.stderr
        assert node[3].is_dir()
