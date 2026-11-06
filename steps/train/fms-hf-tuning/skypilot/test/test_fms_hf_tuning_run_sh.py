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

"""Unit tests for src/fms_hf_tuning_run.sh.

`accelerate`, `nvidia-smi` and the venv python are stubs. The venv python runs
the real src/fms_hf_tuning_job.py on the host python with a fake `tuning`
module (an argparse parser that accepts anything and a no-op parse_dict), so
`build` and `check-output` run for real. The fake accelerate logs its argv and
env, then (like the trainer) writes a model or an adapter into the job's
save_model_dir. These pin the SHELL; training is e2e's job.
"""

import base64
import json
import pathlib
import stat
import subprocess

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
_SCRIPT = _SRC / "fms_hf_tuning_run.sh"

_ACCELERATE = """#!/usr/bin/env bash
python3 - "$@" <<'PY'
import json, os, struct, sys
argv = sys.argv[1:]
keys = ("SFT_TRAINER_CONFIG_JSON_PATH", "TERMINATION_LOG_FILE", "CLEARML_LOG_MODEL", "HF_DATASETS_CACHE")
with open(os.environ["ACC_LOG"], "w") as f:
    json.dump({"argv": argv, "cwd": os.getcwd(), "env": {k: os.environ.get(k) for k in keys}}, f)
if os.environ.get("FAKE_FAIL"):
    if not os.environ.get("FAKE_NO_TERMLOG"):
        open(os.environ["TERMINATION_LOG_FILE"], "a").write("Exception raised during training. bad input")
    sys.exit(int(os.environ["FAKE_FAIL"]))
job = json.load(open(os.environ["SFT_TRAINER_CONFIG_JSON_PATH"]))
out = job["save_model_dir"]
os.makedirs(out, exist_ok=True)
def st(path, n):
    h = {"__metadata__": {"format": "pt"}}
    for i in range(n):
        h[f"t{i}"] = {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]}
    raw = json.dumps(h).encode()
    open(path, "wb").write(struct.pack("<Q", len(raw)) + raw + b"\\0" * (4 * n))
n = 0 if os.environ.get("FAKE_EMPTY") else 2
if job.get("peft_method") in ("lora", "alora"):
    json.dump({"base_model_name_or_path": job["model_name_or_path"]}, open(os.path.join(out, "adapter_config.json"), "w"))
    st(os.path.join(out, "adapter_model.safetensors"), n)
else:
    open(os.path.join(out, "config.json"), "w").write("{}")
    st(os.path.join(out, "model.safetensors"), n)
PY
"""

_TUNING = """
import argparse
class _P(argparse.ArgumentParser):
    def parse_dict(self, d, allow_extra_keys=False):
        return ()
def get_parser():
    p = _P()
    for k in ("model_name_or_path", "output_dir", "save_model_dir", "training_data_path",
              "data_config_path", "report_to", "save_strategy", "logging_strategy",
              "clearml_project", "clearml_task", "peft_method", "alora_invocation_string"):
        p.add_argument("--" + k, type=str)
    p.add_argument("--trackers", type=str, nargs="+")
    p.add_argument("--logging_steps", type=float)
    p.add_argument("--max_steps", type=int)
    p.add_argument("--learning_rate", type=float)
    return p
"""

_PY = """#!/usr/bin/env bash
PYTHONPATH="__FAKES__" exec python3 "$@"
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
    fakes = tmp_path / "fakes"
    (fakes / "tuning").mkdir(parents=True)
    (fakes / "tuning" / "__init__.py").write_text("")
    (fakes / "tuning" / "sft_trainer.py").write_text(_TUNING)
    home = tmp_path / "home"
    (home / ".venv" / "bin").mkdir(parents=True)
    _exe(home / ".venv" / "bin" / "python", _PY.replace("__FAKES__", str(fakes)))
    (home / "accelerate").symlink_to(_SRC / "accelerate")
    for name in ("fms_hf_tuning_job.py", "fms_hf_tuning_entry.py"):
        (home / name).symlink_to(_SRC / name)
    work = tmp_path / "work"
    work.mkdir()
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    return tmp_path, bin_dir, home, work, base


def _b64(obj) -> str:
    return base64.b64encode(json.dumps(obj).encode()).decode()


def _run(node, env=None, args=None, data_config=None, **opts):
    tmp, bin_dir, home, work, base = node
    o = {
        "--base-model": str(base),
        "--output-id": "model",
        "--output-path": "./model_output",
        "--peft-method": "none",
        "--distributed": "ddp",
        "--data-path": "trl-lib/Capybara",
        "--data-config-b64": _b64(data_config or {}),
        "--args-b64": _b64(args or {}),
        "--base-model-id": "",
        "--clearml-mode": "off",
        "--clearml-project": "",
        "--clearml-task": "",
    }
    o.update({f"--{k.replace('_', '-')}": v for k, v in opts.items()})
    argv = ["bash", str(_SCRIPT)]
    for k, v in o.items():
        if v is not None:
            argv += [k, v]
    e = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp),
        "FMS_HF_TUNING_HOME": str(home),
        "ACC_LOG": str(tmp / "acc.json"),
        "SKYPILOT_NUM_GPUS_PER_NODE": "2",
    }
    e.update(env or {})
    e = {k: v for k, v in e.items() if v is not None}
    return subprocess.run(
        argv, cwd=work, env=e, capture_output=True, text=True, check=False
    )


def _acc(node):
    return json.loads((node[0] / "acc.json").read_text())


def _job(node):
    return json.loads((node[3] / "fms_hf_tuning_job.json").read_text())


def _markers(proc):
    return [ln for ln in proc.stdout.splitlines() if ln.startswith("GB_")]


class TestLaunch:
    def test_ddp_two_gpus(self, node):
        proc = _run(node)
        assert proc.returncode == 0, proc.stderr
        argv = _acc(node)["argv"]
        assert argv[0] == "launch"
        assert argv[argv.index("--config_file") + 1].endswith("accelerate/ddp.yaml")
        assert argv[argv.index("--num_processes") + 1] == "2"
        assert argv[argv.index("--num_machines") + 1] == "1"
        assert argv[argv.index("--main_process_port") + 1] == "0"
        assert argv[-1].endswith("fms_hf_tuning_entry.py")
        assert "--mixed_precision" not in argv and "--multi_gpu" not in argv

    def test_one_gpu(self, node):
        proc = _run(node, env={"SKYPILOT_NUM_GPUS_PER_NODE": "1"})
        assert proc.returncode == 0, proc.stderr
        argv = _acc(node)["argv"]
        assert argv[argv.index("--config_file") + 1].endswith(
            "accelerate/single_gpu.yaml"
        )
        assert "--main_process_port" not in argv

    def test_fsdp(self, node):
        assert _run(node, distributed="fsdp").returncode == 0
        argv = _acc(node)["argv"]
        assert argv[argv.index("--config_file") + 1].endswith("accelerate/fsdp.yaml")

    def test_fsdp_on_one_gpu_refused(self, node):
        proc = _run(node, distributed="fsdp", env={"SKYPILOT_NUM_GPUS_PER_NODE": "1"})
        assert proc.returncode == 1
        assert "fsdp needs at least 2 GPUs" in proc.stderr

    def test_gpu_count_falls_back_to_nvidia_smi(self, node):
        assert _run(node, env={"SKYPILOT_NUM_GPUS_PER_NODE": None}).returncode == 0
        argv = _acc(node)["argv"]
        assert argv[argv.index("--num_processes") + 1] == "2"

    def test_no_gpus(self, node):
        (node[1] / "nvidia-smi").unlink()
        proc = _run(node, env={"SKYPILOT_NUM_GPUS_PER_NODE": None})
        assert proc.returncode == 1 and "no GPUs on this node" in proc.stderr

    def test_container_sees_fewer_gpus_than_allocated(self, node):
        proc = _run(node, env={"SKYPILOT_NUM_GPUS_PER_NODE": "4"})
        assert proc.returncode == 1
        assert "#604" in proc.stderr and "can access only 2" in proc.stderr

    def test_env(self, node):
        assert _run(node).returncode == 0
        acc = _acc(node)
        work = node[3]
        assert acc["cwd"] == str(work)
        assert acc["env"]["SFT_TRAINER_CONFIG_JSON_PATH"] == str(
            work / "fms_hf_tuning_job.json"
        )
        assert acc["env"]["TERMINATION_LOG_FILE"] == str(
            work / "fms_hf_tuning_error.log"
        )
        assert acc["env"]["HF_DATASETS_CACHE"] == str(work / "hf_cache" / "datasets")


class TestJob:
    def test_owned_keys(self, node):
        assert (
            _run(node, args={"max_steps": "3", "learning_rate": 1e-5}).returncode == 0
        )
        j = _job(node)
        work = node[3]
        assert j["model_name_or_path"] == str(node[4])
        assert j["output_dir"] == str(work / "fms_hf_tuning_scratch")
        assert j["save_model_dir"] == str(work / "model_output.partial")
        assert j["training_data_path"] == "trl-lib/Capybara"
        assert j["report_to"] == "none" and "peft_method" not in j
        assert j["max_steps"] == 3 and j["learning_rate"] == 1e-5
        assert j["save_strategy"] == "no" and j["logging_strategy"] == "steps"

    def test_data_config(self, node):
        dc = {"datasets": [{"name": "d", "data_paths": ["/staged/ds"]}]}
        assert _run(node, data_path="", data_config=dc).returncode == 0
        j = _job(node)
        written = node[3] / "fms_hf_tuning_data_config.json"
        assert j["data_config_path"] == str(written) and "training_data_path" not in j
        assert json.loads(written.read_text()) == dc

    def test_bad_args_fail_before_launch(self, node):
        proc = _run(node, args={"lerning_rate": 1})
        assert proc.returncode == 1
        assert "args.lerning_rate is not an fms-hf-tuning argument" in proc.stderr
        assert not (node[0] / "acc.json").exists()


class TestClearml:
    def test_off(self, node):
        proc = _run(node)
        assert "ClearML tracking off (clearml_enabled=off)" in proc.stdout
        assert "trackers" not in _job(node)
        assert _acc(node)["env"]["CLEARML_LOG_MODEL"] is None

    def test_on_with_defaults(self, node):
        proc = _run(node, clearml_mode="on", peft_method="lora")
        assert proc.returncode == 0, proc.stderr
        assert (
            "ClearML tracking on (project 'fms-hf-tuning', task 'sft-lora')"
            in proc.stdout
        )
        j = _job(node)
        assert j["trackers"] == ["clearml"]
        assert (j["clearml_project"], j["clearml_task"]) == (
            "fms-hf-tuning",
            "sft-lora",
        )
        assert _acc(node)["env"]["CLEARML_LOG_MODEL"] == "false"

    def test_names_and_log_model_override(self, node):
        proc = _run(
            node,
            clearml_mode="on",
            clearml_project="p",
            clearml_task="t",
            env={"CLEARML_LOG_MODEL": "true"},
        )
        assert proc.returncode == 0, proc.stderr
        j = _job(node)
        assert (j["clearml_project"], j["clearml_task"]) == ("p", "t")
        assert _acc(node)["env"]["CLEARML_LOG_MODEL"] == "true"

    def test_auto_needs_all_three_credentials(self, node):
        creds = {
            "CLEARML_API_HOST": "h",
            "CLEARML_API_ACCESS_KEY": "a",
            "CLEARML_API_SECRET_KEY": "s",
        }
        proc = _run(node, clearml_mode="auto", env=creds)
        assert proc.returncode == 0, proc.stderr
        assert _job(node)["trackers"] == ["clearml"]
        proc = _run(
            node, clearml_mode="auto", env=dict(creds, CLEARML_API_SECRET_KEY=None)
        )
        assert proc.returncode == 0, proc.stderr
        assert "trackers" not in _job(node)


class TestOutput:
    def test_full_model_registered(self, node):
        proc = _run(node)
        assert proc.returncode == 0, proc.stderr
        out = node[3] / "model_output"
        assert (out / "config.json").is_file() and not (
            node[3] / "model_output.partial"
        ).exists()
        assert not (node[3] / "fms_hf_tuning_scratch").exists()
        m = _markers(proc)
        assert m[-1] == f"GB_ARTIFACT_ID:model GB_ARTIFACT_PATH:{out}"
        assert "GB_STEP_METADATA_KEY:peft_method GB_STEP_METADATA_VALUE:none" in m
        assert "GB_STEP_METADATA_KEY:distributed GB_STEP_METADATA_VALUE:ddp" in m
        assert any(
            x.startswith("GB_STEP_METADATA_KEY:fms_hf_tuning_version ") for x in m
        )
        assert any(
            x.startswith("GB_STEP_METADATA_KEY:transformers_version ") for x in m
        )

    def test_adapter_with_base_id(self, node):
        proc = _run(
            node,
            peft_method="alora",
            output_id="adapter",
            base_model_id="org/m",
            args={"alora_invocation_string": "<x>"},
        )
        assert proc.returncode == 0, proc.stderr
        out = node[3] / "model_output"
        cfg = json.loads((out / "adapter_config.json").read_text())
        assert cfg["base_model_name_or_path"] == "org/m"
        assert _markers(proc)[-1] == f"GB_ARTIFACT_ID:adapter GB_ARTIFACT_PATH:{out}"

    @pytest.mark.parametrize("method", ["none", "lora"])
    def test_empty_output_not_registered(self, node, method):
        proc = _run(node, peft_method=method, env={"FAKE_EMPTY": "1"})
        assert proc.returncode == 1
        assert "no weights (0 tensors)" in proc.stderr
        assert _markers(proc) == []
        assert not (node[3] / "model_output").exists()

    def test_absolute_output_path_and_rerun(self, node, tmp_path):
        target = tmp_path / "shared" / "run1"
        assert _run(node, output_path=str(target)).returncode == 0
        proc = _run(node, output_path=str(target) + "/")
        assert proc.returncode == 0, proc.stderr
        assert _markers(proc)[-1].endswith(f"GB_ARTIFACT_PATH:{target}")

    def test_stale_partial_is_replaced(self, node):
        stale = node[3] / "model_output.partial"
        stale.mkdir()
        (stale / "junk").write_text("x")
        assert _run(node).returncode == 0
        assert not (node[3] / "model_output" / "junk").exists()

    def test_non_model_dir_refused(self, node):
        d = node[3] / "model_output"
        d.mkdir()
        (d / "notes.txt").write_text("x")
        proc = _run(node)
        assert proc.returncode == 1 and "not a previous model" in proc.stderr
        assert (d / "notes.txt").exists()

    @pytest.mark.parametrize("path", ["/", ".", "..", "./"])
    def test_degenerate_output_path(self, node, path):
        proc = _run(node, output_path=path)
        assert proc.returncode == 1 and "does not name a directory" in proc.stderr

    def test_output_overlapping_base_model_refused(self, node):
        base = node[4]
        for p in (str(base), str(base / "sub"), str(base.parent)):
            proc = _run(node, output_path=p)
            assert proc.returncode == 1 and "overlaps the base_model" in proc.stderr, p
        link = node[0] / "link"
        link.symlink_to(base)
        proc = _run(node, output_path=str(link))
        assert proc.returncode == 1 and "overlaps the base_model" in proc.stderr

    @pytest.mark.parametrize(
        "path", ["./fms_hf_tuning_scratch", "./fms_hf_tuning_scratch/sub"]
    )
    def test_output_overlapping_scratch_refused(self, node, path):
        scratch = node[3] / "fms_hf_tuning_scratch"
        scratch.mkdir()
        # A previous-model lookalike, so the non-model check doesn't refuse first.
        (scratch / "config.json").write_text('{"keep": 1}')
        proc = _run(node, output_path=path)
        assert proc.returncode == 1
        assert "overlaps the step's scratch dir" in proc.stderr
        assert _markers(proc) == []
        assert not (node[0] / "acc.json").exists()
        assert (scratch / "config.json").read_text() == '{"keep": 1}'
        assert (node[4] / "config.json").is_file()


class TestFailures:
    @pytest.mark.parametrize("code", ["1", "203", "7"])
    def test_training_failure(self, node, code):
        proc = _run(node, env={"FAKE_FAIL": code})
        assert proc.returncode == 1
        assert "termination log" in proc.stderr and "bad input" in proc.stderr
        assert f"training failed (accelerate launch exit {code})" in proc.stderr
        assert "see fms-hf-tuning's termination log above" in proc.stderr
        assert "input error" not in proc.stderr
        assert "nothing registered" in proc.stderr
        assert _markers(proc) == []

    def test_training_failure_without_termination_log(self, node):
        proc = _run(node, env={"FAKE_FAIL": "1", "FAKE_NO_TERMLOG": "1"})
        assert proc.returncode == 1
        assert "no termination log was written" in proc.stderr
        assert "training failed (accelerate launch exit 1)" in proc.stderr
        assert "see the training log above" in proc.stderr
        assert "nothing registered" in proc.stderr
        assert _markers(proc) == []

    @pytest.mark.parametrize(
        "opt,val,needle",
        [
            ("peft_method", "qlora", "--peft-method must be none, lora or alora"),
            ("distributed", "deepspeed", "--distributed must be ddp or fsdp"),
            ("clearml_mode", "yes", "--clearml-mode must be auto, on or off"),
            ("output_id", "a b", "output id contains whitespace"),
            ("base_model", "", "--base-model is required"),
            ("args_b64", "", "--args-b64 is required"),
        ],
    )
    def test_bad_options(self, node, opt, val, needle):
        proc = _run(node, **{opt: val})
        assert proc.returncode == 1
        assert proc.stderr.startswith("fms-hf-tuning: ERROR ") and needle in proc.stderr

    def test_missing_value_and_unknown_option(self, node):
        proc = subprocess.run(
            ["bash", str(_SCRIPT), "--base-model"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 1 and "--base-model needs a value" in proc.stderr
        proc = subprocess.run(
            ["bash", str(_SCRIPT), "--bogus", "x"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 1 and "unknown option '--bogus'" in proc.stderr
