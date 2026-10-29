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

"""Unit tests for src/openinstruct_sft_run.sh.

`accelerate` and `nvidia-smi` are stubs on PATH: the fake accelerate logs its argv,
cwd and tracking env, then (like finetune.py) writes config.json into the
--output_dir it was given, unless told not to. So these pin the SHELL the step
runs: the launch argv, ClearML on/off, the staging swap, and the markers. Real
training is the build tests' job.
"""

import json
import pathlib
import stat
import subprocess

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1] / "src" / "openinstruct_sft_run.sh"
)

_ACCELERATE = """#!/usr/bin/env bash
python3 - "$@" <<'PY'
import json, os, sys
argv = sys.argv[1:]
rec = {"argv": argv, "cwd": os.getcwd(),
       "env": {k: os.environ.get(k) for k in ("WANDB_MODE", "CLEARML_PROJECT", "CLEARML_TASK")}}
with open(os.environ["ACC_LOG"], "w") as f:
    json.dump(rec, f)
if os.environ.get("FAKE_FAIL"):
    sys.exit(3)
out = argv[argv.index("--output_dir") + 1]
os.makedirs(out, exist_ok=True)
if not os.environ.get("FAKE_NO_MODEL"):
    open(os.path.join(out, "config.json"), "w").write("{}")
PY
"""


def _exe(path: pathlib.Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def node(tmp_path):
    """A fake node: stub bin dir, an OPEN_INSTRUCT_HOME, a work dir."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _exe(bin_dir / "accelerate", _ACCELERATE)
    _exe(
        bin_dir / "nvidia-smi",
        'echo "GPU 0: NVIDIA A10G (UUID: x)"\necho "GPU 1: NVIDIA A10G (UUID: y)"\n',
    )
    home = tmp_path / "oi"
    home.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    return tmp_path, bin_dir, home, work


def _run(
    node, *extra_args, env=None, tail=("--dataset_mixer_list", "ds", "1.0"), **opts
):
    tmp, bin_dir, home, work = node
    o = {
        "base-model": "/staged/base",
        "output-id": "model",
        "output-path": "./model_output",
        "mixed-precision": "bf16",
        "deepspeed-config": "stage3_no_offloading_accelerate.conf",
        "clearml-mode": "off",
        "clearml-project": "",
        "clearml-task": "",
    }
    o.update({k.replace("_", "-"): v for k, v in opts.items()})
    argv = ["bash", str(_SCRIPT)]
    for k, v in o.items():
        argv += [f"--{k}", v]
    argv += [*extra_args, "--", *tail]
    e = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp),
        "OPEN_INSTRUCT_HOME": str(home),
        "ACC_LOG": str(tmp / "acc.json"),
        "SKYPILOT_NUM_GPUS_PER_NODE": "1",
        "OPEN_INSTRUCT_COMMIT": "26de99604002d6a7b25a2c5f2ee880e388a21dca",
    }
    e.update(env or {})
    e = {k: v for k, v in e.items() if v is not None}
    proc = subprocess.run(
        argv, capture_output=True, text=True, env=e, cwd=work, check=False
    )
    log = tmp / "acc.json"
    rec = json.loads(log.read_text()) if log.exists() else None
    return proc, rec


def _opt(argv, name):
    return argv[argv.index(name) + 1]


def _markers(proc):
    return [ln for ln in proc.stdout.splitlines() if ln.startswith("GB_")]


class TestLaunch:
    def test_launch_argv(self, node):
        proc, rec = _run(node)
        assert proc.returncode == 0, proc.stderr
        argv = rec["argv"]
        assert argv[: argv.index("open_instruct/finetune.py")] == [
            "launch",
            "--mixed_precision",
            "bf16",
            "--num_machines",
            "1",
            "--num_processes",
            "1",
            "--use_deepspeed",
            "--deepspeed_config_file",
            "configs/ds_configs/stage3_no_offloading_accelerate.conf",
        ]
        assert rec["cwd"] == str(node[2])

    def test_step_owned_flags_then_build_args(self, node):
        proc, rec = _run(
            node, tail=("--dataset_mixer_list", "ds", "1.0", "--max_train_steps", "3")
        )
        ft = rec["argv"][rec["argv"].index("open_instruct/finetune.py") + 1 :]
        stage = str(node[3] / "model_output.partial")
        assert ft == [
            "--model_name_or_path",
            "/staged/base",
            "--output_dir",
            stage,
            "--do_not_randomize_output_dir",
            "--push_to_hub",
            "false",
            "--try_launch_beaker_eval_jobs",
            "false",
            "--try_auto_save_to_beaker",
            "false",
            "--dataset_local_cache_dir",
            str(node[3] / "local_dataset_cache"),
            "--dataset_mixer_list",
            "ds",
            "1.0",
            "--max_train_steps",
            "3",
        ]

    def test_no_deepspeed(self, node):
        _, rec = _run(node, deepspeed_config="")
        assert "--use_deepspeed" not in rec["argv"]

    def test_gpu_count_from_skypilot(self, node):
        _, rec = _run(node, env={"SKYPILOT_NUM_GPUS_PER_NODE": "4"})
        assert _opt(rec["argv"], "--num_processes") == "4"

    def test_gpu_count_falls_back_to_nvidia_smi(self, node):
        _, rec = _run(node, env={"SKYPILOT_NUM_GPUS_PER_NODE": None})
        assert _opt(rec["argv"], "--num_processes") == "2"

    def test_no_gpus_fails_before_training(self, node):
        _exe(node[1] / "nvidia-smi", "exit 9\n")
        proc, rec = _run(node, env={"SKYPILOT_NUM_GPUS_PER_NODE": None})
        assert proc.returncode == 1
        assert "no GPUs" in proc.stderr
        assert rec is None


class TestClearml:
    _CREDS = {
        "CLEARML_API_HOST": "https://h",
        "CLEARML_API_ACCESS_KEY": "a",
        "CLEARML_API_SECRET_KEY": "s",
    }

    def test_off_adds_no_tracking_flags(self, node):
        _, rec = _run(node, clearml_mode="off", env=self._CREDS)
        assert "--with_tracking" not in rec["argv"]
        assert rec["env"]["WANDB_MODE"] == "disabled"

    def test_on_tracks_with_wandb_disabled(self, node):
        _, rec = _run(
            node, clearml_mode="on", clearml_project="proj", clearml_task="t1"
        )
        ft = rec["argv"]
        i = ft.index("--with_tracking")
        assert ft[i : i + 6] == [
            "--with_tracking",
            "--report_to",
            "wandb",
            "clearml",
            "--wandb_entity",
            "none",
        ]
        assert rec["env"] == {
            "WANDB_MODE": "disabled",
            "CLEARML_PROJECT": "proj",
            "CLEARML_TASK": "t1",
        }

    def test_on_default_names(self, node):
        _, rec = _run(node, clearml_mode="on")
        assert rec["env"]["CLEARML_PROJECT"] == "openinstruct-sft"
        assert rec["env"]["CLEARML_TASK"] == "finetune"

    def test_auto_needs_all_three_creds(self, node):
        _, rec = _run(node, clearml_mode="auto", env={"CLEARML_API_HOST": "https://h"})
        assert "--with_tracking" not in rec["argv"]
        _, rec = _run(node, clearml_mode="auto", env=self._CREDS)
        assert "--with_tracking" in rec["argv"]

    def test_bad_mode_refused(self, node):
        proc, rec = _run(node, clearml_mode="maybe")
        assert proc.returncode == 1 and rec is None


class TestOutput:
    def test_markers_last_and_absolute(self, node):
        proc, _ = _run(node)
        out = node[3] / "model_output"
        assert _markers(proc) == [
            "GB_STEP_METADATA_KEY:open_instruct_commit GB_STEP_METADATA_VALUE:26de99604002d6a7b25a2c5f2ee880e388a21dca",
            f"GB_ARTIFACT_ID:model GB_ARTIFACT_PATH:{out}",
        ]
        assert proc.stdout.splitlines()[-1].startswith("GB_ARTIFACT_ID:model ")
        assert (out / "config.json").is_file()
        assert not (node[3] / "model_output.partial").exists()

    def test_absolute_output_path(self, node):
        target = node[0] / "shared" / "m"
        proc, _ = _run(node, output_path=str(target) + "/")
        assert f"GB_ARTIFACT_PATH:{target}" in proc.stdout

    def test_rerun_replaces_previous_model(self, node):
        _run(node)
        (node[3] / "model_output" / "old.bin").write_text("x")
        proc, _ = _run(node)
        assert proc.returncode == 0, proc.stderr
        assert not (node[3] / "model_output" / "old.bin").exists()

    def test_stale_partial_is_cleared(self, node):
        stale = node[3] / "model_output.partial"
        (stale / "step_20").mkdir(parents=True)
        proc, _ = _run(node)
        assert proc.returncode == 0
        assert not (node[3] / "model_output" / "step_20").exists()

    def test_refuses_non_model_dir_before_training(self, node):
        d = node[3] / "model_output"
        d.mkdir()
        (d / "precious.txt").write_text("x")
        proc, rec = _run(node)
        assert proc.returncode == 1 and rec is None
        assert "not a previous model" in proc.stderr
        assert (d / "precious.txt").exists()

    @pytest.mark.parametrize(
        "where",
        ["same", "same-trailing-slash", "relative", "via-symlink", "output-inside"],
    )
    def test_refuses_output_overlapping_the_base_model(self, node, where):
        # The base model has a config.json, so it passes the "previous model"
        # check, and a successful run would `rm -rf` it. Refuse before training.
        tmp, _, _, work = node
        base = work / "base"
        base.mkdir()
        (base / "config.json").write_text("{}")
        (tmp / "link").symlink_to(base)
        out = {
            "same": str(base),
            "same-trailing-slash": f"{base}/",
            "relative": "./base",
            "via-symlink": str(tmp / "link"),
            "output-inside": str(base / "sub"),
        }[where]
        proc, rec = _run(node, base_model=str(base), output_path=out)
        assert proc.returncode == 1 and rec is None
        assert "overlaps the base_model" in proc.stderr
        assert (base / "config.json").is_file()

    def test_refuses_base_model_inside_the_output(self, node):
        _, _, _, work = node
        out = work / "model_output"
        (out / "base").mkdir(parents=True)
        (out / "config.json").write_text("{}")
        (out / "base" / "config.json").write_text("{}")
        proc, rec = _run(node, base_model=str(out / "base"), output_path=str(out))
        assert proc.returncode == 1 and rec is None
        assert "overlaps the base_model" in proc.stderr

    def test_refuses_a_base_model_at_the_staging_dir(self, node):
        # <output_path>.partial is cleared before training, so it must not be
        # the base model either.
        _, _, _, work = node
        base = work / "model_output.partial"
        base.mkdir()
        (base / "config.json").write_text("{}")
        proc, rec = _run(node, base_model=str(base))
        assert proc.returncode == 1 and rec is None
        assert "overlaps the base_model" in proc.stderr
        assert (base / "config.json").is_file()

    def test_sibling_with_a_shared_prefix_is_not_an_overlap(self, node):
        _, _, _, work = node
        base = work / "model"
        base.mkdir()
        (base / "config.json").write_text("{}")
        proc, _ = _run(node, base_model=str(base), output_path=str(work / "model2"))
        assert proc.returncode == 0, proc.stderr
        assert (base / "config.json").is_file()

    def test_no_model_saved_means_no_marker(self, node):
        proc, _ = _run(node, env={"FAKE_NO_MODEL": "1"})
        assert proc.returncode == 1
        assert "no config.json" in proc.stderr
        assert _markers(proc) == []

    def test_training_failure_propagates(self, node):
        proc, _ = _run(node, env={"FAKE_FAIL": "1"})
        assert proc.returncode == 3
        assert _markers(proc) == []

    @pytest.mark.parametrize("bad", ["my model", 'a"b'])
    def test_bad_output_id(self, node, bad):
        proc, rec = _run(node, output_id=bad)
        assert proc.returncode == 1 and rec is None
