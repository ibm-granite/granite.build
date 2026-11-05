#!/usr/bin/env python3

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

"""Unit tests for the granite-switch answerability-lora recipe on AWS.

The recipe and its parameters are rendered the way `gb build start` renders
them, and the result is checked for what the AWS port changes: the EFS-backed
environment, the paths on the EFS, and one train target per METHODS entry,
chained so the trains run one after another. The prepare script and the
hf-peft-sft config it shares with the SLURM recipe are covered by
test_answerability_lora_recipe.py.
"""

import pathlib

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_REPO = pathlib.Path(__file__).resolve().parents[3]
_RECIPE = (
    _REPO / "recipes" / "granite-switch" / "skypilot" / "aws" / "answerability-lora"
)
_ENV = (
    _REPO
    / "configurations"
    / "assets"
    / "environments"
    / "skypilot"
    / "aws"
    / "gb-shared"
)
_EFS_ROOT = "/mnt/gb-shared/"


def _params(*params):
    return [f"{k}={v}" for k, v in (p.split("=", 1) for p in params)]


def _build(tmp_path, *params):
    file_params = get_params_from_file(str(_RECIPE / "parameters.yaml"))
    text = apply_parameters(
        (_RECIPE / "build.yaml").read_text(),
        _params(*params),
        file_params,
        str(tmp_path),
    )
    return yaml.safe_load(text)["granite.build"]


def _trains(build):
    return {n: t for n, t in build["targets"].items() if n.startswith("train-")}


def _hps(target):
    (step,) = target["steps"]
    assert step["step_uri"] == "space://steps/train/hf-peft-sft"
    return step["config"]["hf_peft_sft_config"]


def test_runs_in_the_gb_shared_aws_environment(tmp_path):
    build = _build(tmp_path)
    for target in build["targets"].values():
        assert (
            target["environment_uri"] == "space://environments/skypilot/aws/gb-shared"
        )


def test_default_is_one_lora_train(tmp_path):
    build = _build(tmp_path)
    assert list(build["targets"]) == ["prepare", "train-lora"]
    assert build["name"] == "answerability-lora-it1-r1"
    assert _hps(build["targets"]["train-lora"])["method"] == "lora"


@pytest.mark.parametrize("methods", [["alora", "lora"], ["lora", "alora"]])
def test_methods_make_one_train_each_in_order(tmp_path, methods):
    build = _build(tmp_path, f"METHODS={','.join(methods)}", "RUN_ID=r9")
    assert list(build["targets"]) == ["prepare"] + [f"train-{m}" for m in methods]
    assert build["name"] == f"answerability-{'-'.join(methods)}-r9"
    for m in methods:
        assert _hps(build["targets"][f"train-{m}"])["method"] == m


def test_each_train_waits_for_the_one_before(tmp_path):
    build = _build(tmp_path, "METHODS=alora,lora")
    first, second = build["targets"]["train-alora"], build["targets"]["train-lora"]
    assert "after" not in first["inputs"]
    assert second["inputs"]["after"] == {"binding": "train-alora.adapter"}


def test_every_train_reads_the_one_prepared_dataset(tmp_path):
    build = _build(tmp_path, "METHODS=alora,lora")
    for target in _trains(build).values():
        assert target["inputs"]["dataset"] == {"binding": "prepare.dataset"}


def test_methods_tolerate_spaces(tmp_path):
    build = _build(tmp_path, "METHODS= alora , lora ")
    assert list(_trains(build)) == ["train-alora", "train-lora"]


@pytest.mark.parametrize("methods", ["lora,bogus", "lora,lora", ",", ""])
def test_bad_methods_fail_the_render(tmp_path, methods):
    with pytest.raises(
        Exception, match="METHODS_must_be_distinct_entries_of_lora_or_alora"
    ):
        _build(tmp_path, f"METHODS={methods}")


def test_alora_settings_stay_per_train(tmp_path):
    build = _build(tmp_path, "METHODS=alora,lora")
    alora = _hps(build["targets"]["train-alora"])
    lora = _hps(build["targets"]["train-lora"])
    assert alora["alora_invocation_string"] == "<|im_start|>assistant"
    assert alora["args"]["gradient_checkpointing"] is False
    assert "alora_invocation_string" not in lora
    assert lora["args"]["gradient_checkpointing"] is True


def test_each_adapter_has_its_own_dir_on_the_efs(tmp_path):
    build = _build(tmp_path, "METHODS=alora,lora", "RUN_ID=r9")
    for m in ("alora", "lora"):
        target = build["targets"][f"train-{m}"]
        path = _hps(target)["output_path"]
        assert path.startswith(_EFS_ROOT)
        assert path.endswith(f"/answerability-{m}/r9")
        assert target["outputs"]["adapter"]["uri"] == f"env://{path}"
        assert _hps(target)["clearml_task_name"] == f"answerability-{m}-r9"


def test_prepare_writes_the_dataset_to_the_efs(tmp_path):
    prepare = _build(tmp_path, "RUN_ID=r9")["targets"]["prepare"]
    uri = prepare["outputs"]["dataset"]["uri"]
    assert uri.startswith(f"env://{_EFS_ROOT}") and uri.endswith("/r9")


def test_no_slurm_only_settings_remain(tmp_path):
    prepare = _build(tmp_path)["targets"]["prepare"]
    config = prepare["steps"][0]["config"]
    assert "zone" not in config
    assert "sbatch_options" not in config["launcher_config"]
    assert config["launcher_config"]["resources"] == {"cpus": "4+", "memory": "16+"}


def test_instance_type_only_when_set(tmp_path):
    def resources(*params):
        (target,) = _trains(_build(tmp_path, *params)).values()
        return target["steps"][0]["config"]["launcher_config"]["resources"]

    assert resources() == {"accelerators": "H200:8"}
    assert resources("ACCELERATORS=L40S:1", "INSTANCE_TYPE=g6e.4xlarge") == {
        "accelerators": "L40S:1",
        "instance_type": "g6e.4xlarge",
    }


def test_effective_batch_is_32_by_default(tmp_path):
    (target,) = _trains(_build(tmp_path)).values()
    gpus = int(
        target["steps"][0]["config"]["launcher_config"]["resources"][
            "accelerators"
        ].split(":")[1]
    )
    args = _hps(target)["args"]
    assert (
        gpus * args["per_device_train_batch_size"] * args["gradient_accumulation_steps"]
        == 32
    )


def test_environment_pins_every_step_type_to_the_efs_region():
    env = yaml.safe_load((_ENV / "environment.yaml").read_text())
    config = env["config"]
    region = config["shared_filesystem"]["efs"]["region"]
    assert config["shared_workdir"].startswith(
        config["shared_filesystem"]["mount_point"]
    )
    for step in ("hfpull", "command", "hf-peft-sft"):
        infra = config["steps"][step]["launcher_config"]["resources"]["infra"]
        assert infra == f"aws/{region}"
    # hf pulls are their own steps, cached on the EFS: no inline, no local cache.
    (hf,) = [s for s in env["assetstores"] if s["store_uri"].endswith("/hf")]
    assert hf["pull"][0]["config"] == {}
