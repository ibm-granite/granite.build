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

"""fms_hf_tuning_job against the real fms-hf-tuning parser.

Skipped unless `tuning` (fms-hf-tuning) imports; the repo venv has none. Run in
the CPU venv built from the image pins (README.md, "Tests"). Catches upstream
parser drift: a renamed key, a changed type, a new dataclass.
"""

import base64
import json
import pathlib

import fms_hf_tuning_job as job
import pytest
import yaml

pytest.importorskip("tuning")

from tuning.sft_trainer import (  # noqa: E402  pylint: disable=wrong-import-position
    get_parser,
)

_STEP = pathlib.Path(__file__).resolve().parents[1]


def _dests():
    return {a.dest for a in get_parser()._actions}  # pylint: disable=protected-access


def test_extra_args_are_the_non_dataclass_arguments():
    import dataclasses  # pylint: disable=import-outside-toplevel

    p = get_parser()
    fields = {
        f.name for t in p.dataclass_types for f in dataclasses.fields(t) if f.init
    }
    assert _dests() - fields - {"help"} == set(job.EXTRA_ARGS)


def test_owned_keys_exist_upstream_except_the_wrapper_key():
    # accelerate_launch_args belongs to upstream's build/accelerate_launch.py,
    # which the step does not use; the rest are real parser arguments.
    assert set(job.OWNED) - _dests() == {"accelerate_launch_args"}


def test_step_defaults_are_parser_arguments():
    assert set(job.DEFAULTS) <= _dests()


def _job(args=None, **over):
    kw = dict(
        base_model="/m",
        output_dir="/tmp/scratch",
        save_model_dir="/tmp/out.partial",
        peft_method="none",
        data_path="/ds",
        data_config_path="",
        clearml=False,
        clearml_project="",
        clearml_task="",
    )
    kw.update(over)
    return job.merge_job(args or {}, **kw)


def test_strict_check_accepts_a_digit_style_config():
    args = {
        "learning_rate": "1e-06",
        "logging_steps": "1",
        "max_steps": "15",
        "save_steps": "10",
        "save_strategy": "steps",
        "dataset_text_field": "formatted_chat_data",
        "response_template": "\n### Answer:",
        "optim": "adamw_torch",
        "adam_beta1": 0.9,
        "weight_decay": 0.1,
        "use_flash_attn": True,
        "r": 8,
        "lora_alpha": 8,
        "target_modules": "all-linear",
        "add_special_tokens": ["<|start_of_role|>"],
    }
    out = job.strict_check(_job(args, peft_method="lora"))
    assert out["learning_rate"] == 1e-06 and out["max_steps"] == 15
    assert out["target_modules"] == ["all-linear"] and out["peft_method"] == "lora"


def test_strict_check_accepts_alora_and_clearml():
    out = job.strict_check(
        _job(
            {"alora_invocation_string": "<|im_start|>assistant", "r": 8},
            peft_method="alora",
            clearml=True,
        )
    )
    assert out["trackers"] == ["clearml"] and out["clearml_task"] == "sft-alora"


@pytest.mark.parametrize(
    "args,needle",
    [
        ({"lerning_rate": 1}, "is not an fms-hf-tuning argument"),
        ({"learning_rate": "abc"}, "is not a valid value"),
        ({"packing": "maybe"}, "is not a valid value"),
        ({"max_steps": 1.5}, "is not a valid value"),
        ({"lr_scheduler_type": "nope"}, "is not one of"),
        ({"seed": [1, 2]}, "takes one value"),
    ],
)
def test_strict_check_refuses(args, needle):
    with pytest.raises(job.StepError) as exc:
        job.strict_check(_job(args))
    assert needle in str(exc.value)


def _fixture_args():
    found = []
    for path in sorted((_STEP / "test-data").rglob("build.yaml")):
        text = path.read_text()
        if "$${" in text:  # the exemplar's parameters are not YAML values
            text = text.replace('"$${', '"').replace("$${", "")
        build = yaml.safe_load(text)
        for name, target in build["granite.build"]["targets"].items():
            cfg = target["steps"][0]["config"]["fms_hf_tuning_config"]
            found.append((f"{path.parent.name}/{name}", cfg))
    return found


@pytest.mark.parametrize(
    "name,cfg", _fixture_args(), ids=[n for n, _ in _fixture_args()]
)
def test_every_shipped_args_map_passes_the_strict_check(name, cfg):
    data_config = cfg.get("data_config") or {}
    job.strict_check(
        _job(
            cfg.get("args") or {},
            peft_method=cfg.get("peft_method", "none"),
            data_path="" if data_config else "/ds",
            data_config_path="/w/dc.json" if data_config else "",
        )
    )


def test_build_cli_writes_the_job_and_data_config(tmp_path):
    def b64(o):
        return base64.b64encode(json.dumps(o).encode()).decode()

    dc = {"datasets": [{"name": "d", "data_paths": ["/staged/ds"]}]}
    job.main(
        [
            "build",
            "--args-b64",
            b64({"max_steps": "3"}),
            "--data-config-b64",
            b64(dc),
            "--data-path",
            "",
            "--base-model",
            "/m",
            "--peft-method",
            "lora",
            "--output-dir",
            "/o",
            "--save-model-dir",
            "/s",
            "--clearml",
            "on",
            "--clearml-project",
            "p",
            "--clearml-task",
            "",
            "--job-out",
            str(tmp_path / "j.json"),
            "--data-config-out",
            str(tmp_path / "dc.json"),
        ]
    )
    written = json.loads((tmp_path / "j.json").read_text())
    assert written["max_steps"] == 3 and written["peft_method"] == "lora"
    assert written["data_config_path"] == str(tmp_path / "dc.json")
    assert written["clearml_project"] == "p" and written["clearml_task"] == "sft-lora"
    assert json.loads((tmp_path / "dc.json").read_text()) == dc
