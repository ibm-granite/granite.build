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

"""Render test for the bash fm-tune step's launcher env mapping.

Renders the launcher ``env:`` with gbserver's strict renderer,
``gbserver.utils.template.fill_template(..., strict=True)``: the fill
``targetsteprun.py`` applies to the launcher config at step creation, with
StrictUndefined and the ``raise_error`` global. Checks every value lands in the
right FMTUNE_* var. Mode 1 only.
"""

import json
import pathlib

import pytest
import yaml

from gbserver.utils.template import fill_objtemplate, fill_template

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_REPO = _STEP_DIR.parents[3]
_TEMPLATE = yaml.safe_load((_STEP_DIR / "step-template.yaml").read_text())
_LAUNCHER = _TEMPLATE["environment_configs"]["Bash"]["launchers"]["fm-tune"]
_DATASET = {"dataset": {"binding": {"path": "/staged/data it's"}}}
# The template defaults leave these empty, which the render-time checks reject.
_VALID = {
    "model": "org/tiny",
    "train_file": "d_train.jsonl",
    "validation_file": "d_validation.jsonl",
    "source": {
        "path": "/fm-tune",
        "repo": "",
        "ref": "main",
        "subdir": "autotunex/src/fm-tune",
    },
}


def _fill(cfg_over: dict | None = None, bindings: dict | None = None) -> dict:
    """Fill the whole launcher config, as targetsteprun.py does at step creation."""
    cfg = json.loads(json.dumps(_TEMPLATE["config"]))
    cfg["fm_tune_config"].update(_VALID)
    cfg["fm_tune_config"].update(cfg_over or {})
    ctx = {"config": cfg, "bindings": _DATASET if bindings is None else bindings}
    return fill_objtemplate(_LAUNCHER, ctx, strict=True)


def _env(cfg_over: dict | None = None, bindings: dict | None = None) -> dict:
    return _fill(cfg_over, bindings)["config"]["env"]


def _render_error(cfg_over: dict, bindings: dict | None = None) -> str:
    """The raise_error message a failing fill carries (fill_template wraps it)."""
    with pytest.raises(RuntimeError) as exc:
        _fill(cfg_over, bindings)
    err = exc.value
    while err.__cause__ is not None:
        err = err.__cause__
    assert isinstance(err, ValueError), repr(err)
    return str(err)


def test_defaults_render_strictly():
    env = _env()
    assert env["FMTUNE_DATASET_DIR"] == "/staged/data it's"
    assert env["FMTUNE_BASE_MODEL_DIR"] == ""
    assert env["FMTUNE_TUNING_ALGO"] == "lora"
    assert env["FMTUNE_AUTOTUNE"] == "true"
    assert env["FMTUNE_HPO_CONFIG_JSON"] == "{}"
    assert env["FMTUNE_CONFIG_NAME"] == ""
    # Left empty: fmtune_run.sh defaults them from LLMB_BASH_* (env is unexpanded).
    assert (
        env["FMTUNE_OUTPUT_DIR"]
        == env["FMTUNE_CACHE_DIR"]
        == env["FMTUNE_PYTHON"]
        == ""
    )


def test_an_undefined_value_fails_the_render():
    """The strict fill raises on an undefined variable instead of rendering ''."""
    cfg = json.loads(json.dumps(_TEMPLATE["config"]))
    del cfg["fm_tune_config"]["model"]
    with pytest.raises(RuntimeError):
        fill_template(
            _LAUNCHER["config"]["env"]["FMTUNE_MODEL"], {"config": cfg}, strict=True
        )


@pytest.mark.parametrize(
    "name", ["train.jsonl", "sub/dir/train.jsonl", "a..b.jsonl", "./t.jsonl"]
)
def test_a_split_file_inside_the_dataset_renders(name):
    env = _env({"train_file": name, "validation_file": name})
    assert env["FMTUNE_TRAIN_FILE"] == env["FMTUNE_VALIDATION_FILE"] == name


def test_a_bound_base_model_satisfies_the_model_check():
    b = dict(_DATASET, base_model={"binding": {"path": "/staged/bm"}})
    assert _env({"model": ""}, bindings=b)["FMTUNE_BASE_MODEL_DIR"] == "/staged/bm"


def test_the_check_key_renders_to_nothing():
    """The checks live in a launcher-config key of their own; nothing reads it."""
    assert _fill()["config"]["validate"] == ""


_VAR = {
    "seed": "FMTUNE_SEED",
    "rl_algo": "FMTUNE_RL_ALGO",
    "tuning_algo": "FMTUNE_TUNING_ALGO",
    "backend": "FMTUNE_BACKEND",
    "extra_args": "FMTUNE_EXTRA_ARGS",
}


@pytest.mark.parametrize(
    "key", ["seed", "rl_algo", "tuning_algo", "backend", "extra_args"]
)
def test_a_null_value_renders_empty(key):
    """AutoTuneX's LaunchContext can emit null; it must not reach the script as 'None'."""
    # tuning_algo alone null would trip the "both empty or none" check.
    over = {key: None, "rl_algo": "grpo"} if key == "tuning_algo" else {key: None}
    assert _env(over)[_VAR[key]] == ""


def test_null_hpo_config_and_autotune_render_empty():
    env = _env({"hpo_config": None, "autotune": None})
    # Empty: the script then uses the preset and its default autotune=true.
    assert env["FMTUNE_HPO_CONFIG_JSON"] == env["FMTUNE_AUTOTUNE"] == ""


def test_zero_and_false_are_not_treated_as_null():
    env = _env({"seed": 0, "autotune": False})
    assert (env["FMTUNE_SEED"], env["FMTUNE_AUTOTUNE"]) == ("0", "false")


def test_null_python_and_model_with_a_bound_base_model_render_empty():
    b = dict(_DATASET, base_model={"binding": {"path": "/staged/bm"}})
    env = _env({"python": None, "model": None}, bindings=b)
    assert env["FMTUNE_PYTHON"] == env["FMTUNE_MODEL"] == ""


def test_bound_base_model_renders_its_path():
    b = dict(_DATASET, base_model={"binding": {"path": "/staged/bm"}})
    assert _env(bindings=b)["FMTUNE_BASE_MODEL_DIR"] == "/staged/bm"


def test_hpo_config_renders_as_json():
    hpo = {"tune_config": {"num_samples": {"default": 2}}, "x": "it's"}
    assert json.loads(_env({"hpo_config": hpo})["FMTUNE_HPO_CONFIG_JSON"]) == hpo


def test_autotune_false_renders_lowercase():
    assert _env({"autotune": False})["FMTUNE_AUTOTUNE"] == "false"


def test_nested_source_and_bridge_keys_map():
    env = _env(
        {
            "source": {
                "path": "",
                "repo": "https://g/r.git",
                "ref": "v1",
                "subdir": "a/b",
            },
            "autotunex": {"server_url": "http://h:1", "job_id": "j"},
        }
    )
    assert (env["FMTUNE_SOURCE_REPO"], env["FMTUNE_SOURCE_REF"]) == (
        "https://g/r.git",
        "v1",
    )
    assert env["FMTUNE_SOURCE_SUBDIR"] == "a/b"
    assert (env["FMTUNE_SERVER_URL"], env["FMTUNE_JOB_ID"]) == ("http://h:1", "j")


def test_every_contract_var_is_mapped():
    expected = {
        "FMTUNE_DATASET_DIR",
        "FMTUNE_BASE_MODEL_DIR",
        "FMTUNE_MODEL",
        "FMTUNE_SOURCE_PATH",
        "FMTUNE_SOURCE_REPO",
        "FMTUNE_SOURCE_REF",
        "FMTUNE_SOURCE_SUBDIR",
        "FMTUNE_EXTRA",
        "FMTUNE_TRAIN_FILE",
        "FMTUNE_VALIDATION_FILE",
        "FMTUNE_TUNING_ALGO",
        "FMTUNE_RL_ALGO",
        "FMTUNE_RUN_NAME",
        "FMTUNE_AUTOTUNE",
        "FMTUNE_BACKEND",
        "FMTUNE_SEED",
        "FMTUNE_CONFIG_PRESET",
        "FMTUNE_CONFIG_NAME",
        "FMTUNE_HPO_CONFIG_JSON",
        "FMTUNE_OUTPUT_DIR",
        "FMTUNE_CACHE_DIR",
        "FMTUNE_PYTHON",
        "FMTUNE_EXTRA_ARGS",
        "FMTUNE_SERVER_URL",
        "FMTUNE_JOB_ID",
    }
    assert set(_LAUNCHER["config"]["env"]) == expected


def test_inputs_and_outputs_are_declared():
    assert set(_TEMPLATE["inputs"]["required"]) == {"dataset"}
    assert set(_TEMPLATE["inputs"]["optional"]) == {"base_model"}
    assert set(_TEMPLATE["outputs"]["required"]) == {"model"}


def test_script_path_points_at_the_shipped_src():
    assert _LAUNCHER["config"]["script_path"] == "../../src/fmtune_run.sh"


def test_template_turns_off_the_wrappers_output_dir_scan():
    assert _TEMPLATE["config"]["bash"] == {"skip_finding_output_artifacts": True}


def test_rendered_wrapper_emits_no_marker_of_its_own():
    """The wrapper's subdir scan prints GB_ARTIFACT_ID before it checks the exit
    code, so it fired on failure (build a2313afe). With the template's config.bash
    knob, the wrapper renders the skip branch; fmtune_run.sh's marker is the only one.

    Renders with gbserver's file renderer in strict mode, as the final pass in
    targetsteprun.py (fill_templates_in_dir(..., strict=True)) does.
    """
    from gbserver.utils.template import fill_template as fill_file_template

    wrapper = next(
        (
            _REPO
            / "src"
            / "gbserver"
            / "builtins"
            / "steps"
            / "gbstep"
            / "bash_scripts"
        ).glob("*/llmb_bash_wrapper.sh")
    )
    cfg = json.loads(json.dumps(_TEMPLATE["config"]))
    out = fill_file_template(wrapper.read_text(), {"config": cfg}, strict=True)
    assert "skip making artifacts out of ${LLMB_BASH_OUTPUT_DIR}" in out
    assert "GB_ARTIFACT_ID" not in out
    assert 'find "${LLMB_BASH_OUTPUT_DIR}"' not in out
