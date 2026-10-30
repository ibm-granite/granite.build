"""Contract tests for step-template.yaml.

Every granite-evals family step under steps/eval/ shares this shape and differs only
in `name`, the family's image env var, and its config defaults. `benchmark` selects
which benchmark in the family a target runs; this suite checks every sibling as
well as this one.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve().parent.parent
_STEPS = _HERE.parent.parent
# The granite-evals image families. steps/eval/ also holds eval-example, a
# pre-existing exemplar step that does not follow this contract.
_FAMILIES = (
    "bfcl",
    "bird",
    "ifbench",
    "judged",
    "lmeval",
    "nemoskills",
    "swebench",
    "tau",
    "tbench",
)
_TEMPLATES = sorted(_STEPS / f / "skypilot" / "step-template.yaml" for f in _FAMILIES)


def _load(path):
    return yaml.safe_load(path.read_text())


def _launcher(step):
    return step["environment_configs"]["Skypilot"]["launchers"]["granite"]["config"]


@pytest.fixture(params=_TEMPLATES, ids=lambda p: p.parent.parent.name)
def template(request):
    return request.param


@pytest.fixture
def step(template):
    return _load(template)


@pytest.fixture
def run_script(step):
    return _launcher(step)["run"]


def _as_shell(script):
    script = re.sub(r"\{%.*?%\}", " ", script, flags=re.S)
    return re.sub(r"\{\{.*?\}\}", "X", script, flags=re.S)


def test_this_step_is_found():
    assert _HERE / "step-template.yaml" in _TEMPLATES


def test_name_matches_directory(template, step):
    assert step["name"] == template.parent.parent.name


def test_benchmark_is_a_required_config_field(step, run_script):
    assert step["config"]["benchmark"] == ""
    assert 'BENCHMARK="{{ config.benchmark }}"' in run_script


def test_type_is_a_real_step_type(step):
    from gbcommon.types.stepconfig import StepType

    assert StepType(step["type"]) == StepType.CUSTOM


@pytest.mark.parametrize("strip_optional", [False, True])
def test_run_script_is_valid_bash(run_script, strip_optional):
    script = run_script
    if strip_optional:
        script = re.sub(r"\{%\s*if.*?%\}.*?\{%\s*endif\s*%\}", "", script, flags=re.S)
    result = subprocess.run(
        ["bash", "-n"], input=_as_shell(script), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_script_hygiene(run_script):
    assert "${#" not in run_script  # opens a Jinja comment
    assert "set -o pipefail" in run_script  # the run is behind `| tee`
    assert "unset RANK WORLD_SIZE" in run_script  # vLLM under torch.distributed env


def test_image_is_chosen_at_runtime_not_built(step):
    # The family image is built by steps/eval/Makefile from the Dockerfile one
    # directory up from this template's skypilot/ dir, never by this step.
    assert not (_HERE / "Dockerfile").exists()
    assert (_HERE.parent / "Dockerfile.gpu").exists()
    assert "config.image" in _launcher(step)["image_id"]
    assert _launcher(step)["resources"] == {}


def test_outputs_match_markers(step, run_script):
    # results.json (phase all/score) and generation.json (phase generate); which one a
    # run emits depends on the phase, so both are optional.
    assert step["outputs"] == {
        "optional": {
            "granite_results": {"type": "dataset"},
            "granite_generation": {"type": "dataset"},
        }
    }
    assert set(re.findall(r"GB_ARTIFACT_ID:(\w+)", run_script)) == {
        "granite_results",
        "granite_generation",
    }
    assert run_script.count("GB_ARTIFACT_PATH:${RESULT_FILE}") == 2
    assert 'RESULT_FILE="$OUTPUT_DIR/generation.json"' in run_script


def test_inputs_allow_recipe_ordering_bindings(step):
    # The checkpoint is the config string model_path, and recipes bind only
    # ordering-only inputs (names vary by recipe); nothing is read through
    # bindings.
    assert step["inputs"] == {"allow_unknown": True}


def test_phase_defaults_to_one_job(step, run_script):
    assert step["config"]["phase"] == "all"
    assert '--phase "{{ config.phase }}"' in run_script


def test_monitor_honours_recipe_poll_keys(step):
    mon = step["environment_configs"]["Skypilot"]["monitors"]["skypilot_monitor"]
    assert mon["ref"] == "space://monitors/skypilot"
    assert "config.poll_interval_seconds" in mon["config"]["poll_interval_seconds"]
    assert (
        "config.log_retrieval_interval_seconds"
        in mon["config"]["log_retrieval"]["interval_seconds"]
    )


def test_monitor_regex_is_keyed_on_the_configured_benchmark(step):
    mon = step["environment_configs"]["Skypilot"]["monitors"]["skypilot_monitor"]
    regex = mon["config"]["extra_event_configs"][0]["line_regex"]
    assert "{{ config.benchmark }}" in regex
    assert "vLLM ready at" in regex


def test_every_config_key_reaches_the_script(step, run_script):
    for key in step["config"]:
        if key == "image":
            continue
        assert f"config.{key}" in run_script, key


def test_siblings_share_one_config_contract(step):
    reference = _load(_HERE / "step-template.yaml")
    assert set(step["config"]) == set(reference["config"])
