"""Contract tests for step-template.yaml.

Every steps/sage2-*/ step shares this shape and differs only in `name`, BENCHMARK and
its defaults, so this suite checks every sibling as well as this one.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve().parent.parent
_STEPS = _HERE.parent.parent
_TEMPLATES = sorted(_STEPS.glob("sage2-*/skypilot/step-template.yaml"))


def _load(path):
    return yaml.safe_load(path.read_text())


def _launcher(step):
    return step["environment_configs"]["Skypilot"]["launchers"]["sage2"]["config"]


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


def test_name_matches_directory_and_benchmark(template, step, run_script):
    name = template.parent.parent.name
    assert step["name"] == name
    (bench,) = re.findall(r"^BENCHMARK=(\S+)$", run_script, flags=re.M)
    assert name == f"sage2-{bench}"


def test_type_is_a_real_step_type(step):
    from gbcommon.types.stepconfig import StepType

    assert StepType(step["type"]) == StepType.CUSTOM


@pytest.mark.parametrize("strip_optional", [False, True])
def test_run_script_is_valid_bash(run_script, strip_optional):
    script = run_script
    if strip_optional:
        script = re.sub(r"\{%\s*if.*?%\}.*?\{%\s*endif\s*%\}", "", script, flags=re.S)
    result = subprocess.run(["bash", "-n"], input=_as_shell(script), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_script_hygiene(run_script):
    assert "${#" not in run_script  # opens a Jinja comment
    assert "set -o pipefail" in run_script  # the run is behind `| tee`
    assert "unset RANK WORLD_SIZE" in run_script  # vLLM under torch.distributed env


def test_image_is_chosen_at_runtime_not_built(step):
    assert not (_HERE / "Dockerfile").exists()
    assert "config.sage2_config.image" in _launcher(step)["image_id"]
    assert _launcher(step)["resources"] == {}


def test_single_required_output_matches_marker(step, run_script):
    assert step["outputs"] == {"required": {"sage2_results": {"type": "dataset"}}}
    assert set(re.findall(r"GB_ARTIFACT_ID:(\w+)", run_script)) == {"sage2_results"}
    assert "GB_ARTIFACT_PATH:${RESULT_FILE}" in run_script


def test_monitor_honours_recipe_poll_keys(step):
    mon = step["environment_configs"]["Skypilot"]["monitors"]["skypilot_monitor"]
    assert mon["ref"] == "space://monitors/skypilot"
    assert "config.poll_interval_seconds" in mon["config"]["poll_interval_seconds"]
    assert "config.log_retrieval_interval_seconds" in mon["config"]["log_retrieval"]["interval_seconds"]


def test_every_config_key_reaches_the_script(step, run_script):
    for key in step["config"]["sage2_config"]:
        if key == "image":
            continue
        assert f"config.sage2_config.{key}" in run_script, key


def test_siblings_share_one_config_contract(step):
    reference = _load(_HERE / "step-template.yaml")
    assert set(step["config"]["sage2_config"]) == set(reference["config"]["sage2_config"])
