# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the lm-engine-pretrain step-template.yaml.

These cover the step's promises to a recipe: that the template loads, that every
documented ``config`` key is there with its documented default, and that the
SkyPilot environment config is the shape gbserver validates. The launcher's
``run:`` block is rendered and exercised in ``test_skypilot_render.py`` and
``test_run_pretrain.py``. Mode 1 only (no build).
"""

import pathlib

import pytest
import yaml

_STEP_DIR = pathlib.Path(__file__).resolve().parent.parent
_TEMPLATE_PATH = _STEP_DIR / "step-template.yaml"
_TEMPLATE = yaml.safe_load(_TEMPLATE_PATH.read_text())
_LAUNCHER_NAME = "lm-engine-pretrain"

# The schema this step promises its recipes, as (key, default).
_CONFIG_DEFAULTS = {
    "image": "",
    "phase": "train",
    "train_dir": "",
    "lme": {},
    "ckpt_rungs": "",
    "emit_checkpoint_artifacts": False,
    "checkpoint_watch_interval_seconds": 60,
    "require_progress": True,
    "resume_from_checkpoint_dir": "",
    "nccl_env": {},
    "nccl_debug": "WARN",
    "wandb": {"project": "", "mode": "offline", "name": "", "entity": ""},
}


def _sky() -> dict:
    return _TEMPLATE["environment_configs"]["Skypilot"]


def _launcher() -> dict:
    return _sky()["launchers"][_LAUNCHER_NAME]["config"]


def test_the_template_loads_and_names_the_step():
    assert _TEMPLATE["name"] == _LAUNCHER_NAME
    assert _TEMPLATE["version"] == "0.1.0"


def test_type_is_a_real_step_type():
    from gbcommon.types.stepconfig import StepType

    assert StepType(_TEMPLATE["type"]) == StepType.TRAINING


def test_config_holds_exactly_the_documented_keys():
    assert set(_TEMPLATE["config"]) == set(_CONFIG_DEFAULTS)


@pytest.mark.parametrize("key", sorted(_CONFIG_DEFAULTS))
def test_each_config_key_has_its_documented_default(key):
    """Launcher text renders strictly, so every leaf key needs a default here."""
    assert _TEMPLATE["config"][key] == _CONFIG_DEFAULTS[key]


def test_every_config_key_is_documented_in_usage():
    """A recipe author reads USAGE.md, not the template."""
    usage = (_STEP_DIR / "USAGE.md").read_text()
    for key in _CONFIG_DEFAULTS:
        if key == "wandb":  # documented one leaf per row, as `wandb.<leaf>`
            continue
        assert f"`{key}`" in usage, key
    for leaf in _CONFIG_DEFAULTS["wandb"]:
        assert f"`wandb.{leaf}`" in usage, leaf


def test_the_environment_config_validates_as_gbserver_reads_it():
    from gbserver.types.stepconfig import StepEnvironmentTypeConfig

    env = StepEnvironmentTypeConfig.model_validate(_sky())
    assert env.default_launcher == _LAUNCHER_NAME
    assert _LAUNCHER_NAME in env.launchers


def test_the_launcher_is_a_skypilot_launcher():
    assert _sky()["launchers"][_LAUNCHER_NAME]["type"] == "skypilot"


def test_the_image_comes_from_config_and_is_not_built_here():
    """No Dockerfile next to the Makefile: the family image is built one level up."""
    assert not (_STEP_DIR / "Dockerfile").exists()
    assert "config.image" in _launcher()["image_id"]
    # The template requests no resources itself; the build's launcher_config does.
    assert _launcher()["resources"] == {}


@pytest.mark.parametrize(
    ("image", "expected"),
    [("quay.io/o/lm-engine:v1", "docker:quay.io/o/lm-engine:v1"), ("", "")],
)
def test_image_id_renders_from_config_image(image, expected):
    from gbserver.utils.template import fill_template

    cfg = dict(_TEMPLATE["config"], image=image)
    assert fill_template(_launcher()["image_id"], {"config": cfg}, strict=True) == (
        expected
    )


def test_src_is_file_mounted_and_shipped_verbatim():
    """The Python helpers reach the run through file_mounts, un-Jinja'd."""
    assert _launcher()["file_mounts"] == {"src": "src"}
    gbignore = (_STEP_DIR / ".gbignore").read_text().splitlines()
    assert "src/**" in gbignore
    assert "**/*.md" in gbignore


def test_the_final_checkpoint_is_the_only_declared_output():
    """The ckpt_rungs outputs are named by the build, not by the step."""
    assert _TEMPLATE["outputs"] == {"optional": {"checkpoint": {"type": "model"}}}


def test_inputs_are_unnamed_and_allow_recipe_ordering_bindings():
    assert _TEMPLATE["inputs"] == {"allow_unknown": True}
