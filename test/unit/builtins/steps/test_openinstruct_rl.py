"""Unit tests for the openinstruct-rl (GRPO) step asset.

openinstruct-rl is a declarative step.yaml under
``configurations/assets/environments/skypilot/lsf/steps/openinstruct-rl/``,
companion to openinstruct-sft. These tests load the shipped asset, render its
``run:`` block via the same Jinja renderer production uses
(``gbserver.utils.template.fill_template``), and assert the GRPO command,
quoting, boolean-flag toggling, service-env exports, and that the monitor's
NEWARTIFACT regex matches the line the run script emits.
"""

import re
from pathlib import Path

import yaml

from gbserver.utils.template import fill_template

REPO_ROOT = Path(__file__).resolve().parents[4]
RL_STEP_YAML = (
    REPO_ROOT
    / "configurations/assets/environments/skypilot/lsf/steps/openinstruct-rl/step.yaml"
)


def _load() -> dict:
    return yaml.safe_load(RL_STEP_YAML.read_text())


class TestOpeninstructRlStep:
    def test_step_yaml_exists(self):
        assert RL_STEP_YAML.exists(), f"{RL_STEP_YAML} does not exist"

    def test_step_yaml_identity_and_outputs(self):
        cfg = _load()
        assert cfg["name"] == "openinstruct-rl"
        assert cfg["type"] == "training"
        assert "rl_config" in cfg["config"]
        assert cfg["outputs"]["optional"]["checkpoint"]["type"] == "model"
        # numeric hyperparameters are intentionally quoted strings (avoid YAML
        # float coercion; they are only interpolated into a shell command)
        assert isinstance(cfg["config"]["rl_config"]["learning_rate"], str)
        assert isinstance(cfg["config"]["rl_config"]["beta"], str)
