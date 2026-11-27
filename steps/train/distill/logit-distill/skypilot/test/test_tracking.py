# SPDX-License-Identifier: Apache-2.0

"""Experiment tracking for logit-distill and student-sft: one tool per run (#752).

tracking.py (steps/train/distill/src/gb_steps_post_training/distillation) decides whether
the chosen tool is on, which server it talks to, and whether credentials are present. Its
own self-test is the bulk of the coverage, and it runs here so a suite failure enforces it:
the tool choice, the default servers and user-set ones, netrc parity with wandb 0.31.2,
apply(), and the run-continuity pins.

The rest of this file pins the plumbing: the config keys exist on both trainers'
CustomArguments, this step's template passes them to the renderer only when one is set, and
the renderer emits only the ones that are set.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from gb_steps_post_training.distillation import tracking

_HERE = Path(__file__).resolve().parent
_STEP = _HERE.parent / "step-template.yaml"
_RENDER = _HERE.parent / "src" / "render_gold_config.py"
_PKG = Path(tracking.__file__).resolve().parent
_TRACKING_KEYS = (
    "tracking_tool",
    "clearml_project",
    "clearml_run_name",
    "wandb_entity",
    "wandb_project",
    "wandb_run_name",
)


def test_the_tracking_self_test_passes(capsys):
    assert tracking.self_test() == 0, capsys.readouterr().out


def test_the_key_list_is_trackings_own():
    assert tracking.ALL_CONFIG_FIELDS == _TRACKING_KEYS


def _custom_arguments_fields(path):
    tree = ast.parse(path.read_text())
    [cls] = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "CustomArguments"
    ]
    return {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}


@pytest.mark.parametrize("trainer", ["gold.py", "sft.py"])
def test_every_tracking_key_is_a_trainer_field(trainer):
    """TrlParser rejects an unknown top-level key, so a key the renderer emits must be a field.
    Read from the source, since the trainers need torch and trl to import."""
    missing = set(tracking.ALL_CONFIG_FIELDS) - _custom_arguments_fields(_PKG / trainer)
    assert not missing, f"{trainer} CustomArguments lacks {sorted(missing)}"


def _render_run(tracking_block):
    from jinja2 import Template

    step = yaml.safe_load(_STEP.read_text())
    script = step["environment_configs"]["Skypilot"]["launchers"]["gold"]["config"][
        "run"
    ]
    config = dict(step["config"])
    config["tracking"] = dict(config["tracking"], **tracking_block)
    bindings = {
        "student": {"binding": {"path": "/models/student"}},
        "teacher": {"binding": {"path": "/models/teacher"}},
        "corpus": {"binding": {"path": "/data/corpus/train.jsonl"}},
    }
    return Template(script).render(config=config, bindings=bindings)


class TestTemplate:
    def test_tracking_defaults_to_off(self):
        step = yaml.safe_load(_STEP.read_text())
        assert step["config"]["tracking"] == {
            "tool": "",
            "clearml_project": "",
            "clearml_run_name": "",
            "wandb_entity": "",
            "wandb_project": "",
            "wandb_run_name": "",
        }

    def test_nothing_set_passes_no_tracking_flag(self):
        assert "--tracking-tool" not in _render_run({})
        assert "--wandb-project" not in _render_run({})

    def test_a_chosen_tool_passes_every_tracking_flag(self):
        script = _render_run(
            {"tool": "wandb", "wandb_entity": "ibm", "wandb_project": "kd"}
        )
        for flag, value in (
            ("--tracking-tool", "wandb"),
            ("--wandb-entity", "ibm"),
            ("--wandb-project", "kd"),
            ("--clearml-project", ""),
        ):
            assert f'{flag} "{value}"' in script

    def test_keys_without_a_tool_still_reach_the_trainer(self):
        """So tracking.py can refuse them: dropping them here would turn an expressed intent
        to track into a silent off."""
        script = _render_run({"clearml_project": "d"})
        assert '--clearml-project "d"' in script and '--tracking-tool ""' in script

    def test_the_rendered_script_is_still_valid_bash(self):
        script = _render_run({"tool": "clearml", "clearml_project": "d"})
        result = subprocess.run(
            ["bash", "-n"], input=script, capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr


class TestRenderer:
    _REQUIRED = [
        "--model-name-or-path",
        "/m/student",
        "--teacher-model-name-or-path",
        "/m/teacher",
        "--dataset-name",
        "/d/train.jsonl",
    ]

    def _render(self, tmp_path, *extra):
        out = tmp_path / "gold.yaml"
        cmd = [sys.executable, str(_RENDER), "--output", str(out), "--total-nodes", "1"]
        result = subprocess.run(
            cmd + self._REQUIRED + list(extra),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return yaml.safe_load(out.read_text())

    def test_no_tracking_key_is_emitted_when_none_is_set(self, tmp_path):
        cfg = self._render(tmp_path)
        assert not set(_TRACKING_KEYS) & set(cfg)

    def test_set_keys_are_emitted_stripped_and_empty_ones_are_not(self, tmp_path):
        cfg = self._render(
            tmp_path,
            "--tracking-tool",
            " clearml ",
            "--clearml-project",
            "distill",
            "--wandb-entity",
            "  ",
        )
        assert cfg["tracking_tool"] == "clearml"
        assert cfg["clearml_project"] == "distill"
        assert "wandb_entity" not in cfg and "clearml_run_name" not in cfg
