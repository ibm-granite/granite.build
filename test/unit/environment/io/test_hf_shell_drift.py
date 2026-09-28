import re
from pathlib import Path

import yaml

from gbserver.environment.io.hf_shell import HFPUSH_RUN_SHELL

_STEP_YAML = (
    Path(__file__).resolve().parents[4]
    / "src/gbserver/builtins/steps/skypilot/hfpush/step.yaml"
)


def _step_yaml_run_block() -> str:
    doc = yaml.safe_load(_STEP_YAML.read_text())
    return doc["environment_configs"]["Skypilot"]["launchers"]["hfpush"]["config"][
        "run"
    ]


def test_hfpush_shell_matches_step_yaml_run_block():
    # The single-source shell must remain byte-identical to the shipped
    # skypilot hfpush step.yaml run block (spec §7 drift guard).
    assert HFPUSH_RUN_SHELL == _step_yaml_run_block()


def test_hfpush_shell_emits_pushed_marker():
    assert re.search(
        r'echo "Pushed HF URI: \$\{HF_URI\} for binding \$\{BINDING_ID\}"',
        HFPUSH_RUN_SHELL,
    )
