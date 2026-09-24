import re
from pathlib import Path

import yaml

from gbserver.environment.io.hf_shell import HFPUSH_RUN_SHELL, HFPUSH_UPLOAD_BODY

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


def test_hfpush_upload_body_starts_at_hf_mocked():
    # R1: the inline epilogue (Task 4) exports its own HF_* vars, so the reused
    # body must begin at the hf_mocked() definition — dropping the leading
    # set/trap/echo + Jinja HF_* variable-assignment prologue.
    assert HFPUSH_UPLOAD_BODY.startswith("hf_mocked() {")


def test_hfpush_upload_body_contains_upload_and_marker():
    # Must retain the create_repo + hf upload python and the pushed marker.
    assert "create_repo" in HFPUSH_UPLOAD_BODY
    assert "upload_file" in HFPUSH_UPLOAD_BODY
    assert "upload_folder" in HFPUSH_UPLOAD_BODY
    assert re.search(
        r'echo "Pushed HF URI: \$\{HF_URI\} for binding \$\{BINDING_ID\}"',
        HFPUSH_UPLOAD_BODY,
    )


def test_hfpush_upload_body_has_no_jinja_markers():
    # The reused body must be free of Jinja templating; Task 4 resolves paths
    # at runtime and exports the HF_* vars itself.
    assert "{{" not in HFPUSH_UPLOAD_BODY
