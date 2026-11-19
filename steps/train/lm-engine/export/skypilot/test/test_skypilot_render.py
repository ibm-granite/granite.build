# SPDX-License-Identifier: Apache-2.0

"""Render tests for the lm-engine-export launcher.

The ``run:`` block is filled with gbserver's strict renderer
(``fill_template(..., strict=True)``, the launcher-config fill in
``targetsteprun.py``: StrictUndefined plus the ``raise_error`` global), and the result
is compared against ``golden_run_scripts.json`` for three shapes: a muP ``granite``
rung with full expectations, a bare minimum ``llama`` export of the latest rung with no
expectations, and a ``granitemoehybrid`` rung. The golden file makes an accidental
change to the job's environment visible in review.

What the *script* then does with that environment is ``test_run_export.py``'s job; this
suite stops at the handoff. Mode 1 only (no build).
"""

import json
import pathlib
import re
import subprocess

import pytest
import yaml

from gbserver.types.stepconfig import StepEnvironmentTypeConfig
from gbserver.utils.template import fill_objtemplate, fill_template

_STEP_DIR = pathlib.Path(__file__).resolve().parent.parent
_TEMPLATE = yaml.safe_load((_STEP_DIR / "step-template.yaml").read_text())
_SKY = _TEMPLATE["environment_configs"]["Skypilot"]
_LAUNCHER = _SKY["launchers"]["lm-engine-export"]["config"]
_GOLDEN_PATH = pathlib.Path(__file__).resolve().parent / "golden_run_scripts.json"
_GOLDEN = json.loads(_GOLDEN_PATH.read_text())

_IMAGE = "quay.io/granitebuild/gb-step-lm-engine:v0.1.0"

# The muP expectations a `granite` recipe passes back, as USAGE.md documents them.
_MUP_EXPECTED = {
    "rope_theta": 10000000,
    "vocab_size": 100352,
    "tie_word_embeddings": True,
    "bos_token_id": 100257,
    "eos_token_id": 100257,
    "pad_token_id": 100256,
    "sequence_length": 8192,
    "m_emb": 12,
    "m_residual": 0.263,
    "m_width": 4,
    "attention_multiplier": 0.015625,
}

# The config each golden case renders from, as overrides on the template defaults.
# train_dir, output_dir and tokenizer_path are required by the render, so every case
# sets all three.
_CASES = {
    "granite_rung_with_expectations": {
        "image": _IMAGE,
        "train_dir": "/shared/runs/g4-350m/train",
        "iteration": "4000",
        "output_dir": "/shared/runs/g4-350m/hf/4000",
        "model_type": "granite",
        "tokenizer_path": "/shared/tokenizers/granite-4",
        "expected": _MUP_EXPECTED,
        "parity_tolerance": 0.01,
    },
    "llama_latest_no_expectations": {
        "image": _IMAGE,
        "train_dir": "/shared/runs/g4-350m/train",
        "output_dir": "/shared/runs/g4-350m/hf/latest",
        "model_type": "llama",
        "tokenizer_path": "/shared/tokenizers/granite-4",
    },
    "granitemoehybrid_rung": {
        "image": _IMAGE,
        "train_dir": "/shared/runs/g4-moe/train",
        "iteration": "2000",
        "output_dir": "/shared/runs/g4-moe/hf/2000",
        "model_type": "granitemoehybrid",
        "tokenizer_path": "/shared/tokenizers/granite-4",
        "expected": {"rope_theta": 10000000, "vocab_size": 100352},
    },
}


def _cfg(over: dict | None = None) -> dict:
    cfg = json.loads(json.dumps(_TEMPLATE["config"]))
    cfg.update(over or {})
    return cfg


def _render(over: dict | None = None) -> str:
    return fill_template(_LAUNCHER["run"], {"config": _cfg(over)}, strict=True)


def _render_error(over: dict | None = None) -> str:
    """The ``raise_error`` message a failing fill carries."""
    with pytest.raises(RuntimeError) as exc:
        fill_objtemplate(_LAUNCHER, {"config": _cfg(over)}, strict=True)
    err = exc.value
    while err.__cause__ is not None:
        err = err.__cause__
    assert isinstance(err, ValueError), repr(err)
    return str(err)


# ── the goldens ────────────────────────────────────────────────────────────────


def test_the_golden_file_covers_exactly_the_cases():
    assert set(_GOLDEN) == set(_CASES)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_rendered_run_block_matches_its_golden(case):
    """Regenerate with `python test/golden_run_scripts.py` after a deliberate change."""
    assert _render(_CASES[case]).splitlines() == _GOLDEN[case]


@pytest.mark.parametrize("case", sorted(_CASES))
def test_every_rendered_run_block_is_valid_bash(case):
    proc = subprocess.run(["bash", "-n"], input=_render(_CASES[case]), text=True)
    assert proc.returncode == 0


def test_an_undefined_value_fails_the_render():
    cfg = _cfg(_CASES["granite_rung_with_expectations"])
    del cfg["parity_tolerance"]
    with pytest.raises(RuntimeError):
        fill_template(_LAUNCHER["run"], {"config": cfg}, strict=True)


# ── what the environment says ──────────────────────────────────────────────────


def test_every_cli_input_reaches_the_script_as_an_export():
    run = _render(_CASES["granite_rung_with_expectations"])
    for line in (
        "export LME_TRAIN_DIR='/shared/runs/g4-350m/train'",
        "export LME_OUTPUT_DIR='/shared/runs/g4-350m/hf/4000'",
        "export LME_TOKENIZER_PATH='/shared/tokenizers/granite-4'",
        "export LME_MODEL_TYPE='granite'",
        "export LME_ITERATION='4000'",
        "export LME_PARITY_TOLERANCE='0.01'",
        f"export LME_IMAGE='{_IMAGE}'",
    ):
        assert line in run, line
    assert "export LME_EXPECTED_JSON='" in run


def test_the_code_facts_are_the_steps_own_defaults():
    run = _render(_CASES["llama_latest_no_expectations"])
    assert "export HOME=/tmp" in run
    assert "export TOKENIZERS_PARALLELISM=false" in run


def test_nothing_nccl_or_torchrun_shaped_is_rendered():
    """A single-process CPU export: no distributed topology, no rendezvous."""
    run = _render(_CASES["granite_rung_with_expectations"])
    for absent in ("NCCL", "torchrun", "SKYPILOT_NODE", "CUDA_", "nproc_per_node"):
        assert absent not in run, absent


def test_the_run_block_hands_off_to_the_entry_point():
    assert _render(_CASES["granite_rung_with_expectations"]).endswith(
        "bash ./src/run_export.sh"
    )
    assert (_STEP_DIR / "src" / "run_export.sh").is_file()


def test_the_shell_never_emits_the_artifact_line_itself():
    """export.py prints it once, only after every check passed."""
    for case in _CASES.values():
        assert "GB_ARTIFACT_ID" not in _render(case)
    # The script's comments explain the rule; no line of its code prints one.
    for line in (_STEP_DIR / "src" / "run_export.sh").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            assert "GB_ARTIFACT_ID" not in line, line


def test_the_image_is_requested_as_a_docker_image_id():
    rendered = fill_objtemplate(
        _LAUNCHER,
        {"config": _cfg(_CASES["granite_rung_with_expectations"])},
        strict=True,
    )
    assert rendered["image_id"] == f"docker:{_IMAGE}"
    # The step requests no resources itself; the recipe's launcher_config does.
    assert rendered["resources"] == {}
    assert rendered["file_mounts"] == {"src": "src"}


# ── render-time refusals ───────────────────────────────────────────────────────


@pytest.mark.parametrize("key", ["train_dir", "output_dir", "tokenizer_path"])
def test_a_missing_required_path_is_refused_when_the_step_is_created(key):
    message = _render_error(dict(_CASES["granite_rung_with_expectations"], **{key: ""}))
    assert f"config.{key} is required" in message


def test_an_unsupported_model_type_is_refused():
    assert "config.model_type must be one of" in _render_error(
        dict(_CASES["granite_rung_with_expectations"], model_type="granitemoe")
    )


# ── what bash actually receives ────────────────────────────────────────────────

_STUB = (
    "#!/usr/bin/env bash\n"
    "python3 -c 'import json,os; print(json.dumps({k: v for k, v in "
    'os.environ.items() if k.startswith("LME_") or k in ("HOME", '
    '"TOKENIZERS_PARALLELISM")}))\'\n'
)


def _environment_of(over: dict, tmp_path: pathlib.Path) -> dict:
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src" / "run_export.sh").write_text(_STUB)
    proc = subprocess.run(
        ["bash", "-c", "set -eu\n" + _render(over)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_quotes_and_json_survive_the_shell(tmp_path):
    over = dict(
        _CASES["granite_rung_with_expectations"],
        output_dir="/shared/it's/hf/4000",
        tokenizer_path="/shared/tok it's",
        expected=dict(_MUP_EXPECTED, notes='say "hi" $HOME `x` it\'s'),
    )
    env = _environment_of(over, tmp_path)
    assert env["LME_OUTPUT_DIR"] == "/shared/it's/hf/4000"
    assert env["LME_TOKENIZER_PATH"] == "/shared/tok it's"
    # The JSON reaches the script as JSON, with every metacharacter intact.
    assert json.loads(env["LME_EXPECTED_JSON"]) == over["expected"]
    assert env["LME_EXPECTED_JSON"].count("$HOME") == 1
    assert "`x`" in env["LME_EXPECTED_JSON"]


def test_an_empty_expectation_reaches_the_script_as_an_empty_object(tmp_path):
    env = _environment_of(_CASES["llama_latest_no_expectations"], tmp_path)
    assert env["LME_EXPECTED_JSON"] == "{}"
    assert env["LME_ITERATION"] == ""
    assert env["HOME"] == "/tmp"
    assert env["TOKENIZERS_PARALLELISM"] == "false"


# ── the monitor ────────────────────────────────────────────────────────────────


def test_the_launcher_wires_up_the_shared_skypilot_monitor():
    """Without it nothing scrapes the artifact line, so hf_model never binds."""
    assert _SKY["launchers"]["lm-engine-export"]["monitors"] == ["skypilot_monitor"]
    monitor = _SKY["monitors"]["skypilot_monitor"]
    assert monitor["ref"] == "space://monitors/skypilot"
    StepEnvironmentTypeConfig.model_validate(_SKY)


def test_the_monitor_reports_progress_from_the_job_log():
    events = _SKY["monitors"]["skypilot_monitor"]["config"]["extra_event_configs"]
    assert len(events) == 1
    assert events[0]["event_type"] == "WORKLOAD_STATUS_EVENT"
    regex = events[0]["line_regex"]
    # The lines the script and the driver really print do match.
    assert re.search(regex, "lm-engine-export: iteration='4000' model_type='granite'")
    assert re.search(regex, "export: stage 1/5 unshard: python -m lm_engine")
    assert not re.search(regex, "some unrelated log line")
