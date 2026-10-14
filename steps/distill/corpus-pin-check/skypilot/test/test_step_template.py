"""Contract tests for the corpus-pin-check step-template.yaml.

The run script is rendered with gbserver's own template filler and executed against a
fixture corpus, so an argument that reaches the script in the wrong position fails here.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from gbserver.utils.template import fill_template

_DIR = Path(__file__).resolve().parent.parent
_STEP = _DIR / "step-template.yaml"


@pytest.fixture(scope="module")
def step():
    return yaml.safe_load(_STEP.read_text())


@pytest.fixture(scope="module")
def launcher(step):
    return step["environment_configs"]["Skypilot"]["launchers"]["pin_check"]["config"]


def test_it_is_an_lsf_skypilot_step(step):
    assert step["name"] == "corpus-pin-check"
    assert step["environment_configs"]["Skypilot"]["subtypes"] == ["lsf"]


def test_it_declares_the_output_it_registers(step):
    assert set(step["outputs"]["required"]) == {"pin_check"}


def test_the_script_is_shipped_with_the_step(launcher):
    assert launcher["file_mounts"] == {"src": "src"}
    assert (_DIR / "src" / "check_corpus_pin.py").is_file()


def test_resources_are_left_to_the_build(launcher):
    assert launcher["resources"] == {}


@pytest.mark.parametrize("eval_fraction,accepted", [(0.01, True), (0.02, False)])
def test_the_rendered_script_passes_every_argument_in_place(
    step, launcher, tmp_path, eval_fraction, accepted
):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in ("train.jsonl", "eval.jsonl"):
        (corpus / name).write_text("{}\n")
    (corpus / "corpus_manifest.json").write_text(
        json.dumps(
            {
                "tokenizer_identity": "teacher",
                "policies": {
                    "max_length": 4096,
                    "think_policy": "strip",
                    "documents_policy": "keep",
                    "completion_boundary": "last_message",
                },
                "eval_fraction": 0.01,
            }
        )
    )
    config = {
        "pin_check_config": {
            **step["config"]["pin_check_config"],
            "corpus_dir": str(corpus),
            "teacher_model": "/models/teacher",
            "max_length": 4096,
            "think_policy": "strip",
            "documents_policy": "keep",
            "eval_fraction": eval_fraction,
            "python": sys.executable,
        }
    }
    bindings = {"tokenizer": {"binding": {"path": str(tmp_path / "tok")}}}
    script = fill_template(
        templ=launcher["run"],
        data={"config": config, "bindings": bindings},
        strict=True,
    )

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=_DIR,
        env={**os.environ, "GB_BUILD_WORKDIR": str(tmp_path)},
        capture_output=True,
        text=True,
    )

    out = tmp_path / "corpus-pin" / "corpus_pin.json"
    if accepted:
        assert result.returncode == 0, result.stderr
        assert f"GB_ARTIFACT_ID:pin_check GB_ARTIFACT_PATH:{out}" in result.stdout
    else:
        assert result.returncode != 0
        assert "eval_fraction 0.01 != 0.02" in result.stdout
        assert not out.exists()


def test_the_tokenizer_is_a_required_input(step):
    assert set(step["inputs"]["required"]) == {"tokenizer"}
    assert "tokenizer_dir" not in step["config"]["pin_check_config"]


def test_the_retired_tokenizer_dir_key_is_refused(step, launcher, tmp_path):
    """A recipe still setting the old key would otherwise be ignored silently."""
    config = {
        "pin_check_config": {
            **step["config"]["pin_check_config"],
            "tokenizer_dir": "/old",
            "python": sys.executable,
        }
    }
    bindings = {"tokenizer": {"binding": {"path": str(tmp_path / "tok")}}}
    script = fill_template(
        templ=launcher["run"],
        data={"config": config, "bindings": bindings},
        strict=True,
    )
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=_DIR,
        env={**os.environ, "GB_BUILD_WORKDIR": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "pin_check_config.tokenizer_dir is no longer read" in result.stderr
