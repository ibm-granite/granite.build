"""Contract tests for the corpus-sources step-template.yaml.

The run script is rendered with gbserver's own template filler and then executed, so a
sources list that does not reach the script, or a shell error, fails here rather than on
a queue slot.
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
    return step["environment_configs"]["Skypilot"]["launchers"]["sources"]["config"]


def _render(step, launcher, **overrides):
    config = {"sources_config": {**step["config"]["sources_config"], **overrides}}
    return fill_template(templ=launcher["run"], data={"config": config})


def test_it_is_an_lsf_skypilot_step(step):
    assert step["name"] == "corpus-sources"
    assert step["environment_configs"]["Skypilot"]["subtypes"] == ["lsf"]


def test_it_declares_the_output_it_registers(step):
    assert set(step["outputs"]["required"]) == {"corpus_source"}


def test_the_script_is_shipped_with_the_step(launcher):
    assert launcher["file_mounts"] == {"src": "src"}
    assert (_DIR / "src" / "build_sources.py").is_file()


def test_resources_are_left_to_the_build(launcher):
    assert launcher["resources"] == {}


def test_the_rendered_script_runs_and_registers_the_corpus(step, launcher, tmp_path):
    """End to end through the real renderer: three sources in, one artifact line out."""
    srcs = []
    for name, n in (("general", 8), ("tools", 3), ("rag", 1)):
        p = tmp_path / f"{name}.jsonl"
        p.write_text(
            "".join(json.dumps({"conversations": []}) + "\n" for _ in range(n))
        )
        srcs.append(str(p))
    script = _render(
        step,
        launcher,
        sources=srcs,
        target_rows=0,
        shuffle_seed=3,
        output_dir="sources",
        python=sys.executable,
    )

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=_DIR,
        env={**os.environ, "GB_BUILD_WORKDIR": str(tmp_path)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    out = tmp_path / "sources" / "train.jsonl"
    assert len(out.read_text().splitlines()) == 12
    assert f"GB_ARTIFACT_ID:corpus_source GB_ARTIFACT_PATH:{out}" in result.stdout


def test_an_absolute_output_dir_is_used_as_given(step, launcher, tmp_path):
    src = tmp_path / "a.jsonl"
    src.write_text(json.dumps({"messages": []}) + "\n")
    script = _render(
        step,
        launcher,
        sources=[str(src)],
        output_dir=str(tmp_path / "abs"),
        python=sys.executable,
    )
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=_DIR,
        env={**os.environ, "GB_BUILD_WORKDIR": str(tmp_path / "elsewhere")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "abs" / "train.jsonl").is_file()
