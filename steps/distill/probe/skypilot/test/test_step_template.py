"""Contract tests for the distill-probe step-template.yaml.

The run script is rendered with gbserver's own template filler and executed against a
stand-in interpreter that records its argv, so a setting that does not reach the
script in its place fails here.
"""

import json
import os
import sys
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from gbserver.utils.template import fill_template

_DIR = Path(__file__).resolve().parent.parent


def _image_env(tmp_path, step, python=None):
    """Emulate the baked image: DISTILL_HOME/steps/<step> holds this step's scripts, and a
    `python` first on PATH (the image puts its venv there) is the test's own interpreter."""
    home = tmp_path / "distill-home"
    (home / "steps").mkdir(parents=True)
    (home / "steps" / step).symlink_to(_DIR / "src")
    shim = tmp_path / "shim"
    shim.mkdir()
    py = shim / "python"
    py.write_text(f'#!/bin/sh\nexec "{python or sys.executable}" "$@"\n')
    py.chmod(0o755)
    return {
        **os.environ,
        "GB_BUILD_WORKDIR": str(tmp_path),
        "DISTILL_HOME": str(home),
        "PATH": f"{shim}{os.pathsep}{os.environ['PATH']}",
    }
_STEP = _DIR / "step-template.yaml"


@pytest.fixture(scope="module")
def step():
    return yaml.safe_load(_STEP.read_text())


@pytest.fixture(scope="module")
def launcher(step):
    return step["environment_configs"]["Skypilot"]["launchers"]["probe"]["config"]


def test_it_is_an_lsf_skypilot_step(step):
    assert step["name"] == "probe"
    assert step["environment_configs"]["Skypilot"]["subtypes"] == ["lsf", "slurm"]


def test_it_declares_no_output(step):
    """The answers are decisions for a human, not artifacts for a step."""
    assert "outputs" not in step


def test_the_script_is_shipped_with_the_step(launcher):
    assert "file_mounts" not in launcher
    assert (_DIR / "src" / "distill_probe.py").is_file()


def test_resources_are_left_to_the_build(launcher):
    assert launcher["resources"] == {}


def test_the_hub_is_not_forced_offline(launcher):
    """load-student asks whether transformers' lazy kernel fetch works in this image;
    an offline hub would answer a different question."""
    assert "HF_HUB_OFFLINE" not in (launcher.get("envs") or {})


def test_the_python_is_the_images_venv_on_path(step, launcher):
    """The image puts /stage/.venv/bin first on PATH, so a bare `python` is its venv."""
    assert "python" not in step["config"]["probe_config"]
    assert "/stage/" not in launcher["run"]


@pytest.mark.parametrize("probe", ["load-student", "tokenizer", "tokenizer-fit"])
def test_every_setting_reaches_the_script_in_place(step, launcher, tmp_path, probe):
    argv_log = tmp_path / "argv.json"
    fake = tmp_path / "python"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"json.dump(sys.argv[1:], open({str(argv_log)!r}, 'w'))\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    config = {
        "probe_config": {
            **step["config"]["probe_config"],
            "probe": probe,
            "checkpoint": "/ckpt/epoch_hf_2",
            "corpus": "/data/smoke.jsonl",
        }
    }
    script = fill_template(templ=launcher["run"], data={"config": config})

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=_DIR,
        env=_image_env(tmp_path, "probe", python=str(fake)),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(argv_log.read_text()) == [
        str(tmp_path / "distill-home" / "steps" / "probe" / "distill_probe.py"),
        probe,
        "--checkpoint",
        "/ckpt/epoch_hf_2",
        "--corpus",
        "/data/smoke.jsonl",
        "--rows",
        "200",
    ]

def test_the_slurm_executors_stripped_env_is_restored(launcher):
    """The SLURM executor runs tasks under `env -u PYTHONPATH` with no VIRTUAL_ENV, so the
    image's ENV values for both are gone at run time and each step sets them itself."""
    envs = launcher["envs"]
    assert envs["PYTHONPATH"] == "/opt/distill/src"
    assert envs["VIRTUAL_ENV"] == "/stage/.venv"
