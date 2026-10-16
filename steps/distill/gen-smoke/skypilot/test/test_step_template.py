"""Contract tests for the gen-smoke step-template.yaml.

The run script is rendered with gbserver's own template filler and executed against a
stand-in interpreter that records its argv, so a rung list that does not reach the
script whole (build bb779f1f) fails here.
"""

import json
import os
import stat
import subprocess
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
    return step["environment_configs"]["Skypilot"]["launchers"]["gen_smoke"]["config"]


def test_it_is_an_lsf_skypilot_step(step):
    assert step["name"] == "gen-smoke"
    assert step["environment_configs"]["Skypilot"]["subtypes"] == ["lsf"]


def test_it_declares_the_output_it_registers(step):
    assert set(step["outputs"]["required"]) == {"repetition_report"}


def test_the_script_is_shipped_with_the_step(launcher):
    assert launcher["file_mounts"] == {"src": "src"}
    assert (_DIR / "src" / "gen_smoke.py").is_file()


def test_the_default_gates_the_final_rung(step):
    assert step["config"]["gen_smoke_config"]["gate_final_rung"] is True


@pytest.mark.parametrize("gate,flag", [(True, "true"), (False, "false")])
def test_every_rung_reaches_the_script_as_its_own_argument(
    step, launcher, tmp_path, gate, flag
):
    argv_log = tmp_path / "argv.json"
    fake = tmp_path / "python"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"json.dump(sys.argv[1:], open({str(argv_log)!r}, 'w'))\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    rungs = ["25:/w/export-25", "500:/w/export-500", "2000:/w/export-2000"]
    config = {
        "gen_smoke_config": {
            **step["config"]["gen_smoke_config"],
            "rungs": rungs,
            "gate_final_rung": gate,
            "python": str(fake),
        }
    }
    script = fill_template(templ=launcher["run"], data={"config": config})

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=_DIR,
        env={**os.environ, "GB_BUILD_WORKDIR": str(tmp_path)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    argv = json.loads(argv_log.read_text())
    assert argv == [
        "./src/gen_smoke.py",
        str(tmp_path / "gen-smoke" / "repetition.json"),
        "0.15",
        "256",
        flag,
        *rungs,
    ]
