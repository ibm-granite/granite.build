# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`make test` in a step directory runs its build tests in a usable environment.

A step's Mode-1 tests run under ``steps/<path>/<env>/test/``, outside the repo's
``test/`` tree, so pytest never loads ``test/conftest.py`` for them. Without help
from ``common.mk`` that leaves two traps:

* ``GB_ENVIRONMENT`` unset means gbserver defaults to PROD, and the build-test base
  refuses to run ("Refusing to run storage tests with GB_ENVIRONMENT=PROD").
* An ``hf://`` push resolves its HF Enterprise resource group from the space name
  (``gbspace-<space>``) plus the ``GBTEST_STANDALONE_ENVIRONMENT`` suffix. The
  generated Mode-1 space was named after the step, so the push looked for a
  ``gbspace-<step>`` group that does not exist, and the suffix (which
  ``test/conftest.py`` would have defaulted to STAGING) was missing.

So ``make test`` defaults the two variables when unset and names the Mode-1 space
like the shared one Mode 2 uses. An explicit value from the caller still wins.

The test runs the real steps/common.mk against a throwaway step in a temp tree, with
a stub venv, so it touches nothing outside ``tmp_path``.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
COMMON_MK = REPO_ROOT / "steps" / "common.mk"
SHARED_SPACE_NAME = yaml.safe_load(
    (REPO_ROOT / "configurations" / "spaces" / "local" / "space.yaml").read_text()
)["name"]
_VARS = ("GB_ENVIRONMENT", "GBTEST_STANDALONE_ENVIRONMENT", "GBTEST_MODE")

# The step's only test: record the environment it ran in, and the generated space's
# name, as JSON next to itself.
_PROBE = """\
import json, os, pathlib, yaml

def test_probe():
    here = pathlib.Path(__file__).parent
    space = yaml.safe_load((here.parent / "space" / "space.yaml").read_text())
    seen = {k: os.environ.get(k) for k in %r}
    seen["space_name"] = space["name"]
    (here / "seen.json").write_text(json.dumps(seen))
""" % (_VARS,)


def _make_step(root: pathlib.Path) -> pathlib.Path:
    """Create a minimal step at ``root/steps/fake/skypilot`` with a probe test.

    Args:
        root: temp directory standing in for the repo root.

    Returns:
        The step directory.
    """
    (root / "steps").mkdir()
    shutil.copy(COMMON_MK, root / "steps" / "common.mk")
    step = root / "steps" / "fake" / "skypilot"
    (step / "test").mkdir(parents=True)
    (step / "Makefile").write_text("STEP_NAME := fake\ninclude ../../common.mk\n")
    (step / "step-template.yaml").write_text("name: fake\n")
    (step / ".gbignore").write_text("**/*.md\nsrc/**\n")  # common.mk requires one
    (step / "test" / "test_probe.py").write_text(_PROBE)
    venv_bin = root / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "activate").write_text("")  # the real repo venv is already on PATH
    return step


def _run_make_test(tmp_path: pathlib.Path, extra_env: dict[str, str]) -> dict:
    """Run ``make test`` on a fresh throwaway step and return what its test saw.

    Args:
        tmp_path: per-test temp directory.
        extra_env: variables to set for the run (all of ``_VARS`` are cleared first).

    Returns:
        The probe's JSON: each of ``_VARS`` as the test saw it, plus ``space_name``.
    """
    if shutil.which("make") is None:
        pytest.skip("make not installed")
    step = _make_step(tmp_path)
    # Also drop make's own MAKEFLAGS/MFLAGS/MAKELEVEL: when this test runs under an
    # outer make (`make quick-tests` passes `GBTEST_MODE=mock` on its command line),
    # MAKEFLAGS carries that assignment into the inner make, which exports it to the
    # step's tests, so the run would no longer start from a clean environment.
    _make_vars = ("MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEOVERRIDES")
    env = {
        k: v for k, v in os.environ.items() if k not in _VARS + _make_vars
    } | extra_env
    subprocess.run(
        [
            "make",
            "-s",
            "test",
            f"VENV_DIR={tmp_path / 'venv'}",
            f"PYTHON={sys.executable}",
        ],
        cwd=step,
        env=env,
        check=True,
        capture_output=True,
        timeout=120,
    )
    return json.loads((step / "test" / "seen.json").read_text())


def test_defaults_when_unset(tmp_path):
    """With nothing set, the test sees STANDALONE, STAGING, and the shared space name."""
    seen = _run_make_test(tmp_path, {})
    assert seen["GB_ENVIRONMENT"] == "STANDALONE"
    assert seen["GBTEST_STANDALONE_ENVIRONMENT"] == "STAGING"
    assert seen["space_name"] == SHARED_SPACE_NAME


def test_caller_values_win(tmp_path):
    """Values from the caller's environment are passed through, not overwritten."""
    seen = _run_make_test(
        tmp_path,
        {"GB_ENVIRONMENT": "DEV", "GBTEST_STANDALONE_ENVIRONMENT": "PROD"},
    )
    assert seen["GB_ENVIRONMENT"] == "DEV"
    assert seen["GBTEST_STANDALONE_ENVIRONMENT"] == "PROD"


def test_gbtest_mode_is_left_alone(tmp_path):
    """GBTEST_MODE is not set by make; libgbtest already defaults it to live."""
    assert _run_make_test(tmp_path, {})["GBTEST_MODE"] is None
