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

"""Every step ships its own .gbignore, and common.mk copies it.

gbserver Jinja-renders every file in a step directory except those a ``.gbignore``
lists. Each step source carries one, next to its Makefile, and ``make space`` (which
``make test`` depends on) and ``make publish-step`` both copy it beside the rendered
``step.yaml``, so the step is rendered the same way when tested and when published.

common.mk generates nothing. A step without the file fails both targets with an error
saying so, before anything is published: shipping a step with no ``.gbignore`` would
Jinja-render its docs and scripts, which breaks silently on a cluster.

The test runs the real steps/common.mk against a throwaway step in a temp tree, so it
touches nothing outside ``tmp_path``.
"""

import os
import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
COMMON_MK = REPO_ROOT / "steps" / "common.mk"
AUTHORED = "# step-specific: keep docs and scripts verbatim, plus a data template\n**/*.md\nsrc/**\nassets/*.j2\n"
_MAKE_VARS = ("MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEOVERRIDES")


def _make_step(root: pathlib.Path, gbignore: str | None) -> pathlib.Path:
    """Create a minimal step at ``root/steps/fake/skypilot``.

    Args:
        root: temp directory standing in for the repo root.
        gbignore: the step's own .gbignore text, or None for no file.

    Returns:
        The step directory.
    """
    (root / "steps").mkdir()
    shutil.copy(COMMON_MK, root / "steps" / "common.mk")
    step = root / "steps" / "fake" / "skypilot"
    step.mkdir(parents=True)
    (step / "Makefile").write_text("STEP_NAME := fake\ninclude ../../common.mk\n")
    (step / "step-template.yaml").write_text("name: fake\n")
    (step / "USAGE.md").write_text("# fake\n")
    if gbignore is not None:
        (step / ".gbignore").write_text(gbignore)
    return step


def _make(step: pathlib.Path, *targets: str) -> str:
    """Run make targets in ``step`` with a clean make environment.

    Args:
        step: the step directory.
        targets: make targets to run.

    Returns:
        make's stdout.

    Raises:
        subprocess.CalledProcessError: if make fails.
    """
    if shutil.which("make") is None:
        pytest.skip("make not installed")
    env = {k: v for k, v in os.environ.items() if k not in _MAKE_VARS}
    proc = subprocess.run(
        ["make", "-s", *targets, "PUBLISH_REQUIRE_IMAGE=false"],
        cwd=step,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return proc.stdout


def _space_gbignore(step: pathlib.Path) -> str:
    """Return the .gbignore that ``make space`` wrote for the step."""
    return (step / "space" / "steps" / "fake" / ".gbignore").read_text()


def _published_gbignore(root: pathlib.Path) -> str:
    """Return the .gbignore that ``make publish-step`` wrote for the step."""
    path = root / "configurations/assets/environments/skypilot/steps/fake/.gbignore"
    return path.read_text()


@pytest.mark.parametrize("target", ["space", "publish-step"])
def test_a_step_without_gbignore_is_refused(tmp_path, target):
    """No .gbignore: the target fails, says why, and publishes nothing."""
    step = _make_step(tmp_path, gbignore=None)
    with pytest.raises(subprocess.CalledProcessError) as raised:
        _make(step, target)
    assert ".gbignore" in raised.value.stderr + raised.value.stdout
    assert not (tmp_path / "configurations").exists()


def test_the_step_file_is_used_for_space_and_publish(tmp_path):
    """The step's .gbignore is copied verbatim by both targets."""
    step = _make_step(tmp_path, gbignore=AUTHORED)
    _make(step, "space", "publish-step")
    assert _space_gbignore(step) == AUTHORED
    assert _published_gbignore(tmp_path) == AUTHORED


def test_the_step_file_does_not_show_as_drift(tmp_path):
    """check-published re-renders with the step's file, so it reports no drift."""
    step = _make_step(tmp_path, gbignore=AUTHORED)
    _make(step, "publish-step")
    assert "OK" in _make(step, "check-published")
