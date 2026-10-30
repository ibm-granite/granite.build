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

"""Every custom-image step gets its own repository, named after its directory.

steps/<path>/<env>/ builds $(REGISTRY)/steps/<path>/<env>:<IMAGE_VERSION>, so the image
path mirrors the step directory and the tag is just the version. Runs the real
steps/common.mk against throwaway steps in a temp tree.
"""

import os
import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
COMMON_MK = REPO_ROOT / "steps" / "common.mk"
_MAKE_VARS = ("MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEOVERRIDES")


def _make_step(root: pathlib.Path, rel: str, extra: str = "") -> pathlib.Path:
    """Create a minimal custom-image step at ``root/steps/<rel>``.

    Args:
        root: temp directory standing in for the repo root.
        rel: the step dir relative to steps/, e.g. ``train/foo/skypilot``.
        extra: Makefile lines placed before the include.

    Returns:
        The step directory.
    """
    steps = root / "steps"
    steps.mkdir(exist_ok=True)
    shutil.copy(COMMON_MK, steps / "common.mk")
    step = steps / rel
    step.mkdir(parents=True)
    depth = "../" * len(pathlib.PurePosixPath(rel).parts)
    (step / "Makefile").write_text(
        f"STEP_NAME := {pathlib.PurePosixPath(rel).parts[-2]}\n"
        "REGISTRY := quay.io/org\n"
        f"{extra}"
        f"include {depth}common.mk\n"
        "print-%:\n\t@echo '$($*)'\n"
    )
    (step / "Dockerfile").write_text("FROM scratch\n")
    return step


def _env() -> dict:
    """Return the environment with make's own variables scrubbed (skip if no make)."""
    if shutil.which("make") is None:
        pytest.skip("make not installed")
    return {k: v for k, v in os.environ.items() if k not in _MAKE_VARS}


def _var(step: pathlib.Path, name: str, *overrides: str) -> str:
    """Return a make variable's value as the step's Makefile sees it."""
    proc = subprocess.run(
        ["make", "-s", f"print-{name}", "IMAGE_VERSION=v1", *overrides],
        cwd=step,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return proc.stdout.strip()


def _make(step: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    """Run make in the step with IMAGE_VERSION=v1 (an explicit one in ``args`` wins)."""
    return subprocess.run(
        ["make", "-s", "IMAGE_VERSION=v1", *args],
        cwd=step,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_nested_step_repository_mirrors_its_directory(tmp_path):
    step = _make_step(tmp_path, "train/foo/skypilot")
    assert _var(step, "IMAGE_REF") == "quay.io/org/steps/train/foo/skypilot:v1"


def test_flat_step_repository_mirrors_its_directory(tmp_path):
    step = _make_step(tmp_path, "foo/docker")
    assert _var(step, "IMAGE_REF") == "quay.io/org/steps/foo/docker:v1"


def test_the_tag_is_the_version(tmp_path):
    step = _make_step(tmp_path, "train/foo/skypilot")
    assert _var(step, "IMAGE_TAG") == "v1"


def test_a_step_can_choose_its_repository(tmp_path):
    step = _make_step(tmp_path, "train/foo/skypilot", extra="IMAGE_NAME := flat-foo\n")
    assert _var(step, "IMAGE_REF") == "quay.io/org/flat-foo:v1"


def test_a_step_can_set_the_whole_tag(tmp_path):
    step = _make_step(tmp_path, "train/foo/skypilot", extra="IMAGE_TAG := mytag\n")
    assert _var(step, "IMAGE_REF") == "quay.io/org/steps/train/foo/skypilot:mytag"


def test_command_line_override_wins(tmp_path):
    step = _make_step(tmp_path, "train/foo/skypilot")
    assert _var(step, "IMAGE_REF", "IMAGE_NAME=x/y") == "quay.io/org/x/y:v1"


def test_too_long_tag_is_refused(tmp_path):
    step = _make_step(tmp_path, "train/foo/skypilot")
    proc = _make(step, "image", f"IMAGE_VERSION={'x' * 130}")
    assert proc.returncode != 0
    assert "not a valid OCI tag" in proc.stderr


@pytest.mark.parametrize("bad", ["Train/Foo", "a//b", "a/-b", "a/b/", "/a", "a b"])
def test_invalid_repository_name_is_refused(tmp_path, bad):
    step = _make_step(tmp_path, "train/foo/skypilot")
    proc = _make(step, "image", f"IMAGE_NAME={bad}")
    assert proc.returncode != 0
    assert "not a valid repository name" in proc.stderr


@pytest.mark.parametrize(
    "rel", ["train/foo_bar/skypilot", "data/a.b/docker", "distill/x-y/skypilot"]
)
def test_real_step_paths_make_valid_repository_names(tmp_path, rel):
    step = _make_step(tmp_path, rel)
    proc = _make(step, "-n", "image")
    assert proc.returncode == 0, proc.stderr


def test_valid_tag_passes_the_guard(tmp_path):
    step = _make_step(tmp_path, "train/foo/skypilot")
    proc = _make(step, "-n", "image")
    assert proc.returncode == 0, proc.stderr
    assert "quay.io/org/steps/train/foo/skypilot:v1" in proc.stdout
