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

"""A custom-image step's image repository mirrors its directory under steps/.

steps/<path>/<env>/ builds $(REGISTRY)/steps/<path>/<env>:<tag>, so the registry
layout reads like the source tree (a nested step keeps its category). Runs the
real steps/common.mk against throwaway steps in a temp tree.
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


def _var(step: pathlib.Path, name: str, *overrides: str) -> str:
    """Return a make variable's value as the step's Makefile sees it."""
    if shutil.which("make") is None:
        pytest.skip("make not installed")
    env = {k: v for k, v in os.environ.items() if k not in _MAKE_VARS}
    proc = subprocess.run(
        ["make", "-s", f"print-{name}", "IMAGE_TAG=t1", *overrides],
        cwd=step,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return proc.stdout.strip()


def test_nested_step_image_mirrors_its_directory(tmp_path):
    step = _make_step(tmp_path, "train/foo/skypilot")
    assert _var(step, "IMAGE_REF") == "quay.io/org/steps/train/foo/skypilot:t1"


def test_flat_step_image_mirrors_its_directory(tmp_path):
    step = _make_step(tmp_path, "foo/docker")
    assert _var(step, "IMAGE_REF") == "quay.io/org/steps/foo/docker:t1"


def test_a_step_can_still_choose_its_image_name(tmp_path):
    step = _make_step(tmp_path, "train/foo/skypilot", extra="IMAGE_NAME := flat-foo\n")
    assert _var(step, "IMAGE_REF") == "quay.io/org/flat-foo:t1"


def test_command_line_override_wins(tmp_path):
    step = _make_step(tmp_path, "train/foo/skypilot")
    assert _var(step, "IMAGE_REF", "IMAGE_NAME=x/y") == "quay.io/org/x/y:t1"
