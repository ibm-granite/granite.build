# Copyright Granite.Build Authors
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

"""`make test` warns when the published image is not the one it just built.

A custom-image step's SkyPilot build tests (aws, SLURM, k8s) have the remote node
pull `docker:<IMAGE_REF>` from the registry; the local `make image` build is never
used. With a pinned IMAGE_VERSION, a Dockerfile edit rebuilds locally under a tag
that already exists on the registry, so the tests would pass against the OLD image.
`make test` compares the local image ID with the published image's config digest
(scripts/registry-tag-status.sh --config-digest) and warns loudly when they differ or
cannot be compared, then runs the tests (#565 review: a failure blocked steps whose
image was never published, and anyone testing an image they did not build). The
comparison relies on podman's image IDs, so another DOCKER skips it with a note. Runs the real steps/common.mk against a throwaway custom-image step,
with fake `podman` and `registry-tag-status.sh`.
"""

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
COMMON_MK = REPO_ROOT / "steps" / "common.mk"
_MAKE_VARS = ("MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEOVERRIDES")
_CLEAR = _MAKE_VARS + ("TEST_PUBLISHED", "DOCKER", "REGISTRY_USER", "REGISTRY_PASSWORD")
_LOCAL_ID = "7f9e94997de914cba88372447e60687b31dc82a9a46edbc8e9859df0bb97a933"
_REF = "quay.io/org/steps/train/foo/skypilot:v1"

# `podman build` is a no-op; `podman image inspect` prints $FAKE_LOCAL_ID.
_FAKE_PODMAN = """#!/bin/sh
echo "$*" >> "$FAKE_LOG/podman.calls"
case "$1 $2" in
  "image inspect") printf '%s\\n' "${FAKE_LOCAL_ID}" ;;
esac
exit 0
"""

# Answers --config-digest with $FAKE_PUBLISHED (a digest, `absent` or `unknown`).
_FAKE_STATUS = """#!/bin/sh
echo "$*" >> "$FAKE_LOG/status.calls"
s="${FAKE_PUBLISHED:-absent}"
echo "$s"
case "$s" in sha256:*) exit 0 ;; absent) exit 1 ;; *) exit 2 ;; esac
"""

# The step's only test: leave a marker so we can tell whether pytest ran.
_PROBE = """\
import pathlib

def test_probe():
    (pathlib.Path(__file__).parent / "ran").write_text("yes")
"""


@pytest.fixture
def step(tmp_path: pathlib.Path) -> pathlib.Path:
    """A custom-image step at ``tmp_path/steps/train/foo/skypilot`` with one test."""
    if shutil.which("make") is None:
        pytest.skip("make not installed")
    steps = tmp_path / "steps"
    (steps / "scripts").mkdir(parents=True)
    shutil.copy(COMMON_MK, steps / "common.mk")
    status = steps / "scripts" / "registry-tag-status.sh"
    status.write_text(_FAKE_STATUS)
    status.chmod(0o755)
    d = steps / "train" / "foo" / "skypilot"
    (d / "test").mkdir(parents=True)
    (d / "Makefile").write_text(
        "STEP_NAME := foo\n"
        "REGISTRY := quay.io/org\n"
        "IMAGE_VERSION := v1\n"
        "include ../../../common.mk\n"
    )
    (d / "Dockerfile").write_text("FROM scratch\n")
    (d / "step-template.yaml").write_text(
        "name: foo\nconfig:\n  image_id: docker:${IMAGE_REF}\n"
    )
    (d / ".gbignore").write_text("**/*.md\nsrc/**\n")
    (d / "test" / "test_probe.py").write_text(_PROBE)
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "activate").write_text("")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "podman").write_text(_FAKE_PODMAN)
    (bin_dir / "podman").chmod(0o755)
    return d


def _test(step: pathlib.Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    root = step.parents[3]
    e = {k: v for k, v in os.environ.items() if k not in _CLEAR}
    e.update(
        PATH=f"{root / 'bin'}{os.pathsep}{os.environ['PATH']}",
        FAKE_LOG=str(root / "bin"),
        FAKE_LOCAL_ID=_LOCAL_ID,
    )
    e.update(env)
    return subprocess.run(
        [
            "make",
            "test",
            f"VENV_DIR={root / 'venv'}",
            f"PYTHON={sys.executable}",
            "PYTEST_CAPTURE=-q",
            *args,
        ],
        cwd=step,
        env=e,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _ran(step: pathlib.Path) -> bool:
    return (step / "test" / "ran").exists()


def test_tag_not_published_yet_runs_the_tests(step):
    proc = _test(step, FAKE_PUBLISHED="absent")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _ran(step)


def test_published_image_identical_to_the_local_build_runs_the_tests(step):
    proc = _test(step, FAKE_PUBLISHED=f"sha256:{_LOCAL_ID}")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _ran(step)
    assert "WARNING" not in proc.stdout + proc.stderr


def test_published_image_differing_from_the_local_build_warns_and_runs(step):
    proc = _test(step, FAKE_PUBLISHED="sha256:" + "0" * 64)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _ran(step)
    out = proc.stdout + proc.stderr
    assert "WARNING" in out and _REF in out
    assert "IMAGE_VERSION" in out and "TEST_PUBLISHED=true" in out


def test_unknown_published_state_warns_and_runs(step):
    # e.g. a step never published (quay answers a missing repo with 401), offline.
    proc = _test(step, FAKE_PUBLISHED="unknown")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _ran(step)
    out = proc.stdout + proc.stderr
    assert "WARNING" in out and "could not compare" in out


def test_other_docker_clis_skip_the_comparison(step):
    # docker (containerd store, buildx attestations) reports other image IDs.
    root = step.parents[3]
    shutil.copy(root / "bin" / "podman", root / "bin" / "docker")
    proc = _test(step, "DOCKER=docker", FAKE_PUBLISHED="sha256:" + "0" * 64)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _ran(step)
    assert not (root / "bin" / "status.calls").exists()
    out = proc.stdout + proc.stderr
    assert "WARNING" not in out and "podman" in out


@pytest.mark.parametrize("published", ["sha256:" + "0" * 64, "unknown"])
def test_test_published_true_runs_against_the_published_image(step, published):
    proc = _test(step, "TEST_PUBLISHED=true", FAKE_PUBLISHED=published)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _ran(step)
    status_log = step.parents[3] / "bin" / "status.calls"
    assert not status_log.exists()


def test_the_check_asks_for_the_config_digest_of_image_ref(step):
    _test(step, FAKE_PUBLISHED="absent")
    calls = (step.parents[3] / "bin" / "status.calls").read_text().splitlines()
    assert calls == [f"--config-digest {_REF}"]


def test_a_public_image_step_is_not_checked(step):
    (step / "Dockerfile").unlink()
    (step / "Makefile").write_text("STEP_NAME := foo\ninclude ../../../common.mk\n")
    (step / "step-template.yaml").write_text("name: foo\n")
    proc = _test(step, FAKE_PUBLISHED="unknown")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _ran(step)
    assert not (step.parents[3] / "bin" / "status.calls").exists()
