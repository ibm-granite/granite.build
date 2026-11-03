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

"""`make publish-image`: the optional login and the overwrite guard.

Login: with REGISTRY_USER and REGISTRY_PASSWORD in the environment, publish-image
pipes the password into `<docker> login --password-stdin`, scoped to $(REGISTRY)
for podman (to the registry host for docker), before the push.

Overwrite guard: a tag pins an image, and SLURM's image cache keys its .sqsh by
tag, so publish-image refuses to push a tag that already exists on the registry
(or whose existence it cannot determine) unless PUBLISH_OVERWRITE=true.

New public repository: on quay.io, with QUAY_API_TOKEN in the environment, a
repository the registry reports absent is first created as public through quay's
API (a repo created by a push is private, and SkyPilot pulls anonymously).

Runs the real steps/common.mk against a throwaway step, with a fake `podman` and
`curl` first on PATH that record their argv and stdin, and a fake
`registry-tag-status.sh` that answers present / absent / unknown.
"""

import json
import os
import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
COMMON_MK = REPO_ROOT / "steps" / "common.mk"
_MAKE_VARS = ("MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEOVERRIDES")
_LOGIN_VARS = (
    "REGISTRY_USER",
    "REGISTRY_PASSWORD",
    "PUBLISH_OVERWRITE",
    "DOCKER",
    "QUAY_API_TOKEN",
)
_TOKEN = "s3cr3t-robot-token"
_REF = "quay.io/org/steps/train/foo/skypilot:v1"

# Records each call's argv as a line and, for `login`, its stdin.
_FAKE_PODMAN = """#!/bin/sh
echo "$*" >> "$FAKE_LOG/podman.calls"
if [ "$1" = login ]; then cat > "$FAKE_LOG/podman.login.stdin"; fi
exit 0
"""

# Stands in for steps/scripts/registry-tag-status.sh: answers $FAKE_TAG_STATUS, or
# $FAKE_REPO_STATUS for the `:repo` probe (does the repository exist at all).
_FAKE_STATUS = """#!/bin/sh
echo "$*" >> "$FAKE_LOG/status.calls"
s="${FAKE_TAG_STATUS:-absent}"
case "$*" in *--repository*) s="${FAKE_REPO_STATUS:-present}" ;; esac
echo "$s"
case "$s" in present) exit 0 ;; absent) exit 1 ;; *) exit 2 ;; esac
"""

# Records argv, the stdin config (the Authorization header) and the request body
# file, then answers $FAKE_CURL_CODE (default 201).
_FAKE_CURL = """#!/bin/sh
echo "$*" >> "$FAKE_LOG/curl.calls"
body=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config) [ "$2" = "-" ] && cat > "$FAKE_LOG/curl.config"; shift 2 ;;
    --data-binary) body="${2#@}"; shift 2 ;;
    *) shift ;;
  esac
done
[ -n "$body" ] && cp "$body" "$FAKE_LOG/curl.body"
printf '%s' "${FAKE_CURL_CODE:-201}"
"""


@pytest.fixture
def step(tmp_path: pathlib.Path) -> pathlib.Path:
    """A minimal custom-image step at ``tmp_path/steps/train/foo/skypilot``."""
    steps = tmp_path / "steps"
    steps.mkdir()
    shutil.copy(COMMON_MK, steps / "common.mk")
    (steps / "scripts").mkdir()
    status = steps / "scripts" / "registry-tag-status.sh"
    status.write_text(_FAKE_STATUS)
    status.chmod(0o755)
    step_dir = steps / "train" / "foo" / "skypilot"
    step_dir.mkdir(parents=True)
    (step_dir / "Makefile").write_text(
        "STEP_NAME := foo\n"
        "REGISTRY := quay.io/org\n"
        "IMAGE_VERSION := v1\n"
        "include ../../../common.mk\n"
    )
    (step_dir / "Dockerfile").write_text("FROM scratch\n")
    return step_dir


@pytest.fixture
def fakes(tmp_path: pathlib.Path) -> pathlib.Path:
    """A dir holding a fake `podman`, which also receives the logs."""
    if shutil.which("make") is None:
        pytest.skip("make not installed")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("podman", _FAKE_PODMAN), ("curl", _FAKE_CURL)):
        tool = bin_dir / name
        tool.write_text(body)
        tool.chmod(0o755)
    return bin_dir


def _make(
    step: pathlib.Path, fakes: pathlib.Path, target: str, *args: str, **env: str
) -> subprocess.CompletedProcess:
    """Run `make <target>` with the fakes first on PATH and no login env."""
    base = {k: v for k, v in os.environ.items() if k not in _MAKE_VARS + _LOGIN_VARS}
    base.update(PATH=f"{fakes}{os.pathsep}{os.environ['PATH']}", FAKE_LOG=str(fakes))
    base.update(env)
    return subprocess.run(
        ["make", target, *args],
        cwd=step,
        env=base,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _publish(step, fakes, *args, **env) -> subprocess.CompletedProcess:
    return _make(step, fakes, "publish-image", *args, **env)


def _calls(fakes: pathlib.Path, tool: str = "podman") -> list[str]:
    log = fakes / f"{tool}.calls"
    return log.read_text().splitlines() if log.exists() else []


def _tag_checks(fakes: pathlib.Path) -> list[str]:
    """The overwrite guard's tag checks (not the quay repository probe)."""
    return [c for c in _calls(fakes, "status") if not c.startswith("--repository")]


class TestLogin:
    def test_env_credentials_log_in_before_the_push(self, step, fakes):
        proc = _publish(
            step, fakes, REGISTRY_USER="org+robot", REGISTRY_PASSWORD=_TOKEN
        )
        assert proc.returncode == 0, proc.stderr
        assert _calls(fakes) == [
            "login quay.io/org -u org+robot --password-stdin",
            f"push {_REF}",
        ]
        assert (fakes / "podman.login.stdin").read_text() == _TOKEN

    def test_the_password_never_reaches_makes_output(self, step, fakes):
        proc = _publish(
            step, fakes, REGISTRY_USER="org+robot", REGISTRY_PASSWORD=_TOKEN
        )
        assert proc.returncode == 0, proc.stderr
        assert _TOKEN not in proc.stdout + proc.stderr

    def test_docker_logs_in_to_the_registry_host(self, step, fakes):
        shutil.copy(fakes / "podman", fakes / "docker")
        proc = _publish(
            step, fakes, "DOCKER=docker", REGISTRY_USER="u", REGISTRY_PASSWORD="p"
        )
        assert proc.returncode == 0, proc.stderr
        assert _calls(fakes)[0] == "login quay.io -u u --password-stdin"

    def test_no_credentials_pushes_with_the_existing_login(self, step, fakes):
        proc = _publish(step, fakes)
        assert proc.returncode == 0, proc.stderr
        assert _calls(fakes) == [f"push {_REF}"]

    @pytest.mark.parametrize("only", ["REGISTRY_USER", "REGISTRY_PASSWORD"])
    def test_half_the_credentials_means_no_login(self, step, fakes, only):
        proc = _publish(step, fakes, **{only: "x"})
        assert proc.returncode == 0, proc.stderr
        assert _calls(fakes) == [f"push {_REF}"]


class TestOverwriteGuard:
    def test_new_tag_is_pushed(self, step, fakes):
        proc = _publish(step, fakes, FAKE_TAG_STATUS="absent")
        assert proc.returncode == 0, proc.stderr
        assert _tag_checks(fakes) == [_REF]
        assert _calls(fakes) == [f"push {_REF}"]

    def test_existing_tag_is_refused(self, step, fakes):
        proc = _publish(step, fakes, FAKE_TAG_STATUS="present")
        assert proc.returncode != 0
        assert _calls(fakes) == []
        assert "already exists" in proc.stderr
        assert "IMAGE_VERSION" in proc.stderr
        assert "PUBLISH_OVERWRITE=true" in proc.stderr

    def test_unknown_status_is_refused(self, step, fakes):
        proc = _publish(step, fakes, FAKE_TAG_STATUS="unknown")
        assert proc.returncode != 0
        assert _calls(fakes) == []
        assert "could not determine" in proc.stderr

    @pytest.mark.parametrize("status", ["present", "unknown"])
    def test_overwrite_pushes_anyway(self, step, fakes, status):
        proc = _publish(step, fakes, "PUBLISH_OVERWRITE=true", FAKE_TAG_STATUS=status)
        assert proc.returncode == 0, proc.stderr
        assert _calls(fakes) == [f"push {_REF}"]
        assert _tag_checks(fakes) == []

    @pytest.mark.parametrize("value", ["yes", "1", "True", ""])
    def test_only_exact_true_overwrites(self, step, fakes, value):
        proc = _publish(
            step, fakes, f"PUBLISH_OVERWRITE={value}", FAKE_TAG_STATUS="present"
        )
        assert proc.returncode != 0
        assert _calls(fakes) == []

    def test_the_check_runs_after_login(self, step, fakes):
        # A private repo needs the login (and its credentials) to be checked.
        _publish(
            step,
            fakes,
            REGISTRY_USER="u",
            REGISTRY_PASSWORD="p",
            FAKE_TAG_STATUS="absent",
        )
        assert _calls(fakes)[0].startswith("login ")
        assert _calls(fakes)[1] == f"push {_REF}"


class TestNewPublicRepository:
    """With QUAY_API_TOKEN set, a missing quay.io repository is created public."""

    def _quay_step(self, step):
        mk = step / "Makefile"
        mk.write_text(mk.read_text().replace("quay.io/org", "quay.io/granitebuild"))

    def test_missing_repo_is_created_public_before_the_push(self, step, fakes):
        self._quay_step(step)
        proc = _publish(
            step,
            fakes,
            QUAY_API_TOKEN=_TOKEN,
            FAKE_REPO_STATUS="absent",
            FAKE_TAG_STATUS="absent",
        )
        assert proc.returncode == 0, proc.stderr
        curl = _calls(fakes, "curl")
        assert len(curl) == 1 and "https://quay.io/api/v1/repository" in curl[0]
        body = json.loads((fakes / "curl.body").read_text())
        assert body == {
            "namespace": "granitebuild",
            "repository": "steps/train/foo/skypilot",
            "visibility": "public",
            "description": "granite.build step image (steps/train/foo/skypilot)",
            "repo_kind": "image",
        }
        assert (fakes / "curl.config").read_text().strip() == (
            f'header = "Authorization: Bearer {_TOKEN}"'
        )
        assert _calls(fakes) == [
            "push quay.io/granitebuild/steps/train/foo/skypilot:v1"
        ]
        assert "created public repository" in proc.stdout

    def test_the_token_never_reaches_argv_or_output(self, step, fakes):
        self._quay_step(step)
        proc = _publish(step, fakes, QUAY_API_TOKEN=_TOKEN, FAKE_REPO_STATUS="absent")
        assert proc.returncode == 0, proc.stderr
        assert _TOKEN not in proc.stdout + proc.stderr
        assert all(_TOKEN not in c for c in _calls(fakes, "curl"))

    def test_existing_repo_is_not_recreated(self, step, fakes):
        self._quay_step(step)
        proc = _publish(step, fakes, QUAY_API_TOKEN=_TOKEN, FAKE_REPO_STATUS="present")
        assert proc.returncode == 0, proc.stderr
        assert _calls(fakes, "curl") == []

    def test_without_a_token_a_new_repo_stops_before_the_push(self, step, fakes):
        # A push would create it private, and SkyPilot pulls anonymously: easy to
        # forget to make public, so refuse instead of warning (#565 review).
        self._quay_step(step)
        proc = _publish(step, fakes, FAKE_REPO_STATUS="absent")
        assert proc.returncode != 0
        assert _calls(fakes, "curl") == [] and _calls(fakes) == []
        assert "QUAY_API_TOKEN" in proc.stderr
        assert "public" in proc.stderr

    def test_without_a_token_an_existing_repo_is_pushed(self, step, fakes):
        self._quay_step(step)
        proc = _publish(step, fakes, FAKE_REPO_STATUS="present")
        assert proc.returncode == 0, proc.stderr
        assert _calls(fakes) == [
            "push quay.io/granitebuild/steps/train/foo/skypilot:v1"
        ]

    def test_a_failed_create_stops_before_the_push(self, step, fakes):
        self._quay_step(step)
        proc = _publish(
            step,
            fakes,
            QUAY_API_TOKEN=_TOKEN,
            FAKE_REPO_STATUS="absent",
            FAKE_CURL_CODE="403",
        )
        assert proc.returncode != 0
        assert _calls(fakes) == []
        assert "could not create" in proc.stderr

    def test_unknown_repo_status_is_refused(self, step, fakes):
        self._quay_step(step)
        proc = _publish(step, fakes, QUAY_API_TOKEN=_TOKEN, FAKE_REPO_STATUS="unknown")
        assert proc.returncode != 0
        assert _calls(fakes, "curl") == [] and _calls(fakes) == []
        assert "podman login" in proc.stderr and "REGISTRY_USER" in proc.stderr

    def test_other_registries_never_call_the_quay_api(self, step, fakes):
        mk = step / "Makefile"
        mk.write_text(mk.read_text().replace("quay.io/org", "ghcr.io/org"))
        proc = _publish(step, fakes, QUAY_API_TOKEN=_TOKEN, FAKE_REPO_STATUS="absent")
        assert proc.returncode == 0, proc.stderr
        assert _calls(fakes, "curl") == []
        assert not [c for c in _calls(fakes, "status") if "--repository" in c]


class TestPublishStepImageCheck:
    """publish-step's image check uses the same status script (#568)."""

    def _publish_step(self, step, fakes, *args, **env):
        (step / "step-template.yaml").write_text(
            "name: foo\nconfig:\n  image_id: docker:${IMAGE_REF}\n"
        )
        (step / "USAGE.md").write_text("usage\n")
        (step / ".gbignore").write_text("**/*.md\n")
        return _make(
            step,
            fakes,
            "publish-step",
            f"PUBLISH_STEP_DIR={step.parent / 'out-step'}",
            f"PUBLISH_TEST_DIR={step.parent / 'out-test'}",
            f"PUBLISH_TESTDATA_DIR={step.parent / 'out-data'}",
            *args,
            **env,
        )

    def test_present_image_is_accepted(self, step, fakes):
        proc = self._publish_step(step, fakes, FAKE_TAG_STATUS="present")
        assert "image present on registry" in proc.stdout
        assert _calls(fakes, "status") == [_REF]

    @pytest.mark.parametrize("status", ["absent", "unknown"])
    def test_missing_or_unknown_image_is_refused(self, step, fakes, status):
        proc = self._publish_step(step, fakes, FAKE_TAG_STATUS=status)
        assert proc.returncode != 0
        assert "refusing" in proc.stdout + proc.stderr
        assert not (step.parent / "out-step" / "step.yaml").exists()

    def test_podman_manifest_inspect_is_not_used(self, step, fakes):
        self._publish_step(step, fakes, FAKE_TAG_STATUS="present")
        assert not [c for c in _calls(fakes) if c.startswith("manifest")]
