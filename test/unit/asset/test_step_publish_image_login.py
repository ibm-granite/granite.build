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

"""`make publish-image` can log in with a robot credential from IBM Cloud Secrets Manager.

With REGISTRY_SECRET=<group>/<name>, publish-image fetches a username_password secret
with the `ibmcloud sm` CLI and pipes the password into `<docker> login --password-stdin`
for $(REGISTRY), before the push. Runs the real steps/common.mk against a throwaway
step, with fake `ibmcloud` and `podman` scripts first on PATH that record their argv
and stdin.
"""

import os
import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
COMMON_MK = REPO_ROOT / "steps" / "common.mk"
_MAKE_VARS = ("MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEOVERRIDES")
_LOGIN_VARS = ("REGISTRY_USER", "REGISTRY_PASSWORD", "REGISTRY_SECRET")
_TOKEN = "s3cr3t-robot-token"
_SM_JSON = f'{{"username": "org+robot", "password": "{_TOKEN}", "name": "robot"}}'

# Records its argv (one per line) and its stdin, then prints the canned secret.
_FAKE_IBMCLOUD = """#!/bin/sh
printf '%s\\n' "$@" > "$FAKE_LOG/ibmcloud.args"
[ -n "${FAKE_SM_FAIL:-}" ] && { echo "FAILED: not_found" >&2; exit 1; }
printf '%s' "$FAKE_SM_JSON"
"""

# Records each call's argv as a line and, for `login`, its stdin.
_FAKE_PODMAN = """#!/bin/sh
echo "$*" >> "$FAKE_LOG/podman.calls"
if [ "$1" = login ]; then cat > "$FAKE_LOG/podman.login.stdin"; fi
exit 0
"""


@pytest.fixture
def step(tmp_path: pathlib.Path) -> pathlib.Path:
    """A minimal custom-image step at ``tmp_path/steps/train/foo/skypilot``."""
    steps = tmp_path / "steps"
    steps.mkdir()
    shutil.copy(COMMON_MK, steps / "common.mk")
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
    """A dir holding fake `ibmcloud` and `podman`, which also receives their logs."""
    if shutil.which("make") is None:
        pytest.skip("make not installed")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("ibmcloud", _FAKE_IBMCLOUD), ("podman", _FAKE_PODMAN)):
        tool = bin_dir / name
        tool.write_text(body)
        tool.chmod(0o755)
    return bin_dir


def _publish(
    step: pathlib.Path, fakes: pathlib.Path, *args: str, **env: str
) -> subprocess.CompletedProcess:
    """Run `make publish-image` with the fakes first on PATH and no login env."""
    base = {
        k: v
        for k, v in os.environ.items()
        if k not in _MAKE_VARS + _LOGIN_VARS + ("SECRETS_MANAGER_URL",)
    }
    base.update(
        PATH=f"{fakes}{os.pathsep}{os.environ['PATH']}",
        FAKE_LOG=str(fakes),
        FAKE_SM_JSON=_SM_JSON,
    )
    base.update(env)
    return subprocess.run(
        ["make", "publish-image", *args],
        cwd=step,
        env=base,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _calls(fakes: pathlib.Path) -> list[str]:
    log = fakes / "podman.calls"
    return log.read_text().splitlines() if log.exists() else []


def test_secret_login_pipes_the_password_before_the_push(step, fakes):
    proc = _publish(step, fakes, REGISTRY_SECRET="gb-registry/quay-robot")
    assert proc.returncode == 0, proc.stderr
    calls = _calls(fakes)
    assert calls == [
        "login quay.io/org -u org+robot --password-stdin",
        "push quay.io/org/gb-steps:train-foo-skypilot-v1",
    ]
    assert (fakes / "podman.login.stdin").read_text() == _TOKEN


def test_docker_logs_in_to_the_registry_host(step, fakes):
    shutil.copy(fakes / "podman", fakes / "docker")
    proc = _publish(step, fakes, "DOCKER=docker", REGISTRY_SECRET="g/n")
    assert proc.returncode == 0, proc.stderr
    assert _calls(fakes)[0] == "login quay.io -u org+robot --password-stdin"


def test_secret_is_looked_up_by_group_and_name(step, fakes):
    proc = _publish(step, fakes, REGISTRY_SECRET="gb-registry/quay-robot")
    assert proc.returncode == 0, proc.stderr
    args = (fakes / "ibmcloud.args").read_text().splitlines()
    assert args[:2] == ["sm", "secret-by-name"]
    for flag, value in (
        ("--secret-type", "username_password"),
        ("--secret-group-name", "gb-registry"),
        ("--name", "quay-robot"),
        ("--output", "json"),
        (
            "--service-url",
            "https://3a634e1e-2591-4a19-baa8-34cf8f12defb"
            ".us-east.secrets-manager.appdomain.cloud",
        ),
    ):
        assert args[args.index(flag) + 1] == value, flag


def test_secrets_manager_url_overrides_the_instance(step, fakes):
    proc = _publish(
        step,
        fakes,
        REGISTRY_SECRET="gb-registry/quay-robot",
        SECRETS_MANAGER_URL="https://other.example",
    )
    assert proc.returncode == 0, proc.stderr
    args = (fakes / "ibmcloud.args").read_text().splitlines()
    assert args[args.index("--service-url") + 1] == "https://other.example"


def test_the_password_never_reaches_makes_output(step, fakes):
    proc = _publish(step, fakes, REGISTRY_SECRET="gb-registry/quay-robot")
    assert proc.returncode == 0, proc.stderr
    assert _TOKEN not in proc.stdout + proc.stderr
    assert "gb-registry/quay-robot" in proc.stdout


def test_the_secret_can_be_set_on_the_command_line(step, fakes):
    proc = _publish(step, fakes, "REGISTRY_SECRET=gb-registry/quay-robot")
    assert proc.returncode == 0, proc.stderr
    assert _calls(fakes)[0] == "login quay.io/org -u org+robot --password-stdin"


def test_explicit_env_credentials_win_over_the_secret(step, fakes):
    proc = _publish(
        step,
        fakes,
        REGISTRY_SECRET="gb-registry/quay-robot",
        REGISTRY_USER="me+ci",
        REGISTRY_PASSWORD="env-token",
    )
    assert proc.returncode == 0, proc.stderr
    assert not (fakes / "ibmcloud.args").exists()
    assert _calls(fakes)[0] == "login quay.io/org -u me+ci --password-stdin"
    assert (fakes / "podman.login.stdin").read_text() == "env-token"


def test_no_credentials_pushes_with_the_existing_login(step, fakes):
    proc = _publish(step, fakes)
    assert proc.returncode == 0, proc.stderr
    assert not (fakes / "ibmcloud.args").exists()
    assert _calls(fakes) == ["push quay.io/org/gb-steps:train-foo-skypilot-v1"]


def test_a_failed_lookup_stops_before_the_push(step, fakes):
    proc = _publish(
        step, fakes, REGISTRY_SECRET="gb-registry/quay-robot", FAKE_SM_FAIL="1"
    )
    assert proc.returncode != 0
    assert _calls(fakes) == []
    assert "gb-registry/quay-robot" in proc.stderr


def test_a_secret_without_a_password_stops_before_the_push(step, fakes):
    proc = _publish(
        step,
        fakes,
        REGISTRY_SECRET="gb-registry/quay-robot",
        FAKE_SM_JSON='{"username": "org+robot"}',
    )
    assert proc.returncode != 0
    assert _calls(fakes) == []


@pytest.mark.parametrize("bad", ["quay-robot", "/quay-robot", "gb-registry/"])
def test_a_malformed_secret_name_is_refused(step, fakes, bad):
    proc = _publish(step, fakes, REGISTRY_SECRET=bad)
    assert proc.returncode != 0
    assert "<group>/<name>" in proc.stderr
    assert _calls(fakes) == []
