#!/usr/bin/env python3

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

"""Unit tests for the BlueVela file API.

These tests stub out:
  - ``open_bluevela_tunnel`` (so no SSH / IBM Cloud is contacted)
  - ``lookup_build_target_step`` (so no DB is required)
  - ``authorize_build_access`` (so no auth middleware is required)

What we exercise here is the request/response surface: path-traversal
rejection, size caps, and binary detection. Every request first runs
``ls -1t`` on the step-run parent dir to pick the newest ``launch-*``
entry; the test tunnels return a canned listing for that step.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from gbserver.api import bluevela as bluevela_mod
from gbserver.api.bluevela import bluevela_api
from gbserver.api.bluevela_tunnel import BlueVelaConfig
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.storage.stored_target_run import StoredTargetRun

# --------------------------------------------------------------------- fixtures


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    # Mount the api the same way root_api does, but without AuthMiddleware —
    # we stub authorize_build_access in each test.
    app.mount("/api/v1/bluevela", bluevela_api)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _fake_build_target_step():
    build = StoredBuild(
        uuid="B1",
        name="b",
        space_name="space-a",
        source_uri="",
        username="alice",
    )
    target = StoredTargetRun(
        uuid="TR1",
        build_id="B1",
        environment_uri="env://x",
        name="train",
    )
    step = StoredStepRun(
        uuid="SR1",
        build_id="B1",
        target_id="TR1",
        definition_uri="step://x",
        config={"step": {"name": "step1"}},
    )
    return build, target, step


def _fake_tunnel_cm(tunnel_mock):
    """Return an async context manager yielding (tunnel, BlueVelaConfig)."""

    @asynccontextmanager
    async def _cm(space_name: str, environment_uri: str):
        yield tunnel_mock, BlueVelaConfig(
            login_node="login-1",
            username="ci-user",
            workspace_remote_dir="/ws",
        )

    return _cm


def _tunnel_with_launch_listing():
    """Mock tunnel whose `ls -1t` returns a single launch-NEW entry.

    Use for tests that should fail BEFORE running the listing/stat command
    (path traversal, etc.) — the resolver still needs the `ls -1t` round
    trip to succeed.
    """
    tunnel = MagicMock()

    async def run_remote(cmd, raise_on_error=True):
        if cmd.startswith("ls -1t"):
            return (0, "launch-NEW\n", "")
        return (0, "", "")

    tunnel.run_remote = AsyncMock(side_effect=run_remote)
    return tunnel


def _patches(
    tunnel_mock,
    authorize_raises: Exception | None = None,
):
    build, target, step = _fake_build_target_step()
    lookup = patch.object(
        bluevela_mod,
        "lookup_build_target_step",
        return_value=(build, target, step),
    )
    tunnel = patch.object(
        bluevela_mod, "open_bluevela_tunnel", _fake_tunnel_cm(tunnel_mock)
    )
    auth = patch.object(
        bluevela_mod,
        "authorize_build_access",
        side_effect=(authorize_raises if authorize_raises else (lambda *a, **kw: None)),
    )
    return lookup, tunnel, auth


# ---------------------------------------------------------------------- /files


class TestListFiles:
    def test_happy_path_nonrecursive(self, client):
        tunnel = MagicMock()
        stat_out = (
            "/ws/llm-build-B1/target-train/target-run-TR1/step-step1/step-run-SR1/launch-NEW\t4096\t100\tdirectory\n"
            "/ws/llm-build-B1/target-train/target-run-TR1/step-step1/step-run-SR1/launch-NEW/a.txt\t7\t101\tregular file\n"
        )

        async def run_remote(cmd, raise_on_error=True):
            if cmd.startswith("ls -1t"):
                return (0, "launch-NEW\nlaunch-OLD\n", "")
            return (0, stat_out, "")

        tunnel.run_remote = AsyncMock(side_effect=run_remote)
        lookup, tun, auth = _patches(tunnel)
        with lookup, tun, auth:
            r = client.get(
                "/api/v1/bluevela/builds/B1/files",
                params={"target_name": "train", "step_name": "step1"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["asset_dir"].endswith("launch-NEW")
        paths = {e["path"] for e in body["entries"]}
        assert "" in paths  # the dir itself
        assert "a.txt" in paths

    def test_path_traversal_rejected(self, client):
        tunnel = _tunnel_with_launch_listing()
        lookup, tun, auth = _patches(tunnel)
        with lookup, tun, auth:
            r = client.get(
                "/api/v1/bluevela/builds/B1/files",
                params={
                    "target_name": "train",
                    "step_name": "step1",
                    "path": "../../etc/passwd",
                },
            )
        assert r.status_code == 400
        # run_remote must never have been invoked with the hostile path.
        for call in tunnel.run_remote.await_args_list:
            assert "etc/passwd" not in call.args[0]

    def test_absolute_path_rejected(self, client):
        tunnel = _tunnel_with_launch_listing()
        lookup, tun, auth = _patches(tunnel)
        with lookup, tun, auth:
            r = client.get(
                "/api/v1/bluevela/builds/B1/files",
                params={
                    "target_name": "train",
                    "step_name": "step1",
                    "path": "/etc/passwd",
                },
            )
        assert r.status_code == 400

    def test_null_byte_rejected(self, client):
        tunnel = _tunnel_with_launch_listing()
        lookup, tun, auth = _patches(tunnel)
        with lookup, tun, auth:
            r = client.get(
                "/api/v1/bluevela/builds/B1/files",
                params={
                    "target_name": "train",
                    "step_name": "step1",
                    "path": "a\x00b",
                },
            )
        assert r.status_code == 400

    def test_unauthorized(self, client):
        tunnel = MagicMock()
        tunnel.run_remote = AsyncMock(return_value=(0, "", ""))
        lookup, tun, auth = _patches(
            tunnel,
            authorize_raises=HTTPException(status_code=401, detail="no"),
        )
        with lookup, tun, auth:
            r = client.get(
                "/api/v1/bluevela/builds/B1/files",
                params={"target_name": "train", "step_name": "step1"},
            )
        assert r.status_code == 401


# -------------------------------------------------------------- /files/content


class TestContent:
    def _tunnel_with_file(self, size: int, body: bytes):
        tunnel = MagicMock()

        async def run_remote(cmd, raise_on_error=True):
            if cmd.startswith("ls -1t"):
                return (0, "launch-NEW\n", "")
            if cmd.startswith("stat -c"):
                # size<TAB>type
                return (0, f"{size}\tregular file\n", "")
            if cmd.startswith("head -c"):
                # space-separated decimal bytes
                return (0, " ".join(str(b) for b in body[:8192]), "")
            return (0, "", "")

        tunnel.run_remote = AsyncMock(side_effect=run_remote)

        # Fake SFTP
        sftp = MagicMock()
        fh = MagicMock()
        fh.read = AsyncMock(return_value=body)
        fh.__aenter__ = AsyncMock(return_value=fh)
        fh.__aexit__ = AsyncMock(return_value=None)
        sftp.open = MagicMock(return_value=fh)
        sftp.exit = MagicMock(return_value=None)
        tunnel.start_sftp = AsyncMock(return_value=sftp)
        return tunnel

    def test_text_file_returned(self, client):
        body = b"hello world"
        tunnel = self._tunnel_with_file(size=len(body), body=body)
        lookup, tun, auth = _patches(tunnel)
        with lookup, tun, auth:
            r = client.get(
                "/api/v1/bluevela/builds/B1/files/content",
                params={
                    "target_name": "train",
                    "step_name": "step1",
                    "path": "log.txt",
                },
            )
        assert r.status_code == 200, r.text
        assert r.json()["content"] == "hello world"
        assert r.json()["truncated"] is False

    def test_too_large_returns_413(self, client):
        # 2 MiB > default 1 MiB cap
        tunnel = self._tunnel_with_file(size=2 * 1024 * 1024, body=b"x" * 100)
        lookup, tun, auth = _patches(tunnel)
        with lookup, tun, auth:
            r = client.get(
                "/api/v1/bluevela/builds/B1/files/content",
                params={
                    "target_name": "train",
                    "step_name": "step1",
                    "path": "big.bin",
                },
            )
        assert r.status_code == 413

    def test_binary_returns_415(self, client):
        body = b"abc\x00def"
        tunnel = self._tunnel_with_file(size=len(body), body=body)
        lookup, tun, auth = _patches(tunnel)
        with lookup, tun, auth:
            r = client.get(
                "/api/v1/bluevela/builds/B1/files/content",
                params={
                    "target_name": "train",
                    "step_name": "step1",
                    "path": "img.bin",
                },
            )
        assert r.status_code == 415


# ------------------------------------------------------------- /files/download


class TestDownload:
    def test_streams_and_sets_content_disposition(self, client):
        body = b"payload-bytes"
        tunnel = MagicMock()

        async def run_remote(cmd, raise_on_error=True):
            if cmd.startswith("ls -1t"):
                return (0, "launch-NEW\n", "")
            if cmd.startswith("stat -c"):
                return (0, f"{len(body)}\tregular file\n", "")
            return (0, "", "")

        tunnel.run_remote = AsyncMock(side_effect=run_remote)

        sftp = MagicMock()
        fh = MagicMock()
        # Emit once then EOF.
        reads = [body, b""]
        fh.read = AsyncMock(side_effect=reads)
        fh.__aenter__ = AsyncMock(return_value=fh)
        fh.__aexit__ = AsyncMock(return_value=None)
        sftp.open = MagicMock(return_value=fh)
        sftp.exit = MagicMock(return_value=None)
        tunnel.start_sftp = AsyncMock(return_value=sftp)

        lookup, tun, auth = _patches(tunnel)
        with lookup, tun, auth:
            r = client.get(
                "/api/v1/bluevela/builds/B1/files/download",
                params={
                    "target_name": "train",
                    "step_name": "step1",
                    "path": "out.bin",
                },
            )
        assert r.status_code == 200, r.text
        assert r.content == body
        assert r.headers["content-disposition"].endswith('filename="out.bin"')
        assert r.headers["content-length"] == str(len(body))
        assert r.headers["content-type"] == "application/octet-stream"

    def test_download_too_large_returns_413(self, client):
        tunnel = MagicMock()

        async def run_remote(cmd, raise_on_error=True):
            if cmd.startswith("ls -1t"):
                return (0, "launch-NEW\n", "")
            if cmd.startswith("stat -c"):
                return (0, f"{600 * 1024 * 1024}\tregular file\n", "")
            return (0, "", "")

        tunnel.run_remote = AsyncMock(side_effect=run_remote)
        lookup, tun, auth = _patches(tunnel)
        with lookup, tun, auth:
            r = client.get(
                "/api/v1/bluevela/builds/B1/files/download",
                params={
                    "target_name": "train",
                    "step_name": "step1",
                    "path": "huge.bin",
                },
            )
        assert r.status_code == 413


# ------------------------------------------------------- _resolve_lsf_config


class TestResolveLsfConfig:
    """Direct unit tests for the env-config -> SSH params resolver."""

    def test_non_lsf_environment_returns_400(self):
        from gbserver.api import bluevela_tunnel
        from gbserver.types.environmentconfig import EnvironmentConfig

        env_config = EnvironmentConfig(name="kube-env", type="Kubernetes", config={})
        with patch.object(
            bluevela_tunnel.Environment,
            "load_environment_config",
            return_value=(env_config, MagicMock()),
        ):
            with pytest.raises(HTTPException) as ei:
                bluevela_tunnel._resolve_lsf_config("space://environments/kube")
        assert ei.value.status_code == 400
        assert "Lsf" in str(ei.value.detail)

    def test_lsf_returns_fields(self):
        from gbserver.api import bluevela_tunnel
        from gbserver.types.environmentconfig import EnvironmentConfig

        env_config = EnvironmentConfig(
            name="bluevela",
            type="Lsf",
            config={
                "workspace": {"remote_dir": "/ws"},
                "authentication": {
                    "login_nodes": ["node-a", "node-b"],
                    "login_node_username": "ci-user",
                    "login_node_ssh_key": "key-secret",
                },
            },
        )
        with patch.object(
            bluevela_tunnel.Environment,
            "load_environment_config",
            return_value=(env_config, MagicMock()),
        ):
            login_nodes, username, key, ws = bluevela_tunnel._resolve_lsf_config(
                "space://environments/bluevela"
            )
        assert login_nodes == ["node-a", "node-b"]
        assert username == "ci-user"
        assert key == "key-secret"
        assert ws == "/ws"

    def test_missing_field_returns_503(self):
        from gbserver.api import bluevela_tunnel
        from gbserver.types.environmentconfig import EnvironmentConfig

        env_config = EnvironmentConfig(
            name="bluevela",
            type="Lsf",
            config={
                "workspace": {"remote_dir": "/ws"},
                "authentication": {
                    "login_nodes": ["node-a"],
                    # missing login_node_username and login_node_ssh_key
                },
            },
        )
        with patch.object(
            bluevela_tunnel.Environment,
            "load_environment_config",
            return_value=(env_config, MagicMock()),
        ):
            with pytest.raises(HTTPException) as ei:
                bluevela_tunnel._resolve_lsf_config("space://environments/bluevela")
        assert ei.value.status_code == 503
        detail = str(ei.value.detail)
        assert "login_node_username" in detail
        assert "login_node_ssh_key" in detail


# ---------------------------------------------------------------- /files/debug


class TestStepRunDebug:
    def test_happy_path(self, client):
        tunnel = MagicMock()
        step_run_dir = (
            "/ws/llm-build-B1/target-train/target-run-TR1/step-step1/step-run-SR1"
        )
        find_out = (
            f"{step_run_dir}\t4096\t100\tdirectory\n"
            f"{step_run_dir}/launch-NEW\t4096\t101\tdirectory\n"
            f"{step_run_dir}/launch-NEW/stdout.log\t12\t102\tregular file\n"
            f"{step_run_dir}/launch-OLD\t4096\t99\tdirectory\n"
        )

        async def run_remote(cmd, raise_on_error=True):
            if "find " in cmd:
                return (0, find_out, "")
            return (0, "", "")

        tunnel.run_remote = AsyncMock(side_effect=run_remote)
        lookup, tun, auth = _patches(tunnel)
        with lookup, tun, auth:
            r = client.get(
                "/api/v1/bluevela/builds/B1/files/debug",
                params={"target_name": "train", "step_name": "step1"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["step_run_dir"].endswith("step-run-SR1")
        paths = {e["path"] for e in body["entries"]}
        assert "" in paths
        assert "launch-NEW" in paths
        assert "launch-NEW/stdout.log" in paths
        assert "launch-OLD" in paths

    def test_step_run_dir_missing(self, client):
        tunnel = MagicMock()

        async def run_remote(cmd, raise_on_error=True):
            return (1, "", "find: No such file or directory")

        tunnel.run_remote = AsyncMock(side_effect=run_remote)
        lookup, tun, auth = _patches(tunnel)
        with lookup, tun, auth:
            r = client.get(
                "/api/v1/bluevela/builds/B1/files/debug",
                params={"target_name": "train", "step_name": "step1"},
            )
        assert r.status_code == 404
