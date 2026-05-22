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

"""Unit tests for the build-files REST API.

These tests stub out:
  - ``open_lsf_tunnel`` (so no SSH / IBM Cloud is contacted)
  - ``lookup_build`` (so no DB is required)
  - ``authorize_build_access`` (so no auth middleware is required)
  - ``_pick_environment_uri`` (so no target lookup is required)

What we exercise here is the request/response surface: path-traversal
rejection, build-root resolution, size caps, and auth.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from gbserver.api import build_files as build_files_mod
from gbserver.api.builds import builds_api
from gbserver.api.lsf_tunnel import LsfTunnelConfig
from gbserver.storage.stored_build import StoredBuild

# --------------------------------------------------------------------- fixtures


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    # Mount only the builds_api routes; AuthMiddleware is omitted because
    # we stub authorize_build_access in each test.
    app.mount("/api/v1/builds", builds_api)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _fake_build() -> StoredBuild:
    return StoredBuild(
        uuid="B1",
        name="b",
        space_name="space-a",
        source_uri="",
        username="alice",
    )


def _fake_tunnel_cm(tunnel_mock):
    """Return an async context manager yielding (tunnel, LsfTunnelConfig)."""

    @asynccontextmanager
    async def _cm(space_name: str, environment_uri: str):
        yield tunnel_mock, LsfTunnelConfig(workspace_remote_dir="/ws")

    return _cm


def _patches(
    tunnel_mock,
    *,
    build: StoredBuild | None = None,
    authorize_raises: Exception | None = None,
    lookup_raises: Exception | None = None,
):
    build = build or _fake_build()

    if lookup_raises is not None:
        lookup_build = patch.object(
            build_files_mod, "lookup_build", side_effect=lookup_raises
        )
    else:
        lookup_build = patch.object(build_files_mod, "lookup_build", return_value=build)
    tunnel = patch.object(
        build_files_mod, "open_lsf_tunnel", _fake_tunnel_cm(tunnel_mock)
    )
    auth = patch.object(
        build_files_mod,
        "authorize_build_access",
        side_effect=(authorize_raises if authorize_raises else (lambda *a, **kw: None)),
    )
    pick_env = patch.object(
        build_files_mod, "_pick_environment_uri", return_value="env://x"
    )
    return lookup_build, tunnel, auth, pick_env


def _tunnel_with_listing(entries: str, *, find_out: str | None = None):
    """Mock tunnel for listing flows.

    ``readlink -f`` echoes the input. ``ls -1A`` returns ``entries``. If
    ``find_out`` is provided, recursive ``find`` returns it.
    """
    tunnel = MagicMock()

    async def run_remote(cmd, raise_on_error=True):
        if cmd.startswith("readlink -f"):
            target = cmd.split("--", 1)[1].strip().strip("'\"")
            return (0, target + "\n", "")
        if "find " in cmd:
            return (0, find_out or "", "")
        if cmd.startswith("ls -1A"):
            return (0, entries, "")
        return (0, "", "")

    tunnel.run_remote = AsyncMock(side_effect=run_remote)
    return tunnel


# ---------------------------------------------------------------------- /files


class TestListFiles:
    def test_build_root_listing(self, client):
        tunnel = _tunnel_with_listing("target-train\n.gbstate\n")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"path": "."},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert sorted(body) == [".gbstate", "target-train"]

    def test_recursive_returns_nested_paths(self, client):
        find_out = (
            "/ws/llm-build-B1/a.txt\n"
            "/ws/llm-build-B1/sub\n"
            "/ws/llm-build-B1/sub/nested.log\n"
        )
        tunnel = _tunnel_with_listing("", find_out=find_out)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"path": ".", "recursive": "true"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body == sorted(["a.txt", "sub", "sub/nested.log"])
        # And confirm a `find` command was actually issued.
        cmds = [call.args[0] for call in tunnel.run_remote.await_args_list]
        assert any(c.startswith("set -o pipefail; find ") for c in cmds)

    def test_recursive_default_false(self, client):
        tunnel = _tunnel_with_listing("a.txt\nb.log\n")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"path": "."},
            )
        assert r.status_code == 200, r.text
        cmds = [call.args[0] for call in tunnel.run_remote.await_args_list]
        # Single-level branch: ls used, find absent.
        assert any(c.startswith("ls -1A ") for c in cmds)
        assert not any("find " in c for c in cmds)

    def test_recursive_traversal_still_rejected(self, client):
        tunnel = _tunnel_with_listing("")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"path": "../../etc", "recursive": "true"},
            )
        assert r.status_code == 400
        for call in tunnel.run_remote.await_args_list:
            assert "etc" not in call.args[0]

    def test_missing_path_defaults_to_dot(self, client):
        tunnel = _tunnel_with_listing("target-train\n.gbstate\n")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get("/api/v1/builds/B1/files")
        assert r.status_code == 200, r.text
        assert sorted(r.json()) == [".gbstate", "target-train"]
        # Default of "." resolves to the build root, so ls -1A runs there.
        cmds = [call.args[0] for call in tunnel.run_remote.await_args_list]
        assert any(c.startswith("ls -1A ") and "llm-build-B1" in c for c in cmds)

    def test_path_traversal_rejected(self, client):
        tunnel = _tunnel_with_listing("")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"path": "../../etc/passwd"},
            )
        assert r.status_code == 400
        # readlink/ls must never have been invoked with the hostile path.
        for call in tunnel.run_remote.await_args_list:
            assert "etc/passwd" not in call.args[0]

    def test_absolute_path_rejected(self, client):
        tunnel = _tunnel_with_listing("")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"path": "/etc/passwd"},
            )
        assert r.status_code == 400

    def test_null_byte_rejected(self, client):
        tunnel = _tunnel_with_listing("")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"path": "a\x00b"},
            )
        assert r.status_code == 400

    def test_backslash_rejected(self, client):
        tunnel = _tunnel_with_listing("")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"path": "a\\b"},
            )
        assert r.status_code == 400

    def test_symlink_escape_returns_404(self, client):
        # readlink resolves to /etc/passwd — outside build_root /ws/llm-build-B1.
        tunnel = MagicMock()

        async def run_remote(cmd, raise_on_error=True):
            if cmd.startswith("readlink -f"):
                return (0, "/etc/passwd\n", "")
            return (0, "", "")

        tunnel.run_remote = AsyncMock(side_effect=run_remote)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"path": "evil-symlink"},
            )
        assert r.status_code == 404

    def test_unauthorized(self, client):
        tunnel = MagicMock()
        tunnel.run_remote = AsyncMock(return_value=(0, "", ""))
        lb, tun, auth, env = _patches(
            tunnel,
            authorize_raises=HTTPException(status_code=401, detail="no"),
        )
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"path": "."},
            )
        assert r.status_code == 401

    def test_missing_build_returns_404(self, client):
        tunnel = MagicMock()
        tunnel.run_remote = AsyncMock(return_value=(0, "", ""))
        lb, tun, auth, env = _patches(
            tunnel,
            lookup_raises=HTTPException(status_code=404, detail="build not found"),
        )
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"path": "."},
            )
        assert r.status_code == 404


# ----------------------------------------------------- /files (pattern filter)


def _tunnel_with_grep(grep_rc: int, grep_out: str = "", grep_err: str = ""):
    """Mock tunnel where any piped grep command returns the supplied result.

    ``readlink -f`` echoes the input. Any command containing ``grep -F``
    (the listing's substring filter or the search endpoint) returns
    ``(grep_rc, grep_out, grep_err)``.
    """
    tunnel = MagicMock()

    async def run_remote(cmd, raise_on_error=True):
        if cmd.startswith("readlink -f"):
            target = cmd.split("--", 1)[1].strip().strip("'\"")
            return (0, target + "\n", "")
        if "grep -F" in cmd or "grep -r" in cmd:
            return (grep_rc, grep_out, grep_err)
        return (0, "", "")

    tunnel.run_remote = AsyncMock(side_effect=run_remote)
    return tunnel


class TestListFilesPattern:
    def test_pattern_filters_listing(self, client):
        tunnel = _tunnel_with_grep(0, "a.log\nsub.log\n")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"path": ".", "pattern": ".log"},
            )
        assert r.status_code == 200, r.text
        assert sorted(r.json()) == ["a.log", "sub.log"]
        cmds = [call.args[0] for call in tunnel.run_remote.await_args_list]
        # Filter is applied via piped grep -F, with pipefail and head cap.
        assert any(
            "set -o pipefail" in c and "grep -F --" in c and "head -n" in c
            for c in cmds
        )

    def test_pattern_recursive_filters_listing(self, client):
        find_out = "/ws/llm-build-B1/a.log\n" "/ws/llm-build-B1/sub/nested.log\n"
        tunnel = _tunnel_with_grep(0, find_out)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"path": ".", "recursive": "true", "pattern": ".log"},
            )
        assert r.status_code == 200, r.text
        assert sorted(r.json()) == ["a.log", "sub/nested.log"]
        cmds = [call.args[0] for call in tunnel.run_remote.await_args_list]
        assert any("find " in c and "grep -F --" in c for c in cmds)

    def test_pattern_no_matches_returns_empty(self, client):
        # grep exits 1 with empty stdout when nothing matches.
        tunnel = _tunnel_with_grep(1, "", "")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"path": ".", "pattern": "nope"},
            )
        assert r.status_code == 200, r.text
        assert r.json() == []

    def test_pattern_command_failure_returns_500(self, client):
        # rc>=2 from grep (or upstream under pipefail) is a real failure.
        # stderr that doesn't contain a missing-path signature -> 500.
        tunnel = _tunnel_with_grep(2, "", "grep: out of memory")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"path": ".", "pattern": "x"},
            )
        assert r.status_code == 500

    def test_pattern_with_newline_rejected(self, client):
        tunnel = _tunnel_with_grep(0, "")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"path": ".", "pattern": "a\nb"},
            )
        assert r.status_code == 400
        for call in tunnel.run_remote.await_args_list:
            assert "a\nb" not in call.args[0]


# --------------------------------------------------------------- /files/search


class TestSearchFiles:
    def test_search_returns_hits(self, client):
        out = (
            "/ws/llm-build-B1/a.txt:1:hello world\n"
            "/ws/llm-build-B1/sub/b.txt:42:world!\n"
        )
        tunnel = _tunnel_with_grep(0, out)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files/search",
                params={"pattern": "world"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body == [
            {"path": "a.txt", "line": 1, "text": "hello world"},
            {"path": "sub/b.txt", "line": 42, "text": "world!"},
        ]
        cmds = [call.args[0] for call in tunnel.run_remote.await_args_list]
        # grep -r -n -I -H -F flags must all be present.
        assert any("grep -r -n -I -H -F" in c and "head -n" in c for c in cmds)

    def test_search_no_matches_returns_empty(self, client):
        tunnel = _tunnel_with_grep(1, "", "")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files/search",
                params={"pattern": "nope"},
            )
        assert r.status_code == 200, r.text
        assert r.json() == []

    def test_search_ignore_case_passes_flag(self, client):
        tunnel = _tunnel_with_grep(0, "/ws/llm-build-B1/a.txt:1:HELLO\n")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files/search",
                params={"pattern": "hello", "ignore_case": "true"},
            )
        assert r.status_code == 200, r.text
        cmds = [call.args[0] for call in tunnel.run_remote.await_args_list]
        assert any("grep -r -n -I -H -F -i" in c for c in cmds)

    def test_search_long_text_truncated(self, client):
        long_text = "x" * 2000
        out = f"/ws/llm-build-B1/a.txt:1:{long_text}\n"
        tunnel = _tunnel_with_grep(0, out)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files/search",
                params={"pattern": "x"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body) == 1
        # Default cap is 512 bytes.
        assert len(body[0]["text"]) == 512

    def test_search_text_with_colons_preserved(self, client):
        # Match text contains its own ':' — split must use maxsplit=2.
        out = "/ws/llm-build-B1/a.yaml:7:key: value: nested\n"
        tunnel = _tunnel_with_grep(0, out)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files/search",
                params={"pattern": "value"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body == [{"path": "a.yaml", "line": 7, "text": "key: value: nested"}]

    def test_search_path_traversal_rejected(self, client):
        tunnel = _tunnel_with_grep(0, "")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files/search",
                params={"pattern": "x", "path": "../../etc"},
            )
        assert r.status_code == 400
        for call in tunnel.run_remote.await_args_list:
            assert "etc" not in call.args[0]

    def test_search_pattern_with_null_rejected(self, client):
        tunnel = _tunnel_with_grep(0, "")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files/search",
                params={"pattern": "a\x00b"},
            )
        assert r.status_code == 400

    def test_search_missing_pattern_returns_422(self, client):
        tunnel = _tunnel_with_grep(0, "")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get("/api/v1/builds/B1/files/search")
        assert r.status_code == 422

    def test_search_unauthorized(self, client):
        tunnel = _tunnel_with_grep(0, "")
        lb, tun, auth, env = _patches(
            tunnel,
            authorize_raises=HTTPException(status_code=401, detail="no"),
        )
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files/search",
                params={"pattern": "x"},
            )
        assert r.status_code == 401


# -------------------------------------------------------------- /file/download


class TestDownloadFile:
    def _tunnel_with_file(self, size: int, body: bytes, *, is_dir: bool = False):
        tunnel = MagicMock()

        async def run_remote(cmd, raise_on_error=True):
            if cmd.startswith("readlink -f"):
                target = cmd.split("--", 1)[1].strip().strip("'\"")
                return (0, target + "\n", "")
            if cmd.startswith("stat -c"):
                kind = "directory" if is_dir else "regular file"
                return (0, f"{size}\t{kind}\n", "")
            return (0, "", "")

        tunnel.run_remote = AsyncMock(side_effect=run_remote)

        sftp = MagicMock()
        fh = MagicMock()
        # Stream body in one chunk, then EOF on subsequent reads.
        fh.read = AsyncMock(side_effect=[body, b""])
        fh.__aenter__ = AsyncMock(return_value=fh)
        fh.__aexit__ = AsyncMock(return_value=None)
        sftp.open = MagicMock(return_value=fh)
        sftp.exit = MagicMock(return_value=None)
        tunnel.start_sftp = AsyncMock(return_value=sftp)
        return tunnel

    def test_download_build_root_scope(self, client):
        body = b"yaml: yes\n"
        tunnel = self._tunnel_with_file(size=len(body), body=body)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/file/download",
                params={"path": "builds.yaml"},
            )
        assert r.status_code == 200, r.text
        assert r.content == body
        assert 'filename="builds.yaml"' in r.headers["content-disposition"]

    def test_download_directory_returns_400(self, client):
        tunnel = self._tunnel_with_file(size=4096, body=b"", is_dir=True)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/file/download",
                params={"path": "subdir"},
            )
        assert r.status_code == 400

    def test_download_too_large_returns_413(self, client):
        # 2 GiB > default 1 GiB cap
        tunnel = self._tunnel_with_file(size=2 * 1024**3, body=b"x" * 100)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/file/download",
                params={"path": "big.bin"},
            )
        assert r.status_code == 413

    def test_download_path_traversal_rejected(self, client):
        tunnel = self._tunnel_with_file(size=10, body=b"x" * 10)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/file/download",
                params={"path": "../../etc/passwd"},
            )
        assert r.status_code == 400
        for call in tunnel.run_remote.await_args_list:
            assert "etc/passwd" not in call.args[0]

    def test_download_missing_file_returns_404(self, client):
        tunnel = MagicMock()

        async def run_remote(cmd, raise_on_error=True):
            if cmd.startswith("readlink -f"):
                return (1, "", "readlink: cannot access: No such file")
            return (0, "", "")

        tunnel.run_remote = AsyncMock(side_effect=run_remote)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/file/download",
                params={"path": "missing.txt"},
            )
        assert r.status_code == 404

    def test_download_unauthorized(self, client):
        tunnel = MagicMock()
        tunnel.run_remote = AsyncMock(return_value=(0, "", ""))
        lb, tun, auth, env = _patches(
            tunnel,
            authorize_raises=HTTPException(status_code=401, detail="no"),
        )
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/file/download",
                params={"path": "log.txt"},
            )
        assert r.status_code == 401

    def test_download_missing_path_returns_422(self, client):
        tunnel = MagicMock()
        tunnel.run_remote = AsyncMock(return_value=(0, "", ""))
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get("/api/v1/builds/B1/file/download")
        assert r.status_code == 422

    def test_download_non_ascii_filename(self, client):
        body = b"data"
        tunnel = self._tunnel_with_file(size=len(body), body=body)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/file/download",
                params={"path": "rapport-é.txt"},
            )
        assert r.status_code == 200, r.text
        cd = r.headers["content-disposition"]
        # ASCII fallback: non-ASCII char becomes "?" via "replace".
        assert 'filename="rapport-?.txt"' in cd
        # RFC 5987 form: percent-encoded UTF-8.
        assert "filename*=UTF-8''rapport-%C3%A9.txt" in cd


# ------------------------------------------------------- _resolve_lsf_config


class TestResolveLsfConfig:
    """Direct unit tests for the env-config -> SSH params resolver."""

    def test_non_lsf_environment_returns_400(self):
        from gbserver.api import lsf_tunnel
        from gbserver.types.environmentconfig import EnvironmentConfig

        env_config = EnvironmentConfig(name="kube-env", type="Kubernetes", config={})
        with patch.object(
            lsf_tunnel.Environment,
            "load_environment_config",
            return_value=(env_config, MagicMock()),
        ):
            with pytest.raises(HTTPException) as ei:
                lsf_tunnel._resolve_lsf_config("space://environments/kube")
        assert ei.value.status_code == 400
        assert "Lsf" in str(ei.value.detail)

    def test_lsf_returns_fields(self):
        from gbserver.api import lsf_tunnel
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
            lsf_tunnel.Environment,
            "load_environment_config",
            return_value=(env_config, MagicMock()),
        ):
            login_nodes, username, key, ws = lsf_tunnel._resolve_lsf_config(
                "space://environments/bluevela"
            )
        assert login_nodes == ["node-a", "node-b"]
        assert username == "ci-user"
        assert key == "key-secret"
        assert ws == "/ws"

    def test_missing_field_returns_503(self):
        from gbserver.api import lsf_tunnel
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
            lsf_tunnel.Environment,
            "load_environment_config",
            return_value=(env_config, MagicMock()),
        ):
            with pytest.raises(HTTPException) as ei:
                lsf_tunnel._resolve_lsf_config("space://environments/bluevela")
        assert ei.value.status_code == 503
        detail = str(ei.value.detail)
        assert "login_node_username" in detail
        assert "login_node_ssh_key" in detail
