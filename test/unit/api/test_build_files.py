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

    def test_pattern_regex_flag(self, client):
        tunnel = _tunnel_with_grep(0, "a.log\n")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"path": ".", "pattern": r"\.log$", "regex": "true"},
            )
        assert r.status_code == 200, r.text
        cmds = [call.args[0] for call in tunnel.run_remote.await_args_list]
        assert any("grep -E --" in c for c in cmds)
        assert not any("grep -F --" in c for c in cmds)


# --------------------------------------------------------------- /files (stat)


def _tunnel_with_find_printf(
    rc: int, stdout: str, *, readlink_target: str | None = None
):
    """Mock tunnel for the stat=true find -printf branch of /files."""
    tunnel = MagicMock()

    async def run_remote(cmd, raise_on_error=True):
        if cmd.startswith("readlink -f"):
            target = cmd.split("--", 1)[1].strip().strip("'\"")
            return (0, (readlink_target or target) + "\n", "")
        if "find " in cmd and "-printf" in cmd:
            return (rc, stdout, "")
        return (0, "", "")

    tunnel.run_remote = AsyncMock(side_effect=run_remote)
    return tunnel


class TestListFilesStat:
    def test_stat_returns_file_entries(self, client):
        # find -printf '%P\t%y\t%s\t%T@\n' output. -mindepth 1 -maxdepth 1
        # under build root.
        out = (
            "a.log\tf\t1234\t1700000000.0\n"
            "sub\td\t4096\t1700000100.5\n"
            "ln\tl\t10\t1700000200.0\n"
        )
        tunnel = _tunnel_with_find_printf(0, out)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"stat": "true"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body == [
            {"path": "a.log", "type": "file", "size": 1234, "mtime": 1700000000},
            {"path": "ln", "type": "symlink", "size": 10, "mtime": 1700000200},
            {"path": "sub", "type": "dir", "size": 4096, "mtime": 1700000100},
        ]
        cmds = [call.args[0] for call in tunnel.run_remote.await_args_list]
        # find -mindepth 1 -maxdepth 1 -printf …
        assert any("-mindepth 1" in c and "-maxdepth 1" in c for c in cmds)
        assert any("-printf" in c for c in cmds)

    def test_stat_recursive_omits_maxdepth(self, client):
        out = (
            "a.log\tf\t1\t1700000000\n"
            "sub/nested.log\tf\t2\t1700000001\n"
        )
        tunnel = _tunnel_with_find_printf(0, out)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"stat": "true", "recursive": "true"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert [e["path"] for e in body] == ["a.log", "sub/nested.log"]
        cmds = [call.args[0] for call in tunnel.run_remote.await_args_list]
        assert any(
            "-mindepth 1" in c and "-maxdepth 1" not in c and "-printf" in c
            for c in cmds
        )

    def test_stat_with_pattern_filters_in_python(self, client):
        out = (
            "a.log\tf\t1\t1700000000\n"
            "b.txt\tf\t2\t1700000001\n"
            "c.log\tf\t3\t1700000002\n"
        )
        tunnel = _tunnel_with_find_printf(0, out)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"stat": "true", "pattern": ".log"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert [e["path"] for e in body] == ["a.log", "c.log"]

    def test_stat_with_regex_pattern(self, client):
        out = (
            "a.log\tf\t1\t1700000000\n"
            "b.txt\tf\t2\t1700000001\n"
        )
        tunnel = _tunnel_with_find_printf(0, out)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"stat": "true", "pattern": r"\.log$", "regex": "true"},
            )
        assert r.status_code == 200, r.text
        assert [e["path"] for e in r.json()] == ["a.log"]

    def test_stat_invalid_regex_returns_400(self, client):
        out = "a.log\tf\t1\t1700000000\n"
        tunnel = _tunnel_with_find_printf(0, out)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files",
                params={"stat": "true", "pattern": "[bad", "regex": "true"},
            )
        assert r.status_code == 400


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
        # Strip optional fields for the equality check; covered separately.
        slim = [
            {"path": h["path"], "line": h["line"], "text": h["text"]} for h in body
        ]
        assert slim == [
            {"path": "a.txt", "line": 1, "text": "hello world"},
            {"path": "sub/b.txt", "line": 42, "text": "world!"},
        ]
        assert all(h["is_match"] for h in body)
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

    def test_search_grep_failure_with_stderr_is_500(self, client):
        # rc=1 with non-empty stderr is NOT "no matches" — it's a pipeline
        # failure under pipefail (e.g. head crash, I/O error). Must surface
        # as 500 instead of being masked as an empty result list.
        tunnel = _tunnel_with_grep(1, "", "head: I/O error")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files/search",
                params={"pattern": "x"},
            )
        assert r.status_code == 500, r.text
        assert "I/O error" in r.text

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
        # Match text contains its own ':' — parser must keep the rest of
        # the line intact after the lineno.
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
        assert len(body) == 1
        assert body[0]["path"] == "a.yaml"
        assert body[0]["line"] == 7
        assert body[0]["text"] == "key: value: nested"
        assert body[0]["is_match"] is True

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

    def test_search_path_with_hyphen_digit_hyphen_segments(self, client):
        # Filenames frequently contain "-<digits>-" substrings (UUIDs, run
        # IDs). The parser must anchor on the rightmost <sep>\d+<sep>, not
        # the leftmost — otherwise it splits inside the path and reports a
        # truncated path with the wrong lineno.
        out = (
            "/ws/llm-build-B1/run-1234-abcd/file.txt:42:hit line\n"
            "/ws/llm-build-B1/a-9-b-7-c.log-100-context line\n"
        )
        tunnel = _tunnel_with_grep(0, out)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files/search",
                params={"pattern": "x", "before": 1, "after": 1},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body) == 2
        # Match line: full path preserved, lineno is the trailing digit run.
        assert body[0]["path"] == "run-1234-abcd/file.txt"
        assert body[0]["line"] == 42
        assert body[0]["text"] == "hit line"
        assert body[0]["is_match"] is True
        # Context line: path with multiple "-<digits>-" segments preserved.
        assert body[1]["path"] == "a-9-b-7-c.log"
        assert body[1]["line"] == 100
        assert body[1]["text"] == "context line"
        assert body[1]["is_match"] is False

    def test_search_with_context_before_after(self, client):
        # grep -B1 -A1 output: context lines use ":" between path/lineno is
        # NOT used — they use "-". Match line uses ":". Two distinct hit
        # groups separated by "--".
        out = (
            "/ws/llm-build-B1/a.log-9-warmup\n"
            "/ws/llm-build-B1/a.log:10:Traceback\n"
            "/ws/llm-build-B1/a.log-11-  File ...\n"
            "--\n"
            "/ws/llm-build-B1/a.log-99-...\n"
            "/ws/llm-build-B1/a.log:100:Traceback\n"
            "/ws/llm-build-B1/a.log-101-end\n"
        )
        tunnel = _tunnel_with_grep(0, out)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files/search",
                params={"pattern": "Traceback", "before": 1, "after": 1},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        # Expect 6 entries (2 matches + 4 context), -- separator skipped.
        assert len(body) == 6
        matches = [h for h in body if h["is_match"]]
        contexts = [h for h in body if not h["is_match"]]
        assert len(matches) == 2 and len(contexts) == 4
        assert all(h["path"] == "a.log" for h in body)
        # The grep flags include -B 1 and -A 1.
        cmds = [call.args[0] for call in tunnel.run_remote.await_args_list]
        assert any("-B 1" in c and "-A 1" in c for c in cmds)

    def test_search_context_max_capped(self, client):
        tunnel = _tunnel_with_grep(0, "")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files/search",
                params={"pattern": "x", "before": 9999},
            )
        # Pydantic Query(le=BUILD_FILES_GREP_MAX_CONTEXT) -> 422.
        assert r.status_code == 422

    def test_search_regex_flag(self, client):
        tunnel = _tunnel_with_grep(0, "/ws/llm-build-B1/a.log:1:OOMKilled\n")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files/search",
                params={"pattern": r"\bOOMKilled\b", "regex": "true"},
            )
        assert r.status_code == 200, r.text
        cmds = [call.args[0] for call in tunnel.run_remote.await_args_list]
        # -E flag, no -F flag.
        assert any("grep -r -n -I -H -E" in c for c in cmds)
        assert not any("grep -r -n -I -H -F" in c for c in cmds)

    def test_search_stat_metadata(self, client):
        tunnel = MagicMock()

        async def run_remote(cmd, raise_on_error=True):
            if cmd.startswith("readlink -f"):
                target = cmd.split("--", 1)[1].strip().strip("'\"")
                return (0, target + "\n", "")
            if "grep -r" in cmd:
                return (0, "/ws/llm-build-B1/a.log:1:hit\n", "")
            if cmd.startswith("stat -c"):
                # Single batched stat call for the one distinct hit file.
                return (
                    0,
                    "/ws/llm-build-B1/a.log\t1234\t1700000000\n",
                    "",
                )
            return (0, "", "")

        tunnel.run_remote = AsyncMock(side_effect=run_remote)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files/search",
                params={"pattern": "hit", "stat": "true"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body == [
            {
                "path": "a.log",
                "line": 1,
                "text": "hit",
                "is_match": True,
                "size": 1234,
                "mtime": 1700000000,
            }
        ]

    def test_search_stat_false_omits_metadata_fields(self, client):
        # Defaults still serialize size/mtime as None — existing callers
        # see them as nulls (or ignore unknown fields) but no breakage.
        out = "/ws/llm-build-B1/a.txt:1:hello\n"
        tunnel = _tunnel_with_grep(0, out)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files/search",
                params={"pattern": "hello"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body[0]["is_match"] is True
        assert body[0]["size"] is None
        assert body[0]["mtime"] is None

    def test_search_text_with_carriage_returns(self, client):
        # tqdm-style progress bars rewrite a single source line many
        # times with '\r'. The grep stdout for one such match is one line
        # terminated by '\n', but the line itself contains embedded '\r'.
        # Splitting that on '\r' shatters the record — the parsed `text`
        # ends up empty or a meaningless prefix. We must split on '\n'
        # only and replace '\r' with a space in the rendered text.
        out = "/ws/llm-build-B1/job.log:532:before\rmiddle\rtrain_runtime: 1.364\n"
        tunnel = _tunnel_with_grep(0, out)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files/search",
                params={"pattern": "train_runtime"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body) == 1
        assert body[0]["path"] == "job.log"
        assert body[0]["line"] == 532
        assert body[0]["is_match"] is True
        # Real content survives, '\r' is replaced with spaces.
        assert "train_runtime: 1.364" in body[0]["text"]
        assert "\r" not in body[0]["text"]

    def test_search_long_line_with_embedded_colons_not_corrupted(self, client):
        # A single very long source line containing what looks like a
        # `:NNNN:` triple in its body must not be re-split into a fake
        # second hit. Bug symptom: `path` field contained inline
        # `path:line:text` content because '\r'-splitting (or any other
        # mid-line split) turned the chunk into an apparent grep record.
        # Use a body that includes both an embedded '\r' (the realistic
        # trigger) and a `:5849:` substring (the fake-record bait).
        long_body = (
            "output_values {'a': 1}" + ("\rprogress " * 50)
            + " job.log:5849:more text " + ("x" * 4000)
        )
        out = f"/ws/llm-build-B1/job.log:18:{long_body}\n"
        tunnel = _tunnel_with_grep(0, out)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files/search",
                params={"pattern": "epoch"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        # Exactly one hit — no fake second record from the embedded
        # `:5849:` substring.
        assert len(body) == 1
        # path field is the clean relative path, never inline content.
        assert body[0]["path"] == "job.log"
        assert ":" not in body[0]["path"]
        assert body[0]["line"] == 18
        # text is truncated by the byte cap, with no '\r' surviving.
        assert len(body[0]["text"]) <= 512
        assert "\r" not in body[0]["text"]

    def test_search_does_not_append_trailing_slash(self, client):
        # The grep command must not append a trailing '/' to the search
        # root — that turned single-file `path=` requests into a 500
        # ("grep: <file>/: Not a directory"). With -H, grep emits
        # filenames for both file and directory targets without it.
        tunnel = _tunnel_with_grep(0, "")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/files/search",
                params={"pattern": "x", "path": "outputs/job.log"},
            )
        assert r.status_code == 200, r.text
        cmds = [call.args[0] for call in tunnel.run_remote.await_args_list]
        grep_cmds = [c for c in cmds if "grep -r" in c]
        assert grep_cmds, "expected at least one grep command"
        for c in grep_cmds:
            # The bug appended `<quoted_path>/ ` before the head pipe.
            assert "/job.log/" not in c, c
            assert "outputs/job.log/'" not in c, c


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

    @pytest.mark.asyncio
    async def test_stream_sftp_file_closes_client_on_open_failure(self):
        # If sftp.open() raises after start_sftp() succeeds (e.g. file
        # vanished between the size check and the stream open, or
        # permission denied), the SFTP client must still be closed.
        # Exercised at the helper level because TestClient's streaming
        # error propagation differs from a real ASGI server.
        tunnel = MagicMock()
        sftp = MagicMock()
        sftp.open = MagicMock(side_effect=OSError("permission denied"))
        sftp.exit = MagicMock(return_value=None)
        tunnel.start_sftp = AsyncMock(return_value=sftp)

        gen = build_files_mod._stream_sftp_file(tunnel, "/ws/log.txt")
        with pytest.raises(OSError):
            await gen.__anext__()
        sftp.exit.assert_called_once()


# ----------------------------------------------------- /file/download peek mode


def _tunnel_for_peek(
    peek_stdout: str,
    *,
    is_dir: bool = False,
    size: int = 100,
    peek_rc: int = 0,
    peek_stderr: str = "",
):
    """Mock tunnel for /file/download peek mode.

    ``readlink -f`` echoes its arg. ``stat -c '%s\\t%F'`` returns size and
    type. Any pipefail+head/tail/sed pipeline returns ``peek_stdout``.
    """
    tunnel = MagicMock()

    async def run_remote(cmd, raise_on_error=True):
        if cmd.startswith("readlink -f"):
            target = cmd.split("--", 1)[1].strip().strip("'\"")
            return (0, target + "\n", "")
        if cmd.startswith("stat -c '%s"):
            kind = "directory" if is_dir else "regular file"
            return (0, f"{size}\t{kind}\n", "")
        if "set -o pipefail" in cmd and (
            "head -n" in cmd or "tail -n" in cmd or "sed -n" in cmd
        ):
            return (peek_rc, peek_stdout, peek_stderr)
        return (0, "", "")

    tunnel.run_remote = AsyncMock(side_effect=run_remote)
    return tunnel


class TestDownloadPeek:
    def test_peek_head_returns_text_plain(self, client):
        tunnel = _tunnel_for_peek("line1\nline2\nline3\n")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/file/download",
                params={"path": "stderr.log", "head": 3},
            )
        assert r.status_code == 200, r.text
        assert r.text == "line1\nline2\nline3\n"
        assert r.headers["content-type"].startswith("text/plain")
        assert "content-disposition" not in r.headers
        cmds = [call.args[0] for call in tunnel.run_remote.await_args_list]
        assert any("head -n 3" in c and "head -c " in c for c in cmds)

    def test_peek_tail_returns_text_plain(self, client):
        tunnel = _tunnel_for_peek("end1\nend2\n")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/file/download",
                params={"path": "stderr.log", "tail": 200},
            )
        assert r.status_code == 200, r.text
        assert r.text == "end1\nend2\n"
        cmds = [call.args[0] for call in tunnel.run_remote.await_args_list]
        assert any("tail -n 200" in c for c in cmds)

    def test_peek_range_uses_sed(self, client):
        tunnel = _tunnel_for_peek("middle\n")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/file/download",
                params={"path": "stderr.log", "range": "5-7"},
            )
        assert r.status_code == 200, r.text
        cmds = [call.args[0] for call in tunnel.run_remote.await_args_list]
        assert any("sed -n " in c and "5,7p" in c for c in cmds)

    def test_peek_skips_size_cap(self, client):
        # 50 GiB file — way over BUILD_FILES_DOWNLOAD_MAX_BYTES (1 GiB).
        # Peek mode should still succeed because the cap doesn't apply.
        tunnel = _tunnel_for_peek("tail content\n", size=50 * 1024**3)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/file/download",
                params={"path": "huge.log", "tail": 100},
            )
        assert r.status_code == 200, r.text
        assert r.text == "tail content\n"

    def test_peek_pipefail_sigpipe_treated_as_success(self, client):
        # head -c truncates the producer mid-stream → SIGPIPE → rc=141
        # under pipefail. We must still return the truncated bytes as 200.
        tunnel = _tunnel_for_peek("partial\n", peek_rc=141)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/file/download",
                params={"path": "log", "head": 100},
            )
        assert r.status_code == 200, r.text
        assert r.text == "partial\n"

    def test_peek_directory_returns_400(self, client):
        tunnel = _tunnel_for_peek("", is_dir=True)
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/file/download",
                params={"path": "subdir", "head": 10},
            )
        assert r.status_code == 400

    def test_peek_mutual_exclusion_head_tail(self, client):
        tunnel = _tunnel_for_peek("")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/file/download",
                params={"path": "log", "head": 10, "tail": 10},
            )
        assert r.status_code == 400

    def test_peek_range_malformed_returns_422(self, client):
        # Pydantic Query(pattern=...) -> 422 before we reach the handler.
        tunnel = _tunnel_for_peek("")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/file/download",
                params={"path": "log", "range": "abc"},
            )
        assert r.status_code == 422

    def test_peek_range_inverted_returns_400(self, client):
        tunnel = _tunnel_for_peek("")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/file/download",
                params={"path": "log", "range": "10-5"},
            )
        assert r.status_code == 400

    def test_peek_head_param_capped_at_max(self, client):
        # ge=1, le=BUILD_FILES_PEEK_MAX_LINES (10000) → 422.
        tunnel = _tunnel_for_peek("")
        lb, tun, auth, env = _patches(tunnel)
        with lb, tun, auth, env:
            r = client.get(
                "/api/v1/builds/B1/file/download",
                params={"path": "log", "head": 999999},
            )
        assert r.status_code == 422


# ------------------------------------------------------- _resolve_lsf_config


class TestResolveLsfConfig:
    """Direct unit tests for the env-config -> SSH params resolver."""

    @pytest.mark.asyncio
    async def test_non_lsf_environment_returns_400(self):
        from gbserver.api import lsf_tunnel
        from gbserver.types.environmentconfig import EnvironmentConfig

        env_config = EnvironmentConfig(name="kube-env", type="Kubernetes", config={})
        with patch.object(
            lsf_tunnel.Environment,
            "load_environment_config",
            return_value=(env_config, MagicMock()),
        ):
            with pytest.raises(HTTPException) as ei:
                await lsf_tunnel._resolve_lsf_config("space://environments/kube")
        assert ei.value.status_code == 400
        assert "Lsf" in str(ei.value.detail)

    @pytest.mark.asyncio
    async def test_lsf_returns_fields(self):
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
            login_nodes, username, key, ws = await lsf_tunnel._resolve_lsf_config(
                "space://environments/bluevela"
            )
        assert login_nodes == ["node-a", "node-b"]
        assert username == "ci-user"
        assert key == "key-secret"
        assert ws == "/ws"

    @pytest.mark.asyncio
    async def test_missing_field_returns_503(self):
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
                await lsf_tunnel._resolve_lsf_config("space://environments/bluevela")
        assert ei.value.status_code == 503
        detail = str(ei.value.detail)
        assert "login_node_username" in detail
        assert "login_node_ssh_key" in detail


# ----------------------------------------------------- open_lsf_tunnel failover


class TestOpenLsfTunnelFailover:
    """Failover tests: open_lsf_tunnel retries the next login node when
    SshTunnel.open() raises SshTunnelError, and surfaces 503 only when
    every node fails."""

    @staticmethod
    def _patch_resolvers(login_nodes):
        """Stub _resolve_lsf_config + _fetch_ssh_key_for_space + _write_key_file
        so the test focuses on the retry loop only."""
        from gbserver.api import lsf_tunnel

        resolve = patch.object(
            lsf_tunnel,
            "_resolve_lsf_config",
            new=AsyncMock(return_value=(login_nodes, "u", "secret", "/ws")),
        )
        fetch = patch.object(
            lsf_tunnel,
            "_fetch_ssh_key_for_space",
            new=AsyncMock(return_value="KEY"),
        )
        write = patch.object(
            lsf_tunnel, "_write_key_file", return_value="/tmp/fake.key"
        )
        unlink = patch("os.unlink")
        return resolve, fetch, write, unlink

    @pytest.mark.asyncio
    async def test_single_node_open_failure_returns_503(self):
        from gbserver.api import lsf_tunnel
        from gbserver.utils.ssh_tunnel import SshTunnelError

        resolve, fetch, write, unlink = self._patch_resolvers(["node-a"])

        constructed = []

        def make_tunnel(**kwargs):
            t = MagicMock()
            t.open = AsyncMock(side_effect=SshTunnelError("connect refused"))
            t.close = AsyncMock()
            constructed.append((kwargs["host"], t))
            return t

        with resolve, fetch, write, unlink, patch.object(
            lsf_tunnel, "SshTunnel", side_effect=make_tunnel
        ):
            with pytest.raises(HTTPException) as ei:
                async with lsf_tunnel.open_lsf_tunnel("space-a", "env://x"):
                    pass

        assert ei.value.status_code == 503
        assert "node-a" in str(ei.value.detail)
        assert [host for host, _ in constructed] == ["node-a"]

    @pytest.mark.asyncio
    async def test_first_fails_second_succeeds(self):
        from gbserver.api import lsf_tunnel
        from gbserver.utils.ssh_tunnel import SshTunnelError

        resolve, fetch, write, unlink = self._patch_resolvers(["node-a", "node-b"])

        # Force a deterministic order so we can assert which node served.
        shuffle_noop = patch("random.shuffle", side_effect=lambda lst: None)

        opened_hosts = []

        def make_tunnel(**kwargs):
            t = MagicMock()
            host = kwargs["host"]
            if host == "node-a":
                t.open = AsyncMock(side_effect=SshTunnelError("down"))
            else:
                t.open = AsyncMock(return_value=None)

                async def _run_remote(cmd, raise_on_error=True):
                    # readlink -f canonicalization step
                    return (0, "/ws\n", "")

                t.run_remote = AsyncMock(side_effect=_run_remote)
            t.close = AsyncMock()
            opened_hosts.append(host)
            return t

        with resolve, fetch, write, unlink, shuffle_noop, patch.object(
            lsf_tunnel, "SshTunnel", side_effect=make_tunnel
        ):
            async with lsf_tunnel.open_lsf_tunnel("space-a", "env://x") as (
                tunnel,
                cfg,
            ):
                assert cfg.workspace_remote_dir == "/ws"
                assert tunnel is not None

        # Both nodes were tried; node-b is the one that yielded.
        assert opened_hosts == ["node-a", "node-b"]

    @pytest.mark.asyncio
    async def test_all_nodes_fail_returns_503_with_list(self):
        from gbserver.api import lsf_tunnel
        from gbserver.utils.ssh_tunnel import SshTunnelError

        resolve, fetch, write, unlink = self._patch_resolvers(
            ["node-a", "node-b", "node-c"]
        )

        def make_tunnel(**kwargs):
            t = MagicMock()
            t.open = AsyncMock(side_effect=SshTunnelError(f"fail {kwargs['host']}"))
            t.close = AsyncMock()
            return t

        with resolve, fetch, write, unlink, patch.object(
            lsf_tunnel, "SshTunnel", side_effect=make_tunnel
        ):
            with pytest.raises(HTTPException) as ei:
                async with lsf_tunnel.open_lsf_tunnel("space-a", "env://x"):
                    pass

        assert ei.value.status_code == 503
        detail = str(ei.value.detail)
        assert "node-a" in detail
        assert "node-b" in detail
        assert "node-c" in detail
