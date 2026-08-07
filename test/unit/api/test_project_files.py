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

"""Unit tests for the project-folder REST API (``/api/v1/projects``).

These stub out ``open_lsf_tunnel`` (no SSH / IBM Cloud) and inject a fake
authenticated user onto ``request.state`` via a tiny middleware, so no real
auth stack is required.

The emphasis is the security core that makes this API different from
build-files:

  * a non-member, a nonexistent folder/group, and a malformed folder name all
    return the SAME 404 body (no existence leak);
  * NO data-read command (``readlink``/``ls``/``find``/``grep``/``stat``) is
    ever issued for a non-member — only the two ``getent`` authorization
    lookups;
  * for a member, data commands run AFTER the getent calls, in order.

The generic file-op request/response surface (listing shapes, traversal
rejection, peek modes, download caps) is covered thoroughly by
test_build_files.py against the shared ``remote_files_ops`` code; here we
port the security-relevant subset plus getent-parser unit tests.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from gbserver.api import project_files as project_files_mod
from gbserver.api.lsf_tunnel import LsfTunnelConfig
from gbserver.api.project_files import projects_api
from gbserver.api.project_files_paths import (
    parse_group_members,
    parse_passwd_email,
)

# getent-passwd shape from a login node (GECOS: email;serial;full name):
#   alice:*:1001:2001:alice@example.com;NNNNNN;Alice Example:/u/alice:/bin/bash
MEMBER_EMAIL = "alice@example.com"
MEMBER_USER = "alice"
GROUP_LINE = f"proj_demo:*:2001:{MEMBER_USER},otheruser\n"
PASSWD_LINES = (
    f"{MEMBER_USER}:*:1001:2001:{MEMBER_EMAIL};NNNNNN;Alice Example:"
    "/u/alice:/bin/bash\n"
    "otheruser:*:1002:2001:other.user@example.com;NNNNNN;Other User:"
    "/u/otheruser:/bin/bash\n"
)


# --------------------------------------------------------------------- fixtures


def _make_app(email: str) -> FastAPI:
    """Mount only projects_api, with a middleware that injects a fake user.

    AuthMiddleware is replaced by a trivial middleware that sets
    ``request.state.data`` the same shape the real middleware produces:
    ``{"user": User-like}``. ``email`` is what the requester presents.
    """
    app = FastAPI()

    @app.middleware("http")
    async def _inject_user(request: Request, call_next):
        request.state.data = {"user": SimpleNamespace(login="requester", email=email)}
        return await call_next(request)

    app.mount("/api/v1/projects", projects_api)
    return app


def _client(email: str = MEMBER_EMAIL) -> TestClient:
    return TestClient(_make_app(email))


def _fake_tunnel_cm(tunnel_mock):
    """Async CM yielding (tunnel, LsfTunnelConfig) — cfg is ignored by handlers."""

    @asynccontextmanager
    async def _cm(space_name: str, environment_uri: str):
        yield tunnel_mock, LsfTunnelConfig(workspace_remote_dir="/ws")

    return _cm


def _patch_tunnel(tunnel_mock):
    """Patch open_lsf_tunnel + the server-resolved space/env helpers.

    ``_gpfs_environment_uri`` would otherwise 503 (unset in test env), so we
    pin it to a dummy URI.
    """
    return (
        patch.object(
            project_files_mod, "open_lsf_tunnel", _fake_tunnel_cm(tunnel_mock)
        ),
        patch.object(
            project_files_mod, "_gpfs_environment_uri", return_value="env://x"
        ),
        patch.object(project_files_mod, "_gpfs_space_name", return_value="public"),
    )


class _RecordingTunnel:
    """A tunnel mock whose run_remote dispatches on the command string and
    records every command in call order.

    Defaults model an authorized member of ``proj_demo`` with one file
    ``notes.txt`` under ``/proj/demo``.
    """

    def __init__(
        self,
        *,
        group_stdout: str = GROUP_LINE,
        group_rc: int = 0,
        passwd_stdout: str = PASSWD_LINES,
        ls_stdout: str = "notes.txt\n",
        find_stdout: str = "",
        grep_stdout: str = "",
        grep_rc: int = 0,
        stat_stdout: str = "10\tregular file\n",
        stat_rc: int = 0,
    ):
        self.commands: list[str] = []
        self._group_stdout = group_stdout
        self._group_rc = group_rc
        self._passwd_stdout = passwd_stdout
        self._ls_stdout = ls_stdout
        self._find_stdout = find_stdout
        self._grep_stdout = grep_stdout
        self._grep_rc = grep_rc
        self._stat_stdout = stat_stdout
        self._stat_rc = stat_rc
        self.run_remote = AsyncMock(side_effect=self._run_remote)
        self.start_sftp = AsyncMock()

    async def _run_remote(self, cmd, raise_on_error=True):
        self.commands.append(cmd)
        if cmd.startswith("getent group"):
            return (self._group_rc, self._group_stdout, "")
        if cmd.startswith("getent passwd"):
            return (0, self._passwd_stdout, "")
        if cmd.startswith("readlink -f"):
            target = cmd.split("--", 1)[1].strip().strip("'\"")
            return (0, target + "\n", "")
        if cmd.startswith("stat -c"):
            return (self._stat_rc, self._stat_stdout, "")
        if "grep " in cmd and "getent" not in cmd:
            return (self._grep_rc, self._grep_stdout, "")
        if "find " in cmd:
            return (0, self._find_stdout, "")
        if cmd.startswith("ls -1A") or "ls -1A" in cmd:
            return (0, self._ls_stdout, "")
        return (0, "", "")

    # Convenience predicates over recorded commands.
    def getent_calls(self):
        return [c for c in self.commands if c.startswith("getent")]

    def data_calls(self):
        """Any command that reads the folder (i.e. not a getent lookup)."""
        return [c for c in self.commands if not c.startswith("getent")]


# ----------------------------------------------------------- getent-parser tests


class TestParseGroupMembers:
    def test_normal(self):
        assert parse_group_members("proj_x:*:600:alice,bob,carol\n") == [
            "alice",
            "bob",
            "carol",
        ]

    def test_single_member(self):
        assert parse_group_members("proj_x:*:600:alice\n") == ["alice"]

    def test_empty_member_field(self):
        assert parse_group_members("proj_x:*:600:\n") == []

    def test_empty_output(self):
        assert parse_group_members("") == []
        assert parse_group_members("\n\n") == []

    def test_malformed_too_few_fields(self):
        assert parse_group_members("proj_x:*:600\n") == []

    def test_drops_empty_members_between_commas(self):
        assert parse_group_members("proj_x:*:600:alice,,bob,\n") == ["alice", "bob"]

    def test_takes_first_nonempty_line(self):
        assert parse_group_members("\nproj_x:*:600:alice\nproj_y:*:601:bob\n") == [
            "alice"
        ]


class TestParsePasswdEmail:
    def test_full_gecos_shape(self):
        line = (
            "alice:*:1001:2001:alice@example.com;NNNNNN;Alice Example:"
            "/u/alice:/bin/bash"
        )
        assert parse_passwd_email(line) == "alice@example.com"

    def test_email_only_gecos(self):
        line = "u:*:1:1:person@example.com:/home/u:/bin/bash"
        assert parse_passwd_email(line) == "person@example.com"

    def test_gecos_without_email_returns_none(self):
        # First GECOS sub-field is a name, not an email.
        line = "u:*:1:1:Full Name;serial;more:/home/u:/bin/bash"
        assert parse_passwd_email(line) is None

    def test_empty_gecos_returns_none(self):
        line = "u:*:1:1::/home/u:/bin/bash"
        assert parse_passwd_email(line) is None

    def test_too_few_fields_returns_none(self):
        assert parse_passwd_email("u:*:1:1") is None

    def test_empty_line_returns_none(self):
        assert parse_passwd_email("") is None


# ------------------------------------------------------------ authorization flow


class TestAuthorization:
    def test_member_can_list(self):
        tunnel = _RecordingTunnel()
        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            r = _client().get("/api/v1/projects/demo/files")
        assert r.status_code == 200, r.text
        assert r.json() == ["notes.txt"]

    def test_member_getent_runs_before_data(self):
        tunnel = _RecordingTunnel()
        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            r = _client().get("/api/v1/projects/demo/files")
        assert r.status_code == 200, r.text
        # The two getent lookups come first, in order, then data commands.
        cmds = tunnel.commands
        assert cmds[0].startswith("getent group")
        assert cmds[1].startswith("getent passwd")
        # Every getent call precedes every data (readlink/ls/...) call.
        first_data_idx = next(
            i for i, c in enumerate(cmds) if not c.startswith("getent")
        )
        assert all(cmds[i].startswith("getent") for i in range(first_data_idx))

    def test_group_name_convention(self):
        tunnel = _RecordingTunnel()
        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            _client().get("/api/v1/projects/demo/files")
        assert any("getent group" in c and "proj_demo" in c for c in tunnel.commands)

    def test_non_member_gets_404(self):
        # Requester email not present in the resolved passwd emails.
        tunnel = _RecordingTunnel()
        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            r = _client(email="stranger@example.com").get("/api/v1/projects/demo/files")
        assert r.status_code == 404
        assert r.json()["detail"] == "project folder not found"

    def test_non_member_no_data_read(self):
        # THE security invariant: a non-member triggers ONLY getent, never a
        # data-touching command against /proj.
        tunnel = _RecordingTunnel()
        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            r = _client(email="stranger@example.com").get("/api/v1/projects/demo/files")
        assert r.status_code == 404
        assert tunnel.data_calls() == []
        for c in tunnel.commands:
            assert c.startswith("getent")
            assert "/proj" not in c

    def test_non_member_no_data_read_all_endpoints(self):
        for url in (
            "/api/v1/projects/demo/files",
            "/api/v1/projects/demo/files/search?pattern=x",
            "/api/v1/projects/demo/file/download?path=notes.txt",
        ):
            tunnel = _RecordingTunnel()
            t, env, sp = _patch_tunnel(tunnel)
            with t, env, sp:
                r = _client(email="stranger@example.com").get(url)
            assert r.status_code == 404, url
            assert tunnel.data_calls() == [], url

    def test_nonexistent_group_same_404_as_non_member(self):
        # getent group rc != 0 (group/folder does not exist) → identical 404,
        # and no getent passwd / data command follows.
        tunnel = _RecordingTunnel(group_rc=2, group_stdout="")
        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            r = _client().get("/api/v1/projects/demo/files")
        assert r.status_code == 404
        assert r.json()["detail"] == "project folder not found"
        assert tunnel.data_calls() == []
        # Short-circuits after the failed group lookup — no passwd call.
        assert not any(c.startswith("getent passwd") for c in tunnel.commands)

    def test_empty_group_same_404(self):
        tunnel = _RecordingTunnel(group_stdout="proj_demo:*:600:\n")
        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            r = _client().get("/api/v1/projects/demo/files")
        assert r.status_code == 404
        assert r.json()["detail"] == "project folder not found"
        assert tunnel.data_calls() == []

    def test_empty_requester_email_short_circuits(self):
        # No requester identity → 404 before ANY command runs.
        tunnel = _RecordingTunnel()
        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            r = _client(email="").get("/api/v1/projects/demo/files")
        assert r.status_code == 404
        assert r.json()["detail"] == "project folder not found"
        assert tunnel.commands == []

    def test_email_match_is_case_insensitive(self):
        # GECOS email is mixed-case; requester presents lowercase.
        tunnel = _RecordingTunnel()
        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            r = _client(email=MEMBER_EMAIL.lower()).get("/api/v1/projects/demo/files")
        assert r.status_code == 200, r.text


class TestGetentBatching:
    """A group larger than the getent-passwd batch cap is resolved in chunks,
    and emails from every chunk are unioned — a member who only resolves in a
    later chunk is still authorized (guards against ARG_MAX overflow / partial
    lookups)."""

    def test_member_in_later_chunk_is_authorized(self):
        from gbserver.api import project_files_paths as ppaths

        # Cap batches at 2; build a 5-member group whose target sits in the
        # third chunk. Each getent-passwd chunk returns only its own members'
        # passwd lines, so authz only succeeds if every chunk is queried.
        target_email = "eve@example.com"
        members = ["u1", "u2", "u3", "u4", "eve"]
        group_line = f"proj_demo:*:2001:{','.join(members)}\n"
        passwd_by_user = {
            "u1": "u1:*:1:2001:u1@example.com;N;U1:/u/u1:/bin/bash",
            "u2": "u2:*:2:2001:u2@example.com;N;U2:/u/u2:/bin/bash",
            "u3": "u3:*:3:2001:u3@example.com;N;U3:/u/u3:/bin/bash",
            "u4": "u4:*:4:2001:u4@example.com;N;U4:/u/u4:/bin/bash",
            "eve": f"eve:*:5:2001:{target_email};N;Eve:/u/eve:/bin/bash",
        }
        passwd_calls: list[str] = []

        async def run_remote(cmd, raise_on_error=True):
            if cmd.startswith("getent group"):
                return (0, group_line, "")
            if cmd.startswith("getent passwd"):
                passwd_calls.append(cmd)
                # Return passwd lines only for the users named in this chunk.
                lines = [v for u, v in passwd_by_user.items() if f" {u}" in f" {cmd}"]
                return (0, "\n".join(lines) + "\n" if lines else "", "")
            return (0, "", "")

        tunnel = SimpleNamespace(run_remote=AsyncMock(side_effect=run_remote))

        with patch.object(ppaths, "PROJECT_FILES_GETENT_BATCH_MAX", 2):
            import asyncio

            emails = asyncio.run(ppaths._authorized_emails_for_group(tunnel, "demo"))

        # 5 members / batch of 2 → 3 passwd round-trips, all emails unioned.
        assert len(passwd_calls) == 3
        assert target_email in emails
        assert "u1@example.com" in emails


class TestFolderNameValidation:
    @pytest.mark.parametrize(
        "folder",
        ["a b", "foo$bar", "foo;rm", "foo/bar", "foo`x`", "..", "."],
    )
    def test_malformed_folder_same_404_no_command(self, folder):
        # A bad folder name is rejected identically to "not a member", and
        # crucially issues NO command at all (not even getent) — so a
        # well-formed-but-forbidden name can't be told apart from a malformed
        # one by observing side effects.
        tunnel = _RecordingTunnel()
        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            # "foo/bar" is caught by FastAPI routing (path segment) → 404;
            # the rest hit validate_folder_name → ProjectAccessDenied 404.
            r = _client().get(f"/api/v1/projects/{folder}/files")
        assert r.status_code == 404
        assert tunnel.commands == []

    def test_valid_folder_with_allowed_charset(self):
        tunnel = _RecordingTunnel(group_stdout="proj_demo.v2:*:600:alice\n")
        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            r = _client().get("/api/v1/projects/demo.v2/files")
        assert r.status_code == 200, r.text
        assert any("proj_demo.v2" in c for c in tunnel.getent_calls())


# --------------------------------------------------------- request/response port


class TestListing:
    def test_recursive_listing(self):
        tunnel = _RecordingTunnel(
            find_stdout="/proj/demo/a.txt\n/proj/demo/sub/b.txt\n"
        )
        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            r = _client().get(
                "/api/v1/projects/demo/files", params={"recursive": "true"}
            )
        assert r.status_code == 200, r.text
        assert r.json() == ["a.txt", "sub/b.txt"]

    def test_stat_returns_file_entries(self):
        tunnel = _RecordingTunnel(find_stdout="notes.txt\tf\t10\t1700000000.5\n")
        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            r = _client().get("/api/v1/projects/demo/files", params={"stat": "true"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body == [
            {"path": "notes.txt", "type": "file", "size": 10, "mtime": 1700000000}
        ]

    def test_path_traversal_rejected(self):
        tunnel = _RecordingTunnel()
        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            r = _client().get(
                "/api/v1/projects/demo/files",
                params={"path": "../../etc"},
            )
        assert r.status_code == 400
        # Auth ran (getent) but the traversal target never reached a shell cmd.
        for c in tunnel.commands:
            assert "etc/passwd" not in c


class TestSearch:
    def test_search_returns_hits(self):
        tunnel = _RecordingTunnel(
            grep_stdout="/proj/demo/notes.txt\x0012:hello world\n"
        )
        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            r = _client().get(
                "/api/v1/projects/demo/files/search",
                params={"pattern": "hello"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body == [
            {
                "path": "notes.txt",
                "line": 12,
                "text": "hello world",
                "is_match": True,
                "size": None,
                "mtime": None,
            }
        ]

    def test_search_pattern_control_char_rejected(self):
        tunnel = _RecordingTunnel()
        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            r = _client().get(
                "/api/v1/projects/demo/files/search",
                params={"pattern": "a\nb"},
            )
        assert r.status_code == 400
        # Pure validation happens before any tunnel command.
        assert tunnel.commands == []


class TestDownloadAndPeek:
    def test_download_streams_file(self):
        tunnel = _RecordingTunnel(stat_stdout="4\tregular file\n")
        sftp = MagicMock()
        fh = MagicMock()

        async def fake_read(n):
            return b"data"[:n] if n else b""

        reads = iter([b"data", b""])

        async def read_seq(n):
            return next(reads)

        fh.read = AsyncMock(side_effect=read_seq)
        fh.__aenter__ = AsyncMock(return_value=fh)
        fh.__aexit__ = AsyncMock(return_value=None)
        sftp.open = MagicMock(return_value=fh)
        sftp.exit = MagicMock(return_value=None)
        tunnel.start_sftp = AsyncMock(return_value=sftp)

        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            r = _client().get(
                "/api/v1/projects/demo/file/download",
                params={"path": "notes.txt"},
            )
        assert r.status_code == 200, r.text
        assert r.content == b"data"
        assert r.headers["content-length"] == "4"

    def test_download_directory_returns_400(self):
        tunnel = _RecordingTunnel(stat_stdout="4096\tdirectory\n")
        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            r = _client().get(
                "/api/v1/projects/demo/file/download",
                params={"path": "."},
            )
        assert r.status_code == 400

    def test_download_too_large_returns_413(self):
        tunnel = _RecordingTunnel(stat_stdout=f"{2 * 1024**3}\tregular file\n")
        t, env, sp = _patch_tunnel(tunnel)
        with (
            t,
            env,
            sp,
            patch.object(
                project_files_mod, "PROJECT_FILES_DOWNLOAD_MAX_BYTES", 1 * 1024**3
            ),
        ):
            r = _client().get(
                "/api/v1/projects/demo/file/download",
                params={"path": "big.bin"},
            )
        assert r.status_code == 413

    def test_peek_head_returns_text_plain(self):
        tunnel = _RecordingTunnel(stat_stdout="100\tregular file\n")

        # The peek text comes from _peek_text's run_remote; make the head/sed
        # producer return known text. Override run_remote to add peek handling.
        base = tunnel._run_remote

        async def run_remote(cmd, raise_on_error=True):
            tunnel.commands.append(cmd)
            if cmd.startswith("getent group"):
                return (0, GROUP_LINE, "")
            if cmd.startswith("getent passwd"):
                return (0, PASSWD_LINES, "")
            if cmd.startswith("readlink -f"):
                target = cmd.split("--", 1)[1].strip().strip("'\"")
                return (0, target + "\n", "")
            if cmd.startswith("stat -c"):
                return (0, "100\tregular file\n", "")
            if "head -n" in cmd:
                return (0, "line1\nline2\n", "")
            return (0, "", "")

        tunnel.run_remote = AsyncMock(side_effect=run_remote)

        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            r = _client().get(
                "/api/v1/projects/demo/file/download",
                params={"path": "notes.txt", "head": "2"},
            )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/plain")
        assert r.text == "line1\nline2\n"

    def test_peek_mutual_exclusion_returns_400(self):
        tunnel = _RecordingTunnel()
        t, env, sp = _patch_tunnel(tunnel)
        with t, env, sp:
            r = _client().get(
                "/api/v1/projects/demo/file/download",
                params={"path": "notes.txt", "head": "2", "tail": "2"},
            )
        assert r.status_code == 400
        # Mutual-exclusion is pure validation → no command issued.
        assert tunnel.commands == []


class TestNotConfigured:
    def test_missing_environment_uri_returns_503(self):
        # With the env URI unset (the real default), the endpoint 503s instead
        # of guessing. Patch only the tunnel; leave _gpfs_environment_uri real
        # with the empty default.
        tunnel = _RecordingTunnel()
        with (
            patch.object(project_files_mod, "open_lsf_tunnel", _fake_tunnel_cm(tunnel)),
            patch.object(project_files_mod, "PROJECTS_GPFS_ENVIRONMENT_URI", ""),
        ):
            r = _client().get("/api/v1/projects/demo/files")
        assert r.status_code == 503
        assert tunnel.commands == []
