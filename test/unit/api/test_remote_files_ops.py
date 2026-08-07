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

"""Direct unit tests for the shared ``remote_files_ops`` module.

This module is shared surface (build-files AND environment-files drive it), so
its own units are pinned directly here rather than only incidentally through one
caller's endpoints. Everything is mockable without live access: pure parsers
take strings, and the async helpers take a tunnel stub whose ``run_remote`` /
``start_sftp`` return canned ``stat`` / ``head`` output.

Focus is the low-level behavior that breaks silently and that no endpoint test
targets head-on:
  * ``_parse_grep_line`` — NUL-delimited path/lineno/sep parsing
  * ``_parse_find_printf`` — ``%T@`` float-mtime truncation, type mapping
  * ``_no_match_or_500`` — rc=1 (no match) vs rc=141 (SIGPIPE) vs rc>=2 (error)
  * ``_remote_stat_batch`` — missing-path omission from batched stat
  * ``_validate_peek_args`` — mutual exclusion + range parsing
  * ``peek_file`` — directory rejection
"""

import asyncio
from pathlib import PurePosixPath
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from gbserver.api.remote_files_ops import (
    _content_disposition,
    _no_match_or_500,
    _parse_find_printf,
    _parse_grep_line,
    _remote_stat,
    _remote_stat_batch,
    _validate_peek_args,
    peek_file,
)

ROOT = PurePosixPath("/proj/demo")


def _tunnel(run_remote):
    return type("T", (), {"run_remote": AsyncMock(side_effect=run_remote)})()


# ------------------------------------------------------------ _no_match_or_500


class TestNoMatchOr500:
    def test_rc0_splits_lines_dropping_empties(self):
        assert _no_match_or_500(0, "a\nb\n\n", "", "search") == ["a", "b"]

    def test_rc141_sigpipe_is_success(self):
        # head closed the pipe after the cap; the truncated stdout is the result.
        assert _no_match_or_500(141, "a\nb", "", "search") == ["a", "b"]

    def test_rc0_keeps_embedded_carriage_returns_in_record(self):
        # Splits on '\n' only (not splitlines), so a \r-heavy tqdm line stays
        # one record rather than fragmenting.
        assert _no_match_or_500(0, "x\ry\n", "", "search") == ["x\ry"]

    def test_rc1_empty_streams_is_no_match(self):
        assert _no_match_or_500(1, "", "", "search") == []

    def test_rc1_with_stderr_is_500(self):
        # rc=1 but stderr non-empty → a pipeline-stage failure, not "no match".
        with pytest.raises(HTTPException) as ei:
            _no_match_or_500(1, "", "head: write error", "search")
        assert ei.value.status_code == 500

    def test_rc2_no_such_file_is_404(self):
        with pytest.raises(HTTPException) as ei:
            _no_match_or_500(2, "", "grep: /x: No such file or directory", "search")
        assert ei.value.status_code == 404

    def test_rc2_other_error_is_500(self):
        with pytest.raises(HTTPException) as ei:
            _no_match_or_500(2, "", "grep: something exploded", "search")
        assert ei.value.status_code == 500


# ------------------------------------------------------------- _parse_grep_line


class TestParseGrepLine:
    def test_match_line(self):
        ln = "/proj/demo/notes.txt\x0012:hello world"
        assert _parse_grep_line(ln, ROOT) == ("notes.txt", 12, "hello world", True)

    def test_context_line_uses_dash_separator(self):
        # '-' separator (from -A/-B context) → is_match False.
        ln = "/proj/demo/notes.txt\x0011-prev line"
        assert _parse_grep_line(ln, ROOT) == ("notes.txt", 11, "prev line", False)

    def test_no_nul_returns_none(self):
        # grep's '--' group separator (and any NUL-less line) is dropped.
        assert _parse_grep_line("--", ROOT) is None

    def test_text_may_contain_colon_digits_colon(self):
        # The NUL delimiter means embedded ':<digits>:' in text can't be
        # mistaken for the record boundary.
        ln = "/proj/demo/a.py\x005:x = {1:2, 3:4}"
        assert _parse_grep_line(ln, ROOT) == ("a.py", 5, "x = {1:2, 3:4}", True)

    def test_path_outside_root_returns_none(self):
        ln = "/etc/passwd\x001:root:x:0:0"
        assert _parse_grep_line(ln, ROOT) is None

    def test_malformed_after_nul_returns_none(self):
        ln = "/proj/demo/a.txt\x00not-a-lineno"
        assert _parse_grep_line(ln, ROOT) is None


# ----------------------------------------------------------- _parse_find_printf


class TestParseFindPrintf:
    def test_float_mtime_truncated_to_seconds(self):
        # `%T@` is fractional epoch; int(float(...)) truncates toward zero, so
        # the sub-second part is dropped rather than rounded.
        entry = _parse_find_printf("notes.txt\tf\t10\t1700000000.5", ROOT)
        assert entry is not None
        assert entry.mtime == 1700000000  # .5 dropped, not rounded to ...001
        assert entry.size == 10
        assert entry.type == "file"
        assert entry.path == "/proj/demo/notes.txt"

    def test_type_char_mapping(self):
        assert _parse_find_printf("sub\td\t4096\t1", ROOT).type == "dir"
        assert _parse_find_printf("link\tl\t7\t1", ROOT).type == "symlink"
        # An unrecognized find type char (e.g. 'p' for a pipe) maps to "other".
        assert _parse_find_printf("fifo\tp\t0\t1", ROOT).type == "other"

    def test_too_few_fields_returns_none(self):
        assert _parse_find_printf("notes.txt\tf\t10", ROOT) is None

    def test_empty_relpath_returns_none(self):
        assert _parse_find_printf("\tf\t10\t1700000000", ROOT) is None

    def test_non_numeric_size_returns_none(self):
        assert _parse_find_printf("notes.txt\tf\tNaN\t1700000000", ROOT) is None


# ----------------------------------------------------------- _remote_stat_batch


class TestRemoteStatBatch:
    def test_empty_input_no_call(self):
        called = []

        async def run_remote(cmd, raise_on_error=True):
            called.append(cmd)
            return (0, "", "")

        out = asyncio.run(_remote_stat_batch(_tunnel(run_remote), []))
        assert out == {}
        assert called == []

    def test_missing_path_omitted(self):
        # stat returns rc=1 (some paths missing) and only lines for the paths
        # that exist; the missing one is simply absent from the dict.
        async def run_remote(cmd, raise_on_error=True):
            return (1, "/proj/demo/a.txt\t10\t1700000000\n", "")

        out = asyncio.run(
            _remote_stat_batch(
                _tunnel(run_remote),
                [PurePosixPath("/proj/demo/a.txt"), PurePosixPath("/proj/demo/gone")],
            )
        )
        assert out == {"/proj/demo/a.txt": (10, 1700000000)}
        assert "/proj/demo/gone" not in out

    def test_rc2_returns_empty(self):
        # rc not in (0,1) → hard failure, empty dict (caller leaves meta None).
        async def run_remote(cmd, raise_on_error=True):
            return (2, "", "stat: fatal")

        out = asyncio.run(
            _remote_stat_batch(_tunnel(run_remote), [PurePosixPath("/proj/demo/a")])
        )
        assert out == {}

    def test_short_or_nonnumeric_lines_skipped(self):
        async def run_remote(cmd, raise_on_error=True):
            return (
                0,
                "/proj/demo/a.txt\t10\t1700000000\n"
                "shortline\n"
                "/proj/demo/b.txt\tNaN\t1\n",
                "",
            )

        out = asyncio.run(
            _remote_stat_batch(_tunnel(run_remote), [PurePosixPath("/proj/demo/a.txt")])
        )
        assert out == {"/proj/demo/a.txt": (10, 1700000000)}


# ----------------------------------------------------------- _validate_peek_args


class TestValidatePeekArgs:
    def test_none_when_no_arg(self):
        assert _validate_peek_args(None, None, None) is None

    def test_head(self):
        assert _validate_peek_args(5, None, None) == ("head", (5,))

    def test_tail(self):
        assert _validate_peek_args(None, 5, None) == ("tail", (5,))

    def test_range(self):
        assert _validate_peek_args(None, None, "2-9") == ("range", (2, 9))

    def test_mutual_exclusion_400(self):
        with pytest.raises(HTTPException) as ei:
            _validate_peek_args(5, 5, None)
        assert ei.value.status_code == 400

    def test_range_inverted_400(self):
        with pytest.raises(HTTPException) as ei:
            _validate_peek_args(None, None, "9-2")
        assert ei.value.status_code == 400

    def test_range_zero_start_400(self):
        with pytest.raises(HTTPException) as ei:
            _validate_peek_args(None, None, "0-5")
        assert ei.value.status_code == 400


# ------------------------------------------------------------------- peek_file


class TestPeekFile:
    def test_directory_rejected_400(self):
        async def run_remote(cmd, raise_on_error=True):
            # _remote_stat: size + %F "directory"
            return (0, "4096\tdirectory\n", "")

        with pytest.raises(HTTPException) as ei:
            asyncio.run(
                peek_file(
                    _tunnel(run_remote), PurePosixPath("/proj/demo/d"), ("head", (2,))
                )
            )
        assert ei.value.status_code == 400


# ------------------------------------------------------------------ _remote_stat


class TestRemoteStat:
    def test_file(self):
        async def run_remote(cmd, raise_on_error=True):
            return (0, "42\tregular file\n", "")

        size, is_dir = asyncio.run(
            _remote_stat(_tunnel(run_remote), PurePosixPath("/proj/demo/a"))
        )
        assert (size, is_dir) == (42, False)

    def test_directory(self):
        async def run_remote(cmd, raise_on_error=True):
            return (0, "4096\tdirectory\n", "")

        size, is_dir = asyncio.run(
            _remote_stat(_tunnel(run_remote), PurePosixPath("/proj/demo/d"))
        )
        assert is_dir is True

    def test_missing_is_404(self):
        async def run_remote(cmd, raise_on_error=True):
            return (1, "", "stat: cannot stat '/x': No such file or directory")

        with pytest.raises(HTTPException) as ei:
            asyncio.run(_remote_stat(_tunnel(run_remote), PurePosixPath("/x")))
        assert ei.value.status_code == 404


# ------------------------------------------------------------ _content_disposition


class TestContentDisposition:
    def test_ascii_filename(self):
        v = _content_disposition("notes.txt")
        assert 'filename="notes.txt"' in v
        assert "filename*=UTF-8''notes.txt" in v

    def test_non_ascii_gets_utf8_form_and_ascii_fallback(self):
        v = _content_disposition("résumé.txt")
        # UTF-8 form percent-encodes; ascii fallback replaces non-ascii.
        assert "filename*=UTF-8''" in v
        assert "%" in v.split("UTF-8''", 1)[1]
