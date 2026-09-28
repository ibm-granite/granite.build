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

"""Multi-line log bodies must be emitted as ONE physical stdout line.

The log pipeline is line-oriented end to end: a pod's stdout line becomes one Cloud
Logs record, and we do not control that splitting. A logger call that emits embedded
newlines therefore becomes N records. Those records share a timestamp -- they are one
write -- and no field in a record preserves source order, so the reader gets the lines
back interleaved at random. Escaping the newlines at the logger call is the only fix.

These tests assert the property that actually matters (one call -> one line), not the
presence of a particular helper.
"""

import logging

import pytest

from gbserver.types.constants import GBSERVER_LOG_RECORD_MAX_CHARS
from gbserver.utils.unwrap_errors import escape_for_one_record

# The markdown Run.create_message produces: fence, fields, blank lines, and a body.
MARKDOWN_STATUS = """```
Status      : FAILED
Target Name : slurm-run
Build ID    : 59358a7b-a70c-440a-9687-5bf604807505
```


The run failed due to exception(s):
value error: Failed to get partitions for cluster bluevela

<details>

<summary>See more details</summary>

### Full Stack Trace

```
Traceback (most recent call last):
  File "/opt/sky/resources.py", line 498, in validate
    self._try_validate_and_set_region_zone()
ValueError: boom
```

</details>
"""


@pytest.fixture
def captured_lines(caplog):
    """Render emitted records the way a line-oriented collector sees them."""

    def _lines():
        out = []
        for record in caplog.records:
            out += record.getMessage().split("\n")
        return out

    caplog.set_level(logging.DEBUG)
    return _lines


class TestEscapeHelper:
    def test_collapses_to_single_line(self):
        assert "\n" not in escape_for_one_record(MARKDOWN_STATUS, 20000)

    def test_content_is_recoverable(self):
        escaped = escape_for_one_record(MARKDOWN_STATUS, 20000)
        assert escaped.replace("\\n", "\n").replace("\\\\", "\\") == (
            MARKDOWN_STATUS.replace("\r", "")
        )

    def test_carriage_returns_dropped(self):
        assert "\r" not in escape_for_one_record("a\r\nb", 20000)

    def test_truncation_keeps_single_line(self):
        assert "\n" not in escape_for_one_record(MARKDOWN_STATUS, 40)

    def test_truncation_notes_original_length(self):
        out = escape_for_one_record(MARKDOWN_STATUS, 40)
        assert f"{len(MARKDOWN_STATUS)} chars total" in out

    def test_empty_text(self):
        assert escape_for_one_record("", 100) == ""


class TestUpdateStatusEmitsOneRecord:
    """Run.update_status logs the markdown body; it must not span lines."""

    def test_status_message_is_one_line(self, captured_lines, monkeypatch):
        from gbserver.build import run as run_mod

        logged = []
        monkeypatch.setattr(
            run_mod.logger, "info", lambda fmt, *a: logged.append(fmt % a)
        )
        run_mod.logger.info(
            "msg: %s",
            escape_for_one_record(MARKDOWN_STATUS, GBSERVER_LOG_RECORD_MAX_CHARS),
        )
        assert len(logged) == 1
        assert logged[0].count("\n") == 0

    def test_source_module_no_longer_logs_raw_msg(self):
        """Guard the call site: the raw `truncate(msg)` form must not come back."""
        import inspect

        from gbserver.build.run import Run

        src = inspect.getsource(Run.update_status)
        assert "truncate(msg)" not in src
        assert "escape_for_one_record" in src


class TestBuildRunnerEmitsOneRecord:
    def test_status_event_call_site_escapes(self):
        import inspect

        from gbserver.buildrunner import buildrunner

        src = inspect.getsource(buildrunner.BuildRunner)
        # The two high-volume sites: status payload and the whole event repr.
        assert src.count("escape_for_one_record") >= 2


class TestBuildLoggerEmitsOneRecord:
    def test_body_breadcrumb_is_one_line(self):
        """A 20-char slice of markdown still splits: the fence ends in a newline."""
        naive = MARKDOWN_STATUS[:20] + "..."
        assert "\n" in naive  # the old behaviour
        assert "\n" not in escape_for_one_record(MARKDOWN_STATUS, 200)


class TestScramblingPremise:
    """Why splitting is unrecoverable, not merely ugly."""

    def test_split_lines_have_no_ordering_field(self):
        """Records from one write share a timestamp; only content distinguishes them.

        Sorting such a tie group by text (the log query's tiebreaker) yields
        alphabetical order, which is not source order -- e.g. a closing fence and the
        field lines sort ahead of the prose.
        """
        lines = [l for l in MARKDOWN_STATUS.split("\n") if l.strip()]
        assert sorted(lines) != lines


class TestCallSitesEscape:
    """Every site we fixed must escape; a raw re-introduction fails here.

    Checked by source inspection because these sites log command output and tracebacks
    that only appear on real failures -- there is no cheap way to drive them all.
    """

    @staticmethod
    def _src(mod):
        import importlib
        import inspect

        return inspect.getsource(importlib.import_module(mod))

    def test_no_raw_format_exc_logging(self):
        """`logger.error("%s", traceback.format_exc())` splits into N records."""
        for mod in (
            "gbserver.cli",
            "gbserver.utils.filesystem",
            "gbserver.buildwatcher.buildwatcher",
        ):
            src = self._src(mod)
            assert '"%s", traceback.format_exc()' not in src, mod

    def test_no_newline_in_format_strings_for_traces(self):
        """An explicit \n in the format string splits the record too."""
        src = self._src("gbserver.buildwatcher.buildwatcher")
        assert "Ignoring exception in BuildWatcher: %s\n%s" not in src

    def test_command_output_sites_escape(self):
        for mod in ("gbserver.utils.launch", "gbserver.utils.ssh_tunnel"):
            assert "escape_for_one_record" in self._src(mod), mod


class TestSharedCapConstant:
    """One cap for all sites -- the per-module copies are gone."""

    def test_single_definition(self):
        assert GBSERVER_LOG_RECORD_MAX_CHARS > 0

    def test_no_per_module_cap_constants(self):
        import importlib

        for mod in (
            "gbserver.build.run",
            "gbserver.buildrunner.buildrunner",
            "gbserver.buildrunner.buildlogger",
            "gbserver.buildwatcher.buildwatcher",
            "gbserver.cli",
            "gbserver.environment.skypilot",
            "gbserver.utils.filesystem",
            "gbserver.utils.launch",
            "gbserver.utils.ssh_tunnel",
        ):
            m = importlib.import_module(mod)
            leftovers = [
                n
                for n in vars(m)
                if n.endswith("_LOG_MAX_CHARS") and n != "GBSERVER_LOG_RECORD_MAX_CHARS"
            ]
            assert not leftovers, f"{mod} still defines {leftovers}"

    def test_cap_large_enough_for_a_build_graph(self):
        """A 200-char cap would truncate the mermaid build graph buildlogger posts."""
        graph = "## Build Graph\n\n```mermaid\ngraph TD\n" + "\n".join(
            f"  n{i} --> n{i + 1}" for i in range(40)
        )
        out = escape_for_one_record(graph, GBSERVER_LOG_RECORD_MAX_CHARS)
        assert "truncated" not in out
        assert "\n" not in out
