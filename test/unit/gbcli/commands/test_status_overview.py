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

"""Unit tests for the build-status overview headline.

`_job_overview_lines` is the pure part of `execution_status_plain_output`: it
turns the `details` dict (which carries the `job` summary the service threaded
in) into the status headline plus, for a real retry chain, the per-attempt and
job-summary lines. Testing it directly avoids the rich-rendered output of the
full function, whose `**bold**` markers and ANSI are stripped/added at print
time.
"""

import pytest

from gbcli.commands.command_build import _job_overview_lines

pytestmark = pytest.mark.standalone


def _details(status="failed", build_id="root", job=None):
    return {"build_id": build_id, "status": status, "job": job}


def _job(
    status="success",
    attempts=2,
    build_ids=None,
    succeeded=2,
    failed=0,
    running=0,
    not_run=0,
    total=None,
):
    return {
        "status": status,
        "attempts": attempts,
        "build_ids": build_ids if build_ids is not None else ["root", "retry"],
        "counts": {
            "total": (
                total if total is not None else succeeded + failed + running + not_run
            ),
            "succeeded": succeeded,
            "failed": failed,
            "running": running,
            "not_run": not_run,
        },
    }


def test_headline_is_the_job_status_for_a_real_chain():
    # Issue #222: queried build FAILED, but the retry finished the job.
    headline, attempt_line, summary_line = _job_overview_lines(
        _details(status="failed", build_id="root", job=_job())
    )

    assert "SUCCESS" in headline
    assert "✅" in headline
    # The per-attempt truth is not lost, just moved to its own line.
    assert "FAILED" in attempt_line
    assert "attempt 1 of 2" in attempt_line
    assert "2 of 2" in summary_line


def test_attempt_position_reflects_which_member_was_queried():
    # Querying the retry (second member) says "attempt 2 of 2".
    _headline, attempt_line, _summary = _job_overview_lines(
        _details(status="success", build_id="retry", job=_job())
    )

    assert "attempt 2 of 2" in attempt_line


def test_summary_lists_failed_and_never_ran_counts():
    _h, _a, summary_line = _job_overview_lines(
        _details(
            status="failed",
            build_id="root",
            job=_job(status="failed", succeeded=1, failed=1, not_run=1),
        )
    )

    assert "1 of 3" in summary_line
    assert "1 failed" in summary_line
    assert "1 never ran" in summary_line


def test_running_count_is_shown_and_not_conflated_with_failed():
    _h, _a, summary_line = _job_overview_lines(
        _details(
            status="running",
            build_id="root",
            job=_job(status="running", succeeded=1, failed=0, running=1, not_run=0),
        )
    )

    assert "1 running" in summary_line
    assert "failed" not in summary_line


def test_single_attempt_job_leaves_the_output_unchanged():
    # attempts == 1: the job status equals the build's own status, so headline is
    # the build status and there are no extra lines — byte-identical to before.
    headline, attempt_line, summary_line = _job_overview_lines(
        _details(
            status="failed",
            build_id="root",
            job=_job(
                status="failed", attempts=1, build_ids=["root"], succeeded=0, failed=1
            ),
        )
    )

    assert "FAILED" in headline
    assert attempt_line == ""
    assert summary_line == ""


def test_missing_job_falls_back_to_the_build_status():
    headline, attempt_line, summary_line = _job_overview_lines(
        _details(status="failed", build_id="root", job=None)
    )

    assert "FAILED" in headline
    assert attempt_line == ""
    assert summary_line == ""


def test_queried_build_not_in_chain_degrades_gracefully():
    # Defensive: if the queried build id somehow isn't in build_ids, don't crash;
    # report attempt 0 rather than raising.
    _h, attempt_line, _s = _job_overview_lines(
        _details(status="failed", build_id="ghost", job=_job())
    )

    assert "of 2" in attempt_line  # no exception; position degrades to 0


def test_full_output_renders_the_job_status_end_to_end():
    from gbcli.commands.command_build import execution_status_plain_output

    details = {
        "build_id": "root",
        "name": "b",
        "description": "",
        "status": "failed",
        "started_at": "2020-01-01T00:00:00Z",
        "updated_at": "2020-01-01T00:03:00Z",
        "source_pr": "",
        "retry_of_build_ids": [],
        "retried_by_build_ids": ["retry"],
        "job": _job(),
    }

    rendered = execution_status_plain_output(details, {}, [], show_events=True)

    assert "SUCCESS" in rendered
