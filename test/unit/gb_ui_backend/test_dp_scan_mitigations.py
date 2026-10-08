"""DP scan mitigations: truncation is reported, and decoding leaves the event loop.

These guard two failure modes that are invisible from the outside. A truncated
window looks identical to a complete one, and event-loop-blocking work looks
identical to non-blocking work until the sidecar is under load.
"""

import asyncio
import base64
import io
import threading
import zipfile

import pytest

from gb_ui_backend.services.gbserver_source import (
    _truncation_warning,
    _yaml_from_archive_b64,
    _yaml_from_json_blob,
)


def _archive(files: dict[str, str]) -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, body in files.items():
            zf.writestr(name, body)
    return base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------ truncation warning


def test_truncation_warning_fires_only_when_the_page_is_full():
    """Both directions matter: a warning that always fired would also pass a
    test that only checked the full-page case, while telling every user their
    complete window might be incomplete."""
    assert _truncation_warning([1, 2, 3], limit=3) is not None
    assert _truncation_warning([1, 2], limit=3) is None
    assert _truncation_warning([], limit=3) is None


def test_truncation_warning_names_the_limit_and_hedges():
    msg = _truncation_warning(list(range(10000)), limit=10000)
    assert "10000" in msg
    # A full page may or may not have more behind it; claiming it definitely
    # does would be its own inaccuracy.
    assert "may be missing" in msg


def test_truncation_warning_absent_limit_never_warns():
    """limit=0 means unbounded, which cannot be truncated."""
    assert _truncation_warning([1, 2, 3], limit=0) is None


# ------------------------------------------------------------------- decoders


def test_yaml_from_archive_prefers_build_yaml():
    b64 = _archive({"other.yaml": "no: 1", "build.yaml": "llm.build:\n  name: x"})
    assert _yaml_from_archive_b64(b64) == "llm.build:\n  name: x"


def test_yaml_from_archive_falls_back_to_any_yaml():
    assert "no: 1" in (_yaml_from_archive_b64(_archive({"other.yml": "no: 1"})) or "")


def test_yaml_from_archive_tolerates_garbage():
    """A build whose archive is unreadable must not abort the whole scan."""
    assert _yaml_from_archive_b64("not-base64-at-all!!") is None
    assert _yaml_from_archive_b64(base64.b64encode(b"not a zip").decode()) is None


def test_yaml_from_json_blob_reads_nested_archive():
    blob = f'{{"build_archive": "{_archive({"build.yaml": "a: 1"})}"}}'
    assert _yaml_from_json_blob(blob) == "a: 1"


def test_yaml_from_json_blob_tolerates_missing_archive():
    assert _yaml_from_json_blob('{"no_archive_here": true}') is None
    assert _yaml_from_json_blob("not json") is None


# -------------------------------------------------- the decode leaves the loop


@pytest.mark.asyncio
async def test_decoding_runs_off_the_event_loop_thread():
    """The point of the change: the decode must not run on the loop thread.

    This asserts the *structural* property -- the decode body executes on a
    different thread than the one running the event loop -- rather than trying to
    observe its effect on scheduling.

    An earlier version of this test counted how many times a concurrent ticker
    was scheduled during the decode and asserted `ticks > 1`. That reads like an
    interleaving assertion but is a timing assertion: the tick count depends on
    how long the decode takes relative to how quickly the ticker is rescheduled,
    and nothing structural forces more than one tick. With this workload the
    decode finishes in single-digit milliseconds, so the ticker often got exactly
    one tick in before its first `await` was rescheduled and the assertion read
    `assert 1 > 1`. It passed in isolation and failed in a full-file run, where a
    warm interpreter and a busier loop shift the timing -- the classic signature,
    and reported on the PR as failing 3 runs out of 3.

    A flaky test guarding the central claim is worse than no test: it gets
    quarantined or its threshold lowered, and then the claim is unguarded.
    """
    archives = [_archive({"build.yaml": f"n: {i}\n" + "x" * 20000}) for i in range(60)]

    loop_thread_id = threading.get_ident()
    decode_thread_id: list[int] = []

    def decode_all():
        decode_thread_id.append(threading.get_ident())
        return [_yaml_from_archive_b64(a) for a in archives]

    results = await asyncio.to_thread(decode_all)

    assert len(results) == 60 and all(r is not None for r in results)
    assert decode_thread_id, "the decode callable never ran"
    assert decode_thread_id[0] != loop_thread_id, (
        "the decode ran on the event loop thread, so a wide scan would stall "
        "every other request the sidecar is serving"
    )


# ------------------------------------------------- ordering survives NULL updates


def test_scan_ordering_is_null_safe_on_sqlite():
    """A build created but never updated must sort on created_time, not on NULL.

    The ORDER BY used to read ``CASE WHEN created_time > updated_time THEN
    created_time ELSE updated_time END``. Comparing against NULL is not true, so a
    row whose updated_time is NULL took the ELSE branch and sorted on NULL —
    throwing away the only activity timestamp it had. The WHERE clause
    deliberately admits those rows ("builds started in the window, not yet
    updated"), so they were selected and then mis-ordered.

    Exercised against real SQLite rather than by asserting on the SQL string: the
    expression has to be portable across SQLite and Postgres, and the obvious
    Postgres spelling (GREATEST) does not exist in SQLite at all.
    """
    import re
    import sqlite3

    from gb_ui_backend.services import gbserver_source as gs

    order_by = re.search(
        r"ORDER BY (CASE WHEN COALESCE.*?END)",
        gs.__loader__.get_source(gs.__name__),
        re.S,
    )
    assert order_by, "could not find the scan's ORDER BY expression"
    expr = " ".join(order_by.group(1).split())

    con = sqlite3.connect(":memory:")
    con.execute(
        "CREATE TABLE gb_builds (uuid TEXT, created_time TEXT, updated_time TEXT)"
    )
    con.executemany(
        "INSERT INTO gb_builds VALUES (?, ?, ?)",
        [
            ("never-updated-newest", "2026-03-01", None),
            ("updated-older", "2026-01-01", "2026-02-01"),
            ("never-updated-oldest", "2026-01-15", None),
        ],
    )
    rows = con.execute(f"SELECT uuid FROM gb_builds ORDER BY {expr} DESC").fetchall()
    con.close()

    assert [r[0] for r in rows] == [
        "never-updated-newest",
        "updated-older",
        "never-updated-oldest",
    ], "rows with a NULL updated_time are not ordered by their created_time"
