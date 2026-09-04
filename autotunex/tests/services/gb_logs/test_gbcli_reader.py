"""Unit tests for GbcliLogReader, driven through a faked ``gbcli`` package.

``GbcliLogReader`` imports gbcli lazily, inside ``_fetch``, so registering fake
modules in ``sys.modules`` is enough to drive every path it has: the single-page
read, ``fetch_all``'s paging, cross-page dedup, the page cap, both import-failure
translations, and the two response-error translations.

That matters because gbcli ships only with the opt-in ``granite-build`` extra — a
git dependency, not a PyPI release — so it is absent from the default
``make install`` tree and from CI. These tests used to monkeypatch attributes on
the *real* gbcli behind a ``pytest.importorskip``, which meant they never ran
there and the module sat at 18% coverage while looking tested. Faking also drops
two incidental costs of importing the real package: the ``load_dotenv`` side
effect this directory's ``conftest.py`` documents, and a 1000-record
``BUILD_LOGALL_PAGE_SIZE`` that made every paging test build a thousand rows.

What a fake cannot check is gbcli's own API surface: if upstream renamed
``run_logquery`` or moved ``BUILD_LOGALL_PAGE_SIZE``, every test below would still
pass. :func:`test_the_real_gbcli_exposes_what_the_reader_imports` covers exactly
that, and is the only test here still gated on gbcli being installed.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

import pytest

from autotunex.core.exceptions import GbLogsUnavailableError, GbLogsUpstreamError
from autotunex.services.gb_logs.gbcli_reader import GbcliLogReader

PAGE_SIZE = 3
"""The fake's ``BUILD_LOGALL_PAGE_SIZE``.

Deliberately tiny. The reader treats a short page as "last page", so the paging
tests have to build whole pages; at gbcli's real 1000 that is a thousand records
per page for exactly the same coverage.
"""


class _FakeGbcli:
    """Records what the reader asked gbcli to do, and answers with canned pages."""

    def __init__(self) -> None:
        self._responder: Callable[[int], Any] = lambda _call: {"status": 200, "logs": []}
        self.calls: list[dict[str, Any]] = []
        self.events: list[str] = []

    # --- the two symbols the reader imports ---

    def configure_working_env(self) -> None:
        """Stands in for ``configureGBWorkingEnv``."""
        self.events.append("configured")

    def run_logquery(self, token: str, **kwargs: Any) -> Any:  # noqa: ANN401
        """Stands in for ``run_logquery``; returns whatever the test queued."""
        self.events.append("queried")
        self.calls.append({"token": token, **kwargs})
        return self._responder(len(self.calls))

    # --- test-facing setup ---

    def returns(self, *responses: Any) -> None:  # noqa: ANN401
        """Answer successive calls with ``responses``, in order."""
        queued = list(responses)
        self._responder = lambda call: queued[call - 1]

    def answers_with(self, responder: Callable[[int], Any]) -> None:
        """Answer every call from ``responder(call_number)`` (1-based)."""
        self._responder = responder

    def raises(self, exc: BaseException) -> None:
        """Make every query raise ``exc``, as gbcli does for a bad working env."""

        def _raise(_call: int) -> Any:  # noqa: ANN401
            raise exc

        self._responder = _raise


CLI_CONFIG = "gbcli.utils.cli_config"
GBCONSTANTS = "gbcli.utils.gbconstants"
LOG_QUERY = "gbcli.utils.log_query"


def _install(monkeypatch: pytest.MonkeyPatch, modules: dict[str, ModuleType | None]) -> None:
    """Point ``sys.modules`` at the given fake modules for one test.

    A ``None`` value is not a no-op: CPython's import machinery raises
    ``ImportError`` for a ``sys.modules`` entry that is ``None``, which is how the
    module-absent branches are reached without uninstalling anything.
    """
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def gbcli(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeGbcli]:
    """A complete fake ``gbcli`` package, plus the token the reader reads."""
    fake = _FakeGbcli()

    cli_config = ModuleType(CLI_CONFIG)
    cli_config.configureGBWorkingEnv = fake.configure_working_env  # type: ignore[attr-defined]
    gbconstants = ModuleType(GBCONSTANTS)
    gbconstants.BUILD_LOGALL_PAGE_SIZE = PAGE_SIZE  # type: ignore[attr-defined]
    log_query = ModuleType(LOG_QUERY)
    log_query.run_logquery = fake.run_logquery  # type: ignore[attr-defined]

    _install(
        monkeypatch,
        {
            "gbcli": ModuleType("gbcli"),
            "gbcli.utils": ModuleType("gbcli.utils"),
            CLI_CONFIG: cli_config,
            GBCONSTANTS: gbconstants,
            LOG_QUERY: log_query,
        },
    )
    monkeypatch.setenv("GB_TOKEN", "tok")
    yield fake


def _record(log: str | None) -> dict[str, Any]:
    """A gb log record: its ``text`` is JSON carrying a ``log`` field."""
    return {"logId": log, "timestamp": 1, "text": json.dumps({"log": log})}


def _full_page(prefix: str, *, start_ts: int = 1_000) -> list[dict[str, Any]]:
    """A full ``PAGE_SIZE``-record page, so the reader keeps paging."""
    return [
        {
            "logId": f"{prefix}-{i}",
            "timestamp": start_ts + i,
            "text": json.dumps({"log": f"{prefix}-{i}"}),
        }
        for i in range(PAGE_SIZE)
    ]


# --- the single-page read ---------------------------------------------------


async def test_fetch_returns_extracted_lines_from_the_first_page(gbcli: _FakeGbcli) -> None:
    gbcli.returns({"status": 200, "logs": [_record("line a"), _record("line b")]})
    reader = GbcliLogReader("GB_TOKEN")

    lines = await reader.fetch("build-1", fetch_all=False)

    assert lines == ["line a", "line b"]


async def test_fetch_queries_only_the_requested_build_within_the_window(
    gbcli: _FakeGbcli,
) -> None:
    """The build id and page size the reader sends are part of its gbcli contract."""
    gbcli.returns({"status": 200, "logs": []})
    reader = GbcliLogReader("GB_TOKEN", window_days=7)

    await reader.fetch("build-42", fetch_all=False)

    (call,) = gbcli.calls
    assert call["token"] == "tok"
    assert call["build_id"] == "build-42"
    assert call["page_size"] == PAGE_SIZE
    assert call["sort"] == "asc"
    # A seven-day window, expressed as the span the reader asked for.
    assert call["end_epoch_in_s"] - call["start_epoch_in_s"] == 7 * 86_400


async def test_fetch_raises_unavailable_without_a_token(
    gbcli: _FakeGbcli, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GB_TOKEN", raising=False)
    reader = GbcliLogReader("GB_TOKEN")

    with pytest.raises(GbLogsUnavailableError):
        await reader.fetch("build-1", fetch_all=False)

    assert gbcli.calls == []  # no token means no upstream call at all


async def test_fetch_seeds_the_gb_working_env_before_querying(gbcli: _FakeGbcli) -> None:
    # The reader must run gbcli's env bootstrap (which seeds GB_CONFIG) *before*
    # calling the log server, or the query dereferences an unset GB_CONFIG.
    gbcli.returns({"status": 200, "logs": []})
    reader = GbcliLogReader("GB_TOKEN")

    await reader.fetch("build-1", fetch_all=False)

    assert gbcli.events == ["configured", "queried"]


# --- error translation ------------------------------------------------------


async def test_fetch_raises_upstream_when_the_server_errors(gbcli: _FakeGbcli) -> None:
    gbcli.returns(None)
    reader = GbcliLogReader("GB_TOKEN")

    with pytest.raises(GbLogsUpstreamError):
        await reader.fetch("build-1", fetch_all=False)


async def test_fetch_raises_upstream_for_a_non_200_status(gbcli: _FakeGbcli) -> None:
    gbcli.returns({"status": 500, "logs": []})
    reader = GbcliLogReader("GB_TOKEN")

    with pytest.raises(GbLogsUpstreamError):
        await reader.fetch("build-1", fetch_all=False)


async def test_fetch_translates_an_unexpected_gbcli_error_into_unavailable(
    gbcli: _FakeGbcli,
) -> None:
    # gbcli dereferences ``os.environ["GB_CONFIG"]`` during credential resolution;
    # if the working env is not seeded it raises this bare KeyError deep in the
    # call — the 500 this reader must translate into its declared 503 contract.
    gbcli.raises(KeyError("GB_CONFIG"))
    reader = GbcliLogReader("GB_TOKEN")

    with pytest.raises(GbLogsUnavailableError):
        await reader.fetch("build-1", fetch_all=False)


async def test_fetch_reports_unavailable_when_the_env_bootstrap_is_missing(
    gbcli: _FakeGbcli, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment without gbcli degrades to the 503, not an ImportError."""
    _install(monkeypatch, {CLI_CONFIG: None})
    reader = GbcliLogReader("GB_TOKEN")

    with pytest.raises(GbLogsUnavailableError):
        await reader.fetch("build-1", fetch_all=False)


async def test_fetch_reports_unavailable_when_the_log_query_module_is_missing(
    gbcli: _FakeGbcli, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second import site degrades the same way as the first.

    Separate from the bootstrap case on purpose: the reader imports in two
    stages, and only this one runs *after* ``configureGBWorkingEnv``.
    """
    _install(monkeypatch, {LOG_QUERY: None})
    reader = GbcliLogReader("GB_TOKEN")

    with pytest.raises(GbLogsUnavailableError):
        await reader.fetch("build-1", fetch_all=False)

    assert gbcli.events == ["configured"]  # bootstrap ran, the query never did


# --- line extraction --------------------------------------------------------


async def test_extract_uses_null_placeholder_for_a_missing_log_field(
    gbcli: _FakeGbcli,
) -> None:
    gbcli.returns({"status": 200, "logs": [{"logId": "x", "timestamp": 1, "text": json.dumps({})}]})
    reader = GbcliLogReader("GB_TOKEN")

    lines = await reader.fetch("build-1", fetch_all=False)

    assert lines == ["<null>"]


async def test_extract_falls_back_to_the_raw_text_when_it_is_not_json(
    gbcli: _FakeGbcli,
) -> None:
    """A record whose ``text`` is a bare string is surfaced as-is, not dropped."""
    gbcli.returns(
        {"status": 200, "logs": [{"logId": "x", "timestamp": 1, "text": "not json at all"}]}
    )
    reader = GbcliLogReader("GB_TOKEN")

    lines = await reader.fetch("build-1", fetch_all=False)

    assert lines == ["not json at all"]


async def test_extract_falls_back_to_empty_when_a_record_has_no_text(
    gbcli: _FakeGbcli,
) -> None:
    gbcli.returns({"status": 200, "logs": [{"logId": "x", "timestamp": 1}]})
    reader = GbcliLogReader("GB_TOKEN")

    lines = await reader.fetch("build-1", fetch_all=False)

    assert lines == [""]


# --- fetch_all paging -------------------------------------------------------


async def test_fetch_all_stops_immediately_on_an_empty_first_page(gbcli: _FakeGbcli) -> None:
    gbcli.returns({"status": 200, "logs": []})
    reader = GbcliLogReader("GB_TOKEN")

    lines = await reader.fetch("build-1", fetch_all=True)

    assert lines == []
    assert len(gbcli.calls) == 1


async def test_fetch_all_pages_through_a_full_page_then_stops_at_a_short_page(
    gbcli: _FakeGbcli,
) -> None:
    gbcli.returns(
        {"status": 200, "logs": _full_page("p1")},
        {"status": 200, "logs": [{"logId": "p2-0", "timestamp": 5_000, "text": '{"log": "p2-0"}'}]},
    )
    reader = GbcliLogReader("GB_TOKEN")

    lines = await reader.fetch("build-1", fetch_all=True)

    assert len(gbcli.calls) == 2  # the short page terminated the loop; no third call
    assert lines == [f"p1-{i}" for i in range(PAGE_SIZE)] + ["p2-0"]


async def test_fetch_all_dedups_a_log_id_repeated_across_pages(gbcli: _FakeGbcli) -> None:
    repeated_log_id = f"p1-{PAGE_SIZE - 1}"
    gbcli.returns(
        {"status": 200, "logs": _full_page("p1")},
        {
            "status": 200,
            "logs": [
                # gb re-sent the last record of page1 verbatim across the boundary.
                {
                    "logId": repeated_log_id,
                    "timestamp": 6_000,
                    "text": json.dumps({"log": repeated_log_id}),
                },
                {"logId": "p2-new", "timestamp": 5_000, "text": json.dumps({"log": "p2-new"})},
            ],
        },
    )
    reader = GbcliLogReader("GB_TOKEN")

    lines = await reader.fetch("build-1", fetch_all=True)

    assert lines.count(repeated_log_id) == 1
    assert "p2-new" in lines
    assert len(lines) == PAGE_SIZE + 1


async def test_fetch_all_advances_the_window_to_the_newest_timestamp_in_seconds(
    gbcli: _FakeGbcli,
) -> None:
    """Timestamps arrive in milliseconds, so the next window start is them ÷ 1000.

    Only visible with a realistic timestamp. The reader takes
    ``max(newest_ms / 1000, previous_start + 1)``, so a toy value like ``20_000``
    loses to a window start seven days in the past and the ms→s step never runs —
    which is what the sibling test below covers instead.
    """
    newest_ms = int((time.time() - 60) * 1_000)
    gbcli.returns(
        {"status": 200, "logs": _full_page("p1", start_ts=newest_ms - (PAGE_SIZE - 1))},
        {"status": 200, "logs": []},
    )
    reader = GbcliLogReader("GB_TOKEN")

    await reader.fetch("build-1", fetch_all=True)

    assert gbcli.calls[1]["start_epoch_in_s"] == newest_ms // 1_000


async def test_fetch_all_never_rewinds_the_window_on_a_stale_timestamp(
    gbcli: _FakeGbcli,
) -> None:
    """The other side of that ``max``: a useless timestamp still makes progress.

    A record timestamped near the epoch would otherwise rewind the window to
    1970 and re-read the same page forever.
    """
    gbcli.returns(
        {"status": 200, "logs": _full_page("p1", start_ts=1)},
        {"status": 200, "logs": []},
    )
    reader = GbcliLogReader("GB_TOKEN")

    await reader.fetch("build-1", fetch_all=True)

    assert gbcli.calls[1]["start_epoch_in_s"] == gbcli.calls[0]["start_epoch_in_s"] + 1


async def test_fetch_all_stops_at_the_page_cap_without_looping_forever(
    gbcli: _FakeGbcli,
) -> None:
    # Always a full, never-repeating, strictly-increasing-timestamp page, so
    # neither the short-page nor the all-duplicates termination ever fires —
    # only the page_cap can stop this loop.
    gbcli.answers_with(
        lambda call: {"status": 200, "logs": _full_page(f"call{call}", start_ts=call * 10_000)}
    )
    reader = GbcliLogReader("GB_TOKEN", page_cap=3)

    await reader.fetch("build-1", fetch_all=True)

    assert len(gbcli.calls) == 3


async def test_fetch_all_keeps_records_that_carry_no_id_or_no_newer_timestamp(
    gbcli: _FakeGbcli,
) -> None:
    """Dedup and window-advance are both best-effort: a thin record is still kept.

    ``logId`` is what dedup keys on and ``timestamp`` is what advances the window,
    so a record missing either cannot participate in those — but dropping it would
    lose a log line, which is the one thing this reader exists to return.
    """
    gbcli.returns(
        {
            "status": 200,
            "logs": [
                {"timestamp": 5_000, "text": json.dumps({"log": "no-id"})},
                # Older than the newest seen, so it must not pull the window back.
                {"logId": "b", "timestamp": 3_000, "text": json.dumps({"log": "older"})},
                {"logId": "c", "text": json.dumps({"log": "no-timestamp"})},
            ],
        },
        {"status": 200, "logs": []},
    )
    reader = GbcliLogReader("GB_TOKEN")

    lines = await reader.fetch("build-1", fetch_all=True)

    assert lines == ["no-id", "older", "no-timestamp"]


# --- the real gbcli, when it is installed -----------------------------------


def test_the_real_gbcli_exposes_what_the_reader_imports() -> None:
    """Pin the gbcli API surface the fake above stands in for.

    The only test in this module that needs the real package, and the reason the
    rest can safely fake it: a rename or move upstream fails here rather than
    passing everywhere and breaking in production.
    """
    pytest.importorskip("gbcli.utils.gbconstants")
    from gbcli.utils.cli_config import configureGBWorkingEnv
    from gbcli.utils.gbconstants import BUILD_LOGALL_PAGE_SIZE
    from gbcli.utils.log_query import run_logquery

    assert callable(configureGBWorkingEnv)
    assert callable(run_logquery)
    assert isinstance(BUILD_LOGALL_PAGE_SIZE, int) and BUILD_LOGALL_PAGE_SIZE > 0
