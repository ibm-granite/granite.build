"""Unit tests for GbcliLogReader, with run_logquery monkeypatched."""

from __future__ import annotations

import importlib
import json
from typing import Any

import pytest

# gbcli ships with the granite.build (llmb) backend, which is opt-in and NOT
# installed on the default `make install` path. Skip this module cleanly when it
# is absent, mirroring how GbcliLogReader itself degrades (its gbcli imports are
# lazy). Without this guard a module-scope `from gbcli...` import would *interrupt*
# pytest collection and yield zero tests on the documented install path.
# test_registry.py needs no such guard: it imports our gbcli_reader module, whose
# gbcli imports are deferred to call time.
pytest.importorskip("gbcli.utils.gbconstants")

from autotunex.core.exceptions import GbLogsUnavailableError, GbLogsUpstreamError
from autotunex.services.gb_logs.gbcli_reader import GbcliLogReader

# Read the one gbcli constant the tests need off the (now-importable) module,
# rather than a module-scope `from gbcli...` import that mypy cannot resolve and
# that ruff would rewrap (splitting an inline type-ignore onto the wrong line).
BUILD_LOGALL_PAGE_SIZE = importlib.import_module("gbcli.utils.gbconstants").BUILD_LOGALL_PAGE_SIZE


def _record(log: str | None) -> dict[str, Any]:
    """A gb log record: its ``text`` is JSON carrying a ``log`` field."""
    return {"logId": log, "timestamp": 1, "text": json.dumps({"log": log})}


async def test_fetch_returns_extracted_lines_from_the_first_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_logquery(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return {"status": 200, "logs": [_record("line a"), _record("line b")]}

    monkeypatch.setenv("GB_TOKEN", "tok")
    monkeypatch.setattr("gbcli.utils.log_query.run_logquery", fake_run_logquery)
    reader = GbcliLogReader("GB_TOKEN")

    lines = await reader.fetch("build-1", fetch_all=False)

    assert lines == ["line a", "line b"]


async def test_fetch_raises_unavailable_without_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GB_TOKEN", raising=False)
    reader = GbcliLogReader("GB_TOKEN")

    with pytest.raises(GbLogsUnavailableError):
        await reader.fetch("build-1", fetch_all=False)


async def test_fetch_raises_upstream_when_the_server_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GB_TOKEN", "tok")
    monkeypatch.setattr("gbcli.utils.log_query.run_logquery", lambda *a, **k: None)
    reader = GbcliLogReader("GB_TOKEN")

    with pytest.raises(GbLogsUpstreamError):
        await reader.fetch("build-1", fetch_all=False)


async def test_fetch_translates_an_unexpected_gbcli_error_into_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # gbcli dereferences ``os.environ["GB_CONFIG"]`` during credential resolution;
    # if the working env is not seeded it raises this bare KeyError deep in the
    # call — the 500 this reader must translate into its declared 503 contract.
    def boom(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise KeyError("GB_CONFIG")

    monkeypatch.setenv("GB_TOKEN", "tok")
    monkeypatch.setattr("gbcli.utils.log_query.run_logquery", boom)
    reader = GbcliLogReader("GB_TOKEN")

    with pytest.raises(GbLogsUnavailableError):
        await reader.fetch("build-1", fetch_all=False)


async def test_fetch_seeds_the_gb_working_env_before_querying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The reader must run gbcli's env bootstrap (which seeds GB_CONFIG) *before*
    # calling the log server, or the query dereferences an unset GB_CONFIG.
    order: list[str] = []

    def fake_configure() -> None:
        order.append("configured")

    def fake_run_logquery(*_args: object, **_kwargs: object) -> dict[str, Any]:
        order.append("queried")
        return {"status": 200, "logs": []}

    monkeypatch.setenv("GB_TOKEN", "tok")
    monkeypatch.setattr("gbcli.utils.cli_config.configureGBWorkingEnv", fake_configure)
    monkeypatch.setattr("gbcli.utils.log_query.run_logquery", fake_run_logquery)
    reader = GbcliLogReader("GB_TOKEN")

    await reader.fetch("build-1", fetch_all=False)

    assert order == ["configured", "queried"]


async def test_extract_uses_null_placeholder_for_a_missing_log_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_logquery(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return {"status": 200, "logs": [{"logId": "x", "timestamp": 1, "text": json.dumps({})}]}

    monkeypatch.setenv("GB_TOKEN", "tok")
    monkeypatch.setattr("gbcli.utils.log_query.run_logquery", fake_run_logquery)
    reader = GbcliLogReader("GB_TOKEN")

    lines = await reader.fetch("build-1", fetch_all=False)

    assert lines == ["<null>"]


def _full_page(prefix: str, *, start_ts: int = 1_000) -> list[dict[str, Any]]:
    """A full ``BUILD_LOGALL_PAGE_SIZE``-record page, so the loop keeps paging."""
    return [
        {
            "logId": f"{prefix}-{i}",
            "timestamp": start_ts + i,
            "text": json.dumps({"log": f"{prefix}-{i}"}),
        }
        for i in range(BUILD_LOGALL_PAGE_SIZE)
    ]


async def test_fetch_all_dedups_a_log_id_repeated_across_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repeated_log_id = f"p1-{BUILD_LOGALL_PAGE_SIZE - 1}"
    page1 = _full_page("p1")
    page2 = [
        # gb re-sent the last record of page1 verbatim across the page boundary.
        {
            "logId": repeated_log_id,
            "timestamp": 6_000,
            "text": json.dumps({"log": repeated_log_id}),
        },
        {"logId": "p2-new", "timestamp": 5_000, "text": json.dumps({"log": "p2-new"})},
    ]
    responses = iter([{"status": 200, "logs": page1}, {"status": 200, "logs": page2}])

    def fake_run_logquery(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return next(responses)

    monkeypatch.setenv("GB_TOKEN", "tok")
    monkeypatch.setattr("gbcli.utils.log_query.run_logquery", fake_run_logquery)
    reader = GbcliLogReader("GB_TOKEN")

    lines = await reader.fetch("build-1", fetch_all=True)

    assert lines.count(repeated_log_id) == 1
    assert "p2-new" in lines
    assert len(lines) == BUILD_LOGALL_PAGE_SIZE + 1


async def test_fetch_all_pages_through_a_full_page_then_stops_at_a_short_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page1 = _full_page("p1")
    page2 = [{"logId": "p2-0", "timestamp": 5_000, "text": json.dumps({"log": "p2-0"})}]
    responses = iter([{"status": 200, "logs": page1}, {"status": 200, "logs": page2}])
    calls = 0

    def fake_run_logquery(*_args: object, **_kwargs: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setenv("GB_TOKEN", "tok")
    monkeypatch.setattr("gbcli.utils.log_query.run_logquery", fake_run_logquery)
    reader = GbcliLogReader("GB_TOKEN")

    lines = await reader.fetch("build-1", fetch_all=True)

    assert calls == 2  # the short page terminated the loop; no third call
    assert lines == [f"p1-{i}" for i in range(BUILD_LOGALL_PAGE_SIZE)] + ["p2-0"]


async def test_fetch_all_stops_at_the_page_cap_without_looping_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_run_logquery(*_args: object, **_kwargs: object) -> dict[str, Any]:
        # Always a full, never-repeating, strictly-increasing-timestamp page, so
        # neither the short-page nor the all-duplicates termination ever fires —
        # only the page_cap can stop this loop.
        nonlocal calls
        calls += 1
        return {"status": 200, "logs": _full_page(f"call{calls}", start_ts=calls * 10_000)}

    monkeypatch.setenv("GB_TOKEN", "tok")
    monkeypatch.setattr("gbcli.utils.log_query.run_logquery", fake_run_logquery)
    reader = GbcliLogReader("GB_TOKEN", page_cap=3)

    await reader.fetch("build-1", fetch_all=True)

    assert calls == 3
