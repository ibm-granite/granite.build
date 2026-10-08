"""The batched path extraction must match per-build extraction, and stay off the loop.

`_scan_datasets_async` used to call `_extract_dp_paths` per build inside the
coroutine, alongside a keyword diagnostic that lowercased every build's YAML nine
times and logged every build name in one oversized INFO line. Review measured that
block at roughly as much event-loop time as the archive decode the same PR had
already moved to a worker thread — so the stall was only half fixed.

The diagnostic is gone and the extraction is batched into one `asyncio.to_thread`
hop. These tests guard the two things that could go wrong with that: the answers
changing, and the work quietly coming back onto the loop.
"""

import asyncio
import threading
from datetime import datetime, timezone

import pytest

from gb_ui_backend.api import data_processing as dp

_TOKENIZATION_YAML = """
steps:
  - run: python tokenization2arrow --input_folder cos://bucket/parquet/set-a --output_folder cos://bucket/arrow/set-a
"""

_MEGATRON_YAML = """
steps:
  - run: python run_cos_pipeline.py --arrow_path cos://bucket/arrow/set-b --megatron_path cos://bucket/megatron/set-b
"""

_UNRELATED_YAML = """
steps:
  - run: python train.py --epochs 3
"""


def _build(uuid: str, yaml: str | None):
    return {
        "uuid": uuid,
        "name": f"build-{uuid}",
        "username": "someone",
        "status": "success",
        "created_time": datetime(2026, 9, 1, tzinfo=timezone.utc),
        "updated_time": datetime(2026, 9, 2, tzinfo=timezone.utc),
        "yaml_content": yaml,
    }


class _FakeSource:
    """Stands in for GbserverSource; returns a fixed page of builds."""

    def __init__(self, builds):
        self._builds = builds

    async def list_builds_for_dp_scan(self, days_back: int = 7, limit: int = 0):
        return self._builds, None


@pytest.fixture
def builds():
    return [
        _build("aaa", _TOKENIZATION_YAML),
        _build("bbb", _MEGATRON_YAML),
        _build("ccc", _UNRELATED_YAML),
        _build("ddd", None),  # no archive decoded — must be skipped, not counted
    ]


@pytest.mark.asyncio
async def test_batched_extraction_matches_per_build_extraction(monkeypatch, builds):
    """Same answers as calling _extract_dp_paths directly, build for build."""
    monkeypatch.setattr(dp, "get_gbserver_source", lambda: _FakeSource(builds))

    datasets, scanned, matched, warning = await dp._scan_datasets_async(days=30)

    inline = [
        (b["uuid"], dp._extract_dp_paths(b["yaml_content"]))
        for b in builds
        if b.get("yaml_content")
    ]
    expected_scanned = len(inline)
    expected_matched = sum(1 for _, paths in inline if paths)

    assert scanned == expected_scanned, "a build with no YAML must not be scanned"
    assert matched == expected_matched
    assert warning is None

    # Every matched build must appear in exactly one dataset group.
    grouped = {b["uuid"] for d in datasets for b in d.get("builds", [])}
    assert grouped == {uuid for uuid, paths in inline if paths}


@pytest.mark.asyncio
async def test_extraction_does_not_run_on_the_event_loop_thread(monkeypatch, builds):
    """Structural, not timing-based — same reasoning as the decode's thread check."""
    loop_thread = threading.get_ident()
    seen: list[int] = []
    real = dp._extract_dp_paths

    def recording(yaml_content):
        seen.append(threading.get_ident())
        return real(yaml_content)

    monkeypatch.setattr(dp, "_extract_dp_paths", recording)
    monkeypatch.setattr(dp, "get_gbserver_source", lambda: _FakeSource(builds))

    await dp._scan_datasets_async(days=30)

    assert seen, "_extract_dp_paths was never called"
    assert all(t != loop_thread for t in seen), (
        "path extraction ran on the event loop thread, so a wide scan still stalls "
        "every other request the sidecar is serving"
    )


@pytest.mark.asyncio
async def test_no_build_names_are_logged_at_info(monkeypatch, builds, caplog):
    """The deleted diagnostic logged every build name in a single ~129KB INFO line."""
    import logging

    monkeypatch.setattr(dp, "get_gbserver_source", lambda: _FakeSource(builds))

    with caplog.at_level(logging.INFO, logger=dp.logger.name):
        await dp._scan_datasets_async(days=30)

    for record in caplog.records:
        assert (
            "build-aaa" not in record.getMessage()
        ), "a build name is being logged at INFO — the per-scan diagnostic is back"
