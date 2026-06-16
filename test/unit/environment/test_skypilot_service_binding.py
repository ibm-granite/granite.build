"""Unit tests for the SkyPilot monitor's per-poll log scrape (SERVICE URL bindings).

These exercise _download_and_parse_logs' offset tracking and the per-poll
backoff logic in isolation, mocking sky.download_logs to return a temp dir
with a controllable *.log file. No cluster needed.
"""

import asyncio
from pathlib import Path
from unittest import mock

import pytest

from gbserver.environment.skypilot import Skypilot


def _make_env() -> Skypilot:
    # event_q is required by the base Environment; a plain asyncio.Queue works.
    return Skypilot(event_q=asyncio.Queue())


def test_init_has_offset_and_emitted_state():
    env = _make_env()
    assert env._parsed_log_offsets == {}
    assert env._emitted_binding_ids == {}


def _write_log(dirpath: Path, name: str, lines: list[str]) -> None:
    (dirpath / name).write_text("\n".join(lines) + "\n")


RM_EVENT_CONFIGS = [
    {
        "event_type": "NEWARTIFACT_IN_ENVIRONMENT_EVENT",
        "line_regex": "Starting FastAPI server on .+",
        "event_fields": [
            {"field_name": "binding_id", "field_value_template": "rm_server_url"},
            {
                "field_name": "path",
                "field_regex": "(?<=Starting FastAPI server on )\\S+",
                "is_data": True,
            },
            {
                "field_name": "binding",
                "field_value_template": '{ "path": "http://{{ fields.data.path }}" }',
                "is_json": True,
            },
        ],
    }
]


def _parser_configs():
    from gbserver.environment.environment import EventLogLineParserConfig

    return [EventLogLineParserConfig.model_validate(c) for c in RM_EVENT_CONFIGS]


async def _collect(q: asyncio.Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


@pytest.mark.asyncio
async def test_offset_dedup_no_reemit(tmp_path):
    env = _make_env()
    env._cluster_names["L"] = "gb-x"
    env._job_ids["L"] = 1
    q: asyncio.Queue = asyncio.Queue()
    _write_log(tmp_path, "1.log", ["boot", "Starting FastAPI server on host9:8000"])

    with mock.patch(
        "gbserver.environment.skypilot._download_logs_with_retry",
        return_value=str(tmp_path),
    ):
        await env._download_and_parse_logs(
            cluster_name="gb-x",
            job_id=1,
            launch_id="L",
            event_q=q,
            entityrun_metadata=None,
            event_log_parser_configs=_parser_configs(),
        )
        first = await _collect(q)
        await env._download_and_parse_logs(
            cluster_name="gb-x",
            job_id=1,
            launch_id="L",
            event_q=q,
            entityrun_metadata=None,
            event_log_parser_configs=_parser_configs(),
        )
        second = await _collect(q)

    assert any(getattr(e.payload, "binding_id", None) == "rm_server_url" for e in first)
    assert second == []
    assert "rm_server_url" in env._emitted_binding_ids["L"]


@pytest.mark.asyncio
async def test_incremental_append_emits_only_new(tmp_path):
    env = _make_env()
    env._cluster_names["L"] = "gb-x"
    env._job_ids["L"] = 1
    q: asyncio.Queue = asyncio.Queue()
    _write_log(tmp_path, "1.log", ["loading..."])
    with mock.patch(
        "gbserver.environment.skypilot._download_logs_with_retry",
        return_value=str(tmp_path),
    ):
        await env._download_and_parse_logs(
            cluster_name="gb-x",
            job_id=1,
            launch_id="L",
            event_q=q,
            entityrun_metadata=None,
            event_log_parser_configs=_parser_configs(),
        )
        assert await _collect(q) == []
        _write_log(
            tmp_path, "1.log", ["loading...", "Starting FastAPI server on host9:8000"]
        )
        await env._download_and_parse_logs(
            cluster_name="gb-x",
            job_id=1,
            launch_id="L",
            event_q=q,
            entityrun_metadata=None,
            event_log_parser_configs=_parser_configs(),
        )
        evs = await _collect(q)
    binding = next(
        e for e in evs if getattr(e.payload, "binding_id", None) == "rm_server_url"
    )
    assert binding.payload.binding == {"path": "http://host9:8000"}


@pytest.mark.asyncio
async def test_read_error_does_not_advance_offset(tmp_path):
    env = _make_env()
    env._cluster_names["L"] = "gb-x"
    env._job_ids["L"] = 1
    q: asyncio.Queue = asyncio.Queue()
    _write_log(tmp_path, "1.log", ["Starting FastAPI server on host9:8000"])
    with mock.patch(
        "gbserver.environment.skypilot._download_logs_with_retry",
        return_value=str(tmp_path),
    ):
        with mock.patch.object(
            env, "get_events_from_log_line", side_effect=RuntimeError("boom")
        ):
            await env._download_and_parse_logs(
                cluster_name="gb-x",
                job_id=1,
                launch_id="L",
                event_q=q,
                entityrun_metadata=None,
                event_log_parser_configs=_parser_configs(),
            )
        assert env._parsed_log_offsets.get("L", {}).get(str(tmp_path / "1.log"), 0) == 0
        await env._download_and_parse_logs(
            cluster_name="gb-x",
            job_id=1,
            launch_id="L",
            event_q=q,
            entityrun_metadata=None,
            event_log_parser_configs=_parser_configs(),
        )
        evs = await _collect(q)
    assert any(getattr(e.payload, "binding_id", None) == "rm_server_url" for e in evs)


@pytest.mark.asyncio
async def test_offset_resets_when_log_shrinks(tmp_path):
    env = _make_env()
    env._cluster_names["L"] = "gb-x"
    env._job_ids["L"] = 1
    q: asyncio.Queue = asyncio.Queue()
    _write_log(tmp_path, "1.log", ["a", "b", "Starting FastAPI server on host9:8000"])
    with mock.patch(
        "gbserver.environment.skypilot._download_logs_with_retry",
        return_value=str(tmp_path),
    ):
        await env._download_and_parse_logs(
            cluster_name="gb-x",
            job_id=1,
            launch_id="L",
            event_q=q,
            entityrun_metadata=None,
            event_log_parser_configs=_parser_configs(),
        )
        _ = await _collect(q)
        # log replaced by a SHORTER one that again contains the server line
        _write_log(tmp_path, "1.log", ["Starting FastAPI server on host42:8000"])
        await env._download_and_parse_logs(
            cluster_name="gb-x",
            job_id=1,
            launch_id="L",
            event_q=q,
            entityrun_metadata=None,
            event_log_parser_configs=_parser_configs(),
        )
        evs = await _collect(q)
    # shrink detected → re-read from top → new server line emitted
    binding = next(
        e for e in evs if getattr(e.payload, "binding_id", None) == "rm_server_url"
    )
    assert binding.payload.binding == {"path": "http://host42:8000"}


def _expected_binding_ids(parser_configs):
    from gbserver.environment.skypilot import _static_expected_binding_ids

    return _static_expected_binding_ids(parser_configs)


def test_static_expected_binding_ids_extraction():
    expected, all_static = _expected_binding_ids(_parser_configs())
    assert expected == {"rm_server_url"}
    assert all_static is True


def test_non_static_binding_id_disables_backoff():
    from gbserver.environment.environment import EventLogLineParserConfig

    cfgs = [
        EventLogLineParserConfig.model_validate(
            {
                "event_type": "NEWARTIFACT_IN_ENVIRONMENT_EVENT",
                "line_regex": "X .+",
                "event_fields": [
                    {"field_name": "binding_id", "field_regex": "(?<=X )\\S+"},
                ],
            }
        )
    ]
    expected, all_static = _expected_binding_ids(cfgs)
    assert all_static is False
