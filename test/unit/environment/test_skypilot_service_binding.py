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
