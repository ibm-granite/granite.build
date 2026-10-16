"""Tests for the log API schemas."""

from __future__ import annotations

from datetime import datetime

from autotunex.models.log import LogEntryRead, LogPage


def test_log_entry_read_builds_from_orm_attributes() -> None:
    class Row:
        id = 812
        level = "INFO"
        filename = "train.py"
        message = "epoch 3 loss=0.42"
        iteration = 3
        epoch = 2.5
        timestamp = datetime(2026, 8, 10, 9, 12, 3)

    entry = LogEntryRead.model_validate(Row())

    assert entry.id == 812
    assert entry.level == "INFO"
    assert entry.iteration == 3
    assert entry.epoch == 2.5


def test_log_entry_read_defaults_optional_fields_to_none() -> None:
    entry = LogEntryRead(id=1)

    assert entry.message is None
    assert entry.timestamp is None


def test_log_page_defaults_next_before_id_to_none() -> None:
    page = LogPage(logs=[LogEntryRead(id=1)], has_more=False)

    assert page.next_before_id is None
    assert page.has_more is False
