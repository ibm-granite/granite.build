# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Tests for api_bridge.utils.get_utc_timestamp."""

from datetime import UTC, datetime

from api_bridge.utils import get_utc_timestamp


def test_get_utc_timestamp_reads_the_mysql_zero_date_as_none():
    # pymysql hands back the raw string for MySQL's zero date because it cannot
    # build a datetime from year/month/day 0. A users/trials/results row written
    # without a timestamp holds exactly this value, and parsing it used to raise
    # ValueError and 500 the whole request (e.g. POST /fmtune/api/job/bootstrap).
    assert get_utc_timestamp("0000-00-00 00:00:00") is None


def test_get_utc_timestamp_returns_none_for_empty_input():
    assert get_utc_timestamp(None) is None
    assert get_utc_timestamp("") is None


def test_get_utc_timestamp_labels_a_naive_datetime_as_utc():
    result = get_utc_timestamp(datetime(2026, 8, 18, 13, 15, 0))

    assert result == datetime(2026, 8, 18, 13, 15, 0, tzinfo=UTC).isoformat()


def test_get_utc_timestamp_converts_an_aware_datetime_to_utc():
    # pymysql returns datetime objects for valid DATETIME columns.
    result = get_utc_timestamp(datetime(2026, 8, 18, 13, 15, 0, tzinfo=UTC))

    assert result == datetime(2026, 8, 18, 13, 15, 0, tzinfo=UTC).isoformat()
