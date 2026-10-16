# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for api-bridge."""

import asyncio
import os
from datetime import UTC, datetime
from typing import Any

import pytz

# Constant for system-level user ID (shared configs/datasets)
SYSTEM_USER = "00000000-0000-0000-0000-000000000001"


def utc_now_string():
    return datetime.now(UTC).isoformat()


def get_utc_timestamp(date_input):
    if not date_input:
        return None
    # If it's a string, parse it to datetime first. pymysql only ever hands back a
    # string for a value it could not convert to a datetime — in practice MySQL's
    # zero date '0000-00-00 00:00:00', which users/trials/results rows written
    # without a timestamp carry (those columns are DATETIME NOT NULL with no
    # default). It has no valid year/month/day, so treat it — and anything else
    # unparseable — as absent rather than let it 500 the request.
    if isinstance(date_input, str):
        try:
            date_obj = datetime.strptime(date_input, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    else:
        date_obj = date_input

    # Only localize if the datetime is naive (no timezone info)
    if date_obj.tzinfo is None:
        return pytz.UTC.localize(date_obj).isoformat()
    else:
        return date_obj.astimezone(pytz.UTC).isoformat()


async def run_command(command: str) -> dict[str, Any]:
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return {
        "stdout": stdout.decode(),
        "stderr": stderr.decode(),
        "code": process.returncode,
    }


def is_gb_enabled() -> bool:
    return bool(os.getenv("GB_TOKEN"))


def get_gb_token() -> str:
    return os.getenv("GB_TOKEN")
