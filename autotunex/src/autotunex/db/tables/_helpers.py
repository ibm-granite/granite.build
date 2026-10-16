# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the table modules.

The schema gives most timestamp columns a ``DEFAULT CURRENT_TIMESTAMP``, which
MySQL evaluates server-side. SQLite has no portable equivalent that also yields
an aware value, so the default is applied in Python instead — one definition,
here, so the eight modules cannot drift apart on what "now" means.
"""

from __future__ import annotations

from datetime import UTC, datetime


def utcnow() -> datetime:
    """Return the current time as an aware UTC datetime.

    Aware because :class:`autotunex.db.types.UtcDateTime` rejects naive values
    outright — a naive default would fail at flush time rather than silently
    storing a local timestamp.
    """
    return datetime.now(UTC)
