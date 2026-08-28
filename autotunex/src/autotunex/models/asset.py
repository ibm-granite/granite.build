# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Schema for a job's downloadable output assets (Results panel)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class AssetSummary(BaseModel):
    """One downloadable output file. The UI renders filename/size/modified.

    The optional fields carry through richer descriptors (registry checkpoints)
    without a later contract change. ``path`` (relative to the artifact root) is
    what the download endpoint keys on — not ``filename``, which can repeat
    across directories.
    """

    filename: str
    size: int
    modified: datetime | None = None
    path: str | None = None
    file_hash: str | None = None
    published: bool | None = None
