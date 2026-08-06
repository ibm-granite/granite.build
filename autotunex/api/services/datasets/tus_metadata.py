# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Decode and validate a tus Upload-Metadata dict into a typed UploadIntent.

tuspyserver hands the completion hook an already-decoded ``dict[str, str]``
(the comma-separated ``key <base64-value>`` Upload-Metadata header parsed for
us). This module is the single place that interprets it. It imports nothing
from tuspyserver, so it is fully unit-testable in the sandbox.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_VALID_ROLES = {"source", "train", "validation"}


@dataclass
class UploadIntent:
    dataset_id: str
    filename: str
    role: str  # "source" (auto-split) | "train" | "validation"
    expects: List[str]  # the full set of roles this dataset upload group needs
    train_set_percentage: Optional[int] = None
    column_mapping: Optional[Dict[str, str]] = None


def parse_upload_metadata(md: Dict[str, str]) -> UploadIntent:
    """Validate ``md`` and return an UploadIntent. Raise ValueError on bad input."""
    for required in ("dataset_id", "filename", "role", "expects"):
        if not md.get(required):
            raise ValueError(f"Upload-Metadata missing required field: {required}")

    role = md["role"]
    if role not in _VALID_ROLES:
        raise ValueError(
            f"Upload-Metadata role must be one of {_VALID_ROLES}, got {role!r}"
        )

    expects = [r.strip() for r in md["expects"].split(",") if r.strip()]
    if not expects or any(r not in _VALID_ROLES for r in expects):
        raise ValueError(f"Upload-Metadata expects is invalid: {md['expects']!r}")
    if role not in expects:
        raise ValueError(f"role {role!r} is not in expects {expects}")

    tsp = md.get("train_set_percentage")
    train_set_percentage = int(tsp) if tsp not in (None, "") else None

    cm_raw = md.get("column_mapping")
    column_mapping: Optional[Dict[str, str]] = None
    if cm_raw:
        try:
            column_mapping = json.loads(cm_raw)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Upload-Metadata column_mapping is not valid JSON: {e}")
        if not isinstance(column_mapping, dict):
            raise ValueError(
                f"Upload-Metadata column_mapping must be a JSON object, got {type(column_mapping).__name__}"
            )

    return UploadIntent(
        dataset_id=md["dataset_id"],
        filename=md["filename"],
        role=role,
        expects=expects,
        train_set_percentage=train_set_percentage,
        column_mapping=column_mapping,
    )
