# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Pure mapping from a gbserver cluster status string to our RunStatus.

``granite.build/src/gbserver/types/status.py`` defines nine states; we own six.
The mapping returns ``RunStatus | None`` so "no opinion" is representable in the
type rather than an ``if/elif`` fallthrough, and an unknown string leaves the
job alone rather than risking a mis-map to an irreversible terminal state.
"""

from __future__ import annotations

from autotunex.core.logging import get_logger
from autotunex.models.status import RunStatus

logger = get_logger(__name__)

_MAPPING: dict[str, RunStatus | None] = {
    # Cluster has not started it; our job is already `pending`.
    "submitted": None,
    "pending": None,
    "retry_pending": None,
    # The pending -> running confirmation the launch spec deferred.
    "running": RunStatus.RUNNING,
    "success": RunStatus.COMPLETED,
    # `invalid` is terminal per Status.is_finished(); 2025 ignored it, parking
    # validation-rejected builds forever.
    "failed": RunStatus.ERROR,
    "invalid": RunStatus.ERROR,
    "cancelled": RunStatus.TERMINATED,
    # Not yet terminal; the next sweep will see `cancelled`.
    "cancel_requested": None,
}


def to_run_status(cluster_status: str) -> RunStatus | None:
    """Map a gbserver status to a :class:`RunStatus`, or ``None`` for "no write".

    ``None`` means do not write: either a known not-yet-decisive state
    (``submitted``/``pending``/``retry_pending``/``cancel_requested``) or an
    unknown string, which is logged so a newly-added tenth cluster state is
    noticed rather than silently mis-mapped.
    """
    key = cluster_status.strip().lower()
    if key not in _MAPPING:
        logger.warning("Unknown gbserver status %r; leaving the job unchanged.", cluster_status)
        return None
    return _MAPPING[key]
