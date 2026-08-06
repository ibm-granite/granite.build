# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""In-process rendezvous coordinating the files of one dataset upload group.

Each browser dataset upload may comprise multiple files (train + validation)
sent as independent, concurrently-resumable tus uploads. The LAST file to
complete must run finalize exactly once. This tracks per-dataset completion
and hands out a once-only finalize claim guarded by a per-dataset lock.

SINGLE-PROCESS ASSUMPTION: state lives in this process's memory, exactly like
the existing ``_finalizing_uploads`` chunk guard. If the API is ever run
multi-replica, two completions could land on different workers and miss each
other; the documented fix (out of scope here — no new DB columns) is a
DB-backed completion record. Noted, not solved.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class UploadRendezvous:
    def __init__(self) -> None:
        self._files: Dict[str, Dict[str, str]] = defaultdict(
            dict
        )  # dataset_id -> {role: path}
        self._names: Dict[str, Dict[str, str]] = defaultdict(
            dict
        )  # dataset_id -> {role: client filename}
        self._expects: Dict[str, Set[str]] = {}  # dataset_id -> expected roles
        self._finalized: Set[str] = set()  # dataset_ids already claimed
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _lock(self, dataset_id: str) -> asyncio.Lock:
        return self._locks[dataset_id]

    async def record(
        self,
        dataset_id: str,
        *,
        role: str,
        expects: List[str],
        path: str,
        filename: Optional[str] = None,
    ) -> bool:
        """Record a completed file. Return True iff the full expected set is now present.

        ``filename`` is the real client filename for this role; it is retained so
        finalize can preserve the file's extension (downstream format detection is
        extension-based — e.g. .parquet vs .jsonl).
        """
        async with self._lock(dataset_id):
            self._files[dataset_id][role] = path
            if filename:
                self._names[dataset_id][role] = filename
            self._expects[dataset_id] = set(expects)
            present = set(self._files[dataset_id].keys())
            return self._expects[dataset_id].issubset(present)

    def paths(self, dataset_id: str) -> Dict[str, str]:
        """Role -> staged file path for the group (call after record returns True)."""
        return dict(self._files.get(dataset_id, {}))

    def filenames(self, dataset_id: str) -> Dict[str, str]:
        """Role -> real client filename for the group (call after record returns True).

        A role may be absent if its completion did not carry a filename; callers
        should fall back to a sensible default in that case.
        """
        return dict(self._names.get(dataset_id, {}))

    async def claim_finalize(self, dataset_id: str) -> bool:
        """Return True for the FIRST caller per dataset_id, False thereafter (once-only)."""
        async with self._lock(dataset_id):
            if dataset_id in self._finalized:
                return False
            self._finalized.add(dataset_id)
            return True

    async def release_finalize(self, dataset_id: str) -> None:
        """Release a finalize claim after a FAILED finalize so a retry can re-claim.

        The once-only guarantee holds for a SUCCESSFUL finalize (the claim is left
        in ``_finalized``). On failure the caller releases the claim here, so a
        tus-retried completion re-enters ``claim_finalize`` and re-attempts the
        finalize against the (rolled-back) dataset.
        """
        async with self._lock(dataset_id):
            self._finalized.discard(dataset_id)

    def clear(self, dataset_id: str) -> None:
        """Release per-group file/expects/lock state after finalize or abandonment.

        Intentionally RETAINS the dataset_id in ``_finalized`` so a retried
        completion of an already-finalized group (duplicate PATCH, proxy retry)
        cannot claim finalize again. ``_finalized`` is a bounded set of ids — cheap
        to keep for the process lifetime; the staging sweep handles disk.
        """
        self._files.pop(dataset_id, None)
        self._names.pop(dataset_id, None)
        self._expects.pop(dataset_id, None)
        self._locks.pop(dataset_id, None)
