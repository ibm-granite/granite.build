#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Admin-DB reconciliation for centralized lineage recording.

The admin DB already holds the complete lineage graph: every successful target
run and its input/output artifacts are persisted to admin storage during the
build. ``WandBLineageStore.add_jobstats_for_build_target`` reconstructs a
target's lineage purely from ``storage.target_storage`` — no build events are
involved. So the full lineage for granite.build is recoverable by re-reading the
admin DB alone.

This module makes that reconstruction the *central* recording mechanism, rather
than driving recording off the (in-memory, restart-blind) event stream:

- ``record_target_lineage`` is the single idempotent leaf: "record this one
  (build, target)". Everything that records lineage goes through it — the
  reconciliation scan below, and (later) a manual/CLI selector for pushing
  selected build lineage to the store, with no rework.
- ``reconcile_once`` is the central selector: it scans the admin DB for
  successful target runs and feeds each through the leaf.

Idempotency is what makes a full rescan safe: the underlying store records with
deterministic runIds + ``resume="allow"`` + content-dedupe, so re-recording an
already-recorded target is harmless. Because the scan re-derives the recordable
set from the DB on every pass, a target that succeeded while the recorder was
down is picked up on the next scan — there is no restart blind spot.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

from gbserver.lineage.jobstats import ILineageStore
from gbserver.storage.singleton_storage import SingletonAdminStorage
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def record_target_lineage(
    store: ILineageStore,
    storage: SingletonAdminStorage,
    build_id: str,
    target_id: str,
) -> None:
    """Record lineage for a single (build, target) — the one recording leaf.

    Idempotent: the underlying store dedupes by deterministic runId, so calling
    this for an already-recorded target is a harmless no-op on the backend. Both
    the reconciliation scan and any future manual/selective push feed this same
    leaf.

    Args:
        store: The lineage store to record into.
        storage: Admin storage the store reads the target's lineage from.
        build_id: Build the target belongs to.
        target_id: Target run to record lineage for.
    """
    store.add_jobstats_for_build_target(storage, build_id=build_id, target_id=target_id)


def select_recordable_targets(
    storage: SingletonAdminStorage,
) -> list[StoredTargetRun]:
    """Select the target runs whose lineage should be recorded.

    A target is recordable once it has completed successfully; its lineage is
    fully persisted in admin storage at that point. ``status`` is a queryable
    column on the target storage, so this is a single indexed query rather than a
    full-table scan.

    Returns:
        The successful target runs currently in the admin DB.
    """
    targets = storage.target_storage.get_by_where({"status": Status.SUCCESS.name})
    return [t for t in targets if isinstance(t, StoredTargetRun)]


def reconcile_once(
    store: ILineageStore,
    storage: SingletonAdminStorage,
    already_recorded: Optional[set[str]] = None,
    on_error: Optional[Callable[[str, str, Exception], None]] = None,
) -> set[str]:
    """Reconcile admin-DB lineage into the store once (the central mechanism).

    Scans the admin DB for successful target runs and records each through the
    single leaf. Targets whose uuid is in ``already_recorded`` are skipped to
    bound per-scan work in steady state; recording is idempotent regardless, so
    skipping is an optimization, not a correctness requirement.

    Args:
        store: The lineage store to record into.
        storage: Admin storage to reconcile from.
        already_recorded: Target uuids recorded on a previous scan; skipped this
            pass. Pass an empty set (or None) to force a full rescan — e.g. at
            startup, so anything that succeeded while the recorder was down is
            re-driven.
        on_error: Optional callback ``(build_id, target_id, exc)`` invoked when
            recording a single target raises, so the caller can queue a retry.
            When omitted, a failure is logged and the target is simply retried on
            the next scan (it stays absent from the returned recorded set).

    Returns:
        The set of target uuids successfully recorded so far, i.e.
        ``already_recorded`` plus any newly recorded this pass. The caller
        threads this back in as ``already_recorded`` on the next call.
    """
    recorded: set[str] = set(already_recorded or set())
    targets = select_recordable_targets(storage)

    newly_recorded = 0
    for target in targets:
        if target.uuid in recorded:
            continue
        try:
            record_target_lineage(
                store, storage, build_id=target.build_id, target_id=target.uuid
            )
            recorded.add(target.uuid)
            newly_recorded += 1
        except Exception as exc:  # noqa: BLE001 - reconciliation must not abort
            if on_error is not None:
                on_error(target.build_id, target.uuid, exc)
            else:
                logger.warning(
                    "Failed to record lineage for target %s in build %s; "
                    "will retry on next scan: %s",
                    target.uuid,
                    target.build_id,
                    exc,
                )

    if newly_recorded:
        logger.info("Reconciled lineage for %d target(s)", newly_recorded)
    return recorded


def record_selected_targets(
    store: ILineageStore,
    storage: SingletonAdminStorage,
    targets: Iterable[tuple[str, str]],
) -> None:
    """Record lineage for an explicitly selected set of (build_id, target_id).

    The seam for a future manual/selective push (e.g. a standalone user recording
    a few important builds to a centralized store): a selector supplies the pairs
    and they flow through the same idempotent leaf the reconciliation scan uses.

    Args:
        store: The lineage store to record into.
        storage: Admin storage the store reads lineage from.
        targets: Iterable of (build_id, target_id) pairs to record.
    """
    for build_id, target_id in targets:
        record_target_lineage(store, storage, build_id=build_id, target_id=target_id)
