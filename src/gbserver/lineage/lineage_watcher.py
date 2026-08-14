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

"""Async lineage-recording agent driven by admin-DB reconciliation."""

import threading
import time
from datetime import datetime, timedelta
from typing import Optional

from gbserver.lineage.jobstats import ILineageStore, get_lineage_store
from gbserver.lineage.lineage_reconciler import (
    LINEAGE_WATCHER_CHECKPOINT_KEY,
    LINEAGE_WATCHER_DROPPED_KEY,
    reconcile_once,
)
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class LineageWatcher:
    """Async background thread that reconciles lineage from the admin DB.

    Runs a single background daemon thread that periodically calls
    ``reconcile_once`` (see ``lineage_reconciler``), which scans the admin DB for
    successful target runs and records their lineage into the configured store,
    off the build's hot path.

    Reconciliation — not the event stream — is the authoritative mechanism: the
    admin DB persists the complete lineage graph, so the full lineage is
    recoverable by re-reading it alone. Because each scan re-derives the
    recordable set from the DB, a target that succeeded while this process was
    down is picked up on the next scan; there is no restart blind spot. Recording
    is idempotent (deterministic runIds + resume="allow" + content-dedupe), so a
    re-recorded target is a harmless backend no-op.

    Single-writer guarantee: the watcher is deployed as its own single-replica
    ``lineage-watch`` command/pod (see ``command_lineage_watch.py`` and
    ``dep-lineage-watcher.yaml``), so exactly one process reconciles lineage. It
    must not be wired into any other entrypoint. Even if that were violated,
    idempotent recording means a duplicate watcher would waste I/O but not
    corrupt lineage.

    Steady state uses a ``finished_at`` *time watermark*: each scan asks the
    admin DB only for targets that finished at or after it, so per-scan work
    stays bounded no matter how many builds have accumulated. The watermark lives
    solely in the ``gb_status`` checkpoint (see
    ``lineage_reconciler.LINEAGE_WATCHER_CHECKPOINT_KEY``), rewritten after each
    individually-recorded target and re-read at the top of every scan — so a
    restart resumes from the last successfully-recorded target rather than
    rescanning the whole admin DB, and a key seeded or corrected while the
    watcher runs takes effect on the next scan.

    The checkpoint is never created implicitly: with no checkpoint the watcher
    records *nothing* and every scan is a no-op until the key is seeded
    explicitly (see ``gbserver lineage-seed``). Choosing where centralized
    recording begins — "from now", from a given build, or the full history — is
    the operator's call, so a fresh deployment stays silent rather than picking a
    starting point on its own. When the checkpoint does exist, ``start()``
    verifies its build against the store's own recorded-state
    (``filter_unrecorded``, via a build-scoped ``reconcile_once``) and
    re-records what is missing, closing any gap left by a crash between recording
    and persisting the checkpoint (or vice versa). A small
    ``_WATERMARK_OVERLAP`` is subtracted when querying so a target that
    finished in the same instant as the watermark boundary is never skipped;
    idempotent recording makes the resulting re-reads harmless.

    Which of those newly-finished targets actually get recorded is decided
    per-sink by ``store.filter_unrecorded`` (see ``reconcile_once``): the time
    watermark is sink-neutral, and each sink owns its own recorded-state, so the
    same admin DB can feed W&B and other sinks independently.

    A target whose recording raises is retried on the next scan: the checkpoint
    stops advancing at the failed target for the remainder of that pass (even
    though newer targets in the same pass may still record), so the durable
    watermark never moves past unrecorded lineage and the failed target is
    re-surfaced by the next scan — including after a restart, which is what makes
    retry survive process death rather than depending on in-memory state. A
    target that keeps failing is dropped after ``_MAX_RECORD_ATTEMPTS`` (into
    ``_dropped``, passed as ``skip``) so a persistent failure cannot wedge later
    scans or hold the checkpoint back forever. That drop set is itself persisted
    to ``gb_status`` (``LINEAGE_WATCHER_DROPPED_KEY``) and reloaded by
    ``start()``: because the checkpoint deliberately refuses to advance past an
    unrecorded target, an in-memory-only drop set would let a dropped target
    return after a restart, block the watermark, exhaust its attempts again, and
    repeat on every restart — permanently wedging all newer lineage behind a
    target that is never going to record. Dropping is a terminal decision, so it
    is durable; the per-target attempt *counts* stay in memory, since a restart
    may legitimately retry from zero.
    """

    # A target whose lineage recording keeps failing is retried this many times
    # on subsequent scans before being dropped, so a transient failure (e.g. a
    # network blip) is recovered without a persistent failure wedging the scan.
    _MAX_RECORD_ATTEMPTS = 3

    # Subtracted from the watermark when querying so a target that finished at (or
    # a hair before) the boundary is re-surfaced rather than skipped.
    #
    # The case only this can cover is clock skew: finished_at is stamped from the
    # writer's own clock (``datetime.now()`` in the buildrunner), not the DB's, so
    # a target can land with a timestamp slightly *older* than a watermark that
    # has already advanced — on the wrong side of the cutoff, where no amount of
    # tie-breaking would find it. Equal timestamps are also re-surfaced here, but
    # those have a second safety net: ``_verify_checkpoint`` re-scans the
    # checkpoint's whole build at start(), so a cursor left mid-way through a
    # group of same-instant targets is repaired on the next restart. This overlap
    # is what repairs it *without* waiting for one.
    #
    # Re-reads are harmless: per-sink ``filter_unrecorded`` drops
    # already-recorded targets before anything is written, so the cost is a few
    # extra rows in the paged query, not extra recording work.
    _WATERMARK_OVERLAP = timedelta(seconds=5)

    def __init__(self, monitoring_interval: float = 30.0) -> None:
        """Initialize the LineageWatcher.

        Args:
            monitoring_interval: Sleep duration between reconciliation scans
                (seconds).
        """
        self.monitoring_interval = monitoring_interval
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self._store: Optional[ILineageStore] = None
        # Target uuids dropped after exhausting retries; skipped on later scans
        # so a persistently failing target cannot wedge every scan. Loaded from
        # (and persisted to) gb_status, because the checkpoint refuses to advance
        # past an unrecorded target: an in-memory-only drop set would let a
        # dropped target return after a restart and block the watermark forever.
        self._dropped: set[str] = set()
        # target_uuid -> attempts so far, for targets whose recording failed and
        # should be retried on a subsequent scan.
        self._failed_attempts: dict[str, int] = {}

    def start(self) -> None:
        """Start the watcher thread (daemon=True, does not keep process alive)."""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            logger.error("lineage watcher thread is already running")
            return

        self._store = get_lineage_store()
        storage = get_admin_storage()
        # Load the durable drop set before the first scan, so a target already
        # given up on stays skipped instead of blocking the checkpoint again.
        self._load_dropped(storage)
        self._verify_checkpoint(storage)
        self.worker_thread = threading.Thread(
            target=self._run, name="lineage-watcher", daemon=True
        )
        self.worker_thread.start()
        logger.info("LineageWatcher started")

    def _load_dropped(self, storage: SingletonAdminStorage) -> None:
        """Load the durable set of permanently-given-up-on target uuids.

        A dropped target is one that failed ``_MAX_RECORD_ATTEMPTS`` times; the
        decision to stop trying is permanent, so it must outlive the process.
        Without this the target would return on the next start(), block the
        checkpoint (which never advances past unrecorded lineage), exhaust its
        attempts again, and repeat every restart — wedging all newer lineage
        behind it.
        """
        value = storage.status_storage.get_value(LINEAGE_WATCHER_DROPPED_KEY)
        if value:
            self._dropped = set(value.get("target_ids", []))

    def _persist_dropped(self, storage: SingletonAdminStorage) -> None:
        """Persist the drop set so the decision survives a restart."""
        storage.status_storage.set_value(
            LINEAGE_WATCHER_DROPPED_KEY, {"target_ids": sorted(self._dropped)}
        )

    def _verify_checkpoint(self, storage: SingletonAdminStorage) -> None:
        """Re-record any unrecorded target in the checkpoint's own build.

        Run once at ``start()``. This closes the gap left by a crash between
        recording a target and persisting its checkpoint (or vice versa): the
        checkpoint may name a target whose lineage never reached the sink, and
        the steady-state scan — which starts *at* that watermark — would not
        re-surface everything in its build.

        The checkpoint is never created here, or anywhere else implicitly. When
        ``LINEAGE_WATCHER_CHECKPOINT_KEY`` is absent this is a no-op and so is
        every subsequent scan, until the key is seeded explicitly (see
        ``gbserver lineage-seed``). Auto-seeding it from the newest successful
        target would silently pick a starting point for the operator; deciding
        where centralized recording begins — "from now", from a chosen build, or
        the full history — belongs to whoever seeds it, not to whichever process
        happens to start first.
        """
        checkpoint = storage.status_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
        if checkpoint is None:
            logger.info(
                "No lineage checkpoint (%s) found; recording nothing. Seed it "
                "to choose where centralized lineage recording starts.",
                LINEAGE_WATCHER_CHECKPOINT_KEY,
            )
            return

        # Verify/re-record via reconcile_once — the same central selector the
        # steady-state scan uses — scoped to the checkpoint's own build, so the
        # per-sink filter_unrecorded check, expected-run-count derivation and
        # prerun-skip handling are shared rather than reimplemented here.
        #
        # Failures are logged and swallowed rather than aborting start(): the
        # checkpoint is already durable, so nothing is lost by continuing, and a
        # watcher that refused to start would record nothing at all. Note what
        # each kind of failure costs, though. A target sitting at the watermark is
        # re-surfaced by the very next scan (it falls inside the overlap window).
        # A target further back in this build is NOT — the steady-state scan
        # starts at the watermark and never looks behind it, so this start()-time
        # sweep is its only path back, and a failure here defers it to the next
        # start(). That is the gap this call exists to close, so its failures are
        # worth reading in the log even though they are non-fatal.
        def _on_error(build_id: str, target_id: str, exc: Exception) -> None:
            logger.exception(
                "Failed to re-record checkpoint target %s in build %s on start(): %s",
                target_id,
                build_id,
                exc,
            )

        if self._store is not None:
            # datetime.min: every target of this build is in scope, however long
            # ago it finished. build_id already bounds the scan to one build, so
            # there is nothing for a watermark to bound here — and a build whose
            # targets finished before the checkpoint's own timestamp must still
            # be verified.
            reconcile_once(
                self._store,
                storage,
                finished_after=datetime.min,
                build_id=checkpoint["build_id"],
                on_error=_on_error,
            )

    def _run(self) -> None:
        """Main monitoring loop (runs in daemon thread)."""
        while not self.stop_event.is_set():
            try:
                self._reconcile()
            except Exception:
                logger.exception("LineageWatcher iteration failed")

            time.sleep(self.monitoring_interval)

    def _reconcile(self) -> None:
        """Run one reconciliation scan over the admin DB.

        Delegates target selection and recording to ``reconcile_once`` (the
        central mechanism), passing the ``finished_at`` watermark so steady-state
        scans read only newly-finished targets.

        The watermark is read from the ``gb_status`` checkpoint on every scan
        rather than cached in memory: the checkpoint is the single source of
        truth, and re-reading it means a key seeded (or corrected) while the
        watcher is running takes effect on the next scan instead of at the next
        restart. A missing key is a no-op — "record nothing" until it is seeded
        — and must never fall back to scanning the whole admin DB, which would
        turn an unseeded deployment into a full historical backfill.

        Recording failures are routed to ``_on_record_error`` to drive the
        bounded per-target retry. The checkpoint is persisted immediately after
        each individually-recorded target (``_on_checkpoint_advance``) rather
        than once at the end of the scan, so a mid-scan crash leaves it at the
        last target actually recorded.
        """
        if self._store is None:
            logger.error("lineage store not initialized; start() must run first")
            return
        storage = get_admin_storage()
        watermark = self._checkpoint_watermark(storage)
        if watermark is None:
            # No checkpoint: recording is deliberately off until the key is
            # seeded. Return before querying targets or touching the sink.
            return
        # Query slightly before the watermark so a boundary-timestamp completion
        # is not skipped; idempotent recording makes the overlap re-reads
        # harmless.
        reconcile_once(
            self._store,
            storage,
            finished_after=watermark - self._WATERMARK_OVERLAP,
            on_error=lambda build_id, target_id, exc: (
                self._on_record_error(storage, build_id, target_id, exc)
            ),
            on_success=self._on_record_success,
            on_checkpoint_advance=lambda build_id, finished_at: (
                self._on_checkpoint_advance(storage, build_id, finished_at)
            ),
            skip=self._dropped,
        )

    def _checkpoint_watermark(
        self, storage: SingletonAdminStorage
    ) -> Optional[datetime]:
        """Read the watermark from the durable checkpoint, or ``None`` if unset.

        ``None`` means the key has not been seeded, which the caller treats as
        "record nothing" rather than as "no lower bound".
        """
        checkpoint = storage.status_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
        if checkpoint is None:
            return None
        return datetime.fromisoformat(checkpoint["finished_at"])

    def _on_checkpoint_advance(
        self, storage: SingletonAdminStorage, build_id: str, finished_at: datetime
    ) -> None:
        """Persist the advanced checkpoint.

        Called once per successfully-recorded target (oldest-first), so the
        durable checkpoint always reflects the last target actually recorded —
        never a target merely considered or one that failed to record. The next
        scan reads it back, so this write alone advances the watermark.
        """
        storage.status_storage.set_value(
            LINEAGE_WATCHER_CHECKPOINT_KEY,
            {"build_id": build_id, "finished_at": finished_at.isoformat()},
        )

    def _on_record_success(self, build_id: str, target_id: str) -> None:
        """Clear any retry state for a target that recorded successfully.

        A target that failed a prior scan and then succeeds is reported only
        here — it drops out of the unrecorded set on the next scan, so
        ``_on_record_error`` is never called for it again. Without this, its
        ``_failed_attempts`` entry would linger for the process lifetime and a
        much-later re-failure would resume from a nonzero count.
        """
        self._failed_attempts.pop(target_id, None)

    def _on_record_error(
        self,
        storage: SingletonAdminStorage,
        build_id: str,
        target_id: str,
        exc: Exception,
    ) -> None:
        """Handle a recording failure for one target: retry or drop.

        Keeps a per-target attempt count so a transient failure is retried on the
        next scan, while a persistently failing target is dropped after
        ``_MAX_RECORD_ATTEMPTS`` (added to ``_dropped``, which ``_reconcile``
        passes as ``skip`` to ``reconcile_once``) so it stops being re-recorded —
        it still falls within the watermark window each scan, so the skip set is
        what keeps it from wedging the scan.

        The drop is persisted to ``gb_status``: giving up is permanent, and since
        the checkpoint never advances past an unrecorded target, a drop that was
        forgotten on restart would block the watermark forever.

        The attempt *counts* stay in memory on purpose — they are a
        within-process backoff, and a restart legitimately re-tries a target from
        zero (the failure may well have been the crash itself). Only the terminal
        decision is durable.
        """
        attempts = self._failed_attempts.get(target_id, 0) + 1
        if attempts >= self._MAX_RECORD_ATTEMPTS:
            self._failed_attempts.pop(target_id, None)
            # Mark dropped so a persistent failure does not wedge every scan,
            # and persist it so a restart does not resurrect the target.
            self._dropped.add(target_id)
            self._persist_dropped(storage)
            logger.exception(
                "Dropping lineage for target %s in build %s after %d attempts: %s",
                target_id,
                build_id,
                attempts,
                exc,
            )
        else:
            self._failed_attempts[target_id] = attempts
            logger.warning(
                "Failed to record lineage for target %s in build %s "
                "(attempt %d/%d); will retry on next scan: %s",
                target_id,
                build_id,
                attempts,
                self._MAX_RECORD_ATTEMPTS,
                exc,
            )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the watcher thread to stop and wait for it to exit.

        Joins the worker thread (bounded by ``timeout``) so shutdown does not
        race an in-flight scan, and resets state so the watcher can be started
        again.

        Args:
            timeout: Maximum seconds to wait for the worker thread to exit.
        """
        logger.info("Stopping LineageWatcher")
        self.stop_event.set()
        thread = self.worker_thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(
                    "LineageWatcher thread did not stop within %.1fs", timeout
                )
        self.worker_thread = None
        self.stop_event.clear()
        # Nothing watermark-related to reset: it lives only in the gb_status
        # checkpoint, which the next start() re-verifies and every scan re-reads.
