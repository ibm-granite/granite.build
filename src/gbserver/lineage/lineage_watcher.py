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
from datetime import datetime, timedelta, timezone
from typing import Optional

from gbserver.lineage.jobstats import ILineageStore, get_lineage_store
from gbserver.lineage.lineage_reconciler import (
    LINEAGE_WATCHER_CHECKPOINT_KEY,
    LINEAGE_WATCHER_DROPPED_KEY,
    as_utc_naive,
    reconcile_once,
)
from gbserver.lineage.lineage_seeding import BACKFILL_BUILD_ID
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
    explicitly (see ``gbserver lineage-watch --base-build-id``). Choosing where
    centralized recording begins — "from now", from a given build, or the full
    history — is the operator's call, so a fresh deployment stays silent rather
    than picking a starting point on its own. When the checkpoint does exist,
    ``start()`` verifies its build against the store's own recorded-state
    (``filter_unrecorded``, via a build-scoped ``reconcile_once``) and re-records
    what is missing, closing any gap left by a crash between recording
    and persisting the checkpoint (or vice versa). That sweep is scoped to one
    build, though, so the general safety net is ``_WATERMARK_OVERLAP``:
    subtracted when querying, it re-surfaces any target that landed behind an
    already-advanced watermark — whether from clock skew or from builds
    interleaving in the global finished_at queue — for every build rather than
    just the checkpoint's. Idempotent recording makes the re-reads harmless.

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
    # The case only this can cover is a target landing *behind* an already-advanced
    # watermark, where it sits on the wrong side of the cutoff and no amount of
    # tie-breaking would find it. Two things put it there:
    #
    #  - Clock skew: finished_at is stamped from the writer's own clock
    #    (``datetime.now()`` in the buildrunner), not the DB's, so a target can be
    #    written with a timestamp slightly older than the current watermark.
    #  - Concurrent builds: the scan is a single global queue ordered by
    #    finished_at, not a per-build walk, so targets of different builds
    #    interleave. A target that commits its finished_at a moment after a
    #    faster build's target already pushed the watermark past that instant is
    #    behind the cutoff by the time it is visible to a scan.
    #
    # Equal timestamps are re-surfaced here too, and those have a second safety
    # net — but only a partial one, which is why this window carries the real
    # weight. ``_verify_checkpoint`` re-scans at start(), yet it is scoped to the
    # *checkpoint's own build*, and which build that is depends on whichever
    # target recorded last (see ``_on_checkpoint_advance``). With builds running
    # concurrently that is effectively arbitrary among the active ones, so a gap
    # in any other build is not covered by it at all. This overlap is
    # build-agnostic and repairs all of them, without waiting for a restart.
    #
    # Sized in minutes rather than seconds for that reason: seconds cover only
    # clock skew, while interleaved commits across builds can land a target
    # further back. Re-reads are harmless — per-sink ``filter_unrecorded`` drops
    # already-recorded targets before anything is written, so the cost of a wider
    # window is extra rows in the paged query and one sink query over them, not
    # extra recording work.
    _WATERMARK_OVERLAP = timedelta(minutes=1)

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
        ``gbserver lineage-watch --base-build-id``). Auto-seeding it from the
        newest successful target would silently pick a starting point for the
        operator; deciding where centralized recording begins — "from now", from
        a chosen build, or the full history — belongs to whoever seeds it, not to
        whichever process happens to start first.
        """
        checkpoint = storage.status_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
        if checkpoint is None:
            logger.info(
                "No lineage checkpoint (%s) found; recording nothing. Seed it "
                "to choose where centralized lineage recording starts.",
                LINEAGE_WATCHER_CHECKPOINT_KEY,
            )
            return
        build_id = checkpoint.get("build_id")
        if build_id is None:
            # Malformed checkpoint. start() guards only per-target recording
            # errors via on_error, so raising KeyError here would abort start()
            # entirely and leave the watcher not running at all. Skipping the
            # verification sweep instead is strictly better: the steady-state
            # scan still works off the watermark, and it re-runs on next start().
            logger.error(
                "lineage checkpoint %s has no build_id (%r); skipping the "
                "start-up verification sweep for its build",
                LINEAGE_WATCHER_CHECKPOINT_KEY,
                checkpoint,
            )
            return

        # Verify/re-record via reconcile_once — the same central selector the
        # steady-state scan uses — scoped to the checkpoint's own build, so the
        # per-sink filter_unrecorded check, expected-run-count derivation and
        # prerun-skip handling are shared rather than reimplemented here.
        #
        # Failures are recorded and swallowed rather than aborting start(): the
        # checkpoint is already durable, so nothing is lost by continuing, and a
        # watcher that refused to start would record nothing at all. Note what
        # each kind of failure costs, though. A target sitting at the watermark is
        # re-surfaced by the very next scan (it falls inside the overlap window).
        # A target further back in this build is NOT — the steady-state scan
        # starts at the watermark and never looks behind it, so this start()-time
        # sweep is its only path back, and a failure here defers it to the next
        # start(). That is the gap this call exists to close, so its failures are
        # worth reading in the log even though they are non-fatal.
        #
        # Failures route through the same ``_on_record_error`` bookkeeping the
        # steady-state scan uses, not a log-only callback. A target that fails
        # this sweep is one the sweep is the *only* path back for, so it must be
        # able to exhaust ``_MAX_RECORD_ATTEMPTS`` and be dropped durably;
        # otherwise a permanently-unrecordable target behind the watermark blocks
        # checkpoint advancement for this build on every restart, forever.
        def _on_error(build_id: str, target_id: str, exc: Exception) -> None:
            self._on_record_error(storage, build_id, target_id, exc)

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
                build_id=build_id,
                # Same durable drop set the steady-state scan honours: a target
                # already dropped for exceeding _MAX_RECORD_ATTEMPTS must not be
                # re-attempted here, or every restart would burn the attempt
                # budget again and block checkpoint advancement for the sweep.
                skip=self._dropped,
                on_error=_on_error,
                on_success=self._on_record_success,
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
        # Query behind the watermark so a target that landed on the older side of
        # it — clock skew, or a concurrent build committing out of order — is
        # re-surfaced instead of skipped; idempotent recording makes the overlap
        # re-reads harmless. See ``_WATERMARK_OVERLAP``.
        #
        # Clamped at datetime.min: the --all backfill anchor *is* datetime.min, and
        # subtracting from it raises OverflowError — which, being raised before any
        # recording, would fail every scan forever and record nothing at all.
        # Nothing can have finished before datetime.min anyway, so there is no
        # boundary case for the overlap to protect there.
        if watermark - datetime.min < self._WATERMARK_OVERLAP:
            finished_after = datetime.min
        else:
            finished_after = watermark - self._WATERMARK_OVERLAP
        # Stamped before the scan, not after: a target that finishes while the
        # scan is running must stay in scope for the next one. Retiring the
        # anchor to an after-the-fact timestamp would step over it.
        # Naive UTC, matching every other watermark in this module (see
        # as_utc_naive); utcnow() itself is deprecated.
        scan_started = datetime.now(timezone.utc).replace(tzinfo=None)
        reconcile_once(
            self._store,
            storage,
            finished_after=finished_after,
            on_error=lambda build_id, target_id, exc: (
                self._on_record_error(storage, build_id, target_id, exc)
            ),
            on_success=self._on_record_success,
            on_checkpoint_advance=lambda build_id, finished_at: (
                self._on_checkpoint_advance(storage, build_id, finished_at)
            ),
            on_scan_complete=lambda untouched: (
                self._retire_backfill_anchor(
                    storage, watermark, untouched, scan_started
                )
            ),
            skip=self._dropped,
        )

    def _checkpoint_watermark(
        self, storage: SingletonAdminStorage
    ) -> Optional[datetime]:
        """Read the watermark from the durable checkpoint, or ``None`` if unset.

        ``None`` means the key has not been seeded, which the caller treats as
        "record nothing" rather than as "no lower bound". A checkpoint that is
        present but malformed (missing/unparseable ``finished_at``) is treated
        the same way: recording stays off until it is corrected, which is the
        safe direction — the alternative is raising out of every scan.

        The parsed value is normalized to naive UTC. ``finished_at`` is written
        straight from a stored target, which a backend or DB driver may hand
        back timezone-aware; leaving it aware would make the caller's
        ``watermark - datetime.min`` arithmetic raise ``TypeError`` on every
        scan and record nothing at all.

        ``OverflowError`` is caught alongside the parse errors because the
        normalization itself can raise it: ``as_utc_naive`` shifts an aware value
        to UTC, which overflows for a timestamp within the UTC offset of
        ``datetime.min``/``datetime.max`` — e.g. the ``--base-build-id all`` anchor
        (``datetime.min``) read back aware with a positive offset. Uncaught it
        would escape to ``_run``'s blanket handler and fail every scan forever,
        the same wedge this guard exists to prevent for unparseable values.
        """
        checkpoint = storage.status_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
        if checkpoint is None:
            return None
        raw = checkpoint.get("finished_at")
        if raw is None:
            logger.error(
                "lineage checkpoint %s has no finished_at (%r); recording stays "
                "off until it is re-seeded",
                LINEAGE_WATCHER_CHECKPOINT_KEY,
                checkpoint,
            )
            return None
        try:
            return as_utc_naive(datetime.fromisoformat(raw))
        except (TypeError, ValueError, OverflowError):
            logger.error(
                "lineage checkpoint %s has an unparseable finished_at (%r); "
                "recording stays off until it is re-seeded",
                LINEAGE_WATCHER_CHECKPOINT_KEY,
                raw,
            )
            return None

    def _retire_backfill_anchor(
        self,
        storage: SingletonAdminStorage,
        watermark: datetime,
        watermark_untouched: bool,
        scan_started: datetime,
    ) -> None:
        """Move a spent ``datetime.min`` backfill anchor up to the scan time.

        ``--base-build-id all`` anchors the checkpoint at ``datetime.min`` so the first
        scan walks the entire history. Normally the first recorded target
        advances the checkpoint off that anchor via ``_on_checkpoint_advance``.
        When the backfill records nothing, though — an empty DB, or one where
        every candidate is in the drop set — nothing advances it, and *every*
        subsequent scan re-walks the whole table at ``finished_after=datetime.min``
        instead of reading only newly-finished rows.

        Retiring the anchor is safe only under both conditions checked here:

        - The watermark really is the backfill anchor. A normal watermark must
          never be moved by anything but a recorded target, so this is scoped to
          ``datetime.min`` and nothing else.
        - ``watermark_untouched``: the pass recorded nothing, failed nothing and
          dropped nothing (see ``reconcile_once``). Any of those would mean
          unrecorded lineage sits *behind* the new anchor, and the steady-state
          scan never looks behind its watermark, so moving it would strand that
          lineage permanently. A record in the same pass has already advanced the
          checkpoint to the right place via ``_on_checkpoint_advance``; moving it
          again to the scan time would step over every target after it.
        """
        if not watermark_untouched or watermark != datetime.min:
            return
        logger.info(
            "Full-history lineage backfill completed with nothing left to "
            "record; advancing the checkpoint off the datetime.min anchor to %s "
            "so later scans read only newly-finished targets.",
            scan_started,
        )
        self._on_checkpoint_advance(storage, BACKFILL_BUILD_ID, scan_started)

    def _on_checkpoint_advance(
        self, storage: SingletonAdminStorage, build_id: str, finished_at: datetime
    ) -> None:
        """Persist the advanced checkpoint.

        Called once per successfully-recorded target (oldest-first), so the
        durable checkpoint always reflects the last target actually recorded —
        never a target merely considered or one that failed to record. The next
        scan reads it back, so this write alone advances the watermark.

        The watermark is monotonic: a target inside the ``_WATERMARK_OVERLAP``
        window legitimately finished *behind* the current watermark (clock skew
        or interleaved builds — see that constant), and recording it must not
        drag the durable watermark back down with it. Without this guard such a
        target lowers the watermark, and if no newer target in the same pass
        pushes it back up it stays lowered, re-reading that window on every
        later scan. Recording still happens either way; only the watermark write
        is suppressed.

        The comparison is against the *durable* value rather than an in-process
        high-water mark so a restart cannot forget it and re-open the same
        regression. The ``datetime.min`` backfill anchor compares below every
        real timestamp, so ``_retire_backfill_anchor`` still advances off it.

        ``finished_at`` arrives straight off a ``StoredTargetRun`` (see
        ``reconcile_once``), so it is timezone-*aware* in production: it is
        stamped from ``BuildEvent.timestamp``, which defaults to
        ``get_time()`` (``datetime.now().astimezone()``). The watermark read back
        by ``_checkpoint_watermark`` is naive UTC, so comparing the two raises
        ``TypeError: can't compare offset-naive and offset-aware datetimes``.
        Normalizing here is what keeps the guard from turning every real
        deployment's checkpoint write into a spurious *recording* failure — the
        raise happens inside ``reconcile_once``'s per-target ``try``, so it is
        misattributed to recording (which already succeeded), blocks the
        checkpoint, and after ``_MAX_RECORD_ATTEMPTS`` scans lands the target in
        the durable dropped set. It also keeps the persisted ``isoformat()``
        consistently naive UTC, matching what the seeding path writes.
        """
        finished_at = as_utc_naive(finished_at)
        current = self._checkpoint_watermark(storage)
        if current is not None and finished_at <= current:
            logger.debug(
                "Not moving the lineage checkpoint from %s back to %s for build "
                "%s; the target finished within the overlap window and the "
                "watermark is monotonic.",
                current,
                finished_at,
                build_id,
            )
            return
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
