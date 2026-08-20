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
from datetime import datetime, timezone
from typing import Optional

from gbserver.lineage.jobstats import ILineageStore, get_lineage_store
from gbserver.lineage.lineage_reconciler import (
    LINEAGE_WATCHER_CHECKPOINT_KEY,
    LINEAGE_WATCHER_DROPPED_KEY,
    UTC_MIN,
    as_aware,
    expected_run_count,
    get_oldest_successful_target,
    reconcile_once,
    select_recordable_targets,
)
from gbserver.lineage.lineage_seeding import BACKFILL_BUILD_ID
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.storage.stored_build import StoredBuild
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

    Steady state keeps per-scan work bounded with a ``finished_at`` checkpoint,
    but does not query straight from it: each scan bounds its walk at the
    *anchor* row — the checkpoint build's oldest successful target — so a target
    whose row surfaced behind the checkpoint is still reached (see
    ``_reconcile``). The checkpoint lives solely in the ``gb_kv_pairs`` key (see
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
    build, though, so the general safety net is the scan's *anchor*: each scan
    bounds its walk by the checkpoint build's oldest successful target rather than
    by a timestamp, and because the query filters on status alone the sweep back to
    that row re-surfaces the straggling targets of other builds, not just the
    checkpoint's. Idempotent recording makes the re-reads harmless. See
    ``_reconcile`` for why a row anchor and not a time window, and for the residual
    gap it does not close.

    That sweep is bounded by *which* builds it may record for, not only by how far
    back it reaches: the watcher tracks an in-memory allowlist of builds — the
    checkpoint build as a permanent floor, plus every build seen since or found
    inside the anchored range — and the scan records only for those. Without it an
    anchor at a recent build still selects every older build's targets behind it,
    which on a deployment with history means hundreds of unrelated candidates per
    scan, and a re-record storm of exactly that size the first time the sink fails
    to answer. Builds older than the floor are ignored for the life of the
    process. See ``_refresh_allowed_builds``,
    ``_admit_builds_in_anchored_range`` and ``_retire_finished_builds``.

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
    to ``gb_kv_pairs`` (``LINEAGE_WATCHER_DROPPED_KEY``) and reloaded by
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
        # (and persisted to) gb_kv_pairs, because the checkpoint refuses to advance
        # past an unrecorded target: an in-memory-only drop set would let a
        # dropped target return after a restart and block the watermark forever.
        self._dropped: set[str] = set()
        # target_uuid -> attempts so far, for targets whose recording failed and
        # should be retried on a subsequent scan.
        self._failed_attempts: dict[str, int] = {}
        # Builds whose targets this watcher may record for: the checkpoint build
        # (the floor) plus every build seen since, minus those retired by
        # _retire_finished_builds. Passed to reconcile_once as
        # `allowed_build_ids`, and the reason an anchored retreat cannot wander
        # into unrelated history (see select_recordable_targets).
        #
        # Deliberately in-memory and NOT persisted: it is a working set, and the
        # floor makes it reconstructible — a restart reseeds from the durable
        # checkpoint and re-adds whatever is still live. Persisting it would add
        # an unbounded value to gb_kv_pairs for no recoverable state.
        #
        # Safe across restarts only because the checkpoint never advances past an
        # unrecorded target (see _on_checkpoint_advance and reconcile_once's
        # contiguity rule). A mark that outran pending work would leave the
        # regenerated set unable to reach it, since the walk starts at the floor.
        self._allowed_build_ids: set[str] = set()
        # Whether the floor has been seeded into _allowed_build_ids. Seeding
        # needs a checkpoint, which may not exist yet at start(), so like
        # _checkpoint_verified this is retried at the top of each scan.
        self._allowlist_seeded = False
        # Whether the start-up verification sweep has run against a checkpoint.
        # It cannot run while the key is absent, so it stays pending and is
        # retried at the top of each scan until a checkpoint is seeded — a
        # watcher started before seeding must still get the sweep, otherwise the
        # unrecorded targets behind the seeded watermark would have no path back
        # until the next restart. Latches on the first True and is never cleared:
        # the sweep repairs a crash gap in the checkpoint this process started
        # from, which a later checkpoint does not reintroduce (see
        # _verify_checkpoint).
        self._checkpoint_verified = False
        # Whether the "no checkpoint yet" notice has been logged. The check runs
        # every scan, so without this the message would repeat forever at the
        # monitoring interval; it is a one-time operator notice, not a per-scan
        # event. Reset once a checkpoint is seen — defensive only: nothing in the
        # codebase deletes the key (seeding either keeps or overwrites it, see
        # seed_if_absent), so the absent -> present -> absent cycle this reset
        # covers is not a path any caller can currently drive.
        self._missing_checkpoint_logged = False

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
        self._checkpoint_verified = self._verify_checkpoint(storage)
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
        value = storage.kv_pair_storage.get_value(LINEAGE_WATCHER_DROPPED_KEY)
        if value:
            self._dropped = set(value.get("target_ids", []))

    def _persist_dropped(self, storage: SingletonAdminStorage) -> None:
        """Persist the drop set so the decision survives a restart."""
        storage.kv_pair_storage.set_value(
            LINEAGE_WATCHER_DROPPED_KEY, {"target_ids": sorted(self._dropped)}
        )

    def _verify_checkpoint(self, storage: SingletonAdminStorage) -> bool:
        """Re-record any unrecorded target in the checkpoint's own build.

        Runs once against the checkpoint: at ``start()`` when the key is already
        seeded, otherwise at the first scan that finds one. Returns ``True`` when
        the sweep is done with (including the malformed-checkpoint case, which no
        amount of retrying fixes) and ``False`` while it is still pending because
        no checkpoint exists yet.

        Once it returns ``True`` the caller latches ``_checkpoint_verified`` and
        this never runs again for the process's lifetime — deliberately, and it is
        not a recovery gap. The sweep exists to close a *crash* gap (below), which
        is a property of the checkpoint the process started from, not something a
        later checkpoint reintroduces: a ``--force-build-id`` overwrite resolves a
        fresh anchor from a chosen build, so there is no interrupted
        record-then-persist to repair, and re-sweeping would re-drive that build
        for nothing. There is likewise no delete-then-reseed cycle to handle,
        since no code path removes the key.

        This closes the gap left by a crash between
        recording a target and persisting its checkpoint (or vice versa): the
        checkpoint may name a target whose lineage never reached the sink, and
        the steady-state scan — which starts *at* that watermark — would not
        re-surface everything in its build.

        The checkpoint is never created here, or anywhere else implicitly. When
        ``LINEAGE_WATCHER_CHECKPOINT_KEY`` is absent this defers (and so does
        every scan) until the key is seeded explicitly (see
        ``gbserver lineage-watch --base-build-id``). Auto-seeding it from the
        newest successful target would silently pick a starting point for the
        operator; deciding where centralized recording begins — "from now", from
        a chosen build, or the full history — belongs to whoever seeds it, not to
        whichever process happens to start first.
        """
        checkpoint = storage.kv_pair_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
        if checkpoint is None:
            if self._missing_checkpoint_logged:
                return False
            self._missing_checkpoint_logged = True
            logger.info(
                "No lineage checkpoint (%s) found; recording nothing for now. "
                "Seed it to choose where centralized lineage recording starts "
                "\u2014 the watcher keeps checking and picks it up on the next "
                "scan, without a restart.",
                LINEAGE_WATCHER_CHECKPOINT_KEY,
            )
            return False
        self._missing_checkpoint_logged = False
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
            return True

        # Verify/re-record via reconcile_once — the same central selector the
        # steady-state scan uses — scoped to the checkpoint's own build, so the
        # per-sink filter_unrecorded check, expected-run-count derivation and
        # prerun-skip handling are shared rather than reimplemented here.
        #
        # Failures are recorded and swallowed rather than aborting start(): the
        # checkpoint is already durable, so nothing is lost by continuing, and a
        # watcher that refused to start would record nothing at all. Note what
        # each kind of failure costs, though. A target at or after the scan's
        # anchor is re-surfaced by the very next scan. A target further back than
        # the anchor is NOT — the steady-state scan stops at that row and never
        # looks behind it, so this start()-time
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
            # UTC_MIN: every target of this build is in scope, however long
            # ago it finished. build_id already bounds the scan to one build, so
            # there is nothing for a watermark to bound here — and a build whose
            # targets finished before the checkpoint's own timestamp must still
            # be verified.
            reconcile_once(
                self._store,
                storage,
                finished_after=UTC_MIN,
                build_id=build_id,
                # Same durable drop set the steady-state scan honours: a target
                # already dropped for exceeding _MAX_RECORD_ATTEMPTS must not be
                # re-attempted here, or every restart would burn the attempt
                # budget again and block checkpoint advancement for the sweep.
                skip=self._dropped,
                on_error=_on_error,
                on_success=self._on_record_success,
            )

        # The sweep is a once-per-checkpoint pass, so report it as done. Falling
        # off the end here would return None, leaving `_checkpoint_verified`
        # falsy and re-running the whole build-scoped scan on every single
        # iteration of the monitoring loop — one wasted sink round-trip per
        # scan, forever, plus a log line that reads like a wedged watcher
        # stuck on the same target.
        return True

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
        central mechanism), bounding the walk at the anchor row derived from the
        checkpoint so a scan reads only the targets above it.

        The checkpoint is read from ``gb_kv_pairs`` on every scan
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
        if not self._checkpoint_verified:
            # A checkpoint seeded after start() still needs the verification
            # sweep, and only this retry can give it one: the steady-state scan
            # below stops at the anchor row and never looks behind it, so
            # unrecorded targets that finished before the checkpoint build's
            # oldest one would otherwise wait for a restart.
            self._checkpoint_verified = self._verify_checkpoint(storage)
        watermark = self._checkpoint_watermark(storage)
        if watermark is None:
            # No checkpoint: recording is deliberately off until the key is
            # seeded. Return before querying targets or touching the sink.
            return
        # Bound the walk by a *row*, not a timestamp. ``finished_at`` is stamped
        # when the build event is created, not when its row is committed, so a row
        # can become visible already carrying a timestamp behind an advanced
        # watermark; a time cutoff excludes it from the query outright and the
        # monotonic watermark never re-surfaces it. Anchoring on the checkpoint
        # build's *oldest* successful target instead lets the walk retreat to a
        # known row, and since the query never filters on build_id, that sweep
        # also picks up the straggling targets of every concurrent build.
        #
        # The oldest rather than the checkpoint's own finished_at: the checkpoint
        # names whichever target recorded last, so a build whose targets finished
        # at staggered times would leave its earlier ones behind that row.
        #
        # The anchor can fail to resolve: a malformed checkpoint (no build_id), or
        # a build whose successful targets are all gone or carry no finish time.
        # There is then no row to aim at, so the scan falls back to the plain
        # watermark cutoff — see below for what that costs.
        checkpoint_build_id = self._checkpoint_build_id(storage)
        # Seed the allowlist floor from the checkpoint build before selecting, so
        # the very first scan after a (re)start is already bounded. Seeding needs
        # the checkpoint, so it is retried here rather than done once in start().
        if not self._allowlist_seeded and checkpoint_build_id is not None:
            self._allowed_build_ids.add(checkpoint_build_id)
            self._allowlist_seeded = True
            logger.info(
                "Lineage allowlist seeded with checkpoint build %s as its floor; "
                "builds older than it are ignored for the life of this process.",
                checkpoint_build_id,
            )
        stop_at: Optional[tuple[str, datetime]] = None
        if checkpoint_build_id is not None:
            oldest = get_oldest_successful_target(storage, build_id=checkpoint_build_id)
            if oldest is not None and oldest.finished_at is not None:
                stop_at = (checkpoint_build_id, as_aware(oldest.finished_at))
        # Order matters here. Admission needs the resolved anchor to know what
        # "within range" means, and both must run before the filtered scan below:
        # a build admitted after the scan would have its targets dropped on this
        # pass and only picked up on the next one.
        self._admit_builds_in_anchored_range(storage, stop_at)
        self._refresh_allowed_builds(storage)
        # The anchor is the real bound, so the timestamp cutoff opens all the way
        # up; select_recordable_targets stops at the anchor row instead.
        #
        # With no anchor the walk falls back to the watermark cutoff, which is
        # inclusive of the boundary row (the selector compares `<`) but reaches no
        # further back. A target that finished strictly behind the watermark is
        # therefore out of reach on this scan, and stays so while the anchor cannot
        # be resolved — the one case where this design is no better than the fixed
        # window it replaced. Nothing is subtracted to soften it: without a row to
        # aim at there is no principled width to pick, and inventing one is exactly
        # what the old overlap did. Logged as a warning so it is visible instead of
        # silent.
        if stop_at is not None:
            finished_after = UTC_MIN
        else:
            finished_after = watermark
            logger.warning(
                "Lineage scan could not resolve an anchor row for checkpoint "
                "build %s (no successful target with a finish time); falling back "
                "to the bare %s watermark for this scan. A target that finished "
                "behind it will not be re-surfaced.",
                checkpoint_build_id,
                watermark,
            )
        # Stamped before the scan, not after: a target that finishes while the
        # scan is running must stay in scope for the next one. Retiring the
        # anchor to an after-the-fact timestamp would step over it.
        # Aware, so it compares against the targets' own aware timestamps (see
        # as_aware). UTC here because this is a sentinel stamped from the clock
        # rather than a value read off a gb_targets row; utcnow() is deprecated.
        scan_started = datetime.now(timezone.utc)
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
            # Log-only: names the build whose target set the watermark, so a
            # steady-state line reads as "starting here because of that row"
            # rather than as a bare epoch. This names the build for the log; the
            # build's role in *selection* is carried by ``stop_at`` below, which
            # bounds the walk at that build's oldest target.
            watermark_build_id=checkpoint_build_id,
            stop_at=stop_at,
            # Bounds *which* builds the anchored retreat may record for. Only
            # once seeded: an unseeded set is empty, and an empty membership
            # filter selects nothing, which would silently stall recording
            # rather than merely widening it. Unseeded means no checkpoint build
            # resolved, in which case the scan is already on its fallback path.
            allowed_build_ids=(
                self._allowed_build_ids if self._allowlist_seeded else None
            ),
        )
        # After the scan, never before: retirement asks the sink whether a build's
        # lineage is recorded, and running it first would ask that question about
        # targets this pass was about to record — retiring a build a moment before
        # its own targets were written, and (because a retired build is not
        # re-admitted from build state) dropping them.
        self._retire_finished_builds(storage)

    def _refresh_allowed_builds(self, storage: SingletonAdminStorage) -> None:
        """Add builds that have appeared since the last refresh.

        Purely additive: a build enters the set when first seen and leaves only
        through ``_retire_finished_builds``, which runs *after* the scan rather
        than from here — retiring before recording would ask the sink about
        targets the pass had not written yet. Nothing is ever dropped for
        being *old*, because "old" would have to be judged on ``finished_at``,
        which can place a late-committing row behind the watermark — the very case
        the anchored walk exists to recover (see ``select_recordable_targets``).

        Only builds that are not yet finished are added *here*. A build that was
        already finished before this watcher ever saw it is either behind the
        floor (deliberately out of scope) or already recorded, so adding it would
        re-open the history the floor exists to exclude.

        That rule alone is too narrow, though, and the gap is the one the anchored
        walk exists for: a build concurrent with the floor can finish *inside* the
        anchored range and be finished by the time any refresh runs — it is
        neither the checkpoint build nor unfinished, so it would never be added,
        and filtering would drop the very target the retreat reached back for. So
        membership is also granted from below, by
        ``_admit_builds_in_anchored_range``, which admits any build whose target
        the walk actually surfaced at or after the anchor. This method covers
        builds that are still live; that one covers builds that finished inside
        the window.
        """
        if not self._allowlist_seeded:
            # Without a floor there is nothing to add *to*: an unseeded set is
            # empty and is not used as a filter, so populating it here would
            # start bounding the scan by a set that has no floor behind it.
            return
        try:
            live = storage.build_storage.get_by_where({})
        except (
            Exception
        ) as exc:  # noqa: BLE001 — a refresh failure must not stop the scan
            # Fail *open* on the set: keep the builds already tracked and scan
            # with them. Losing an addition delays a new build's lineage by a
            # scan; raising here would stop recording entirely.
            logger.warning(
                "Could not refresh the lineage build allowlist (%s); scanning "
                "with the %d build(s) already tracked.",
                exc,
                len(self._allowed_build_ids),
            )
            return
        added = 0
        for build in live:
            if build.uuid in self._allowed_build_ids:
                continue
            if build.status.is_finished():
                continue
            self._allowed_build_ids.add(build.uuid)
            added += 1
        if added:
            logger.info(
                "Lineage allowlist tracking %d build(s) (%d newly added).",
                len(self._allowed_build_ids),
                added,
            )

    def _admit_builds_in_anchored_range(
        self,
        storage: SingletonAdminStorage,
        stop_at: Optional[tuple[str, datetime]],
    ) -> None:
        """Admit every build whose targets lie within the anchored range.

        This is what keeps the allowlist from turning the anchored retreat into a
        lineage-loss bug. The retreat exists because a target's row can become
        visible carrying a ``finished_at`` behind the watermark, and because a
        concurrent build can finish between two targets of the checkpoint's build.
        Such a build is already finished by the time a refresh looks at build rows,
        so ``_refresh_allowed_builds`` will not add it — and a membership filter
        would then discard the target the walk retreated to collect.

        Admitting from the *swept range* rather than from build state closes that:
        anything the unfiltered walk can still reach is in scope by definition, so
        the filter only ever removes what lies outside the anchor. The range is
        bounded by the same anchor the scan uses, so this is not a full-history
        read — and it is exactly the range the scan was going to walk anyway.

        Without an anchor there is no bounded range to admit from, so this is a
        no-op and the scan runs on its fallback path.
        """
        if stop_at is None or not self._allowlist_seeded:
            return
        try:
            in_range = select_recordable_targets(
                storage, finished_after=UTC_MIN, stop_at=stop_at
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 — admission failure must not stop the scan
            logger.warning(
                "Could not admit builds from the anchored range (%s); scanning "
                "with the %d build(s) already tracked.",
                exc,
                len(self._allowed_build_ids),
            )
            return
        admitted = {
            t.build_id
            for t in in_range
            if t.build_id not in self._allowed_build_ids
            # A build retired earlier in this process is deliberately done: its
            # lineage was confirmed in the sink, so re-admitting it here would
            # undo retirement on every scan and pin it forever.
            and t.uuid not in self._dropped
        }
        if admitted:
            self._allowed_build_ids |= admitted
            logger.info(
                "Admitted %d build(s) whose targets fall within the anchored "
                "range: %s. Allowlist now tracks %d build(s).",
                len(admitted),
                ", ".join(sorted(admitted)),
                len(self._allowed_build_ids),
            )

    def _retire_finished_builds(self, storage: SingletonAdminStorage) -> None:
        """Drop builds that are finished, have nothing pending, and are recorded.

        Keeps the in-memory set proportional to real concurrency rather than to
        the process's uptime. All three conditions are required, and each guards a
        distinct way a build can still owe work:

        - ``status.is_finished()``: excludes ``RETRY_PENDING`` (a build queued for
          a build-level retry is still going to produce targets) as well as
          ``RUNNING``/``PENDING``.
        - Nothing pending locally: no target of this build is mid-retry in
          ``_failed_attempts``. Targets in ``_dropped`` do NOT count as pending —
          they were given up on deliberately, and treating them as pending would
          pin their build in the set forever.
        - Recorded in the sink: local state cannot answer this. Only the sink
          knows whether the runs actually landed, so this asks it.

        Retirement is irreversible for this process: a retired build is not
        re-added (``_refresh_allowed_builds`` skips finished builds), and the walk
        will no longer select its targets. So every uncertain case must retain,
        never retire — including a sink that fails to answer.
        """
        if self._store is None or not self._allowed_build_ids:
            return
        candidates: list[str] = []
        for build_id in self._allowed_build_ids:
            try:
                build = storage.build_storage.get_by_uuid(build_id)
            except (
                Exception
            ) as exc:  # noqa: BLE001 — unreadable build: retain, retry next scan
                logger.debug(
                    "Could not read build %s while considering it for lineage "
                    "allowlist retirement (%s); retaining it.",
                    build_id,
                    exc,
                )
                continue
            # get_by_uuid is typed as scalar-or-list (it returns a list when given
            # one), so narrow explicitly rather than trusting the scalar shape.
            if not isinstance(build, StoredBuild):
                continue
            if not build.status.is_finished():
                continue
            candidates.append(build_id)
        for build_id in candidates:
            try:
                confirmed = self._build_lineage_is_confirmed(storage, build_id)
            except (
                Exception
            ) as exc:  # noqa: BLE001 — retirement must never break the scan
                # Retirement is a memory optimization running *after* the pass
                # already recorded. Letting a storage error escape here would
                # abort a scan whose real work is done, so a failure only means
                # "not confirmed": retain the build and re-check next scan.
                logger.warning(
                    "Could not confirm whether build %s can leave the lineage "
                    "allowlist (%s); retaining it.",
                    build_id,
                    exc,
                )
                continue
            if not confirmed:
                continue
            self._allowed_build_ids.discard(build_id)
            logger.info(
                "Retired build %s from the lineage allowlist: finished, nothing "
                "pending, and its lineage is recorded in the sink. Allowlist now "
                "tracks %d build(s).",
                build_id,
                len(self._allowed_build_ids),
            )

    def _build_lineage_is_confirmed(
        self, storage: SingletonAdminStorage, build_id: str
    ) -> bool:
        """Whether ``build_id`` has nothing pending and its lineage is in the sink.

        The last two of the three retirement conditions (the caller has already
        checked that the build is finished). Split out so the caller can treat any
        failure as "not confirmed" in one place: every path here either answers
        definitively or raises, and a raise must never mean "retire".
        """
        if self._store is None:
            return False
        # Re-select this build's successful targets to ask the sink about them.
        # Scoped to the one build and only reached for finished builds, so this is
        # not the unbounded walk `allowed_build_ids` exists to prevent.
        targets = select_recordable_targets(
            storage, finished_after=UTC_MIN, build_id=build_id
        )
        if any(t.uuid in self._failed_attempts for t in targets):
            # Mid-retry: still owes work locally.
            return False
        pending = {t.uuid for t in targets if t.uuid not in self._dropped}
        if not pending:
            # Nothing left to confirm: every target either recorded or was
            # deliberately dropped.
            return True
        # A sink that fails to answer returns the full candidate set (fail-open),
        # which is indistinguishable from a real "none recorded" verdict.
        # Retirement is irreversible, so it must act only on a real answer: this
        # flag turns an unanswered query into "not confirmed", independently of
        # what the returned set looks like.
        sink_failed = False

        def _note_failure(_exc: Exception) -> None:
            nonlocal sink_failed
            sink_failed = True

        expected = {t.uuid: expected_run_count(t) for t in targets}
        unrecorded = self._store.filter_unrecorded(
            pending, expected, on_query_error=_note_failure
        )
        return not sink_failed and not unrecorded

    def _checkpoint_build_id(self, storage: SingletonAdminStorage) -> Optional[str]:
        """Return the checkpoint's ``build_id``, or ``None`` if unset/malformed.

        Read separately from ``_checkpoint_watermark`` rather than returned
        alongside it: this value is for the log only, so a missing or malformed
        one must never affect whether a scan runs. Every failure mode collapses to
        ``None``, which simply omits the build from the log line.
        """
        checkpoint = storage.kv_pair_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
        if checkpoint is None:
            return None
        build_id = checkpoint.get("build_id")
        return build_id if isinstance(build_id, str) else None

    def _checkpoint_watermark(
        self, storage: SingletonAdminStorage
    ) -> Optional[datetime]:
        """Read the watermark from the durable checkpoint, or ``None`` if unset.

        ``None`` means the key has not been seeded, which the caller treats as
        "record nothing" rather than as "no lower bound". A checkpoint that is
        present but malformed (missing/unparseable ``finished_at``) is treated
        the same way: recording stays off until it is corrected, which is the
        safe direction — the alternative is raising out of every scan.

        The parsed value is made aware, keeping whatever offset it carries. The
        checkpoint holds ``finished_at`` as a plain string inside a JSON dict, so
        it is backend-opaque and round-trips its offset on both SQLite and
        Postgres; every writer persists it aware-ISO, which is why the naive case
        is defensive rather than expected. Filling in the local offset for a naive
        value is what lets the caller compare it against the targets' own
        timestamps without a ``TypeError`` from mixing awareness. Note the offset
        that gets filled in is the *local* one, so a caller testing for the
        ``datetime.min`` backfill anchor must compare within the watermark's own
        offset rather than against ``UTC_MIN`` (see ``_retire_backfill_anchor``).
        The offset is not rewritten to UTC — aware values already compare as
        instants, and rewriting made the key disagree textually with the
        ``gb_targets`` row behind it (see ``_on_checkpoint_advance``).

        ``OverflowError`` is still caught alongside the parse errors: it is
        reachable from arithmetic on an extreme parsed value (the
        ``--base-build-id all`` anchor is ``datetime.min``). Uncaught it would
        escape to ``_run``'s blanket handler and fail every scan forever, the same
        wedge this guard exists to prevent for unparseable values.
        """
        checkpoint = storage.kv_pair_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
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
            return as_aware(datetime.fromisoformat(raw))
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
          ``datetime.min`` and nothing else. Compared against ``datetime.min`` in
          the watermark's *own* offset, not against ``UTC_MIN``: the anchor is
          written as ``UTC_MIN.isoformat()``, but ``as_aware`` preserves whatever
          offset the backend returned, and a backend that drops the offset
          (SQLite does; Postgres ``timestamptz`` does not) hands back a naive
          ``datetime.min`` that is then re-tagged *local*. That is the same
          instant only when the reader sits at UTC, so a bare ``!= UTC_MIN`` would
          leave the anchor in place on every non-UTC deployment and re-walk the
          whole target table on every scan, forever.
        - ``watermark_untouched``: the pass recorded nothing, failed nothing and
          dropped nothing (see ``reconcile_once``). Any of those would mean
          unrecorded lineage sits *behind* the new anchor, and the steady-state
          scan never looks behind its watermark, so moving it would strand that
          lineage permanently. A record in the same pass has already advanced the
          checkpoint to the right place via ``_on_checkpoint_advance``; moving it
          again to the scan time would step over every target after it.
        """
        anchor = datetime.min.replace(tzinfo=watermark.tzinfo)
        if not watermark_untouched or watermark != anchor:
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

        The watermark is monotonic: a target swept in behind the current
        watermark legitimately finished there (its row became visible late, or
        builds interleaved — see the anchor rationale in ``_reconcile``), and
        recording it must not drag the durable watermark back down with it.
        Without this guard such a target lowers the watermark, and if no newer
        target in the same pass pushes it back up it stays lowered, re-reading
        that range on every later scan. Recording still happens either way; only
        the watermark write is suppressed.

        The comparison is against the *durable* value rather than an in-process
        high-water mark so a restart cannot forget it and re-open the same
        regression. The ``datetime.min`` backfill anchor compares below every
        real timestamp, so ``_retire_backfill_anchor`` still advances off it.

        ``finished_at`` arrives straight off a ``StoredTargetRun`` (see
        ``reconcile_once``), so it is timezone-*aware* in production: it is
        stamped from ``BuildEvent.timestamp``, which defaults to
        ``get_time()`` (``datetime.now().astimezone()``) — aware *local*, not
        UTC. ``as_aware`` only fills in an offset when one is missing, which the
        JSON-column read path does not produce, so the value persisted here is the
        ``gb_targets`` row's own timestamp verbatim, in the same form that table
        holds it. ``gb_targets`` is untouched by this; the checkpoint is what
        adopts its form.

        Rewriting the offset to UTC instead would still compare correctly — aware
        datetimes compare as instants — but it made the checkpoint disagree
        *textually* with the row it came from, so the same target read three hours
        apart depending on whether it was loaded from ``gb_targets`` or
        ``gb_kv_pairs``, which is unreadable in a log and in the key itself.

        Ensuring awareness is still required: the watermark read back by
        ``_checkpoint_watermark`` was once naive, and comparing the two raised
        ``TypeError: can't compare offset-naive and offset-aware datetimes``.
        That raise happens inside ``reconcile_once``'s per-target ``try``, so it
        is misattributed to recording (which already succeeded), blocks the
        checkpoint, and after ``_MAX_RECORD_ATTEMPTS`` scans lands the target in
        the durable dropped set.
        """
        finished_at = as_aware(finished_at)
        current = self._checkpoint_watermark(storage)
        if current is not None and finished_at <= current:
            logger.debug(
                "Not moving the lineage checkpoint from %s back to %s for build "
                "%s; the target finished behind the watermark and the watermark "
                "is monotonic.",
                current,
                finished_at,
                build_id,
            )
            return
        storage.kv_pair_storage.set_value(
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
        it still falls within the scan's anchored range each scan, so the skip set is
        what keeps it from wedging the scan.

        The drop is persisted to ``gb_kv_pairs``: giving up is permanent, and since
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
        # Nothing watermark-related to reset: it lives only in the gb_kv_pairs
        # checkpoint, which the next start() re-verifies and every scan re-reads.
