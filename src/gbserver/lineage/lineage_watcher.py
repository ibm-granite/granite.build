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
    LINEAGE_WATCHER_INCOMPLETE_KEY,
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

    A single daemon thread periodically calls ``reconcile_once`` (see
    ``lineage_reconciler``), which scans the admin DB for successful target runs and
    records their lineage into the configured store, off the build's hot path.

    Reconciliation — not the event stream — is authoritative: the admin DB persists
    the complete lineage graph, so full lineage is recoverable by re-reading it
    alone. Each scan re-derives the recordable set from the DB, so a target that
    succeeded while this process was down is picked up next scan; there is no restart
    blind spot. Recording is idempotent (deterministic runIds + resume="allow" +
    content-dedupe), making a re-record a harmless no-op.

    Single-writer: deployed as its own single-replica ``lineage-watch`` command/pod
    (see ``command_lineage_watch.py``, ``dep-lineage-watcher.yaml``), so exactly one
    process reconciles lineage; it must not be wired into another entrypoint. Even
    violated, idempotent recording means a duplicate wastes I/O without corrupting.

    How a scan is bounded, in two independent ways — each documented in full at
    its own site:

    - *How far back*: not the ``finished_at`` checkpoint directly but the *anchor* row
      it names (the checkpoint build's oldest successful target), so a row that
      surfaced behind the watermark is still reached. The checkpoint lives solely in
      ``LINEAGE_WATCHER_CHECKPOINT_KEY``, rewritten per recorded target and re-read
      each scan. Never created implicitly — an unseeded watcher records *nothing*,
      leaving the operator to choose where recording begins (``--base-build-id``).
      See ``_reconcile``, ``_verify_checkpoint``.
    - *Which builds*: an in-memory allowlist floored at the checkpoint build. Without
      it an anchor at a recent build still selects every older build's targets behind
      it — hundreds of unrelated candidates per scan on a deployment with history, and
      a re-record storm of that size the first time the sink fails to answer. See
      ``_refresh_allowed_builds``, ``_admit_builds_in_anchored_range``,
      ``_retire_finished_builds``.

    Which selected targets get recorded is decided per-sink by
    ``store.filter_unrecorded``: the watermark is sink-neutral and each sink owns its
    recorded-state, so one admin DB can feed W&B and other sinks independently.

    Failures retry on later scans and the checkpoint stops advancing at the failed
    target, so the durable watermark never moves past unrecorded lineage, restarts
    included. A target that keeps failing is dropped durably
    (``LINEAGE_WATCHER_DROPPED_KEY``); one whose run id the sink permanently rejected
    is dropped on first sighting, since deterministic ids leave no new id to try. See
    ``_on_record_error``, ``_is_permanent_sink_rejection``.
    """

    # A target whose lineage recording keeps failing is retried this many times
    # on subsequent scans before being dropped, so a transient failure (e.g. a
    # network blip) is recovered without a persistent failure wedging the scan.
    _MAX_RECORD_ATTEMPTS = 3

    # Substring identifying a sink rejection that no retry can clear: the run id
    # was created and then deleted in the sink, which remembers the id as deleted
    # and refuses it permanently ("... was previously created and deleted; try a
    # new run id"). Matched on the message rather than the exception type because
    # wandb raises the same CommError for ordinary transient network failures,
    # which MUST stay retryable — the type alone cannot separate them.
    #
    # The sink's advice ("try a new run id") is not actionable here: run ids are
    # derived deterministically from the target and output uuids
    # (WandBLineageStore._build_events_for_target), and that determinism is what
    # makes re-recording idempotent. Changing it to dodge a deleted id would
    # trade a bounded, visible gap for silent duplicate lineage everywhere else.
    _PERMANENT_SINK_REJECTION = "previously created and deleted"

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
        # Target uuids dropped after exhausting retries; skipped on later scans so a
        # persistently failing target cannot wedge every scan. Persisted to
        # gb_kv_pairs because the checkpoint refuses to advance past an unrecorded
        # target: an in-memory-only drop set would let a dropped target return after a
        # restart and block the watermark forever.
        self._dropped: set[str] = set()
        # target_uuid -> attempts so far, for targets to retry on a later scan.
        self._failed_attempts: dict[str, int] = {}
        # Builds whose targets this watcher may record for: the checkpoint build
        # (the floor) plus every build seen since, minus those retired. Passed to
        # reconcile_once as `allowed_build_ids` — the reason an anchored retreat
        # cannot wander into unrelated history.
        #
        # NOT persisted: a working set the floor makes reconstructible, so a restart
        # reseeds from the checkpoint and re-adds whatever is live; persisting would
        # add an unbounded gb_kv_pairs value for no recoverable state. Safe across
        # restarts only because the checkpoint never advances past an unrecorded
        # target — a mark that outran pending work would leave the regenerated set
        # unable to reach it, since the walk starts at the floor.
        self._allowed_build_ids: set[str] = set()
        # Builds retired by _retire_finished_builds. Discarding from
        # _allowed_build_ids is not enough: anchored-range admission keys off
        # absence from that set, so a retired build would match its "not yet
        # tracked" test and return on the next scan. In-memory like the allowlist
        # — a restart re-derives it from the sink.
        self._retired_build_ids: set[str] = set()
        # Whether the floor has been seeded into _allowed_build_ids. Needs a
        # checkpoint, which may not exist at start(), so retried each scan.
        self._allowlist_seeded = False
        # Whether the start-up verification sweep has run. It cannot run while the
        # key is absent, so it is retried each scan until one is seeded — a watcher
        # started before seeding must still get the sweep, or targets behind the
        # seeded watermark have no path back until the next restart. Latches on the
        # first True and is never cleared (see _verify_checkpoint).
        self._checkpoint_verified = False
        # Whether the "no checkpoint yet" notice has been logged, so a one-time
        # operator notice does not repeat every scan forever. Reset once a
        # checkpoint is seen — defensive only: nothing deletes the key, so the
        # absent -> present -> absent cycle is not currently reachable.
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

    def _record_incomplete_build(
        self, storage: SingletonAdminStorage, build_id: str, dropped: set[str]
    ) -> None:
        """Record that ``build_id`` was retired with some lineage never recorded.

        Retiring such a build is deliberate — a run deleted from wandb cannot be
        regenerated, so retaining it would pin the build forever for lineage that can
        never land. This adds the audit trail: the drop is already logged at ERROR,
        but logs rotate and "was this build's lineage complete?" is asked long after.
        Read back from ``LINEAGE_WATCHER_INCOMPLETE_KEY``.

        Read-modify-write on one key, safe only because a single watcher thread
        retires builds. Failures are swallowed — this runs inside retirement, itself
        a post-scan optimization, so losing the entry must never abort a scan whose
        recording succeeded; the original ERROR line remains either way.
        """
        try:
            value = storage.kv_pair_storage.get_value(LINEAGE_WATCHER_INCOMPLETE_KEY)
            builds = dict((value or {}).get("builds", {}))
            builds[build_id] = {
                "target_ids": sorted(dropped),
                "retired_at": datetime.now(timezone.utc).isoformat(),
            }
            storage.kv_pair_storage.set_value(
                LINEAGE_WATCHER_INCOMPLETE_KEY, {"builds": builds}
            )
        except Exception as exc:  # noqa: BLE001 — audit write must not break a scan
            logger.warning(
                "Could not record build %s as having incomplete lineage (%s); its "
                "%d dropped target(s) are still in the ERROR log above.",
                build_id,
                exc,
                len(dropped),
            )

    def _verify_checkpoint(self, storage: SingletonAdminStorage) -> bool:
        """Re-record any unrecorded target in the checkpoint's own build.

        Runs once: at ``start()`` when the key is already seeded, otherwise at the
        first scan that finds one. ``True`` means done with (including the malformed
        case, which retrying cannot fix), ``False`` still pending for want of a
        checkpoint.

        On ``True`` the caller latches ``_checkpoint_verified`` and this never runs
        again — deliberately, not a recovery gap. It closes the gap left by a crash
        between recording a target and persisting its checkpoint (or vice versa),
        where the checkpoint may name a target whose lineage never reached the sink
        and the scan, starting *at* that watermark, would not re-surface its build.
        That is a property of the checkpoint this process started from: a
        ``--force-build-id`` overwrite resolves a fresh anchor with nothing to
        repair, and no code path deletes the key.

        The checkpoint is never created here or anywhere else implicitly: when the
        key is absent this defers, as does every scan (see the class docstring).
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

        # Via reconcile_once, scoped to the checkpoint's build, so
        # filter_unrecorded, expected-run-count derivation and prerun-skip handling
        # are shared rather than reimplemented.
        #
        # Failures are swallowed rather than aborting start() (the checkpoint is
        # already durable, and a watcher that refused to start would record
        # nothing), but they cost differently: a target at or after the anchor is
        # re-surfaced by the next scan, one further back is NOT, so this sweep is
        # its only path back. They route through the same _on_record_error
        # bookkeeping, not a log-only callback, so such a target can still exhaust
        # _MAX_RECORD_ATTEMPTS and be dropped durably — otherwise it blocks
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

        Delegates selection and recording to ``reconcile_once``, bounding the walk
        at the anchor row derived from the checkpoint.

        The checkpoint is re-read every scan rather than cached: it is the single
        source of truth, so a key seeded or corrected mid-run takes effect on the
        next scan instead of the next restart. A missing key is a no-op — "record
        nothing" until seeded — and must never fall back to scanning the whole admin
        DB, which would turn an unseeded deployment into a full backfill.

        Failures route to ``_on_record_error`` for the bounded per-target retry. The
        checkpoint is persisted after each individually-recorded target, not once at
        the end, so a mid-scan crash leaves it at the last target recorded.
        """
        if self._store is None:
            logger.error("lineage store not initialized; start() must run first")
            return
        storage = get_admin_storage()
        if not self._checkpoint_verified:
            # A checkpoint seeded after start() still needs the sweep, and only this
            # retry can give it one: the scan below stops at the anchor row and never
            # looks behind it, so targets that finished before the checkpoint build's
            # oldest would otherwise wait for a restart.
            self._checkpoint_verified = self._verify_checkpoint(storage)
        watermark = self._checkpoint_watermark(storage)
        if watermark is None:
            # No checkpoint: recording is off until seeded. Return before querying
            # targets or touching the sink.
            return
        # Bound the walk by a *row*, not a timestamp. finished_at is stamped when the
        # build event is created, not when its row is committed, so a row can become
        # visible already carrying a timestamp behind an advanced watermark; a time
        # cutoff excludes it outright and the monotonic watermark never re-surfaces
        # it. Anchoring on the checkpoint build's *oldest* successful target lets the
        # walk retreat to a known row, and since the query never filters on build_id
        # it also picks up concurrent builds' stragglers. The oldest, not the
        # checkpoint's own finished_at: it names whichever target recorded last, so
        # staggered finishes would leave earlier ones behind the row. It can fail to
        # resolve, in which case the scan falls back to the watermark — see below.
        checkpoint_build_id = self._checkpoint_build_id(storage)
        # Seed the floor before selecting, so the first scan after a (re)start is
        # already bounded. Needs the checkpoint, so retried here, not in start().
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
        # Order matters: admission needs the resolved anchor to know what "within
        # range" means, and both must run before the filtered scan — a build admitted
        # after it would have its targets dropped until the next pass.
        self._admit_builds_in_anchored_range(storage, stop_at)
        self._refresh_allowed_builds(storage)
        # The anchor is the real bound, so the timestamp cutoff opens all the way
        # up; select_recordable_targets stops at the anchor row instead.
        #
        # With no anchor the walk falls back to the watermark cutoff: inclusive of the
        # boundary row but reaching no further, so a target strictly behind it is out
        # of reach while the anchor cannot be resolved — the one case this design is
        # no better than the fixed window it replaced. Nothing is subtracted to soften
        # it: without a row to aim at there is no principled width, and inventing one
        # is what the old overlap did. Warned so it is visible.
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
        # Stamped before the scan, not after: a target finishing mid-scan must stay in
        # scope for the next one, and an after-the-fact stamp would step over it.
        # Aware, to compare against the targets' own aware timestamps; UTC because
        # this is a sentinel off the clock, not a gb_targets value.
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
            # Log-only: names the build whose target set the watermark, so the line
            # reads as "starting here because of that row" rather than a bare epoch.
            # Its role in *selection* is carried by stop_at below.
            watermark_build_id=checkpoint_build_id,
            stop_at=stop_at,
            # Bounds *which* builds the retreat may record for. Only once seeded: an
            # empty membership filter selects nothing, silently stalling recording
            # rather than merely widening it, and unseeded means no checkpoint build
            # resolved — the scan is already on its fallback path.
            allowed_build_ids=(
                self._allowed_build_ids if self._allowlist_seeded else None
            ),
        )
        # After the scan, never before: retirement asks the sink whether a build's
        # lineage is recorded, and running it first would ask about targets this pass
        # had yet to write — retiring a build just before its own targets landed and,
        # since a retired build is not re-admitted from build state, dropping them.
        self._retire_finished_builds(storage)

    def _refresh_allowed_builds(self, storage: SingletonAdminStorage) -> None:
        """Add builds that have appeared since the last refresh.

        Purely additive: a build enters when first seen and leaves only through
        ``_retire_finished_builds``, which runs *after* the scan — retiring first
        would ask the sink about targets the pass had not written yet. Nothing is
        dropped for being *old*: that would be judged on ``finished_at``, which can
        place a late-committing row behind the watermark — the case the anchored walk
        exists to recover.

        Only builds not yet finished are added *here*. One already finished before
        this watcher saw it is either behind the floor (deliberately out of scope) or
        already recorded, so adding it would re-open the history the floor excludes.

        That rule alone is too narrow: a build concurrent with the floor can finish
        *inside* the anchored range and already be finished when a refresh runs —
        neither the checkpoint build nor unfinished, so never added here, and
        filtering would drop the very target the retreat reached back for.
        ``_admit_builds_in_anchored_range`` grants membership from below for those.
        This method covers builds still live; that one, builds that finished inside
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
            # Also keeps retired builds out of this path without consulting
            # _retired_build_ids: retirement requires is_finished().
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
        lineage-loss bug. A concurrent build can finish between two targets of the
        checkpoint's build and so be finished before any refresh sees it, meaning
        ``_refresh_allowed_builds`` will not add it — and a membership filter would
        then discard the target the walk retreated to collect.

        Admitting from the *swept range* rather than from build state closes that:
        whatever the walk can reach is in scope by definition, so the filter only
        removes what lies outside the anchor. Bounded by the same anchor the scan
        uses, so this is not a full-history read — it is the range the scan was
        going to walk anyway. Without an anchor there is no bounded range, so this
        is a no-op and the scan runs on its fallback path.
        """
        if stop_at is None or not self._allowlist_seeded:
            return
        try:
            # stop_at is the bound, not finished_after: UTC_MIN opens the
            # timestamp cutoff all the way up and the walk stops at the anchor
            # row, same as the scan's own call. Not a full-history read.
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
            # A retired build is deliberately done; re-admitting would undo
            # retirement every scan. Not _dropped: that holds *target* uuids.
            and t.build_id not in self._retired_build_ids
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
        uptime. All three conditions are required, each guarding a distinct way a
        build can still owe work:

        - ``status.is_finished()``: excludes ``RETRY_PENDING`` (a build queued for a
          build-level retry will still produce targets) and ``RUNNING``/``PENDING``.
        - Nothing pending locally: no target mid-retry in ``_failed_attempts``.
          Targets in ``_dropped`` do NOT count as pending — treating them so would
          pin the build forever, and a run deleted from wandb cannot be regenerated
          (its id derives from the target and the sink refuses a deleted id), so
          that pin would never release. Retiring anyway is the deliberate choice;
          the gap is recorded under ``LINEAGE_WATCHER_INCOMPLETE_KEY`` rather than
          left to a rotating log, so completeness stays analyzable.
        - Recorded in the sink: only the sink knows whether the runs landed.

        Retirement is irreversible here, and both admission paths are closed to keep
        it so: ``_refresh_allowed_builds`` skips finished builds, and
        ``_admit_builds_in_anchored_range`` (admitting from the swept range, so it
        never sees that check) tests ``_retired_build_ids``. Every uncertain case must
        retain, never retire — including a sink that fails to answer.
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
                confirmed, dropped = self._build_lineage_is_confirmed(storage, build_id)
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
            if dropped:
                # Retired with lineage knowingly missing. Recorded before the
                # discard so a crash between the two cannot lose the gap: a
                # re-recorded entry is harmless, an unrecorded one is invisible.
                self._record_incomplete_build(storage, build_id, dropped)
            self._allowed_build_ids.discard(build_id)
            # Keeps _admit_builds_in_anchored_range from re-admitting it: that
            # path admits from the swept range, so it never sees the
            # finished-build check _refresh_allowed_builds applies.
            self._retired_build_ids.add(build_id)
            if dropped:
                # Not "recorded in the sink": these targets never landed there.
                logger.info(
                    "Retired build %s from the lineage allowlist: finished and "
                    "nothing pending, but %d target(s) were dropped, so its "
                    "lineage in the sink is incomplete and will stay that way "
                    "(recorded under %s). Allowlist now tracks %d build(s).",
                    build_id,
                    len(dropped),
                    LINEAGE_WATCHER_INCOMPLETE_KEY,
                    len(self._allowed_build_ids),
                )
            else:
                logger.info(
                    "Retired build %s from the lineage allowlist: finished, nothing "
                    "pending, and its lineage is recorded in the sink. Allowlist now "
                    "tracks %d build(s).",
                    build_id,
                    len(self._allowed_build_ids),
                )

    def _build_lineage_is_confirmed(
        self, storage: SingletonAdminStorage, build_id: str
    ) -> tuple[bool, set[str]]:
        """Whether ``build_id`` has nothing pending and its lineage is in the sink.

        The last two of the three retirement conditions (the caller has already
        checked that the build is finished). Split out so the caller can treat any
        failure as "not confirmed" in one place: every path here either answers
        definitively or raises, and a raise must never mean "retire".

        Returns ``(confirmed, dropped_target_ids)``. The second element is what
        makes a retirement's completeness auditable: "confirmed" covers both
        *recorded* and *deliberately dropped*, and only the caller can tell those
        apart to record the gap. Empty means every target actually recorded.
        """
        if self._store is None:
            return False, set()
        # Re-select this build's successful targets to ask the sink about them.
        # Scoped to the one build and only reached for finished builds, so this is
        # not the unbounded walk `allowed_build_ids` exists to prevent.
        targets = select_recordable_targets(
            storage, finished_after=UTC_MIN, build_id=build_id
        )
        if any(t.uuid in self._failed_attempts for t in targets):
            # Mid-retry: still owes work locally.
            return False, set()
        dropped = {t.uuid for t in targets if t.uuid in self._dropped}
        pending = {t.uuid for t in targets if t.uuid not in self._dropped}
        if not pending:
            # Nothing left to confirm: every target either recorded or was
            # deliberately dropped. Report the dropped ones so the caller can
            # record the gap — if `dropped` covers every target, this build's
            # lineage never landed at all.
            return True, dropped
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
        return (not sink_failed and not unrecorded), dropped

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

        The parsed value is made aware, keeping whatever offset it carries — the key
        holds ``finished_at`` as a string in a JSON dict, so it is backend-opaque and
        round-trips its offset on SQLite and Postgres alike, and since every writer
        persists aware-ISO the naive case is defensive. Filling a missing offset lets
        the caller compare against targets' own timestamps without a ``TypeError``;
        the offset filled in is *local*, so a caller testing for the ``datetime.min``
        anchor must compare within the watermark's own offset (see
        ``_retire_backfill_anchor``). Not rewritten to UTC — see
        ``_on_checkpoint_advance``.

        ``OverflowError`` is caught alongside the parse errors: reachable from
        arithmetic on an extreme value (the ``all`` anchor is ``datetime.min``), and
        uncaught it would escape to ``_run`` and fail every scan forever.
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

        ``--base-build-id all`` anchors the checkpoint at ``datetime.min`` so the
        first scan walks all history, and normally the first recorded target advances
        it off that anchor. When the backfill records nothing — empty DB, or every
        candidate dropped — nothing advances it and every later scan re-walks the
        whole table. Safe only under both conditions checked here:

        - The watermark really is the anchor (a normal one moves only via a recorded
          target). Compared in the watermark's *own* offset, not against ``UTC_MIN``:
          a backend that drops the offset (SQLite does, Postgres ``timestamptz`` does
          not) hands back a naive ``datetime.min`` re-tagged *local* — the same instant
          only at UTC, so a bare ``!= UTC_MIN`` would leave the anchor in place on
          every non-UTC deployment, re-walking the table forever.
        - ``watermark_untouched``: the pass recorded, failed and dropped nothing.
          Otherwise unrecorded lineage sits *behind* the new anchor, which the scan
          never looks behind, stranding it. A record in the same pass has already
          advanced the checkpoint; moving it again would step over later targets.
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

        The watermark is monotonic: a target swept in behind it legitimately finished
        there (late-visible row, or interleaved builds) and must not drag the durable
        value down, which would re-read that range on every later scan. Recording
        still happens; only the watermark write is suppressed. Compared against the
        *durable* value, not an in-process high-water mark, so a restart cannot forget
        it; the ``datetime.min`` anchor sits below every real timestamp, so
        ``_retire_backfill_anchor`` still advances off it.

        ``finished_at`` comes off a ``StoredTargetRun``, so in production it is aware
        *local* (from ``BuildEvent.timestamp`` → ``get_time()``), not UTC. ``as_aware``
        only fills a missing offset, so what is persisted is the ``gb_targets`` row's
        timestamp verbatim — the checkpoint adopts that table's form and ``gb_targets``
        is untouched. Rewriting to UTC would still compare correctly but made the key
        disagree *textually* with its source row, unreadable in a log.

        Ensuring awareness is still required: ``_checkpoint_watermark`` once returned
        naive, and comparing raised ``TypeError: can't compare offset-naive and
        offset-aware datetimes`` inside ``reconcile_once``'s per-target ``try`` —
        misattributed to recording (already succeeded), blocking the checkpoint, and
        after ``_MAX_RECORD_ATTEMPTS`` scans landing the target in the dropped set.
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

        The drop is persisted: giving up is permanent, and since the checkpoint never
        advances past an unrecorded target, a drop forgotten on restart would block
        the watermark forever. The attempt *counts* stay in memory — a within-process
        backoff, and a restart legitimately retries from zero (the failure may have
        been the crash itself). Only the terminal decision is durable.
        """
        if self._is_permanent_sink_rejection(exc):
            # No retry can clear this, so spend none: it would be
            # _MAX_RECORD_ATTEMPTS scans of guaranteed failures, holding the
            # checkpoint at this target throughout. Drop on first sighting and say
            # why — an operational condition (someone deleted runs in the sink),
            # not a flaky write, and the operator's response differs.
            self._failed_attempts.pop(target_id, None)
            self._dropped.add(target_id)
            self._persist_dropped(storage)
            logger.error(
                "The lineage sink permanently rejected the run id for target %s "
                "in build %s: its run was created and then deleted in the sink, "
                "which will not accept that id again. Run ids are derived "
                "deterministically from the target, so no retry can succeed; "
                "dropping this target's lineage without retrying. Its lineage "
                "will stay absent from the sink unless the deleted run is "
                "restored there. Underlying error: %s",
                target_id,
                build_id,
                exc,
            )
            return
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

    @classmethod
    def _is_permanent_sink_rejection(cls, exc: Exception) -> bool:
        """Whether ``exc`` is a sink rejection that retrying can never clear.

        Walks the exception chain rather than checking only ``exc``: the failure
        surfaces from the store's own ``except`` blocks, so the rejection can
        arrive wrapped (``raise ... from`` sets ``__cause__``; a bare re-raise
        inside a handler sets ``__context__``). Matching only the outermost
        exception would miss it and spend the full retry budget.

        Deliberately conservative — a false negative just means the target takes
        the normal retry path (the pre-existing behavior), while a false positive
        would drop a recoverable target's lineage without retrying. That is why
        this matches a specific message rather than an exception type.
        """
        seen: set[int] = set()
        current: Optional[BaseException] = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if cls._PERMANENT_SINK_REJECTION in str(current):
                return True
            current = current.__cause__ or current.__context__
        return False

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
