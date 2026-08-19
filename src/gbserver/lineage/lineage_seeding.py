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

"""Placing the LineageWatcher's ``gb_kv_pairs`` checkpoint.

The watcher never creates its checkpoint implicitly: with no
``lineage_store_latest_build_id`` key it records nothing at all (see
``lineage_watcher.LineageWatcher._verify_checkpoint``). Deciding where
centralized recording begins — "from now", from a chosen build, or the platform's
whole history — belongs to an operator, not to whichever process starts first.

``gbserver lineage-watch --base-build-id`` is how that decision is expressed. It is
seed-*if-absent*: an existing checkpoint is never overwritten, which is what
makes the flag safe to leave in a pod spec permanently, since a re-seed on every
restart would either skip accumulated lineage (anchor moved forward) or re-drive
the whole history (anchor moved back).

Three anchors, expressed as a single spec string (``from-latest``, ``all``, or a
build id) so no invalid combination is representable.

Both build-derived anchors land on a build's *oldest* completion, because the
watermark is inclusive but forward-only: anchoring at a build's newest target
would exclude that build's own earlier targets, and lose any concurrent build's
targets in that window for good. They differ only in how the build is chosen —
``from-latest`` takes the build of the most recent completion, a build id takes
the one named. So "from now on" means from the start of the newest build, not
from the middle of it.
"""

from gbserver.lineage.lineage_reconciler import (
    LINEAGE_WATCHER_CHECKPOINT_KEY,
    UTC_MIN,
    as_utc_aware,
    get_most_recent_successful_target,
    get_oldest_successful_target,
)
from gbserver.storage.singleton_storage import SingletonAdminStorage
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# Spec values that name an anchor rather than a build id.
SEED_FROM_LATEST = "from-latest"
SEED_ALL = "all"

# Sentinel build_id for the `all` checkpoint. The watcher only reads the
# checkpoint's build_id to re-verify that build at start(); a build id that
# matches nothing simply finds no targets to verify, which is exactly right for a
# backfill anchor that deliberately predates every real build.
BACKFILL_BUILD_ID = "__lineage_backfill__"


class LineageSeedError(Exception):
    """No checkpoint could be built for the requested anchor."""


def _build_checkpoint(storage: SingletonAdminStorage, spec: str) -> dict:
    """Build (but do not persist) the checkpoint value for ``spec``.

    Args:
        storage: Admin storage to resolve the anchor target against.
        spec: ``"from-latest"`` (anchor at the oldest successful target of the
            *newest* build), ``"all"`` (anchor at ``UTC_MIN``, i.e. the full
            history), or a build id (anchor at that build's oldest successful
            target). Either way the anchored build is recorded whole.

    Returns:
        ``{"build_id": str, "finished_at": <ISO 8601 str, aware UTC>}``.

    Raises:
        LineageSeedError: When the anchor resolves to no successful target — an
            empty DB, or a build id that does not exist or never succeeded.
    """
    if spec == SEED_ALL:
        # UTC_MIN: older than any real finished_at, so nothing is excluded. Aware,
        # matching every other watermark — a naive datetime.min would raise
        # TypeError the moment it met an aware finished_at.
        return {
            "build_id": BACKFILL_BUILD_ID,
            "finished_at": UTC_MIN.isoformat(),
        }

    if spec == SEED_FROM_LATEST:
        # Resolve the build in two steps: the most recent completion names the
        # build to start from, but the anchor is then that build's *oldest*
        # target — the same treatment a build id gets, with the build chosen
        # automatically instead of by the operator.
        #
        # Anchoring directly at the newest completion would start recording
        # mid-build: the watermark is inclusive but forward-only, so the rest of
        # that build's already-finished targets would be skipped, while the
        # checkpoint still names their build. "From now on" means from the start
        # of the newest build, not from the middle of it.
        latest = get_most_recent_successful_target(storage)
        build_id = latest.build_id if latest is not None else None
        target = (
            get_oldest_successful_target(storage, build_id=build_id)
            if build_id is not None
            else None
        )
    else:
        # A specific build: anchor at that build's *oldest* target, not its
        # newest. The watermark is inclusive but forward-only, so anchoring at the
        # newest would silently exclude every earlier target of the very build the
        # operator asked to record — and any concurrent build's targets in that
        # window, which no later scan re-surfaces (see
        # get_oldest_successful_target).
        build_id = spec
        target = get_oldest_successful_target(storage, build_id=build_id)
    # The selectors only return targets that have a finished_at, but their return
    # type does not say so; check both so the anchor is provably non-null rather
    # than assumed.
    if target is None or target.finished_at is None:
        scope = f"build {build_id}" if build_id else "the admin DB"
        raise LineageSeedError(
            f"No successful target with a finish time found in {scope}; "
            "nothing to anchor a checkpoint at."
        )
    # Serialize as an aware UTC instant, the single format every writer of this
    # key uses (see LineageWatcher._on_checkpoint_advance). `finished_at` is a
    # DateTime(timezone=True) column, so Postgres hands it back aware while
    # SQLite drops the offset; normalizing here means the stored string is one
    # unambiguous instant either way, and keeping the "+00:00" makes it
    # round-trip losslessly instead of becoming a naive value that a reader has
    # to guess the offset of.
    return {
        "build_id": target.build_id,
        "finished_at": as_utc_aware(target.finished_at).isoformat(),
    }


def seed_if_absent(
    storage: SingletonAdminStorage, spec: str, force: bool = False
) -> bool:
    """Seed the checkpoint, by default only when one does not already exist.

    Leaving an existing checkpoint alone is the whole point: the flag is meant to
    live permanently in a Deployment spec, and re-seeding on every pod restart
    would either skip lineage (anchor moved forward) or re-drive the full history
    (anchor moved back).

    ``force`` overrides that, replacing an existing checkpoint with the requested
    anchor. It exists for the case the seed-if-absent rule cannot fix on its own:
    a checkpoint left at a wrong or unusable value (seeded at the wrong build, or
    written in a stale format), where recording stays stuck until someone moves it
    by hand. Moving the anchor *backwards* re-drives lineage that was already
    recorded, which is idempotent at the sink but not free, and moving it forwards
    skips lineage permanently — so it must never live in a Deployment spec, where
    it would re-apply on every restart.

    Returns:
        True if the checkpoint was written, False if one already existed and
        was kept.

    Raises:
        LineageSeedError: When the anchor cannot be resolved.
    """
    existing = storage.kv_pair_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
    if existing is not None and force:
        # Resolve the new anchor before overwriting: if it cannot be resolved this
        # raises, and the existing checkpoint must survive that rather than being
        # cleared by a failed re-seed.
        checkpoint = _build_checkpoint(storage, spec)
        storage.kv_pair_storage.set_value(LINEAGE_WATCHER_CHECKPOINT_KEY, checkpoint)
        logger.warning(
            "Overwrote lineage checkpoint %s: %s -> %s (--force-build-id). "
            "Lineage between the two anchors is re-driven if the anchor moved "
            "back, or skipped for good if it moved forward.",
            LINEAGE_WATCHER_CHECKPOINT_KEY,
            existing,
            checkpoint,
        )
        return True
    if existing is not None:
        logger.info(
            "Lineage checkpoint %s already exists (%s); keeping it and ignoring "
            "the requested seed (%s).",
            LINEAGE_WATCHER_CHECKPOINT_KEY,
            existing,
            spec,
        )
        return False

    checkpoint = _build_checkpoint(storage, spec)
    storage.kv_pair_storage.set_value(LINEAGE_WATCHER_CHECKPOINT_KEY, checkpoint)
    logger.info(
        "Seeded lineage checkpoint %s = %s. The watcher records targets that "
        "finish at or after this point on its next scan.",
        LINEAGE_WATCHER_CHECKPOINT_KEY,
        checkpoint,
    )
    return True
