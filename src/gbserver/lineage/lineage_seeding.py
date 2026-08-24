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
``lineage_watcher.LineageWatcher._read_checkpoint``). Deciding where
centralized recording begins — "from now", from a chosen build, or the platform's
whole history — belongs to an operator, not to whichever process starts first.

``gbserver lineage-watch --base-build-id`` is how that decision is expressed. It is
seed-*if-absent*: an existing checkpoint is never overwritten, which is what
makes the flag safe to leave in a pod spec permanently, since a re-seed on every
restart would either skip accumulated lineage (anchor moved forward) or re-drive
the whole history (anchor moved back).

Three anchors, expressed as a single spec string (``from-latest``, ``all``, or a
build id) so no invalid combination is representable.

The anchor is a *build*, and the anchored build is recorded whole: the watcher
selects it along with every build created at or after it, and reconciles each
build's targets as a unit. There is therefore no way to start mid-build, and no
need to reach for a build's oldest target to avoid doing so. The two
build-derived anchors differ only in how the build is chosen — ``from-latest``
takes the newest build, a build id takes the one named.
"""

from gbserver.lineage.lineage_reconciler import (
    LINEAGE_WATCHER_CHECKPOINT_KEY,
    LINEAGE_WATCHER_CHECKPOINT_VERSION,
    UTC_MIN,
    as_aware,
    get_most_recent_build,
)
from gbserver.storage.singleton_storage import SingletonAdminStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# Spec values that name an anchor rather than a build id.
SEED_FROM_LATEST = "from-latest"
SEED_ALL = "all"

# Sentinel build_id for the `all` checkpoint. It names no real build, and the
# watcher special-cases it: rather than reading a build row for its creation time,
# it resolves the cutoff to UTC_MIN, which is exactly right for a backfill anchor
# that deliberately predates every real build.
BACKFILL_BUILD_ID = "__lineage_backfill__"


class LineageSeedError(Exception):
    """No checkpoint could be built for the requested anchor."""


def _build_checkpoint(storage: SingletonAdminStorage, spec: str) -> dict:
    """Build (but do not persist) the checkpoint value for ``spec``.

    The anchor is a *build*, not a target. That is inherent to the build-scoped
    scan: the watcher selects the anchor build and everything created at or after
    it, and reconciles each build's targets whole — so there is no way to start
    mid-build and no need to reach for a build's oldest target to avoid it.

    Args:
        storage: Admin storage to resolve the anchor build against.
        spec: ``"from-latest"`` (anchor at the newest build), ``"all"`` (anchor at
            ``UTC_MIN``, i.e. the full history), or a build id.

    Returns:
        ``{"build_id": str, "created_time": <ISO 8601 str, aware>, "version": int}``.

    Raises:
        LineageSeedError: When the anchor resolves to no build — an empty DB, or a
            build id that does not exist or carries no creation time.
    """
    if spec == SEED_ALL:
        # UTC_MIN: older than any real created_time, so nothing is excluded. Aware,
        # matching every other timestamp here — a naive datetime.min would raise
        # TypeError the moment it met an aware created_time.
        return {
            "build_id": BACKFILL_BUILD_ID,
            "created_time": UTC_MIN.isoformat(),
            "version": LINEAGE_WATCHER_CHECKPOINT_VERSION,
        }

    if spec == SEED_FROM_LATEST:
        build = get_most_recent_build(storage)
        build_id = build.uuid if build is not None else None
    else:
        build_id = spec
        found = storage.build_storage.get_by_uuid(build_id)
        build = found if isinstance(found, StoredBuild) else None

    if build is None or build.created_time is None:
        scope = f"build {build_id}" if build_id else "the admin DB"
        raise LineageSeedError(
            f"No build with a creation time found for {scope}; "
            "nothing to anchor a checkpoint at."
        )
    # Serialize keeping the build's own offset. created_time is written by the
    # storage layer via get_utc_time(), so it is aware UTC on Postgres, while
    # SQLite's DateTime(timezone=True) column drops the offset on the way out;
    # filling it in here means the stored string is one unambiguous instant either
    # way instead of a naive value a reader has to guess the offset of.
    return {
        "build_id": build.uuid,
        "created_time": as_aware(build.created_time).isoformat(),
        "version": LINEAGE_WATCHER_CHECKPOINT_VERSION,
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
