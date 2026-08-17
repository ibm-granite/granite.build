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

"""Placing the LineageWatcher's ``gb_status`` checkpoint.

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
"""

from datetime import datetime

from gbserver.lineage.lineage_reconciler import (
    LINEAGE_WATCHER_CHECKPOINT_KEY,
    get_most_recent_successful_target,
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
        spec: ``"from-latest"``, ``"all"``, or a build id.

    Returns:
        ``{"build_id": str, "finished_at": <ISO 8601 str>}``.

    Raises:
        LineageSeedError: When the anchor resolves to no successful target — an
            empty DB, or a build id that does not exist or never succeeded.
    """
    if spec == SEED_ALL:
        # datetime.min: older than any real finished_at, so nothing is excluded.
        return {
            "build_id": BACKFILL_BUILD_ID,
            "finished_at": datetime.min.isoformat(),
        }

    build_id = None if spec == SEED_FROM_LATEST else spec
    target = get_most_recent_successful_target(storage, build_id=build_id)
    # get_most_recent_successful_target only returns targets that have a
    # finished_at, but its return type does not say so; check both so the anchor
    # is provably non-null rather than assumed.
    if target is None or target.finished_at is None:
        scope = f"build {build_id}" if build_id else "the admin DB"
        raise LineageSeedError(
            f"No successful target with a finish time found in {scope}; "
            "nothing to anchor a checkpoint at."
        )
    return {
        "build_id": target.build_id,
        "finished_at": target.finished_at.isoformat(),
    }


def seed_if_absent(storage: SingletonAdminStorage, spec: str) -> bool:
    """Seed the checkpoint only when one does not already exist.

    Leaving an existing checkpoint alone is the whole point: the flag is meant to
    live permanently in a Deployment spec, and re-seeding on every pod restart
    would either skip lineage (anchor moved forward) or re-drive the full history
    (anchor moved back).

    Returns:
        True if a checkpoint was written, False if one already existed.

    Raises:
        LineageSeedError: When no checkpoint exists and the anchor cannot be
            resolved.
    """
    existing = storage.status_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
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
    storage.status_storage.set_value(LINEAGE_WATCHER_CHECKPOINT_KEY, checkpoint)
    logger.info(
        "Seeded lineage checkpoint %s = %s. The watcher records targets that "
        "finish at or after this point on its next scan.",
        LINEAGE_WATCHER_CHECKPOINT_KEY,
        checkpoint,
    )
    return True
