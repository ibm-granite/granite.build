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

"""``gbserver lineage-seed`` — choose where centralized lineage recording starts.

The LineageWatcher never creates its checkpoint implicitly: with no
``lineage_store_latest_build_id`` key in ``gb_status`` it records nothing at all
(see ``lineage_watcher.LineageWatcher._verify_checkpoint``). That is deliberate —
a fresh deployment must not decide on its own whether to start "from now" or to
backfill the platform's whole history. This command is how that decision is made
and written down.

Three ways to place the checkpoint:

- ``--from-latest``: anchor at the newest successful target. Recording starts
  "from now"; everything older is left alone.
- ``--build-id <id>``: anchor at a chosen build's newest successful target, so
  everything that finished after that build is picked up.
- ``--all``: anchor before all recorded history, so the watcher walks every
  successful target the platform has ever run. The per-sink
  ``filter_unrecorded`` check means targets already in the sink are skipped
  rather than duplicated, but the first scan is a full-table walk plus a large
  sink query — hence opt-in.
"""

from datetime import datetime

import click

from gbserver.lineage.jobstats import get_lineage_store
from gbserver.lineage.lineage_reconciler import (
    LINEAGE_WATCHER_CHECKPOINT_KEY,
    get_most_recent_successful_target,
)
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# Sentinel build_id for the --all checkpoint. The watcher only reads the
# checkpoint's build_id to re-verify that build at start(); a build id that
# matches nothing simply finds no targets to verify, which is exactly right for a
# backfill anchor that deliberately predates every real build.
_BACKFILL_BUILD_ID = "__lineage_backfill__"


@click.command()
@click.option(
    "--from-latest",
    "from_latest",
    is_flag=True,
    help="Start recording from the newest successful target (i.e. from now on).",
)
@click.option(
    "--build-id",
    "build_id",
    required=False,
    type=str,
    help="Start recording from this build's newest successful target.",
)
@click.option(
    "--all",
    "seed_all",
    is_flag=True,
    help=(
        "Record all history: walk every successful target ever run. Already-"
        "recorded targets are skipped, but the first scan is expensive."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite an existing checkpoint instead of refusing.",
)
@pass_environment
def cli(
    ctx: CliEnvironment,
    from_latest: bool,
    build_id: str,
    seed_all: bool,
    force: bool,
):
    """Seed the lineage checkpoint so the watcher knows where to start."""
    chosen = [
        name
        for name, given in (
            ("--from-latest", from_latest),
            ("--build-id", bool(build_id)),
            ("--all", seed_all),
        )
        if given
    ]
    if len(chosen) != 1:
        raise click.UsageError(
            "Pass exactly one of --from-latest, --build-id, or --all "
            f"(got: {', '.join(chosen) if chosen else 'none'})."
        )

    store = get_lineage_store()
    if not store.records_centralized_lineage:
        # Standalone / GBSERVER_LINEAGE_PROVIDER=none: the watcher's recording
        # leaf is a no-op there, so a checkpoint would have nothing to drive.
        logger.info(
            "Configured lineage store does not record centralized lineage; "
            "there is nothing for a checkpoint to start. Not seeding."
        )
        return

    storage = get_admin_storage()
    existing = storage.status_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
    if existing is not None and not force:
        # Re-seeding moves the watermark. Moving it forward skips lineage that
        # was never recorded; moving it back re-drives recorded targets. Neither
        # should happen by accident, so require --force.
        raise click.ClickException(
            f"A lineage checkpoint already exists: {existing}. "
            "The watcher is already recording from there; pass --force to "
            "overwrite it (this moves the watermark and may skip or re-drive "
            "lineage)."
        )

    if seed_all:
        # datetime.min: older than any real finished_at, so nothing is excluded.
        checkpoint = {
            "build_id": _BACKFILL_BUILD_ID,
            "finished_at": datetime.min.isoformat(),
        }
    else:
        target = get_most_recent_successful_target(storage, build_id=build_id)
        # get_most_recent_successful_target only returns targets that have a
        # finished_at, but its return type does not say so; check both so the
        # anchor is provably non-null rather than assumed.
        if target is None or target.finished_at is None:
            scope = f"build {build_id}" if build_id else "the admin DB"
            raise click.ClickException(
                f"No successful target with a finish time found in {scope}; "
                "nothing to anchor a checkpoint at."
            )
        checkpoint = {
            "build_id": target.build_id,
            "finished_at": target.finished_at.isoformat(),
        }

    storage.status_storage.set_value(LINEAGE_WATCHER_CHECKPOINT_KEY, checkpoint)
    logger.info(
        "Seeded lineage checkpoint %s = %s. The watcher records targets that "
        "finish at or after this point on its next scan.",
        LINEAGE_WATCHER_CHECKPOINT_KEY,
        checkpoint,
    )
