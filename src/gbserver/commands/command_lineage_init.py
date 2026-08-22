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

"""Place the lineage checkpoint without starting the watcher.

``lineage-watch --base-build-id`` already seeds the checkpoint, but it then runs
the watcher forever, so initializing a fresh environment meant starting it and
killing it once the key appeared. That works, but it records lineage for however
long it happened to run -- an unclear amount of work on an environment being set
up, and awkward to script.

This command does the seeding half only: resolve the anchor, write
``gb_kv_pairs``, exit. The watcher then starts clean, from a mark someone chose
deliberately.
"""

import json

import click

from gbserver.lineage.lineage_reconciler import (
    LINEAGE_WATCHER_CHECKPOINT_KEY,
    LINEAGE_WATCHER_DROPPED_KEY,
)
from gbserver.lineage.lineage_seeding import LineageSeedError, seed_if_absent
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def _clear_dropped_targets(storage) -> None:
    """Empty the durable dropped-*target* set so those targets are retried.

    Writes an empty list rather than deleting the key, so the shape stays what
    ``LineageWatcher._load_dropped`` expects on its next start.

    A running watcher does NOT see this immediately: ``_dropped`` is loaded once in
    ``start()``, not per scan (unlike the checkpoint, which is re-read every scan).
    So this takes effect on the watcher's next restart, and the caller is told so
    rather than left to wonder why the targets are still skipped.
    """
    existing = storage.kv_pair_storage.get_value(LINEAGE_WATCHER_DROPPED_KEY)
    target_ids = (existing or {}).get("target_ids", [])
    if not target_ids:
        click.echo(f"{LINEAGE_WATCHER_DROPPED_KEY}: nothing to clear.")
        return
    storage.kv_pair_storage.set_value(LINEAGE_WATCHER_DROPPED_KEY, {"target_ids": []})
    click.echo(
        f"Cleared {len(target_ids)} dropped target(s) from "
        f"{LINEAGE_WATCHER_DROPPED_KEY}: {', '.join(sorted(target_ids))}. "
        "Restart the lineage watcher for this to take effect -- it loads the drop "
        "set once at start(), not on every scan."
    )


@click.command()
@click.option(
    "--build-id",
    required=False,
    default=None,
    type=str,
    metavar="from-latest|all|BUILD_ID",
    help=(
        "Anchor to seed: 'from-latest' starts at the most recent build, 'all' "
        "walks the full history (expensive first scan), any other value is "
        "treated as a build id. The anchor is the build itself: it and every "
        "build created after it are processed, so the anchored build is recorded "
        "whole."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help=(
        "Replace an existing checkpoint instead of keeping it. Moving the anchor "
        "back re-drives lineage already recorded; moving it forward skips lineage "
        "for good."
    ),
)
@click.option(
    "--clear-dropped-targets",
    is_flag=True,
    default=False,
    help=(
        "Clear the set of TARGET ids the watcher permanently gave up on (after "
        "exhausting its record attempts) so they are retried. Requires a watcher "
        "restart to take effect: the drop set is loaded once at start(), unlike "
        "the checkpoint, which is re-read every scan. Moving the anchor back does "
        "NOT clear them, which is why this exists -- the drop decision is durable "
        "on purpose, and a dropped target is otherwise skipped forever. Only "
        "useful once whatever made recording fail is fixed."
    ),
)
@click.option(
    "--show",
    is_flag=True,
    default=False,
    help=(
        "Print the current checkpoint and dropped-target set, then exit without "
        "writing anything."
    ),
)
@pass_environment
def cli(
    ctx: CliEnvironment,
    build_id: str,
    force: bool,
    clear_dropped_targets: bool,
    show: bool,
) -> None:
    """Seed the lineage checkpoint (gb_kv_pairs) without running the watcher."""
    storage = get_admin_storage()

    if show:
        existing = storage.kv_pair_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
        if existing is None:
            click.echo(
                f"No lineage checkpoint under {LINEAGE_WATCHER_CHECKPOINT_KEY}. "
                "The watcher records nothing until one is seeded."
            )
        else:
            click.echo(f"{LINEAGE_WATCHER_CHECKPOINT_KEY} = {json.dumps(existing)}")
        # Report the drop set too: a dropped target is skipped on every scan, so an
        # operator asking "why is this target's lineage missing?" needs to see it
        # next to the checkpoint rather than querying gb_kv_pairs by hand.
        dropped = storage.kv_pair_storage.get_value(LINEAGE_WATCHER_DROPPED_KEY)
        target_ids = (dropped or {}).get("target_ids", [])
        if target_ids:
            click.echo(
                f"{LINEAGE_WATCHER_DROPPED_KEY}: {len(target_ids)} target(s) "
                f"permanently skipped: {', '.join(sorted(target_ids))}"
            )
        else:
            click.echo(f"{LINEAGE_WATCHER_DROPPED_KEY}: no targets dropped.")
        return

    if clear_dropped_targets:
        _clear_dropped_targets(storage)
        if build_id is None:
            # Clearing alone is a complete operation: the checkpoint is untouched
            # and the watcher retries the targets on its next scan.
            return

    if build_id is None:
        raise click.ClickException(
            "Nothing to do: pass --build-id to seed the checkpoint, "
            "--clear-dropped-targets to retry dropped targets, or --show to "
            "inspect the current state."
        )

    if not build_id.strip():
        # Same reason lineage-watch rejects this: an empty string passes click's
        # required check but resolves to no anchor, so the operator would believe a
        # checkpoint was placed when none was.
        raise click.ClickException(
            "--build-id was given an empty value; pass 'from-latest', 'all', or a "
            "build id."
        )

    try:
        wrote = seed_if_absent(storage, build_id, force=force)
    except LineageSeedError as exc:
        raise click.ClickException(str(exc)) from exc

    value = storage.kv_pair_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)
    if wrote:
        click.echo(f"Seeded {LINEAGE_WATCHER_CHECKPOINT_KEY} = {json.dumps(value)}")
    else:
        # Not an error: seed-if-absent is the safe default, and re-running this
        # command on an initialized environment should be a no-op rather than a
        # failure.
        click.echo(
            f"{LINEAGE_WATCHER_CHECKPOINT_KEY} already set to {json.dumps(value)}; "
            "kept it. Pass --force to replace it."
        )
