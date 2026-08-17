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

"""``gbserver lineage-watch`` — the centralized lineage recording process.

Runs the LineageWatcher, which periodically reconciles the admin DB into the
configured lineage store (see ``lineage_reconciler`` / ``lineage_watcher``).
Deployed as its own single-replica pod (``dep-lineage-watcher.yaml``) so the
single-writer guarantee is a deployment fact and lineage recording is isolated
from the build watcher's failure domain, restarts, and resource contention.

``--base-build-id`` is optional. Without it the watcher reads the ``gb_status``
checkpoint and, when the key is absent, records nothing until it is seeded. With
it, an *absent* key is seeded before the first scan, so a fresh deployment does
not need a separate exec/init-container step just to become useful. It never
overwrites an existing checkpoint, which is what makes it safe to leave in a pod
spec across restarts.
"""

import traceback

import click

from gbserver.lineage.jobstats import get_lineage_store
from gbserver.lineage.lineage_seeding import LineageSeedError, seed_if_absent
from gbserver.lineage.lineage_watcher import LineageWatcher
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


@click.command()
@click.option(
    "--interval",
    required=False,
    type=float,
    default=30.0,
    show_default=True,
    help="Seconds between admin-DB reconciliation scans.",
)
@click.option(
    "--base-build-id",
    required=False,
    type=str,
    default=None,
    metavar="from-latest|all|BUILD_ID",
    help=(
        "Seed the lineage checkpoint before the first scan, but only if it is "
        "not already set: 'from-latest' starts recording from now, 'all' walks "
        "the full history (expensive first scan), any other value is treated as "
        "a build id. Omit to use whatever is already in gb_status (recording "
        "nothing while the key is absent). Never overwrites an existing "
        "checkpoint."
    ),
)
@pass_environment
def cli(ctx: CliEnvironment, interval: float, base_build_id: str):
    """Start the centralized lineage recording watcher."""
    store = get_lineage_store()
    if not store.records_centralized_lineage:
        # Standalone / GBSERVER_LINEAGE_PROVIDER=none: nothing to record. Do not
        # busy-idle; log and exit so the process/pod is a clear no-op.
        logger.info(
            "Configured lineage store does not record centralized lineage; "
            "lineage-watch has nothing to do. Exiting."
        )
        return

    if base_build_id is not None:
        # Seed-if-absent, before start(): the watcher's own _verify_checkpoint
        # and first scan both read the key, so placing it here means the very
        # first scan is already driven by it. A failure to resolve the anchor is
        # fatal on purpose — the operator asked for a specific starting point,
        # and silently starting up with no checkpoint (recording nothing) would
        # look like a working watcher that never records.
        try:
            seed_if_absent(get_admin_storage(), base_build_id)
        except LineageSeedError as exc:
            raise click.ClickException(str(exc)) from exc

    lineage_watcher = LineageWatcher(monitoring_interval=interval)
    try:
        logger.info("Starting lineage watcher")
        lineage_watcher.start()
        # Keep the process alive; the watcher runs in a daemon thread. Block on
        # the stop event rather than sleep-looping so a shutdown signal wakes the
        # main thread immediately instead of after up to `interval` seconds.
        lineage_watcher.stop_event.wait()
    except KeyboardInterrupt:
        logger.info("Lineage watcher interrupted")
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"Lineage watcher exception: {e}")
    finally:
        logger.warning("Lineage watcher stopped!")
        lineage_watcher.stop()
