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
"""

import traceback

import click

from gbserver.lineage.jobstats import get_lineage_store
from gbserver.lineage.lineage_watcher import LineageWatcher
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
@pass_environment
def cli(ctx: CliEnvironment, interval: float):
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
