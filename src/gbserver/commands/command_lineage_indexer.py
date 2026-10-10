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

"""``gbserver lineage-indexer`` — incrementally fill ``gb_lineage_index``.

Walks jobs by ``(timestamp, job_id)`` -- ``gb_targets`` in standalone, the
configured lineage store everywhere else -- and writes them to the index. See :mod:`gbserver.lineage.indexer`.

Outside standalone it indexes nothing until ``gbserver lineage-index-init`` seeds
its checkpoint; it re-reads the checkpoint every scan, so no restart is needed.
"""

import traceback

import click

from gbserver.lineage.indexer import create_indexer, resolve_indexer_source
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
    help="Seconds between index scans.",
)
@pass_environment
def cli(ctx: CliEnvironment, interval: float):
    """Start the lineage indexer."""
    source = resolve_indexer_source()
    indexer = create_indexer(source, monitoring_interval=interval)

    if indexer is None:
        return

    try:
        logger.info("Starting lineage indexer (source=%s)", source)
        indexer.start()
        indexer.stop_event.wait()
    except KeyboardInterrupt:
        logger.info("Lineage indexer interrupted")
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"Lineage indexer exception: {e}")
    finally:
        logger.warning("Lineage indexer stopped!")
        indexer.stop()
