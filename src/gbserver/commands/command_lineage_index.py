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

"""``gbserver lineage-index`` — incrementally fill ``gb_lineage_index``.

Reads new lineage from ``gb_build`` (``--source admin_db``, standalone default) or
from the configured lineage store (``--source lineage_store``, default elsewhere)
and writes it to the index. See :mod:`gbserver.lineage.indexer`.
"""

import traceback

import click

from gbserver.lineage.indexer import (
    INDEXER_CHECKPOINT_KEY,
    INDEXER_SOURCE_ADMIN_DB,
    VALID_INDEXER_SOURCES,
    UnknownIndexerSource,
    create_indexer,
    resolve_indexer_source,
)
from gbserver.lineage.lineage_seeding import LineageSeedError, seed_if_absent
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
    help="Seconds between index scans.",
)
@click.option(
    "--source",
    required=False,
    type=click.Choice(VALID_INDEXER_SOURCES),
    default=None,
    help=(
        "Where to read lineage from. Defaults to GBSERVER_LINEAGE_INDEXER_SOURCE, "
        "else admin_db in standalone and lineage_store elsewhere."
    ),
)
@click.option(
    "--base-build-id",
    required=False,
    type=str,
    default=None,
    help=(
        "admin_db only: seed the indexer checkpoint when absent. 'from-latest', "
        "'all', or a build id. Never overwrites an existing checkpoint."
    ),
)
@pass_environment
def cli(ctx: CliEnvironment, interval: float, source: str, base_build_id: str):
    """Start the lineage indexer."""
    try:
        source = resolve_indexer_source(source)
    except UnknownIndexerSource as exc:
        raise click.ClickException(str(exc)) from exc

    if base_build_id is not None:
        if source != INDEXER_SOURCE_ADMIN_DB:
            raise click.ClickException(
                "--base-build-id only applies to --source admin_db."
            )
        if not base_build_id.strip():
            raise click.ClickException(
                "--base-build-id was given an empty value; pass 'from-latest', "
                "'all', or a build id."
            )
        try:
            seed_if_absent(
                get_admin_storage(), base_build_id, key=INDEXER_CHECKPOINT_KEY
            )
        except LineageSeedError as exc:
            raise click.ClickException(str(exc)) from exc

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
