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


import os
import traceback
from pathlib import Path
from typing import Optional

import click

from gbserver.asset.assetstore import Assetstore
from gbserver.buildwatcher.buildwatcher import BuildWatcher
from gbserver.lineage.jobstats import get_lineage_store
from gbserver.lineage.lineage_watcher import LineageWatcher
from gbserver.types.constants import ENV_VAR_DEFAULT_GITHUB_TOKEN, GBSERVER_GITHUB_TOKEN
from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


@click.command()
@click.option(
    "--gh-token",
    default=GBSERVER_GITHUB_TOKEN,
    type=str,
    show_default=False,
    help=f"Set the token to use with GitHub. If not provided we will skip GitHub monitoring. Default is defined by {ENV_VAR_DEFAULT_GITHUB_TOKEN} env var.",
)
@click.option(
    "--config",
    required=False,
    type=click.Path(path_type=Path, exists=True, file_okay=True, dir_okay=False),
    default=None,
    help="Path to a config file.",
)
@click.option(
    "--watch/--no-watch",
    required=False,
    type=bool,
    default=True,
    help="Whether to watch the config file for changes.",
)
@click.option(
    "--asset_stores_dir",
    type=click.Path(exists=True),
    help="Path to asset stores config dir",
)
@pass_environment
def cli(
    ctx: CliEnvironment,
    gh_token: str,
    config: Optional[Path],
    watch: bool,
    asset_stores_dir: Path,
):
    """Start the pending build watcher/monitor."""
    if asset_stores_dir:
        logger.info("loading assets from path: %s", asset_stores_dir)
        Assetstore.load_assetstores_from_dir(Path(asset_stores_dir))
    build_watcher: Optional[BuildWatcher] = None
    lineage_watcher: Optional[LineageWatcher] = None
    try:
        logger.info("Starting build monitor")
        build_watcher = BuildWatcher(
            config_path=config,
            watch_for_config_changes=watch,
            gh_token=gh_token,
        )

        # Start the lineage watcher (records lineage asynchronously from gb_events).
        # Only start if the configured lineage store records to a centralized store
        # (skip for noop stores in standalone mode).
        store = get_lineage_store()
        if store.records_centralized_lineage:
            logger.info("Starting lineage watcher")
            lineage_watcher = LineageWatcher()
            lineage_watcher.start()

        build_watcher.start_and_wait()
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"Build monitor exception from start_and_wait(): {e}")
    finally:
        logger.warning("Build monitor stopped!")
        if lineage_watcher is not None:
            lineage_watcher.stop()
        if build_watcher is not None:
            build_watcher.stop()
