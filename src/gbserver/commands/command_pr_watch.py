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

"""Command pr watch module."""

import os
import sys
from pathlib import Path
from typing import Optional

import click

from gbserver.github.githubmanager import GitHubManager
from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


@click.command()
@click.option(
    "--gh-token",
    default=lambda: os.environ.get("GBSERVER_GITHUB_TOKEN", ""),
    type=str,
    help="Set the token to use with GitHub. If not provided we will skip GitHub monitoring.",
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
@pass_environment
def cli(
    ctx: CliEnvironment,
    gh_token: str,
    config: Optional[Path],
    watch: bool,
):
    """Start the PR Watcher/Monitor."""
    if gh_token == "":
        logger.error("no GitHub token specified, GitHub monitoring exiting!")
        sys.exit(1)
    gh_manager: Optional[GitHubManager] = None
    try:
        logger.info("Starting github monitor")
        logger.info("pr_watcher_config: %s", config)
        gh_manager = GitHubManager(
            token=gh_token,
            config_path=config,
            watch_for_config_changes=watch,
        )
        logger.debug("gh_manager: %s", gh_manager)
        gh_manager.start_and_wait()
    finally:
        logger.warning("GitHubManager stopped!")
        if gh_manager is not None:
            gh_manager.stop()
