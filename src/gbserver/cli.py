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

"""The entrypoint and CLI interface root command."""

import importlib
import os
import traceback
from pathlib import Path
from typing import List, Optional, Self

import click

from gbserver.storage import singleton_storage
from gbserver.types.constants import (
    CONTEXT_SETTINGS,
    DEFAULT_DIR_PERMS,
    DEFAULT_LOG_LEVEL,
    ENV_VAR_GBSERVER_ADMIN_TABLE_PREFIX,
)
from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


class GraniteBuildServerCLI(click.Group):
    """
    The root command class.
    This find the other sub-commands dynamically.
    """

    def list_commands(self: Self, ctx: click.Context) -> List[str]:
        rv = []
        command_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "commands")
        )
        for filename in os.listdir(command_folder):
            if filename.endswith(".py") and filename.startswith("command_"):
                cmd = filename[8:-3]
                cmd = cmd.replace("_", "-")
                rv.append(cmd)
        rv.sort()
        return rv

    def get_command(
        self: Self, ctx: click.Context, cmd_name: str
    ) -> Optional[click.Command]:
        try:
            cmd_name = cmd_name.replace("-", "_")
            mod = importlib.import_module(f"gbserver.commands.command_{cmd_name}")
        except ImportError as e:
            logger.error("%s", traceback.format_exc())
            logger.error(e)
            return None
        return mod.cli


@click.command(cls=GraniteBuildServerCLI, context_settings=CONTEXT_SETTINGS)
@click.option(
    "--log-level",
    default=DEFAULT_LOG_LEVEL.lower(),
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    help="Set the logging level",
)
@click.option(
    "--log-file",
    default="",
    type=str,
    help="Path to a file where logs will be saved.",
)
@click.option(
    "--gb-admin-table-prefix",
    default=None,
    type=str,
    help="A common prefix to use for the admin table names - used primarily for testing.",
)
@click.option(
    "--server-runtime-config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to a server runtime config file",
)
@pass_environment
def gbserver(
    ctx: CliEnvironment,
    log_level: str,
    log_file: str,
    gb_admin_table_prefix: str,
    server_runtime_config: Optional[Path],
):
    """Granite.Build command line interface."""
    # NOTE: The server_runtime_config flag here is a placeholder to avoid click parsing errors.
    # We process server_runtime_config earlier in the gbserverenvconfig.py
    ctx.log_level = log_level.lower()
    if log_file != "":
        ctx.log_path = Path(log_file).resolve()
        ctx.log_path.parent.mkdir(mode=DEFAULT_DIR_PERMS, parents=True, exist_ok=True)
    configure_logging(
        level=ctx.log_level,
        log_file=str(ctx.log_path) if ctx.log_path is not None else None,
    )
    if gb_admin_table_prefix is not None:
        logger.warning(
            "Global admin table name prefix set to '%s'!!!", gb_admin_table_prefix
        )
        # The environment variable is passed to a child process
        os.environ[ENV_VAR_GBSERVER_ADMIN_TABLE_PREFIX] = gb_admin_table_prefix
        singleton_storage.set_storage_prefix(gb_admin_table_prefix)
