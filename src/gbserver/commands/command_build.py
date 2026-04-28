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

"""
The main command to run builds.
"""

import asyncio
import getpass
from pathlib import Path
from typing import List, Optional

import click

from gbcommon.uri.git import GitURI
from gbserver.build import Build
from gbserver.build.buildrun import BuildRun
from gbserver.build.space import Space
from gbserver.storage import singleton_storage
from gbserver.types.buildevent import BuildEvent
from gbserver.types.constants import truncate
from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

LOCAL_BUILD_USER = getpass.getuser() or "none"


def get_space_uri_from_space_name(space_name: str) -> Optional[str]:
    """Get the space config URI given the space name."""
    storage = singleton_storage.get_admin_storage().space_storage
    space = storage.get_by_name(space_name)
    if space is None:
        from gbserver.utils.optional_imports import HAS_LAKEHOUSE

        if HAS_LAKEHOUSE:
            from gbserver.storage.lh.space_storage import LhSpaceStorage

            is_lh = isinstance(storage, LhSpaceStorage)
        else:
            is_lh = False
        if is_lh:
            logger.error(
                "Could not find space %s in table %s.%s",
                space_name,
                storage.namespace,
                storage.table_name,
            )
        else:
            logger.error(
                "Could not find space %s in table %s",
                space_name,
                storage.get_table_name(),
            )
        return None
    uri = space.git_repo_uri
    logger.info("Found space %s with uri %s", space_name, uri)
    return GitURI.get_gb_space_config_uri(uri=uri)


def get_space(
    space_name: Optional[str],
    space_config_uri: Optional[str],
    username: str = LOCAL_BUILD_USER,
) -> Optional[Space]:
    """
    Get the Space instance constructed from one of the inputs.
    If the space repo uri is not a 'file://` uri, then make convert it for use by the build (to git+ssh...)
    Return None and issue an error message if there are any problems.
    """
    assert space_name is not None or space_config_uri is not None
    space = None
    if space_name is not None:
        if space_config_uri is not None:
            logger.warning(
                "space_name %s overriding space_config_uri %s. Ignoring the latter.",
                space_name,
                space_config_uri,
            )
        # Supporting only git uris in build management
        space_config_uri = get_space_uri_from_space_name(space_name=space_name)
        if space_config_uri is None:
            return None  # error message already issued.
    logger.info("Using Space uri %s", space_config_uri)
    assert space_config_uri is not None, "space_config_uri is None"
    space = Space(uri=space_config_uri, username=username)
    return space


@click.group("build")
def cli():
    """build."""


@cli.command()
@click.argument(
    "build_dir",
    default=Path(".").resolve(),
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--space-config-uri",
    help="URI pointing to a space config (not to be confused with a space:// URI)",
)
@click.option(
    "--space-name",
    default=None,
    type=str,
    help="Name of the registered space to use when running the build",
)
@click.option("--targets", "-t", multiple=True, help="List of targets to process.")
@click.option(
    "--user-name",
    default=LOCAL_BUILD_USER,
    type=str,
    help="Name of the user when running the build",
)
@click.option(
    "--cancel_on_error",
    is_flag=True,
    help="Cancel the entire build on first step error.",
)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    default=False,
    help="Do a dry run instead of a full build",
)
@pass_environment
def run(
    ctx: CliEnvironment,
    build_dir: Path,
    space_config_uri: Optional[str] = None,
    space_name: Optional[str] = None,
    targets: Optional[List[str]] = None,
    user_name: str = LOCAL_BUILD_USER,
    cancel_on_error: bool = False,
    dry_run: bool = False,
):
    """Run a build"""
    space = None
    if space_name is not None or space_config_uri is not None:
        space = get_space(space_name, space_config_uri, username=user_name)
        if space is None:
            return  # Error message issued.
    build = Build(
        build_dir=Path(build_dir),
        space=space,
        targets=targets,
        username=LOCAL_BUILD_USER,
    )
    build_run = BuildRun(build=build, cancel_on_error=cancel_on_error, dry_run=dry_run)
    asyncio.run(build_run.run_and_wait())


@cli.command()
@click.argument(
    "build_dir",
    default=Path(".").resolve(),
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--space-config-uri",
    help="URI pointing to a space config (not to be confused with a space:// URI)",
)
@click.option(
    "--space-name",
    default=None,
    type=str,
    help="Name of the registered space to use when running the build",
)
@click.option(
    "--targets",
    "-t",
    multiple=True,
    help="List of targets to process.",
)
@click.option(
    "--user-name",
    default=LOCAL_BUILD_USER,
    type=str,
    help="Name of the user when running the build",
)
@click.option(
    "--cancel_on_error",
    is_flag=True,
    help="Cancel the entire build on first step error.",
)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    default=False,
    help="Do a dry run instead of a full build",
)
@pass_environment
def run_and_monitor(
    ctx: CliEnvironment,
    build_dir: Path,
    space_config_uri: Optional[str] = None,
    space_name: Optional[str] = None,
    targets: Optional[List[str]] = None,
    user_name: str = LOCAL_BUILD_USER,
    cancel_on_error: bool = False,
    dry_run: bool = False,
):
    """Run a build and print all the emitted events (This DOES NOT terminate. Use Ctrl+C to terminate.)."""
    event_q: asyncio.Queue[asyncio.Event] = asyncio.Queue()
    space = None
    if space_name is not None or space_config_uri is not None:
        space = get_space(space_name, space_config_uri, username=user_name)
        if space is None:
            return  # Error message issued.

    build = Build(
        build_dir=build_dir,
        space=space,
        event_q=event_q,
        targets=targets,
        username=user_name,
    )
    build_run = BuildRun(
        build=build, event_q=event_q, cancel_on_error=cancel_on_error, dry_run=dry_run
    )

    async def run_build_and_wait_on_queue() -> None:
        """Run the build and wait for events forever."""
        build_run.async_run()
        while True:
            event = await event_q.get()
            assert isinstance(event, BuildEvent), f"invalid event: {event}"
            event_str = truncate(str(event))
            logger.info(
                "\x1b[0;35mGot a new event: %s : %s\x1b[0m", event.type, event_str
            )

    asyncio.run(run_build_and_wait_on_queue())
