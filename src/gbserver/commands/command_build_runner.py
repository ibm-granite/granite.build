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

"""Start the build-runner to manage a build."""

from pathlib import Path
from typing import Optional

import click

from gbserver.asset.assetstore import Assetstore
from gbserver.buildwatcher.build_utils import finalize_build_status
from gbserver.buildwatcher.buildrunner import BuildRunner
from gbserver.commands.utils import MutexOption
from gbserver.storage import singleton_storage
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.buildconfig import BUILD_FILENAME, BuildConfig
from gbserver.types.constants import (
    COMMAND_RUN_BUILD_WATCH_BUILD_NAME,
    DEFAULT_GH_API_ENDPOINT,
    DEFAULT_ROOT_WORKSPACE_DIR,
    ENV_VAR_DEFAULT_GITHUB_TOKEN,
    GBSERVER_GITHUB_TOKEN,
    PUBLIC_SPACE_NAME,
)
from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def load_build(
    build_dir: Path, space_name: str, username: str, targets: Optional[list[str]]
) -> Optional[StoredBuild]:
    """Load a build from the build directory."""
    space = singleton_storage.get_admin_storage().space_storage.get_by_name(name=space_name)
    if space is None:
        logger.error("Could not find space with name %s in space storage", space_name)
        return None
    build_path = Path(build_dir).resolve()
    assert build_path.is_dir(), f"build_path {build_path} is not a valid directory"
    build_yaml_path = build_path / BUILD_FILENAME
    build_config = BuildConfig.from_yaml(path=build_yaml_path)
    logger.debug("build_config: %s", build_config)
    if isinstance(targets, tuple):
        targets = list(targets)
    if targets is not None and len(targets) == 0:
        targets = None
    stored_build = StoredBuild.create(
        name=COMMAND_RUN_BUILD_WATCH_BUILD_NAME,
        space_name=space_name,
        source_uri="",
        username=username,
        build_yaml_path=build_yaml_path,
        targets=targets,
    )
    logger.info(
        """Created inmemory build using...
    build: %s
    uuid: %s
    user name: %s
    space name: %s
    targets: %s""",
        build_yaml_path,
        stored_build.uuid,
        stored_build.username,
        stored_build.space_name,
        stored_build.targets,
    )
    return stored_build


@click.command(context_settings={"show_default": True})
@click.option(
    "--build-id",
    type=str,
    help=f"Id of a {Status.PENDING} build in build storage to run.",
    cls=MutexOption,
    not_required_if=["build-dir"],
)
@click.option(
    "--build-dir",
    type=click.Path(exists=True),
    help="Directory holding a single build to be run by the build watcher.",
    cls=MutexOption,
    not_required_if=["build-id"],
)
@click.option(
    "--username",
    default="gb-local-user",
    type=str,
    help="Username assigned to a build loaded from a directory.",
    cls=MutexOption,
    not_required_if=["build-id"],
)
@click.option(
    "--space-name",
    default=PUBLIC_SPACE_NAME,
    type=str,
    help="Name of the space assigned to a build loaded from a directory.",
    cls=MutexOption,
    not_required_if=["build-id"],
)
@click.option(
    "--target",
    "-t",
    multiple=True,
    help="One or more targets to process from a build loaded from a directory.",
    cls=MutexOption,
    not_required_if=["build-id"],
)
@click.option(
    "--gh-token",
    default=GBSERVER_GITHUB_TOKEN,
    type=str,
    show_default=False,
    help=f"Set the token to use with GitHub. If not provided we will skip GitHub logging. Default is defined by {ENV_VAR_DEFAULT_GITHUB_TOKEN} env var.",
)
@click.option(
    "--monitoring-interval",
    default=5,
    type=int,
    show_default=True,
    help="Sets the interval (in seconds) between event processing and other build monitoring operations",
)
@click.option(
    "--workspace-dir",
    required=False,
    default=Path(DEFAULT_ROOT_WORKSPACE_DIR),
    type=click.Path(),
    help="Workspace directory to use to run the build.",
)
@click.option(
    "--asset-stores-dir",
    type=click.Path(exists=True),
    help="Path to asset stores config dir",
)
@click.option(
    "--space-config-uri",
    help="URI pointing to a space assigned to the build. Overrides --space-name.",
)
@click.option(
    "--gh-api-endpoint",
    help="URI pointing to a github API to use to add updates to the build's PR.",
    default=DEFAULT_GH_API_ENDPOINT,
    cls=MutexOption,
    not_required_if=["build-dir"],
)
@click.option(
    "--ignore-build-not-pending",
    is_flag=True,
    help=f"Run stored build under a given build id even if its status is not {Status.PENDING}.  Primarily for debugging.",
    cls=MutexOption,
    not_required_if=["build-dir"],
)
@click.option(
    "--create-pr",
    "create_pr",
    is_flag=True,
    default=False,
    help="Create a PR for the build during setup.",
)
@click.option(
    "--enable-resume",
    "enable_resume",
    is_flag=True,
    default=False,
    help="Allow resuming an already RUNNING build instead of failing or restarting it.",
)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    default=False,
    help="Do a dry run instead of a full build",
)
@pass_environment
def cli(
    ctx: CliEnvironment,
    gh_token: str,
    build_id: str,
    workspace_dir: Path,
    space_name: str,
    username: str,
    build_dir: Path,
    asset_stores_dir: Path,
    space_config_uri: Optional[str],
    ignore_build_not_pending: bool,
    target: Optional[list[str]],
    monitoring_interval: int,
    gh_api_endpoint: str,
    create_pr: bool,
    enable_resume: bool,
    dry_run: bool = False,
):
    """Start build in build storage or loaded from a specified directory"""
    if asset_stores_dir:
        logger.info("loading assets from path: %s", asset_stores_dir)
        Assetstore.load_assetstores_from_dir(Path(asset_stores_dir))

    # Get the StoredBuild.
    build_storage = singleton_storage.get_admin_storage().build_storage
    if build_id is not None:
        stored_build: StoredBuild = build_storage.get_by_uuid(build_id)  # type: ignore[assignment]
        if stored_build is None:
            logger.error("Could not find build with id %s in build storage", build_id)
            return
        if stored_build.status != Status.PENDING:
            if ignore_build_not_pending:
                logger.warning(
                    "Build with id %s has status %s. Resetting to PENDING and running.",
                    build_id,
                    stored_build.status,
                )
                stored_build.status = Status.PENDING
                # BuildRunner won't process events if the build "is_finished()" so make sure it is not.
                stored_build = singleton_storage.get_admin_storage().build_storage.update_fields(  # type: ignore[assignment]
                    stored_build.uuid,
                    {"status": stored_build.status},
                )
            else:  # A non-pending build.  Something wrong here. CANCEL_REQUESTED, FAILED,
                if stored_build.status in [Status.RUNNING, Status.CANCEL_REQUESTED]:
                    msg = f"Build with id {build_id} has status {stored_build.status}, Marking as {Status.FAILED.name}"
                    logger.error("%s", msg)
                    finalize_build_status(
                        build_id=build_id, status=Status.FAILED, failure_reason=msg
                    )
                    # set_build_status(build_id=build_id, status=Status.FAILED)
                else:
                    logger.error(
                        "Build with id %s has unexpected status %s != %s. Ignoring.",
                        build_id,
                        stored_build.status,
                        Status.PENDING,
                    )
                return
    else:
        stored_build = load_build(
            build_dir=build_dir,
            space_name=space_name,
            username=username,
            targets=target,
        )
        if stored_build is None:
            return  # And error message was already issued

    # Start the build.
    build_runner = BuildRunner(
        build=stored_build,
        gh_api_endpoint=gh_api_endpoint,
        monitoring_interval=monitoring_interval,
        gh_token=gh_token,
        workspace_dir=workspace_dir,
        space_uri=space_config_uri,
        create_pr=create_pr,  # space_name is ignored if providing a space_uri
        enable_resume=enable_resume,
        dry_run=dry_run,
    )
    build_runner.start_and_wait()  # Returns on completion, cancellation or failure.

    finished_build_id = stored_build.uuid
    finished_stored_build: StoredBuild = build_storage.get_by_uuid(finished_build_id)  # type: ignore[assignment]
    if finished_stored_build is None:
        # This should NEVER be the case, but we are occasionally seeing this with LH.
        logger.error("Build with id %s could not be found after completion?!", finished_build_id)
    else:
        logger.info(
            "Build with id %s completed with status=%s",
            finished_build_id,
            finished_stored_build.status,
        )
