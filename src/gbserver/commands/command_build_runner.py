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

import os
import signal
import threading
from pathlib import Path
from typing import Optional

import click
from click.core import ParameterSource

from gbserver.asset.assetstore import Assetstore
from gbserver.buildrunner.build_utils import finalize_build_status
from gbserver.buildrunner.buildrunner import BuildRunner
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
    MIN_MONITORING_INTERVAL_SECONDS,
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
    space = singleton_storage.get_admin_storage().space_storage.get_by_name(
        name=space_name
    )
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


#: Options that all select the build's space; at most one may be provided.
_SPACE_OPTION_NAMES = ("space_name", "space_config_uri", "space_dir")


def _check_space_options_exclusive() -> None:
    """Ensure at most one space-selecting option was explicitly provided.

    ``--space-name``, ``--space-config-uri`` and ``--space-dir`` all choose the
    build's space, so only one may be set. ``--space-name`` has a default, so an
    option is only counted when the user set it explicitly (command line or
    environment), not when it falls back to its default. This is enforced here
    rather than via ``MutexOption`` because that helper matches option names
    without normalizing hyphens to underscores, so it currently never fires.

    Raises:
        click.UsageError: if more than one of the space options was explicitly
            provided.
    """
    ctx = click.get_current_context()
    provided = [
        name
        for name in _SPACE_OPTION_NAMES
        if ctx.get_parameter_source(name) != ParameterSource.DEFAULT
    ]
    if len(provided) > 1:
        flags = ", ".join("--" + name.replace("_", "-") for name in provided)
        raise click.UsageError(
            f"Options {flags} are mutually exclusive; provide only one."
        )


def _apply_termination(build_runner: BuildRunner, signum: int) -> None:
    """Cancel (SIGINT) or fail (SIGTERM) the running build.

    Args:
        build_runner (BuildRunner): the runner whose build should be terminated.
        signum (int): the received signal number.
    """
    if signum == signal.SIGTERM:
        logger.warning("Received SIGTERM - marking build as FAILED...")
        click.echo("\nReceived SIGTERM - failing build...", err=True)
        build_runner.stop_and_fail("Build runner received SIGTERM")
    else:  # SIGINT / Ctrl+C
        logger.warning("Ctrl+C received - cancelling build...")
        click.echo("\nCancelling build (Ctrl+C)...", err=True)
        build_runner.stop()


def run_build_handling_signals(build_runner: BuildRunner) -> None:
    """Run the build to completion, reacting to termination signals.

    ``BuildRunner.start_and_wait`` blocks until the build finishes, so it runs on
    a background (daemon) thread while the main thread waits. SIGINT (Ctrl+C)
    cancels the build (CANCELLED); SIGTERM fails it (FAILED). Both call into the
    runner from the main thread - the runner's stop()/stop_and_fail() are designed
    to be invoked from another thread - and the main thread keeps waiting (in
    bounded steps) until the build thread finishes, so the terminal status is fully
    persisted before returning.

    A *second* termination signal force-exits the process. Graceful shutdown can
    hang (e.g. a stuck workload teardown); swallowing repeat signals would then
    leave the process unkillable via Ctrl+C, so the second signal is honored as a
    hard exit. The build thread is a daemon so this exit is never blocked by it.

    Args:
        build_runner (BuildRunner): the configured BuildRunner whose build should
            be run.

    Raises:
        Exception: re-raises on the main thread any exception raised by
            ``start_and_wait`` on the background thread (e.g. a storage failure or
            a re-raised build exception). Without this, the exception would go to
            threading.excepthook, the wait would return normally, and the CLI would
            exit 0 on a build that never ran.
    """
    received: dict[str, int] = {}
    error: dict[str, BaseException] = {}

    def _run_build():
        # Capture any failure so it can be re-raised on the main thread after the
        # thread is joined; a bare thread target would otherwise swallow it (exit 0).
        try:
            build_runner.start_and_wait()
        except Exception as exc:  # noqa: BLE001 - re-raised verbatim below
            error["exc"] = exc

    def _record_signal(signum, _frame):
        # Keep the handler tiny; the main loop below does the real work. A repeat
        # signal means the operator wants out now (graceful shutdown is slow or
        # hung), so hard-exit instead of swallowing it. os._exit is async-signal-
        # safe and bypasses the (possibly stuck) build thread.
        if received:
            os._exit(128 + signum)
        received["signum"] = signum

    prev_int = signal.signal(signal.SIGINT, _record_signal)
    prev_term = signal.signal(signal.SIGTERM, _record_signal)
    build_thread = threading.Thread(
        target=_run_build, name="build-runner-cli", daemon=True
    )
    terminating = False
    try:
        build_thread.start()
        while build_thread.is_alive():
            # Bounded join so the main thread stays responsive to signals rather
            # than blocking indefinitely in an unbounded join().
            build_thread.join(timeout=0.5)
            if received and not terminating:
                terminating = True
                _apply_termination(build_runner, received["signum"])
                # Keep looping (bounded) until the build thread finishes; a repeat
                # signal now force-exits via _record_signal above.
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)

    # Propagate a build/storage failure so the CLI exits non-zero (a graceful
    # SIGINT/SIGTERM leaves start_and_wait returning normally, so nothing is
    # captured and the caller proceeds to log the final status as before).
    if "exc" in error:
        raise error["exc"]


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
    type=click.IntRange(min=MIN_MONITORING_INTERVAL_SECONDS),
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
    help="URI pointing to a space assigned to the build. Mutually exclusive with "
    "--space-name and --space-dir.",
)
@click.option(
    "--space-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to a local space directory (a directory containing a space.yaml). "
    "Convenience form of --space-config-uri for a file-based space; resolved to a "
    "file:// URI. Mutually exclusive with --space-name and --space-config-uri.",
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
    space_dir: Optional[str],
    ignore_build_not_pending: bool,
    target: Optional[list[str]],
    monitoring_interval: int,
    gh_api_endpoint: str,
    create_pr: bool,
    dry_run: bool = False,
):
    """Start build in build storage or loaded from a specified directory"""
    _check_space_options_exclusive()

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
                _updated = (
                    singleton_storage.get_admin_storage().build_storage.update_fields(
                        stored_build.uuid,
                        {"status": stored_build.status},
                    )
                )
                assert isinstance(_updated, StoredBuild)
                stored_build = _updated
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

    # --space-dir is a convenience form of --space-config-uri for a local,
    # file-based space directory; resolve it to an absolute file:// URI (the
    # two options are mutually exclusive, so only one is ever set).
    if space_dir is not None:
        space_config_uri = Path(space_dir).resolve().as_uri()
        logger.info(
            "using space directory %s as space URI %s", space_dir, space_config_uri
        )

    # Start the build.
    build_runner = BuildRunner(
        build=stored_build,
        gh_api_endpoint=gh_api_endpoint,
        monitoring_interval=monitoring_interval,
        gh_token=gh_token,
        workspace_dir=workspace_dir,
        space_uri=space_config_uri,
        create_pr=create_pr,  # space_name is ignored if providing a space_uri
        dry_run=dry_run,
    )
    # Runs on a background thread so SIGINT/SIGTERM can cancel/fail the build.
    # Returns on completion, cancellation or failure (a build/storage exception is
    # re-raised here and exits non-zero).
    run_build_handling_signals(build_runner)

    # Report on the FINAL build in the (possibly retried) chain, not the original:
    # the retry loop reassigns build_runner.stored_build to each retry, so its
    # status is the outcome that determines the exit code.
    finished_build_id = build_runner.stored_build.uuid
    finished_stored_build: StoredBuild = build_storage.get_by_uuid(finished_build_id)  # type: ignore[assignment]
    if finished_stored_build is None:
        # This should NEVER be the case, but we are occasionally seeing this with LH.
        logger.error(
            "Build with id %s could not be found after completion?!", finished_build_id
        )
        raise SystemExit(1)

    logger.info(
        "Build with id %s completed with status=%s",
        finished_build_id,
        finished_stored_build.status,
    )
    # Exit non-zero on a failed build so callers (e.g. orchestrators keying off the
    # exit code) don't read a failure as success. SUCCESS and CANCELLED (a
    # deliberate SIGINT cancel) exit 0; FAILED (incl. SIGTERM) and INVALID do not.
    if finished_stored_build.status in (Status.FAILED, Status.INVALID):
        raise SystemExit(1)
