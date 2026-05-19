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


import subprocess
from typing import Optional

import click

from gbserver.commands.utils import set_failed_build_status
from gbserver.storage import singleton_storage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.constants import (
    GB_ENVIRONMENT,
    GB_ENVIRONMENT_CONFIG,
    GB_METADATA_STORAGE,
    GBSERVER_SQL_DBNAME,
    GBSERVER_SQL_SCHEMA,
)
from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.types.status import Status


@click.command(context_settings={"show_default": True})
@click.option(
    "--operation",
    type=click.Choice(
        [
            "fail-zombie-builds",
            "fix-zombie-targets",
            "fix-zombie-steps",
            "fail-pending-without-pr",
        ]
    ),
    help=f"Controls the type of operation to perform. ",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help=f"If not set then make changes to the tables.  If set then only show what would be done.",
)
@pass_environment
def cli(
    ctx: CliEnvironment,
    operation: str,
    dry_run: bool,
):
    """
    Provides ability make edits to the GB admin metadata tables. Available operations include,
    fail-zombie-builds:  Marks all builds that are RUNNING but have no active pods/jobs as FAILED.
    fix-zombie-targets:  For targets that are RUNNING, if their builds are not RUNNING, then copy the build status.
    fix-zombie-steps:  For steps that are RUNNING, if their builds are not RUNNING, then copy the build status.
    fail-pending-without-pr: Mark PENDING builds that do not have a PR uri as FAILED.
    """
    print(f"\nPerforming admin table operations with the following:")
    print(f"Operation             : {operation}")
    print(f"Dry run               : {dry_run}")
    print(f"G.B environment       : {GB_ENVIRONMENT}")
    print(f"Admin/metadata storage: {GB_METADATA_STORAGE}")
    print(f"OpenShift project     : {GB_ENVIRONMENT_CONFIG.default_pod_namespace}")
    print(f"SQL db.schema         : {GBSERVER_SQL_DBNAME}.{GBSERVER_SQL_SCHEMA}")

    print("Do you want to proceed? (yes/no)")
    user_input = input().lower()
    if user_input != "yes":
        print("Operation aborted!")
        return

    if operation == "fail-zombie-builds":
        fail_zombie_builds(dry_run)
    elif operation == "fix-zombie-targets":
        fix_zombie_targets(dry_run)
    elif operation == "fix-zombie-steps":
        fix_zombie_steps(dry_run)
    elif operation == "fail-pending-without-pr":
        fail_pending_without_pr(dry_run)
    else:
        print(f"Invalid operation: {operation}")


def fix_zombie_steps(dry_run: bool):
    running_steps = get_steps(status=Status.RUNNING)
    total_steps = len(running_steps)
    print(f"Examining build status of {total_steps} RUNNING steps")
    # Group the steps by build.
    steps_by_build_id = group_by_build_id(running_steps)

    # Update steps per build.
    step_storage = singleton_storage.get_admin_storage().step_storage
    build_storage = singleton_storage.get_admin_storage().build_storage
    updated = []  # type: ignore[var-annotated]
    failed = []  # type: ignore[var-annotated]
    running = []
    deleted = []  # type: ignore[var-annotated]
    index = 0
    for build_id, steps in steps_by_build_id.items():
        index += len(steps)
        print(
            f"{index}/{total_steps}: Processing {len(steps)} steps from build {build_id}",
            end="",
        )
        build = build_storage.get_by_uuid(build_id)
        assert isinstance(build, StoredBuild)
        print(f" {build.source_uri}")
        if build is None:
            delete_items_without_a_build(steps, step_storage, deleted, failed, dry_run)
            continue
        assert isinstance(build, StoredBuild)
        if build.status == Status.RUNNING:
            ids = [step.uuid for step in steps]
            print(
                f"Skipping running steps {ids} that are part of running build {build.uuid}"
            )
            running.extend(ids)
            continue
        for step in steps:
            copy_build_status(build, step, step_storage, updated, failed, dry_run)
    print(f"{len(deleted)} steps deleted \n{deleted}")
    print(f"{len(updated)} steps updated \n{updated}")
    if len(failed) > 0:
        print(f"{len(failed)} steps failed to update\n{failed}")
    print(f"{len(running)} running steps not updated.\n{running}")


def fix_zombie_targets(dry_run: bool):
    running_targets = get_targets(status=Status.RUNNING)
    total_targets = len(running_targets)
    print(f"Examining build status of {total_targets} RUNNING targets")
    # Group the targets by build.
    targets_by_build_id = group_by_build_id(running_targets)

    # Update targets per build.
    target_storage = singleton_storage.get_admin_storage().target_storage
    build_storage = singleton_storage.get_admin_storage().build_storage
    updated = []  # type: ignore[var-annotated]
    failed = []  # type: ignore[var-annotated]
    running = []
    deleted = []  # type: ignore[var-annotated]
    index = 0
    for build_id, targets in targets_by_build_id.items():
        index += len(targets)
        print(
            f"{index}/{total_targets}: Processing {len(targets)} targets from build {build_id}",
            end="",
        )
        build = build_storage.get_by_uuid(build_id)
        assert isinstance(build, StoredBuild)
        print(f" {build.source_uri}")
        if build is None:
            delete_items_without_a_build(
                targets, target_storage, deleted, failed, dry_run
            )
            continue
        assert isinstance(build, StoredBuild)
        if build.status == Status.RUNNING:
            ids = [target.uuid for target in targets]
            print(
                f"Skipping running targets {ids} that are part of running build {build.uuid}"
            )
            running.extend(ids)
            continue
        for target in targets:
            copy_build_status(build, target, target_storage, updated, failed, dry_run)
    print(f"{len(deleted)} targets deleted \n{deleted}")
    print(f"{len(updated)} targets updated \n{updated}")
    if len(failed) > 0:
        print(f"{len(failed)} targets failed to update\n{failed}")
    print(f"{len(running)} running targets not updated.\n{running}")


def group_by_build_id(targets_or_steps: list):
    by_build_id = {}  # type: ignore[var-annotated]
    for item in targets_or_steps:
        targets = by_build_id.get(item.build_id, None)
        if targets is None:
            targets = []
            by_build_id[item.build_id] = targets
        targets.append(item)
    return by_build_id


def delete_items_without_a_build(
    targets_or_steps: list, storage, deleted, failed, dry_run
):
    if len(targets_or_steps) == 0:  # Should not happend
        return
    is_target = isinstance(targets_or_steps[0], StoredTargetRun)
    assert is_target or isinstance(targets_or_steps[0], StoredStepRun)
    item_name = "target" if is_target else "step"
    for item in targets_or_steps:
        msg = f"Deleting {item_name} {item.uuid} that does not have an associated build {item.build_id}."
        if dry_run:
            print(f"NOT {msg}")
            deleted.append(item.uuid)
        else:
            try:
                print(msg)
                storage.delete(item.uuid)
                deleted.append(item.uuid)
            except Exception as e:
                print(f"Failed {msg}: {e}")
                failed.append(item.uuid)


def copy_build_status(build, target_or_step, storage, updated, failed, dry_run):
    is_target = isinstance(target_or_step, StoredTargetRun)
    assert is_target or isinstance(target_or_step, StoredStepRun)
    item_name = "target" if is_target else "step"
    item = storage.get_by_uuid(target_or_step.uuid)
    if item.status.is_finished():
        print(f"{item_name} {item.uuid} is no longer RUNNING. Skipping update.")
        return
    msg = f"Updating {item_name} {target_or_step.uuid} with staus={target_or_step.status} of build {build.uuid} to build status={build.status}"
    target_or_step.status = build.status
    if dry_run:
        print(f"NOT {msg}")
        updated.append(target_or_step.uuid)
    else:
        try:
            print(msg)
            storage.update(target_or_step, update_updated_time=False)
            updated.append(target_or_step.uuid)
        except Exception as e:
            print(f"Failed {msg}: {e}")
            failed.append(target_or_step.uuid)


def fail_pending_without_pr(dry_run: bool):
    pending_builds = get_builds(status=Status.PENDING)
    print(f"Found {len(pending_builds)} pending builds.")
    updated = []
    failed = []
    running = []  # type: ignore[var-annotated]

    for build in pending_builds:
        if build.source_uri is not None and build.source_uri != "":
            continue
        build_id = build.uuid
        msg = f"Marking build {build_id} without a PR uri as failed."
        try:
            if dry_run:
                print(f"NOT {msg}")
            else:
                print(msg)
                set_failed_build_status(build_id)
            updated.append(build_id)
        except Exception as e:
            print(f"Failed {msg}")
            failed.append(build_id)
    print(f"{len(updated)} builds updated \n{updated}")
    if len(failed) > 0:
        print(f"{len(failed)} builds failed to update\n{failed}")
    print(f"{len(running)} running builds not updated.\n{running}")


def get_zombie_build_ids() -> tuple[bool, list[str]]:
    project = GB_ENVIRONMENT_CONFIG.default_pod_namespace
    if not oc_project(project):
        return False, []  # And error message was issued.

    # Get the list of pods BEFORE  asking for the RUNNING builds,
    success, pods_and_jobs_before = get_pods_and_jobs()
    if not success:
        return False, []  # And error message was issued.

    print(f"Looking up RUNNING builds")
    running_builds = get_builds(status=Status.RUNNING)
    build_count = len(running_builds)
    print(f"Found {build_count} running builds.")

    # And get the list of pods AFTER asking for the RUNNING builds, to avoid a raice condition
    success, pods_and_jobs_after = get_pods_and_jobs()
    if not success:
        return False, []  # And error message was issued.

    zombies_ids = []
    for build in running_builds:
        build_id = build.uuid
        if not build_id in pods_and_jobs_before and not build_id in pods_and_jobs_after:
            zombies_ids.append(build_id)
    return True, zombies_ids


def fail_zombie_builds(dry_run: bool):

    updated = []
    failed = []
    running = []  # type: ignore[var-annotated]

    success, zombie_build_ids = get_zombie_build_ids()
    if not success:
        return  # And error message was issued
    zombie_count = len(zombie_build_ids)
    print(f"Found {zombie_count} zombie builds.")

    index = 0
    for build_id in zombie_build_ids:
        index += 1
        msg = f"{index}/{zombie_count}: Marking RUNNING build {build_id} which has no pod or job as failed."
        try:
            if dry_run:
                print(f"NOT {msg}")
            else:
                print(f"{msg}")
                set_failed_build_status(build_id)
            updated.append(build_id)
        except Exception as e:
            print(f"Failed {msg}: {e}")
            failed.append(build_id)
    print(f"{len(updated)} builds updated \n{updated}")
    if len(failed) > 0:
        print(f"{len(failed)} builds failed to update\n{failed}")
    print(f"{len(running)} running builds not updated.\n{running}")


def get_pods_and_jobs() -> tuple[bool, str]:
    success, running_pods = oc_get(
        obj="pods", check_presence="gbserver-build-watch"
    )  # A sanity check on oc API
    if not success:
        return False, ""
    success, running_jobs = oc_get(obj="jobs", check_presence="gb-build-runner")
    return success, running_pods + running_jobs


def get_builds(status: Status) -> list[StoredBuild]:
    build_storage = singleton_storage.get_admin_storage().build_storage
    builds = build_storage.get_by_where({"status": status.name})
    return builds


def get_targets(status: Status) -> list[StoredTargetRun]:
    target_storage = singleton_storage.get_admin_storage().target_storage
    items = target_storage.get_by_where({"status": status.name})
    return items


def get_steps(status: Status) -> list[StoredBuild]:
    step_storage = singleton_storage.get_admin_storage().step_storage
    items = step_storage.get_by_where({"status": status.name})
    return items  # type: ignore[return-value]


def oc_get(obj: str, check_presence: Optional[str]) -> tuple[bool, str]:
    """Get the full stdout of the 'oc get <obj>' command.

    Args:
        obj (str): one of jobs or pods.
        check_presence (Optional[str]): string to make sure is in the stdout.

    Returns:
        bool: indicates if command succeeded.
        str: full stdout output of the successful command.
    """
    result = subprocess.run(["oc", "get", obj], capture_output=True, text=True)
    if result.returncode == 0:
        output = result.stdout
    else:
        print(f"Could not 'oc get {obj}'.\nError: {result.stderr}.")
        return False, ""
    if check_presence is not None and not check_presence in output:
        print(
            f"Did not see {check_presence} when getting {obj}.  Assuming oc API failure."
        )
        return False, ""
    return True, result.stdout


def oc_project(project: str) -> bool:
    print(f"Setting oc project to {project}")
    result = subprocess.run(["oc", "project", project], capture_output=True, text=True)
    if result.returncode != 0:
        print(
            f"Could not switch oc project to {project}.\nError {result.stderr}Are you logged in with oc to the G.B (RIS3) cluster?"
        )
        return False
    return True
