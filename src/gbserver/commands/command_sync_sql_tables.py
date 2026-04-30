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


"""Command sync sql tables module."""

import json
import traceback
from typing import Any

import click

from gbserver.storage import singleton_storage, storage
from gbserver.storage.artifact_registry import IArtifactRegistry
from gbserver.storage.build_storage import IStoredBuildStorage
from gbserver.storage.lh.lh_storage import BaseLakehouseItemStorage
from gbserver.storage.lh.storage_factory import LhStorageFactory
from gbserver.storage.space_storage import IStoredSpaceStorage
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.sql.storage_factory import SQLStorageFactory
from gbserver.storage.steprun_storage import IStoredStepRunStorage
from gbserver.storage.storage import BaseStoredItem, IItemStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.target_run_storage import IStoredTargetRunStorage
from gbserver.types.constants import (
    GB_ARTIFACT_REGISTRY_TABLE_NAME,
    GB_BUILDS_TABLE_NAME,
    GB_ENVIRONMENT,
    GB_ENVIRONMENT_CONFIG,
    GB_METADATA_STORAGE,
    GB_SPACES_TABLE_NAME,
    GB_STEP_RUNS_TABLE_NAME,
    GB_TARGET_RUNS_TABLE_NAME,
    GBSERVER_SQL_DBNAME,
    GBSERVER_SQL_SCHEMA,
    LAKEHOUSE_ENVIRONMENT,
)
from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.types.status import Status
from gbserver.utils.utils import get_differing_attributes

enable_storage_modifications = False
src_prefix: str = ""


def get_dest_prefix():
    """Get the dest prefix."""
    dest_prefix = singleton_storage.get_admin_storage().table_name_prefix
    return dest_prefix


def get_src_prefix():
    """Get the src prefix."""
    return src_prefix


def get_artifact_registry_pair(
    dest_factory, src_factory
) -> tuple[IArtifactRegistry, IArtifactRegistry]:
    """Get the artifact registry pair."""
    dest_prefix = get_dest_prefix()
    src_prefix = get_src_prefix()

    dest_table_name = dest_prefix + GB_ARTIFACT_REGISTRY_TABLE_NAME
    dest = dest_factory.create_artifact_registry(dest_table_name)
    src_table_name = src_prefix + GB_ARTIFACT_REGISTRY_TABLE_NAME
    src = src_factory.create_artifact_registry(src_table_name)
    return dest, src


def get_build_storage_pair(
    dest_factory, src_factory
) -> tuple[IStoredBuildStorage, IStoredBuildStorage]:
    """Get the build storage pair."""
    dest_prefix = get_dest_prefix()
    src_prefix = get_src_prefix()

    dest_table_name = dest_prefix + GB_BUILDS_TABLE_NAME
    dest = dest_factory.create_build_storage(dest_table_name)
    src_table_name = src_prefix + GB_BUILDS_TABLE_NAME
    src = src_factory.create_build_storage(src_table_name)
    return dest, src


def get_target_storage_pair(
    dest_factory, src_factory
) -> tuple[IStoredTargetRunStorage, IStoredTargetRunStorage]:
    """Get the target storage pair."""
    dest_prefix = get_dest_prefix()
    src_prefix = get_src_prefix()
    dest_table_name = dest_prefix + GB_TARGET_RUNS_TABLE_NAME
    dest = dest_factory.create_target_storage(dest_table_name)
    src_table_name = src_prefix + GB_TARGET_RUNS_TABLE_NAME
    src = src_factory.create_target_storage(src_table_name)
    return dest, src


def get_step_storage_pair(
    dest_factory, src_factory
) -> tuple[IStoredStepRunStorage, IStoredStepRunStorage]:
    """Get the step storage pair."""
    dest_prefix = get_dest_prefix()
    src_prefix = get_src_prefix()
    dest_table_name = dest_prefix + GB_STEP_RUNS_TABLE_NAME
    dest = dest_factory.create_step_storage(dest_table_name)
    src_table_name = src_prefix + GB_STEP_RUNS_TABLE_NAME
    src = src_factory.create_step_storage(src_table_name)
    return dest, src


def get_space_storage_pair(
    dest_factory, src_factory
) -> tuple[IStoredSpaceStorage, IStoredSpaceStorage]:
    """Get the space storage pair."""
    dest_prefix = get_dest_prefix()
    src_prefix = get_src_prefix()
    dest_table_name = dest_prefix + GB_SPACES_TABLE_NAME
    dest = dest_factory.create_space_storage(dest_table_name)
    src_table_name = src_prefix + GB_SPACES_TABLE_NAME
    src = src_factory.create_space_storage(src_table_name)
    return dest, src


@click.command(context_settings={"show_default": True})
# @click.option(
#     "--tables",
#     help="Specifies which table(s) to clear.  One of all, builds, artifacts, spaces",
#     default="all"
# )
@click.option(
    "--operation",
    type=click.Choice(["sync", "snapshot"]),
    help=f"Controls the type of operation to perform.  One of sync-builds, copy",
)
@click.option(
    "--enable-storage-mods",
    is_flag=True,
    help=f"If set then make changes to the SQL tables.  If not set then only show what would be done.",
)
@click.option(
    "--source-table-prefix",
    default="",
    help=f"Specify source tables prefix. Use --gb-admin-table-prefix to specify destination tables prefix",
)
@pass_environment
def cli(
    ctx: CliEnvironment,
    operation: str,
    enable_storage_mods: bool,
    source_table_prefix: str,
    # overwrite:bool,
    # clear:bool,
):
    """
    Migrate data from the Lakehouse tables to the SQL tables.
    The following operations are supported:
        snapshot - clears the SQL tables and then copies the data from Lakehouse to SQL .
        sync - copies new spaces and newly finished builds/targets/steps/artifacts from Lakehouse to SQL.
    """

    global enable_storage_modifications
    enable_storage_modifications = enable_storage_mods
    global src_prefix
    src_prefix = source_table_prefix

    print(f"\nPerforming the following:")
    print(f"Operation                : {operation}")
    print(f"Enable storage mods      : {enable_storage_modifications}")
    print(f"G.B environment          : {GB_ENVIRONMENT}")
    print(f"Admin/metadata source    : {GB_METADATA_STORAGE}")
    print(f"OpenShift project        : {GB_ENVIRONMENT_CONFIG.default_pod_namespace}")
    print(f"Lakehouse environment    : {LAKEHOUSE_ENVIRONMENT}")
    print(f"SQL destination          : {GBSERVER_SQL_DBNAME}.{GBSERVER_SQL_SCHEMA}")
    if src_prefix != "":
        print(f"Source table prefix      : {src_prefix}")
    print(f"Destination table prefix : {get_dest_prefix()}")
    print("Do you want to proceed? (yes/no)")
    user_input = input().lower()
    if user_input != "yes":
        print("Operation aborted!")
        return

    dest_factory = SQLStorageFactory()
    src_factory = LhStorageFactory()
    if operation == "snapshot":
        snapshot(dest_factory, src_factory)
    elif operation == "sync":
        results = sync_admin_storage(dest_factory, src_factory)
        with open("sync-results.json", "w") as json_file:
            json.dump(results, json_file, indent=4)
    else:
        print("Unknown operation")


def get_dict_of_items(storage: IItemStorage):
    """Get the dict of items."""
    items = storage.get_by_uuid(None)  # Get all items
    item_dict = {}
    for item in items:
        item_dict[item.uuid] = item
    return item_dict


def allow_item_overwrite(src_item: BaseStoredItem, dest_item: BaseStoredItem) -> bool:
    """Determine if the given source item is allowed to overwrite the destinant item.
    Things not allowed:
    Builds that are PENDING in the source and not PENDING in the destination (presumably finished)
    """
    if not isinstance(src_item, StoredBuild):
        return True
    assert isinstance(dest_item, StoredBuild)
    if src_item.status == Status.PENDING and dest_item.status != Status.PENDING:
        # These builds may have been moved from pending status in the destination/sql environment.
        return False
    unequal = get_differing_attributes(src_item, dest_item)
    if len(unequal) == 1 and storage.UPDATED_TIME_FIELD_NAME in unequal:
        # Ignore changes in ONLY the update time only since this is written on update into the destination.
        return False
    return True


def get_deltas(
    item_name: str, dest_item_dict, src_item_dict
) -> tuple[list[BaseStoredItem], list[BaseStoredItem], list[BaseStoredItem]]:
    """Get the deltas."""
    missing: list[BaseStoredItem] = []
    modified: list[BaseStoredItem] = []
    ignored: list[BaseStoredItem] = []
    for uuid, src_item in src_item_dict.items():
        dest_item = dest_item_dict.get(uuid, None)
        if dest_item is None:
            missing.append(src_item)
        elif src_item != dest_item and allow_item_overwrite(src_item, dest_item):
            # print(f"src={src_item}")
            # print(f"dest={dest_item}")
            # print(f"unequal = {get_unequal_attributes (src_item,dest_item)}")
            modified.append(src_item)
        else:
            ignored.append(src_item)
    return missing, modified, ignored


def sync_storage(
    item_name: str, sql_storage: IItemStorage, lh_storage: IItemStorage
) -> dict[str, list[str]]:
    """Sync storage."""
    assert isinstance(sql_storage, BaseSQLItemStorage) and isinstance(
        lh_storage, BaseLakehouseItemStorage
    )
    lh_item_dict = get_dict_of_items(lh_storage)
    sql_item_dict = get_dict_of_items(sql_storage)
    missing, modified, ignored = get_deltas(item_name, sql_item_dict, lh_item_dict)

    total_items = len(lh_item_dict)
    print(f"Found {total_items} total {item_name}(s) in source Lakehouse storage.")
    print(f"Found {len(ignored)} {item_name}(s) to be left unchanged in SQL.")
    print(f"Found {len(missing)} missing {item_name}(s) and will add these.")
    print(f"Found {len(modified)} modified {item_name}(s) and will update these.")

    count = 0
    added = []
    total_items = len(missing)
    for item in missing:
        count += 1
        uuid = item.uuid
        msg = f"Adding new {item_name} {uuid} to SQL from Lakehouse ({count} of {total_items})"
        if enable_storage_modifications:
            print(msg)
            sql_storage.add(item)
            stored_item = sql_storage.get_by_uuid(uuid)
            assert stored_item == item
        else:
            print(f"NOT {msg}")
        added.append(uuid)

    total_items = len(modified)
    count = 0
    updated = []
    for item in modified:
        count += 1
        uuid = item.uuid
        msg = f"Updating {item_name} {uuid} in SQL from Lakehouse ({count} of {total_items})"
        if enable_storage_modifications:
            print(msg)
            sql_storage.update(item=item, update_updated_time=False)
            stored_item = sql_storage.get_by_uuid(uuid)
            assert stored_item == item
        else:
            print(f"NOT {msg}")
        updated.append(uuid)

    print(f"Added {len(added)} and updated {len(updated)} {item_name}s.")
    return {"added": added, "updated": updated}


def sync_admin_storage(
    dest_factory: SQLStorageFactory, src_factory: LhStorageFactory
) -> dict[str, Any]:
    """Sync admin storage."""
    assert isinstance(dest_factory, SQLStorageFactory)
    assert isinstance(src_factory, LhStorageFactory)
    print(
        f"\nSyncing admin tables from Lakehouse {LAKEHOUSE_ENVIRONMENT} environment to SQL {GBSERVER_SQL_SCHEMA} schema\n"
    )

    print(f"Syncing builds")
    dest_storage, src_storage = get_build_storage_pair(dest_factory, src_factory)
    builds = sync_storage("build", dest_storage, src_storage)

    print(f"Syncing targets")
    dest_storage, src_storage = get_target_storage_pair(dest_factory, src_factory)
    targets = sync_storage("target", dest_storage, src_storage)

    print(f"Syncing steps")
    dest_storage, src_storage = get_step_storage_pair(dest_factory, src_factory)
    steps = sync_storage("step", dest_storage, src_storage)

    print(f"Syncing artifacts")
    dest_storage, src_storage = get_artifact_registry_pair(dest_factory, src_factory)
    artifacts = sync_storage("artifact", dest_storage, src_storage)

    print(f"Syncing spaces")
    dest_storage, src_storage = get_space_storage_pair(dest_factory, src_factory)
    spaces = sync_storage("space", dest_storage, src_storage)

    return {
        "builds": builds,
        "targets": targets,
        "steps": steps,
        "artifacts": artifacts,
        "spaces": spaces,
    }


def snapshot(dest_factory: SQLStorageFactory, src_factory: LhStorageFactory):
    """Snapshot."""
    assert isinstance(dest_factory, SQLStorageFactory)
    assert isinstance(src_factory, LhStorageFactory)

    print(
        f"\nClearing SQL and copying from Lakehouse {LAKEHOUSE_ENVIRONMENT} environment to SQL {GBSERVER_SQL_SCHEMA} schema\n"
    )

    dest, src = get_artifact_registry_pair(dest_factory, src_factory)
    snapshot_storage_pair(dest, src)

    dest, src = get_build_storage_pair(dest_factory, src_factory)
    snapshot_storage_pair(dest, src)

    dest, src = get_target_storage_pair(dest_factory, src_factory)
    snapshot_storage_pair(dest, src)

    dest, src = get_step_storage_pair(dest_factory, src_factory)
    snapshot_storage_pair(dest, src)

    dest, src = get_space_storage_pair(dest_factory, src_factory)
    snapshot_storage_pair(dest, src)


def snapshot_storage_pair(dest: IItemStorage, src: IItemStorage):
    """Snapshot storage pair."""
    print(
        f"\nSyncing {type(src).__name__} table={src.get_table_name()} into {type(dest).__name__} table={dest.get_table_name()}"
    )
    items: list = src.get_by_uuid(None)
    item_count = len(items)
    print(f"Found {item_count} items in {type(src).__name__} table={src.get_table_name()}")
    # return
    msg = f"Deleting table {type(dest).__name__} table={dest.get_table_name()}"
    if enable_storage_modifications:
        print(msg)
        dest.delete_table()
    else:
        print(f"NOT {msg}")
    print(
        f"Begin writing {item_count} items to {type(dest).__name__} table={dest.get_table_name()}"
    )
    count = 0
    try:
        msg = f"Adding {item_count} items to {type(dest).__name__} table={dest.get_table_name()}"
        if enable_storage_modifications:
            print(msg)
            dest.add(items)
        else:
            print(f"NOT {msg}")
    except Exception as exc:
        print(f"ERROR: Exception writing item #{count}: {exc}")
        msg = f"Deleting table {type(dest).__name__} table={dest.get_table_name()}"
        if enable_storage_modifications:
            print(msg)
            dest.delete_table()
        else:
            print(f"NOT {msg}")  # But, should never really get here.
        count = 0

    # Check results
    if enable_storage_modifications:
        items: list = dest.get_by_uuid(None)
        dest_item_count = len(items)
        if item_count != dest_item_count:
            print(
                f"WARNING: {item_count} items found in {type(src).__name__} table={src.get_table_name()}, but only {dest_item_count} found in {type(dest).__name__} table={dest.get_table_name()}"
            )
        else:
            print(
                f"Done writing {dest_item_count} of {item_count} items to {type(dest).__name__} table={dest.get_table_name()}"
            )
