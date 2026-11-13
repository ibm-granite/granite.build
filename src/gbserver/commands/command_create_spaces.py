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


from pathlib import Path
from typing import List, Optional, cast

import click
import yaml
from pydantic import BaseModel

from gbserver.spaces.db_space import is_db_uri, parse_db_uri
from gbserver.spaces.local_secrets_isolation import check_local_secrets_collision
from gbserver.spaces.repo_identity import same_repo
from gbserver.storage import singleton_storage
from gbserver.storage.stored_space import StoredSpace
from gbserver.types.constants import (
    LAKEHOUSE_ENVIRONMENT,
    PUBLIC_SPACE_GIT_URI,
    PUBLIC_SPACE_LH_NAMESPACE,
    PUBLIC_SPACE_NAME,
)
from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

predefined_spaces = [
    StoredSpace(
        name=PUBLIC_SPACE_NAME,
        git_repo_uri=PUBLIC_SPACE_GIT_URI,
        lakehouse_namespace=PUBLIC_SPACE_LH_NAMESPACE,
    ),
    # StoredSpace(
    #     name=TEST_SPACE_NAME,
    #     git_repo_uri=TEST_SPACE_GIT_URI,
    #     lakehouse_namespace=TEST_SPACE_LH_NAMESPACE,
    # ),
]


def _check_alias_targets(storage, spaces: List[StoredSpace], clear: bool) -> None:
    """Refuse db:// aliases whose target row would not exist afterwards.

    Raises:
        click.ClickException: An alias names a row that is neither registered
            and kept (not removed by --clear) nor among the spaces being added.
    """
    names = {s.name for s in spaces}
    for space in spaces:
        if not is_db_uri(space.space_uri):
            continue
        target = parse_db_uri(cast(str, space.space_uri)).space_name
        if target == space.name or target in names:
            continue
        if clear or storage.get_by_name(target) is None:
            raise click.ClickException(
                f"space {space.name} is served from {space.space_uri}, but"
                f" {target} would not exist; add it to the spaces to create"
            )


def _check_local_secrets_collisions(
    storage, spaces: List[StoredSpace], clear: bool = False
) -> None:
    """Refuse spaces that would share another space's default local secrets.

    Each space is checked against the spaces that stay registered (minus any
    being re-created under the same name) and the ones accepted before it here.

    Args:
        storage: The admin space storage.
        spaces: The spaces about to be added.

    Raises:
        LocalSecretsCollisionError: See check_local_secrets_collision().
    """
    names = {s.name for s in spaces}
    accepted = (
        []
        if clear
        else [
            s
            for s in cast(List[StoredSpace], storage.get_by_uuid(None) or [])
            if s.name not in names
        ]
    )
    for space in spaces:
        check_local_secrets_collision(space, accepted)
        accepted.append(space)


def _keep_db_snapshots(storage, spaces: List[StoredSpace]) -> List[StoredSpace]:
    """Carry registered rows' space_uri and snapshots over to specs re-creating them.

    A spec for the same repo (e.g. the predefined public space,
    PUBLIC_SPACE_GIT_URI) keeps a db-served row's space_uri and snapshots
    unless it sets its own, so re-running create-spaces neither moves the space
    back to git nor reaches git. Read before --clear/--replace drop the rows.

    Args:
        storage: The admin space storage.
        spaces: The spaces about to be added.

    Returns:
        The specs, with space_uri and snapshots carried over where they apply.
    """
    result = []
    for space in spaces:
        existing = storage.get_by_name(space.name)
        if (
            existing is not None
            and existing.space_uri
            and same_repo(existing.git_repo_uri, space.git_repo_uri)
        ):
            logger.info(
                "space %s is served from %s; keeping it", space.name, existing.space_uri
            )
            space = space.model_copy(
                update={
                    "space_uri": space.space_uri or existing.space_uri,
                    "space": space.space or existing.space,
                    "assets": space.assets or existing.assets,
                }
            )
        result.append(space)
    return result


class CLICreateSpacesConfig(BaseModel):
    lakehouse_environment: str
    spaces: List[StoredSpace]


@click.command()
@click.option(
    "--clear",
    is_flag=True,
    help="If set, clear the spaces table before adding the spaces",
)
@click.option("--force", is_flag=True, help="Force clearing of PROD tables")
@click.option(
    "--replace",
    is_flag=True,
    help="Replace by deleting the space if it already exists.",
)
@click.option(
    "--spaces-path",
    required=False,
    default=None,
    type=click.Path(path_type=Path, exists=True, file_okay=True, dir_okay=False),
    help="Path to YAML file defining the LH environment and the spaces to create",
)
@pass_environment
def cli(
    ctx: CliEnvironment,
    clear: bool,
    force: bool,
    replace: bool,
    spaces_path: Optional[Path],
):
    """Create the fixed/predefined spaces"""
    logger.info(
        "running with clear: %s force: %s replace: %s spaces: %s",
        clear,
        force,
        replace,
        spaces_path,
    )
    spaces_to_create = predefined_spaces
    if spaces_path is not None:
        assert spaces_path.is_file(), f"{spaces_path} is not a file"
        with open(spaces_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        space_config = CLICreateSpacesConfig.model_validate(data)
        logger.debug("space_config: %s", space_config)
        if space_config.lakehouse_environment != LAKEHOUSE_ENVIRONMENT:
            logger.error(
                "Environment of requested spaces %s does not match"
                + "current environment %s taken from LAKEHOUSE_ENVIRONMENT env var.",
                space_config.lakehouse_environment,
                LAKEHOUSE_ENVIRONMENT,
            )
            return
        spaces_to_create = space_config.spaces

    logger.info("spaces to create: %s", spaces_to_create)

    logger.info("Working on spaces in Lakehouse %s environment", LAKEHOUSE_ENVIRONMENT)
    storage = singleton_storage.get_admin_storage().space_storage
    spaces_to_create = _keep_db_snapshots(storage, spaces_to_create)
    # Before --clear/--replace delete anything: a refusal must not leave the
    # table emptied, and aliases' target rows must still be readable.
    _check_alias_targets(storage, spaces_to_create, clear)
    _check_local_secrets_collisions(storage, spaces_to_create, clear)
    if clear:
        if force or LAKEHOUSE_ENVIRONMENT.lower() != "prod":
            logger.info("Deleting the space table")
            storage.delete_table()
        else:
            logger.info(
                "Skipped clearing of PROD space table.  Use --force to override"
            )

    logger.info("Check for existing spaces")
    for space in spaces_to_create:
        record = storage.get_by_name(space.name)
        if record is not None:
            if replace:
                logger.info("Removing pre-existing space %s", space.name)
                storage.delete(record.uuid)
            else:
                logger.info("Skipped clearing of pre-existing space %s", space.name)

    logger.info("Create all the spaces")
    storage.add(spaces_to_create)

    # Do this first so logging happens before the table below
    all_spaces = cast(List[StoredSpace], storage.get_by_uuid(None))

    fmt = "{:15s} {:50} {:s}"
    print("\n")
    print(fmt.format("Space", "Lakehouse Namespace", "Git Repo"))
    for space in all_spaces:
        assert isinstance(space, StoredSpace), "Not a StoredSpace"
        print(fmt.format(space.name, space.lakehouse_namespace, space.git_repo_uri))
