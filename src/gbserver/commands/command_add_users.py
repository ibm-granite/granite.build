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
from typing import Dict, List, Literal

import click
import yaml
from pydantic import BaseModel

from gbserver.storage import singleton_storage
from gbserver.storage.stored_space_user import StoredSpaceUser
from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class SpaceUserEntry(BaseModel):
    username: str
    role: Literal["admin", "member"]


class AddUsersConfig(BaseModel):
    spaces: Dict[str, List[SpaceUserEntry]]


@click.command()
@click.argument(
    "users_file",
    type=click.Path(path_type=Path, exists=True, file_okay=True, dir_okay=False),
)
@pass_environment
def cli(ctx: CliEnvironment, users_file: Path):
    """Add users to spaces from a YAML file"""
    logger.info("Reading users file: %s", users_file)

    with open(users_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    config = AddUsersConfig.model_validate(data)

    storage = singleton_storage.get_admin_storage().space_user_storage

    for space_name, entries in config.spaces.items():
        for entry in entries:
            space_user = StoredSpaceUser(
                username=entry.username,
                role=entry.role,
                space_name=space_name,
            )
            try:
                storage.add(space_user)
                logger.info(
                    "Added user %s with role %s to space %s",
                    entry.username,
                    entry.role,
                    space_name,
                )
            except Exception as e:
                logger.warning(
                    "Failed to add user %s to space %s: %s",
                    entry.username,
                    space_name,
                    e,
                )
