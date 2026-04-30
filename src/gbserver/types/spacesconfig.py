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

"""Yaml file to give to create-spaces command."""

from typing import Dict, List, Optional, Self, cast

from pydantic import BaseModel, Field, PrivateAttr

from gbserver.storage.space_storage import IStoredSpaceStorage
from gbserver.storage.stored_space import StoredSpace
from gbserver.types.config import Config
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class CLISpaceConfig(BaseModel):
    """Details about the space."""

    name: str


class CLISpacesConfig(Config):
    """Configure a list of spaces to watch."""

    # If the space list is empty, we will watch all spaces
    spaces: List[CLISpaceConfig] = Field(default_factory=list)
    # https://docs.pydantic.dev/latest/concepts/models/#private-model-attributes
    _spaces: Dict[str, StoredSpace] = PrivateAttr(default_factory=dict)

    def initialize(
        self: Self,
        spaces_storage: IStoredSpaceStorage,
        prev_config: Optional[Self] = None,
    ) -> None:
        """Fetch the space information based on the names."""
        assert isinstance(spaces_storage, IStoredSpaceStorage)
        logger.info("got a previous config: %s", prev_config)
        if len(self.spaces) == 0:
            if prev_config is not None:
                self.spaces = prev_config.spaces
        cache = {} if prev_config is None else prev_config._spaces
        if len(self.spaces) == 0:
            # can only occur once because after that we have a prev config
            logger.warning("no spaces were configured, fetching all spaces")
            items = cast(List[StoredSpace], spaces_storage.get_by_where())
            if len(items) == 0:
                raise ValueError("failed to fetch spaces, found no spaces")
            for space in items:
                self.spaces.append(CLISpaceConfig(name=space.name))
                cache[space.name] = space
        for config_space in self.spaces:
            cached_result = cache.get(config_space.name, None)
            logger.info("cached_result: %s", cached_result)
            if cached_result is not None:
                self._spaces[cached_result.name] = cached_result
                continue
            items = spaces_storage.get_by_where({"name": config_space.name})
            if len(items) == 0:
                logger.error("failed to find a space for name: %s", config_space.name)
                continue
            if len(items) > 1:
                logger.warning("more than one space found: %s %s", config_space.name, items)
            space = items[0]
            assert isinstance(space, StoredSpace)
            cache[space.name] = space
            self._spaces[space.name] = space
