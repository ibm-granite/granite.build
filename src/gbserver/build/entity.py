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

"""
The entity. All entities like Build, BuildTarget etc inherity this
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Self

from gbserver.types.config import Config
from gbserver.types.validation import GBValidationErrors


class Entity(ABC):
    """The base class for all entities."""

    # instance attributes
    val_errors: Optional[GBValidationErrors] = None
    force_fetch: bool = False

    def __init__(
        self: Self,
        type: str,
        config: Config,
        dir: Optional[Path] = None,
        context: Optional[str] = None,
        validate: bool = True,
        force_fetch: bool = False,
    ) -> None:
        """Loads a entity"""

        self.type = type
        self.config = config
        self.dir = dir
        self.context = context
        self.force_fetch = force_fetch
        self.assimilate()
        if validate:
            if self.val_errors is None:
                self.val_errors = GBValidationErrors()
            try:
                self.val_errors.add(self.validate())
            except Exception as e:
                self.val_errors.add(e)
            self.val_errors.raise_if_invalid()

    @abstractmethod
    def assimilate(self: Self) -> None:
        """Process the information in the entity"""

    def validate(self: Self) -> GBValidationErrors:
        """Validate the entity."""
        return GBValidationErrors()
