#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

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
