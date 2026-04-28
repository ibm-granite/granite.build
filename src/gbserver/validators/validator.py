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
Base class for validators.
"""

import importlib
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Self, Tuple, Type

from gbserver.types.validation import GBValidationErrors, GBValidatorConfig
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class GBValidator(ABC):
    """Abstract base class for all built-in validators to inherit."""

    # class attributes
    validator_types: Dict[str, Type[Self]] = {}
    # instance attributes
    validator_config: GBValidatorConfig

    def __init__(
        self: Self, validator_config: GBValidatorConfig, **kwargs: dict
    ) -> None:
        self.validator_config = validator_config
        super().__init__()

    @abstractmethod
    def validate(self: Self, data: Any, **kwargs: dict) -> GBValidationErrors:
        """Validate the data."""
        raise NotImplementedError("validate is not implemented")

    @staticmethod
    def is_static() -> bool:
        """Static validators can be run at any time."""
        return False

    @staticmethod
    def __get_module_and_class_names(filename: str) -> Tuple[str, str]:
        """Get the name of the module and validator class, given a file name."""
        s0 = filename.removesuffix(".py")
        s1 = s0.removesuffix("_gbvalidator").removesuffix("_validator")
        s2 = "".join([x.capitalize() for x in s1.split("_")])
        s3 = s2 + "GBValidator"
        return s0, s3

    @classmethod
    def __load_validator_type(cls: Type[Self], filename: str) -> None:
        module_name, validator_classname = cls.__get_module_and_class_names(filename)
        logger.info(
            "module_name %s validator_classname %s",
            module_name,
            validator_classname,
        )
        my_module = importlib.import_module(
            f".{module_name}", package="gbserver.validators"
        )
        logger.info("validator module %s", my_module)
        handler_class = getattr(my_module, validator_classname, None)
        if handler_class is None:
            logger.error(
                "Module %s does not contain the class named %s",
                module_name,
                validator_classname,
            )
            return
        if isinstance(handler_class, type) and issubclass(handler_class, cls):
            key = module_name.removesuffix("_validator")
            cls.validator_types[key] = handler_class
        else:
            logger.error(
                "Ignoring %s since it is not a subclass of GBValidator class",
                validator_classname,
            )

    @classmethod
    def load_validator_types(cls: Type[Self]) -> None:
        """Gather the list of built-in validator classes."""
        logger.info("load_validator_types start")
        cls.validator_types = {}
        curr_file = Path(__file__)
        curr_dir = curr_file.parent
        assert curr_dir.is_dir()
        for filename in os.listdir(curr_dir):
            if (not filename.endswith(".py")) or (
                filename in ("__init__.py", "validator.py")
            ):
                continue
            module_path = curr_dir / filename
            logger.info("validator module_path %s", module_path)
            if not module_path.is_file():
                continue
            try:
                cls.__load_validator_type(filename=filename)
            except ImportError as imp_e:
                logger.error("failed to import validator %s : %s", module_path, imp_e)
            except Exception as e:
                logger.error("failed to load validator at %s : %s", module_path, e)
        logger.info("validator_types: %s", cls.validator_types)
        logger.info("load_validator_types end")

    @classmethod
    def get_validator(
        cls: Type[Self], validator_config: GBValidatorConfig, **kwargs: dict
    ) -> Self:
        """Factory method for getting a validator of a given type."""
        if validator_config.type not in cls.validator_types:
            raise ValueError(f"unknown validator type: {validator_config}")
        validator_class = cls.validator_types[validator_config.type]
        validator = validator_class(validator_config=validator_config, **kwargs)
        return validator
