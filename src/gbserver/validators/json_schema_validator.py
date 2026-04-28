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
A validator that checks the given data against a JSON schema.
"""

import json
from pathlib import Path
from typing import Any, Optional, Self

from jsonschema import Draft202012Validator, ValidationError
from pydantic import BaseModel, model_validator

from gbserver.types.validation import GBValidationErrors, GBValidatorConfig
from gbserver.utils.logger import get_logger
from gbserver.validators.validator import GBValidator

logger = get_logger(__name__)


class JsonSchemaGBValidatorConfig(BaseModel):
    """Config for the JSON schema validator."""

    schema_path: Optional[Path] = None
    # cannot use 'schema' , already defined in BaseModel
    # https://docs.pydantic.dev/1.10/usage/schema/#getting-schema-of-a-specified-type
    json_schema: Optional[dict] = None

    @model_validator(mode="after")
    def validate_config_fields(self: Self) -> Self:
        """Validate the fields of the config."""
        if self.schema_path is None and self.json_schema is None:
            raise ValueError("one of 'schema_path' or 'json_schema' must be specified")
        if self.schema_path is not None and self.json_schema is not None:
            raise ValueError("both 'schema_path' and 'json_schema' cannot be specified")
        return self


class JsonSchemaGBValidator(GBValidator):
    "Checks the given data against a JSON schema."

    # instance attributes
    config: JsonSchemaGBValidatorConfig
    validator: Draft202012Validator

    def __init__(
        self: Self, validator_config: GBValidatorConfig, **kwargs: dict
    ) -> None:
        assert (
            validator_config.type == "json_schema"
        ), f"invalid type, validator_config: {validator_config}"
        self.config = JsonSchemaGBValidatorConfig.model_validate(
            validator_config.config
        )
        rel_schema_path = self.config.schema_path
        if rel_schema_path is not None:
            assert (
                "context" in kwargs
            ), f"context is required to load the schema file {rel_schema_path}"
            context = kwargs["context"]
            assert isinstance(
                context, dict
            ), f"expected context dict, actual: {context}"
            assert (
                "dir" in context
            ), f"context.dir is required to load the schema file {rel_schema_path}"
            my_dir = context["dir"]
            assert isinstance(my_dir, Path), f"expected dir Path, actual: {my_dir}"
            assert my_dir.is_dir(), f"expected '{my_dir}' to be a directory"
            logger.info(
                "loading schema %s from the directory: %s", rel_schema_path, my_dir
            )
            schema_path = my_dir / rel_schema_path
            assert (
                schema_path.is_file()
            ), f"expected '{schema_path}' to be a schema file"
            with open(schema_path, "r", encoding="utf-8") as f:
                self.config.json_schema = json.load(f)
        schema = self.config.json_schema
        assert isinstance(
            schema, dict
        ), f"expected schema to be a dict, actual: {schema}"
        Draft202012Validator.check_schema(schema=schema)
        self.validator = Draft202012Validator(schema=schema)
        super().__init__(validator_config=validator_config, **kwargs)

    def validate(self: Self, data: Any, **kwargs: dict) -> GBValidationErrors:
        """Validate the data against the schema."""
        errors = GBValidationErrors()
        for error in self.validator.iter_errors(data):
            assert isinstance(error, ValidationError)
            errors.add(error)
        return errors

    @staticmethod
    def is_static() -> bool:
        return True
