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
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel

from gbserver.asset.asset import Asset
from gbserver.types.validation import GBValidationErrors, GBValidatorConfig
from gbserver.utils.logger import get_logger
from gbserver.validators.validator import GBValidator

logger = get_logger(__name__)


class CustomCodeGBValidatorConfig(BaseModel):
    """Config for the custom code validator."""

    validator_uri: str
    entrypoint_path: str = "validator.py"


class CustomCodeGBValidator(GBValidator):
    "Checks the given data using a custom script."

    # instance attributes
    config: CustomCodeGBValidatorConfig

    def __init__(self: Self, validator_config: GBValidatorConfig, **kwargs: dict) -> None:
        assert (
            validator_config.type == "custom_code"
        ), f"invalid type, validator_config: {validator_config}"
        self.config = CustomCodeGBValidatorConfig.model_validate(validator_config.config)
        validator_uri = self.config.validator_uri
        logger.info("validator_uri: %s", validator_uri)
        # self.validator_asset = Asset(uri=validator_uri, context=context)
        self.validator_asset = Asset(uri=validator_uri)
        logger.info("validator_asset: %s", self.validator_asset)
        self.working_dir = Path(tempfile.mkdtemp())
        self.validator_asset_dir = self.working_dir / self.validator_asset.urihash()
        logger.info("validator_asset_dir: %s", self.validator_asset_dir)
        self.validator_asset.sync(dest=self.validator_asset_dir, force=True)
        logger.info("validator_asset copied into: %s", self.validator_asset_dir)
        self.validator_entrypoint = self.validator_asset_dir / self.config.entrypoint_path
        if not self.validator_entrypoint.is_file():
            error = ValueError(f"expected '{self.validator_entrypoint}' to be a file")
            if self.validator_asset_dir.is_dir():
                subfilesanddirs = list(self.validator_asset_dir.iterdir())
                subdirs = [subdir for subdir in subfilesanddirs if subdir.is_dir()]
                if len(subdirs) == 1:
                    self.validator_asset_dir = subdirs[0]
                    self.validator_entrypoint = (
                        self.validator_asset_dir / self.config.entrypoint_path
                    )
                    if not self.validator_entrypoint.is_file():
                        raise ValueError(f"expected '{self.validator_entrypoint}' to be a file")
                else:
                    raise error
            else:
                raise error
        super().__init__(validator_config=validator_config, **kwargs)

    def validate(self: Self, data: Any, **kwargs: dict) -> GBValidationErrors:
        """Validate the data against the schema."""
        errors = GBValidationErrors()
        cmd = [
            "python3",
            str(self.validator_entrypoint),
            "--input",
            "input.json",
            "--output",
            "output.json",
        ]
        try:
            input_path = self.working_dir / "input.json"
            full_input_data = {"data": data}
            if kwargs.get("context"):
                full_input_data["context"] = kwargs.get("context")
            with open(input_path, "w", encoding="utf-8") as f:
                json.dump(full_input_data, f, indent=4)
            proc = subprocess.run(
                cmd, cwd=self.working_dir, check=True, capture_output=True, text=True
            )
            logger.info("custom code validator succeeded")
            if proc.stdout != "":
                logger.info("stdout: %s", proc.stdout)
            if proc.stderr != "":
                logger.info("stderr: %s", proc.stderr)
            output_path = self.working_dir / "output.json"
            if not output_path.is_file():
                raise ValueError(f"expected output file at path '{output_path}'")
            with open(output_path, "r", encoding="utf-8") as f:
                output_str = f.read()
            logger.info("output_str: %s", output_str)
            errors = GBValidationErrors.model_validate_json(output_str)
            logger.info("errors: %s", errors)
        except Exception as e:
            logger.error("custom code validator failed with error: %s", e)
            logger.error("%s", traceback.format_exc())
            errors.add(e)
        return errors

    @staticmethod
    def is_static() -> bool:
        return True
