# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import yaml
from yaml.representer import SafeRepresenter
from typing import Dict, Any, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class YAMLManager:
    def __init__(self, file_path: str, verbose: bool = True):
        self.file_path = Path(file_path)
        self.verbose = verbose

        # Register custom representers for YAML serialization
        self._register_representers()

        if self.verbose:
            logger.info(
                f"YAMLManager initialized with path: {self.file_path.absolute()}"
            )

        # Create directory if it doesn't exist
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.error(
                f"Permission denied creating directory: {self.file_path.parent}"
            )
            raise
        except Exception as e:
            logger.error(f"Error creating directory {self.file_path.parent}: {e}")
            raise

        # Create file if it doesn't exist
        if not self.file_path.exists():
            if self.verbose:
                logger.info(f"Creating new YAML file at: {self.file_path.absolute()}")
            self.create_empty_yaml()

    def _register_representers(self) -> None:
        """Register custom representers for YAML serialization."""

        def custom_string_representer(dumper, data):
            # Force double quotes for specific keys like "/tmp/config.yaml"
            if isinstance(data, str) and data == "/tmp/config.yaml":
                logger.debug(f"Applying double quotes to key: {data}")
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
            # Use literal block style for multi-line strings
            if isinstance(data, str) and "\n" in data:
                logger.debug(
                    f"Applying literal block style to multi-line string: {data[:50]}..."
                )
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            # Default string representation for other cases
            return SafeRepresenter.represent_str(dumper, data)

        # Register the unified representer for strings
        yaml.add_representer(str, custom_string_representer, Dumper=yaml.SafeDumper)

    def create_empty_yaml(self) -> None:
        """Create an empty YAML file."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as file:
                yaml.dump({}, file, Dumper=yaml.SafeDumper)
            if self.verbose:
                logger.info(f"Created empty YAML file: {self.file_path.absolute()}")
        except PermissionError:
            logger.error(f"Permission denied creating file: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error creating YAML file: {e}")
            raise

    def read_yaml(self) -> Union[Dict[str, Any], list, str, int, float, bool, None]:
        """Read and return YAML content."""
        try:
            if not self.file_path.exists():
                logger.warning(f"YAML file does not exist: {self.file_path.absolute()}")
                return {}

            with open(self.file_path, "r", encoding="utf-8") as file:
                content = yaml.safe_load(file)
                return content if content is not None else {}

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in file {self.file_path}: {e}")
            raise
        except PermissionError:
            logger.error(f"Permission denied reading file: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading YAML file {self.file_path}: {e}")
            raise

    def write_yaml(
        self, data: Union[Dict[str, Any], list, str, int, float, bool, None]
    ) -> None:
        """Write data to YAML file."""
        try:
            if self.verbose:
                logger.debug(f"Writing data to YAML file: {self.file_path}")
            with open(self.file_path, "w", encoding="utf-8") as file:
                yaml.dump(
                    data,
                    file,
                    default_flow_style=False,
                    indent=2,
                    sort_keys=False,
                    allow_unicode=True,
                    Dumper=yaml.SafeDumper,
                )
            if self.verbose:
                logger.info(
                    f"Successfully wrote to YAML file: {self.file_path.absolute()}"
                )
        except PermissionError:
            logger.error(f"Permission denied writing to file: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error writing to YAML file {self.file_path}: {e}")
            raise

    def update_yaml(self, key: str, value: Any) -> None:
        """Update a specific key in the YAML file."""
        try:
            data = self.read_yaml()
            if not isinstance(data, dict):
                raise ValueError("Cannot update key in non-dictionary YAML content")

            data[key] = value
            self.write_yaml(data)

        except Exception as e:
            logger.error(f"Error updating YAML file: {e}")
            raise

    def delete_key(self, key: str) -> bool:
        """Delete a key from the YAML file. Returns True if key existed and was deleted."""
        try:
            data = self.read_yaml()
            if not isinstance(data, dict):
                raise ValueError("Cannot delete key from non-dictionary YAML content")

            if key in data:
                del data[key]
                self.write_yaml(data)
                return True
            return False

        except Exception as e:
            logger.error(f"Error deleting key from YAML file: {e}")
            raise
