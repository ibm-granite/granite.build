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

"""Handle reading/writing assets in different stores/locations."""

import glob
import importlib
import os
import re
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Self, Type

from gbcommon.uri.uri import URI
from gbserver.types.artifact import ArtifactType
from gbserver.types.assetstoreconfig import AssetStoreConfig
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

ASSETSTORE_YAML = "store.yaml"


class Assetstore(ABC):
    """A class for storing and accessing any asset"""

    assetstore_types: Dict[type[URI], type["Assetstore"]] = {}
    _thread_local = threading.local()

    def __init__(
        self,
        store_config: Optional[AssetStoreConfig] = None,
        secrets: dict = None,  # type: ignore[assignment]
        **kwargs,
    ):
        self.type = self.__class__.__name__
        self.config: AssetStoreConfig = store_config  # type: ignore[assignment]
        self.secrets = secrets

    def can_handle(self, uri: URI):
        """Check if handle."""
        if (
            self.config.base_uri is not None
            and self.config.base_uri != ""
            and URI.get_uristr(uri).startswith(self.config.base_uri)
        ):
            return True
        if (
            self.config.uri_regex is not None
            and self.config.uri_regex != ""
            and re.search(self.config.uri_regex, URI.get_uristr(uri))
        ):
            return True
        return False

    def can_handle_len(self, uri: URI) -> int:
        """
        Returns the length of the prefix/regex match as well.
        Returns 0 for a non-match.
        """
        if (
            self.config.base_uri is not None
            and self.config.base_uri != ""
            and URI.get_uristr(uri).startswith(self.config.base_uri)
        ):
            return len(self.config.base_uri)
        if self.config.uri_regex is not None and self.config.uri_regex != "":
            uri_regex_match = re.search(self.config.uri_regex, URI.get_uristr(uri))
            if uri_regex_match:
                return len(uri_regex_match.group(0))
        return 0

    def get_secrets(self) -> dict:
        """Get the secrets."""
        return self.secrets

    def get_metadata(self, uri: URI) -> Dict:
        """Get the metadata."""
        return {}

    @classmethod
    def load_assetstores_from_dir(
        cls,
        dir: Optional[Path] = None,
        context: Optional[str] = None,
        secrets: dict = None,  # type: ignore[assignment]
    ):
        """Parse assetstores from dir."""
        if dir is None:
            return
        store_yamls = glob.glob(str(dir / "**" / ASSETSTORE_YAML), recursive=True)
        for store_yaml in store_yamls:
            cls.load_asset_store(Path(store_yaml), context=context, secrets=secrets)

    @classmethod
    def load_asset_store(
        cls,
        path: Optional[Path] = None,
        context: Optional[str] = None,
        secrets: dict = None,  # type: ignore[assignment]
    ) -> "Assetstore":
        """Parse asset store."""
        if path is None:
            raise ValueError("path must not be None")
        if path.is_dir():
            store_yamls = glob.glob(str(path / "**" / ASSETSTORE_YAML), recursive=True)
            store_yaml = Path(store_yamls[0])
        else:
            store_yaml = path
        store_config = AssetStoreConfig.from_yaml(store_yaml, context=context)
        test_uri = store_config.base_uri
        if test_uri is None or test_uri == "":
            test_uri = store_config.uri_regex
        mycls, _ = URI.get_uri_class(test_uri)
        assetstore = cls.assetstore_types[mycls](store_config, secrets=secrets)
        if not hasattr(cls._thread_local, "assetstores"):
            # base_uri -> asset store
            # assetstores : Dict[str, "Assetstore"] = {}
            cls._thread_local.assetstores = {}
            Assetstore.load_assetstores_from_dir(
                Path(__file__).parent.parent / "builtins" / "assetstores" / "file"
            )
        cls._thread_local.assetstores[store_config.base_uri] = assetstore
        return assetstore

    @classmethod
    def _load_assetstore_types(cls):
        if len(cls.assetstore_types) != 0:
            return
        package_dir = os.path.dirname(__file__)

        for filename in os.listdir(package_dir):
            if (
                filename.endswith(".py")
                and filename != "__init__.py"
                and filename != "asset.py"
                and filename != os.path.basename(__file__)
            ):
                assetstore_module_name = filename[:-3]
                assetstore_classname = assetstore_module_name.capitalize()
                try:
                    module = importlib.import_module(
                        f".{assetstore_module_name}", package="gbserver.asset"
                    )
                    if hasattr(module, assetstore_classname):
                        handler_class = getattr(module, assetstore_classname)
                        if isinstance(handler_class, type) and issubclass(handler_class, cls):
                            for uriclass in handler_class.get_supported_uri_classes():
                                cls.assetstore_types[uriclass] = handler_class
                        else:
                            logger.error(
                                f"Ignoring {assetstore_classname} since it is not a subclass of AssetStore class"
                            )
                    else:
                        logger.error(
                            f"Module {assetstore_module_name} does not contain expected AssetStore type class {assetstore_classname}."
                        )
                except ImportError as e:
                    logger.error(f"Error importing module {assetstore_module_name}: {e}")
                except Exception as e:
                    logger.error(f"Error loading AssetStore type from {assetstore_classname}: {e}")

    @classmethod
    @abstractmethod
    def get_supported_uri_classes(cls: Type[Self]) -> Type[URI]:
        """Get the supported uri classes."""
        raise NotImplementedError("get_supported_uri_classes is not implemented")

    def get_asset_type(self: Self, uri: URI) -> ArtifactType:
        """Get the asset type."""
        return ArtifactType.UNDEFINED
