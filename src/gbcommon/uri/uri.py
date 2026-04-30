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

"""Uri module."""

import hashlib
import importlib
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Self, Tuple, Type, Union
from urllib.parse import ParseResult, urlparse

from gbserver.types.artifact import ArtifactType
from gbserver.types.spaceconfig import SpaceConfig
from gbserver.utils.logger import get_logger
from gbserver.utils.template import fill_template

logger = get_logger(__name__)


class URI(ABC):
    """U R I implementation."""

    uri_handler_classes: ClassVar[Dict[str, Type[Self]]] = {}
    _thread_local = threading.local()

    def __init__(
        self: Self,
        uri: Optional[ParseResult] = None,
        context: Optional[str] = None,
        secrets: Optional[dict] = None,
        **kwargs: dict,
    ) -> None:
        """Initialize any uri"""
        self.uri = uri
        self.context = context
        self.secrets = secrets

    @staticmethod
    def get_supported_schemes() -> List[str]:
        """Return supported uri schemes as list"""
        return []

    @staticmethod
    def get_uristr(uri: Union[str, "URI"]) -> str:
        """Get the uristr."""
        if isinstance(uri, URI):
            if uri.uri is None:
                return ""
            if hasattr(uri, "custom_str"):
                return uri.custom_str()
            return uri.uri.geturl()
        elif isinstance(uri, ParseResult):
            return uri.geturl()
        else:
            return uri

    def get_secrets(self: Self) -> Optional[dict]:
        """Get the secrets."""
        return self.secrets

    def __str__(self: Self) -> str:
        return self.get_uristr(self)

    def get_metadata(self: Self) -> Any:
        """Get the metadata."""
        return {"uri": self.get_uristr(self)}

    @staticmethod
    def _load_urihandlers() -> None:
        package_dir = os.path.dirname(__file__)

        for filename in os.listdir(package_dir):
            if (
                filename.endswith(".py")
                and filename != "__init__.py"
                and filename != "utils.py"
                and filename != os.path.basename(__file__)
            ):
                urihandler_modulename = filename[:-3]
                urihandler_name = urihandler_modulename.capitalize() + "URI"
                try:
                    module = importlib.import_module(
                        f".{urihandler_modulename}", package="gbcommon.uri"
                    )
                    if hasattr(module, urihandler_name):
                        handler_class = getattr(module, urihandler_name)
                        if isinstance(handler_class, type) and issubclass(handler_class, URI):
                            supported_schemes = handler_class.get_supported_schemes()
                            for supported_scheme in supported_schemes:
                                URI.uri_handler_classes[supported_scheme] = handler_class
                        else:
                            logger.error(
                                f"Ignoring {urihandler_name} since it is not a subclass or URI class"
                            )
                    else:
                        logger.error(
                            f"Module {urihandler_modulename} does not contain expected uri_hander class {urihandler_name}."
                        )
                except ImportError as e:
                    logger.error(f"Error importing module {urihandler_name}: {e}")
                except Exception as e:
                    logger.error(f"Error loading uri handler from {urihandler_name}: {e}")

    @classmethod
    def get_uri_class(
        cls: Type[Self],
        uri: Union["URI", str],
        default_scheme: str = "git",
    ) -> Tuple[Type[Self], ParseResult]:
        """
        Resolve URIs into appropriate URI class.

        Returns: (URI class, URI parsed into an object)
        """
        if uri is None:
            raise ValueError("uri cannot be None")
        if isinstance(uri, str) and uri == "":
            raise ValueError("uri cannot be empty")
        if isinstance(uri, URI):
            uri = URI.get_uristr(uri)
        uri = fill_template(uri, URI.get_space_config(), True)
        parsed_uri = urlparse(uri, default_scheme)
        if parsed_uri.scheme not in cls.uri_handler_classes:
            logger.error("Unknown URI scheme %s uri: %s", parsed_uri.scheme, uri)
            raise UnknownURIScheme(f"Unknown URI scheme {parsed_uri.scheme} uri: {uri}")
        mycls = cls.uri_handler_classes[parsed_uri.scheme]
        return mycls, parsed_uri

    @classmethod
    def get_uri(
        cls: Type[Self],
        uri: Union["URI", str],
        default_scheme: str = "git",
        context: Optional[str] = None,
        secrets: Optional[dict] = None,
        **kwargs,
    ) -> Self:
        """Resolve URIs into objects"""
        mycls, parsed_uri = cls.get_uri_class(
            uri=uri,
            default_scheme=default_scheme,
        )
        return mycls(uri=parsed_uri, context=context, secrets=secrets, **kwargs)

    @abstractmethod
    def exists(self: Self, force: bool = False) -> bool:
        """Returns if the resource exists or not"""

    @abstractmethod
    def is_accessible(self) -> bool:
        """Returns if an artifact is accessible in the current context or not"""

    def hash(self) -> str:
        """Hash."""
        assert self.uri is not None, "the URI is empty"
        return hashlib.sha256(self.uri.geturl().encode()).hexdigest()

    @abstractmethod
    def pull(self: Self, dest: Path, force: bool = False) -> bool:
        """Pulls the contents of the uri to a local directory"""

    @abstractmethod
    def delete(self: Self) -> bool:
        """Delete the resource referred to by this URI.

        Returns:
            True if deletion succeeded, False on any error.
        """

    def get_artifact_type(self) -> ArtifactType:
        """Return the artifact type for this URI.

        Subclasses should override this to provide type-specific mappings.

        Returns:
            ArtifactType: UNDEFINED by default; overridden by LhURI and HfURI.
        """
        return ArtifactType.UNDEFINED

    def append_path(self, path: str):
        """Add path."""
        base_path = self.uri.path.rstrip("/")  # type: ignore[union-attr]
        new_segment = path.lstrip("/")
        new_path = f"{base_path}/{new_segment}"
        self.uri = self.uri._replace(path=new_path)  # type: ignore[union-attr]

    @classmethod
    def set_space_config(cls, space_config: SpaceConfig = None):  # type: ignore[assignment]
        """Set the space config."""
        cls._thread_local.space_config = {
            "space": {"variables": space_config.variables, "name": space_config.name}
        }

    @classmethod
    def get_space_config(cls) -> dict:
        """Get the space config."""
        if cls._thread_local is not None and hasattr(cls._thread_local, "space_config"):
            return cls._thread_local.space_config
        return {}


class UnknownURIScheme(Exception):
    """Unknown URI Scheme exception"""
