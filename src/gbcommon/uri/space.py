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

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Self
from urllib.parse import ParseResult

from gbcommon.uri.uri import URI
from gbserver.utils.logger import get_logger

GBSPACE_SCHEME = "gb"
SPACE_SCHEME = "space"
STEPS_PREFIX = "steps/"

logger = get_logger(__name__)


class SpaceURI(URI):

    _thread_local = threading.local()

    def __new__(self, uri: ParseResult, **kwargs: dict) -> Self:
        if not hasattr(SpaceURI._thread_local, "base_uris"):
            default_base_uris = ["file:"]
            SpaceURI._thread_local.base_uris = default_base_uris
            logger.warning(
                "the space base_uris have not been initialized. Setting it to: %s",
                default_base_uris,
            )
        if not hasattr(SpaceURI._thread_local, "space_secrets"):
            SpaceURI._thread_local.space_secrets = {}
            logger.warning("the space space_secrets have not been initialized.")
        uristr = uri.geturl()
        uri_suffix = uristr
        if uristr.startswith(GBSPACE_SCHEME):
            uri_suffix = uristr.removeprefix(GBSPACE_SCHEME + "://")
        elif uristr.startswith(SPACE_SCHEME):
            uri_suffix = uristr.removeprefix(SPACE_SCHEME + "://")
        candidates = SpaceURI._build_candidates(uri_suffix)
        for candidate in candidates:
            for base_uri in SpaceURI._thread_local.base_uris:
                resolved: URI = URI.get_uri(
                    base_uri, "file", secrets=SpaceURI._thread_local.space_secrets
                )
                resolved.append_path(candidate)
                if resolved.exists():
                    return resolved  # type: ignore[return-value]
        raise ValueError(f"Unresolvable space uri : {uristr}")

    @staticmethod
    def _build_candidates(uri_suffix: str) -> List[str]:
        """Return the ordered list of suffixes to try against each base_uri.

        For ``space://steps/<rest>`` the active env's step_type chain (from the
        thread-local) is prepended so the resolver tries the most-preferred
        step_type-specific path first, falling back through the chain to the
        env-agnostic ``steps/<rest>`` location.  All other URI prefixes pass
        through unchanged so this only affects step lookups.
        """
        if not uri_suffix.startswith(STEPS_PREFIX):
            return [uri_suffix]
        step_types: List[str] = (
            getattr(SpaceURI._thread_local, "current_step_types", None) or []
        )
        rest = uri_suffix[len(STEPS_PREFIX):]
        candidates = [f"{STEPS_PREFIX}{st}/{rest}" for st in step_types]
        candidates.append(uri_suffix)  # env-agnostic fallback
        return candidates

    @classmethod
    def set_baseuris(cls, base_uris: List[str], space_secrets: dict):
        cls._thread_local.space_secrets = space_secrets
        cls._thread_local.base_uris = base_uris

    @classmethod
    @contextmanager
    def with_current_env_step_types(
        cls, step_types: Optional[List[str]]
    ) -> Iterator[None]:
        """Scope a step_type chain on the thread-local for the duration of the
        ``with`` block.

        Saves and restores any previous value so that nested or sibling target
        processing in the same thread doesn't leak.  ``None`` or an empty list
        means "no step_type-narrowing" — the resolver behaves as if the field
        had never been set.

        Args:
            step_types: Ordered chain of step_type strings, most-preferred first.
                Pass ``None`` or an empty list to opt out of narrowing.
        """
        prev = getattr(cls._thread_local, "current_step_types", None)
        cls._thread_local.current_step_types = list(step_types) if step_types else None
        try:
            yield
        finally:
            if prev is None:
                if hasattr(cls._thread_local, "current_step_types"):
                    del cls._thread_local.current_step_types
            else:
                cls._thread_local.current_step_types = prev

    @staticmethod
    def get_supported_schemes() -> List[str]:
        """Return supported uri schemes as list"""
        return [GBSPACE_SCHEME, SPACE_SCHEME]

    def exists(self: Self, force: bool = False) -> bool:
        """# TODO: fix this"""
        return True  # TODO: fix this

    def is_accessible(self) -> bool:
        """# TODO: fix this"""
        return True  # TODO: fix this

    def pull(self: Self, dest: Path, force: bool = False) -> bool:
        """# TODO: fix this"""
        return True  # TODO: fix this

    def delete(self: Self) -> bool:
        raise NotImplementedError("SpaceURI delete is not implemented")
