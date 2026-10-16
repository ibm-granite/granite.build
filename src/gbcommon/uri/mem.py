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
Reference a value held in the build's shared in-memory state.

A ``mem://`` URI is an opaque key into a per-build dictionary
(``BuildRun.shared_mem_state``), not a filesystem path. Unlike ``env://``,
it does NOT normalise or munge its path, because the value it carries (e.g. a
service URL ``http://host:8000``) must survive verbatim — round-tripping such a
value through filesystem-path normalisation is exactly what corrupts it.
"""

from pathlib import Path
from typing import Dict, List, Optional, Self
from urllib.parse import ParseResult

from gbcommon.uri.uri import URI
from gbserver.types.constants import MEM_URI_SCHEME


class MemURI(URI):
    """Reference a value in the build's shared in-memory state dict."""

    def __init__(
        self: Self, uri: ParseResult, context: Optional[str] = None, **kwargs: Dict
    ) -> None:
        self.context = context
        super().__init__(uri=uri, context=context, **kwargs)

    @staticmethod
    def get_supported_schemes() -> List[str]:
        """Return supported uri schemes as list"""
        return [MEM_URI_SCHEME]

    def exists(self: Self, force: bool = False) -> bool:
        # Presence is decided by the in-memory store at pull time, not here.
        return True

    def is_accessible(self: Self) -> bool:
        return self.exists()

    def pull(self: Self, dest: Path, force: bool = False) -> bool:
        # No filesystem transfer — the value lives in shared memory.
        return True

    def delete(self: Self) -> bool:
        raise NotImplementedError("MemURI delete is not implemented")

    def custom_str(self: Self) -> str:
        """Stringify verbatim — the URI is an opaque key, so it must be a
        stable, unmodified string (no path normalisation)."""
        if not self.uri:
            return ""
        return self.uri.geturl()
