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
URI for files/folders in the local filesystem.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Self
from urllib.parse import ParseResult, urlparse, urlunparse

from gbcommon.uri.uri import URI
from gbserver.types.constants import FILE_SCHEME
from gbserver.utils.filesystem import sync_or_copy


class FileURI(URI):
    """Reference a file/folder in the local filesystem."""

    def __init__(
        self: Self, uri: ParseResult, context: Optional[str] = None, **kwargs: Dict
    ) -> None:
        self.context = context
        if (
            not Path(uri.path).is_absolute()
            and self.context is not None
            and self.context != ""
        ):
            uri = urlparse(FILE_SCHEME + ":///" + self.context + "/" + uri.path)
        super().__init__(uri=uri, context=context, **kwargs)

    @staticmethod
    def get_supported_schemes() -> List[str]:
        """Return supported uri schemes as list"""
        return [FILE_SCHEME]

    def exists(self: Self, force: bool = False) -> bool:
        assert self.uri is not None, "self.uri is None"
        return Path(self.uri.path).exists()

    def is_accessible(self: Self) -> bool:
        return self.exists()

    def pull(
        self: Self,
        dest: Path,
        force: bool = False,
        raise_errors: bool = False,
        copy_dir_contents: bool = False,
    ) -> bool:
        """Pull this file/dir to ``dest``.

        By default a directory source is copied as a whole (rsync without a
        trailing slash nests it under ``dest`` as ``dest/<basename>/``), matching
        long-standing behavior that other callers (step/space asset
        materialization) rely on.

        ``copy_dir_contents=True`` instead copies a directory source's CONTENTS
        into ``dest`` (no extra nesting level), for callers where ``dest`` IS the
        artifact's final location. Opt-in so default callers are unaffected.

        ``raise_errors=True`` surfaces a failed copy as an exception instead of a
        ``False`` return (for callers that must not silently treat a failed copy
        as success).
        """
        assert self.uri is not None, "self.uri is None"
        src = self.uri.path
        if copy_dir_contents and os.path.isdir(src) and not src.endswith(os.sep):
            src = src + os.sep
        return sync_or_copy(src, dest, raise_errors=raise_errors)

    def delete(self: Self) -> bool:
        """Delete the file or directory at this URI's path.

        Returns:
            True if deletion succeeded, False if the path does not exist or on error.
        """
        assert self.uri is not None, "self.uri is None"
        path = Path(self.uri.path)
        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                return False
            return True
        except Exception:
            return False


def absolutize_file_uri(uri: URI) -> URI:
    """Resolve a relative ``file:`` URI to an absolute one against the cwd.

    A relative ``file:outputs/...`` URI is physically read/written relative to
    the process working directory (e.g. an artifact push shells out to rsync
    from ``os.getcwd()``), but it would otherwise be *registered* verbatim as a
    relative URI — meaningless to any later consumer (e.g. ``gb artifact
    download``) that runs from a different directory. Rewriting it to
    ``file:///<cwd>/outputs/...`` makes the URI match the file's real on-disk
    location.

    Absolute ``file:`` URIs and all non-``file:`` schemes (``hf:``, ``lh:``,
    ``cos:``, ``env:``, ...) are returned unchanged.
    """
    if uri.uri is None or uri.uri.scheme != FILE_SCHEME:
        return uri
    path = uri.uri.path
    if Path(path).is_absolute():
        return uri
    abs_path = os.path.normpath(os.path.join(os.getcwd(), path))
    # normpath strips a trailing slash; restore it for directory URIs so the
    # round-tripped URI keeps its original (dir vs file) shape.
    if path.endswith("/") and not abs_path.endswith("/"):
        abs_path += "/"
    return URI.get_uri(urlunparse((FILE_SCHEME, "", abs_path, "", "", "")))
