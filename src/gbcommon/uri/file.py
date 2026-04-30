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

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Self

from gbcommon.uri.uri import URI
from gbserver.types.constants import FILE_SCHEME
from gbserver.utils.filesystem import sync_or_copy


class FileURI(URI):
    """Reference a file/folder in the local filesystem."""

    def __init__(self: Self, uri: URI, context: Optional[str] = None, **kwargs: Dict) -> None:
        self.context = context
        if not Path(uri.path).is_absolute() and self.context is not None and self.context != "":
            uri = FileURI(uri=FILE_SCHEME + ":///" + self.context + "/" + uri.path)
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

    def pull(self: Self, dest: Path, force: bool = False) -> bool:
        assert self.uri is not None, "self.uri is None"
        return sync_or_copy(self.uri.path, dest)

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
