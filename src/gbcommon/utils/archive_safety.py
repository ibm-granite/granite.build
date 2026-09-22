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

"""Zip-archive safety checks, shared across distributions.

This lives in ``gbcommon`` rather than ``gbserver.utils.archive`` because
``granite-build-analytics`` ships ``gb_ui_backend`` and ``gbcommon`` **without**
``gbserver``, and ``gb_ui_backend`` needs this guard when it decodes build
archives out of the database.

Importing it from ``gbserver`` worked in-tree and failed in that distribution with
``No module named 'gbserver'``. The failure was invisible for a long time because
the call site wrapped the import in ``except Exception``, so the archive simply
never decoded and the Data Processing page reported no datasets rather than an
error.

The function is pure stdlib, so it has no business living in the server package.
"""

import zipfile

MAX_ZIP_ENTRIES = 1000
MAX_ZIP_UNCOMPRESSED_BYTES = 50 * 1024 * 1024  # 50 MB


def check_zip_safe(
    zf: zipfile.ZipFile,
    max_entries: int = MAX_ZIP_ENTRIES,
    max_uncompressed_bytes: int = MAX_ZIP_UNCOMPRESSED_BYTES,
) -> None:
    """Guard against zip-bomb archives before reading any entry.

    Raises ValueError if the archive has more entries, or more total
    uncompressed size, than the given caps.
    """
    infos = zf.infolist()
    if len(infos) > max_entries:
        raise ValueError(f"archive has too many entries ({len(infos)} > {max_entries})")
    total_size = sum(info.file_size for info in infos)
    if total_size > max_uncompressed_bytes:
        raise ValueError(
            f"archive uncompressed size too large "
            f"({total_size} > {max_uncompressed_bytes} bytes)"
        )
