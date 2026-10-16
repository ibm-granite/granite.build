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

import io
import zipfile

import pytest

from gbserver.utils.archive import check_zip_safe


def _make_zip(entries: dict[str, bytes]) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    buf.seek(0)
    return zipfile.ZipFile(buf)


class TestCheckZipSafe:
    def test_small_archive_passes(self):
        zf = _make_zip({"build.yaml": b"key: value"})
        check_zip_safe(zf)  # should not raise

    def test_too_many_entries_rejected(self):
        zf = _make_zip({f"file_{i}.txt": b"x" for i in range(10)})
        with pytest.raises(ValueError, match="too many entries"):
            check_zip_safe(zf, max_entries=5)

    def test_uncompressed_size_too_large_rejected(self):
        zf = _make_zip({"big.txt": b"x" * 1000})
        with pytest.raises(ValueError, match="uncompressed size too large"):
            check_zip_safe(zf, max_uncompressed_bytes=100)

    def test_zip_bomb_style_archive_rejected_on_uncompressed_size(self):
        # A highly-compressible payload: tiny on disk, huge once decompressed —
        # the guard must check declared uncompressed size, not compressed size.
        zf = _make_zip({"bomb.txt": b"0" * 10_000_000})
        with pytest.raises(ValueError, match="uncompressed size too large"):
            check_zip_safe(zf, max_uncompressed_bytes=1_000_000)
