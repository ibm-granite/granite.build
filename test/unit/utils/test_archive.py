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
import tarfile
import zipfile
from pathlib import Path

import pytest

from gbserver.utils.archive import check_tar_safe, check_zip_safe, extract_archive


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


def _make_tar(members: list[tarfile.TarInfo], payloads: dict[str, bytes]) -> bytes:
    """Build an in-memory tar from explicit headers (allows malicious names)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for info in members:
            data = payloads.get(info.name)
            if data is not None:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            else:
                tar.addfile(info)
    return buf.getvalue()


def _file(name: str, data: bytes = b"x") -> tuple[tarfile.TarInfo, bytes]:
    return tarfile.TarInfo(name), data


class TestExtractArchiveTar:
    def test_benign_tar_extracts(self, tmp_path: Path):
        info, data = _file("build.yaml", b"key: value")
        blob = _make_tar([info], {info.name: data})
        assert extract_archive(blob, tmp_path / "out", archive_format="tar")
        assert (tmp_path / "out" / "build.yaml").read_bytes() == b"key: value"

    def test_parent_traversal_rejected(self, tmp_path: Path):
        info, data = _file("../escape.txt")
        blob = _make_tar([info], {info.name: data})
        out = tmp_path / "nested" / "out"
        with pytest.raises(ValueError, match="unsafe tar member"):
            extract_archive(blob, out, archive_format="tar")
        assert not (tmp_path / "nested" / "escape.txt").exists()

    def test_absolute_path_confined_to_output_dir(self, tmp_path: Path):
        # The data filter strips the leading "/" rather than rejecting it.
        target = tmp_path / "abs_escape.txt"
        info, data = _file(str(target))
        blob = _make_tar([info], {info.name: data})
        out = tmp_path / "out"
        assert extract_archive(blob, out, archive_format="tar")
        assert not target.exists()
        assert (out / str(target).lstrip("/")).is_file()

    def test_symlink_escape_rejected(self, tmp_path: Path):
        link = tarfile.TarInfo("link")
        link.type = tarfile.SYMTYPE
        link.linkname = str(tmp_path / "victim")
        info, data = _file("link/pwned.txt")
        blob = _make_tar([link, info], {info.name: data})
        with pytest.raises(ValueError):
            extract_archive(blob, tmp_path / "out", archive_format="tar")
        assert not (tmp_path / "victim").exists()

    def test_too_many_entries_rejected(self, tmp_path: Path):
        infos = [tarfile.TarInfo(f"f{i}") for i in range(10)]
        blob = _make_tar(infos, {i.name: b"x" for i in infos})
        with pytest.raises(ValueError, match="too many entries"):
            extract_archive(blob, tmp_path / "out", archive_format="tar", max_entries=5)

    def test_uncompressed_size_too_large_rejected(self, tmp_path: Path):
        info, data = _file("big.txt", b"x" * 1000)
        blob = _make_tar([info], {info.name: data})
        with pytest.raises(ValueError, match="uncompressed size too large"):
            extract_archive(
                blob,
                tmp_path / "out",
                archive_format="tar",
                max_uncompressed_bytes=100,
            )
        assert not (tmp_path / "out" / "big.txt").exists()


class TestExtractArchiveZip:
    def test_benign_zip_extracts_with_autodetect(self, tmp_path: Path):
        zf = _make_zip({"build.yaml": b"key: value"})
        blob = zf.fp.getvalue()  # type: ignore[union-attr]
        assert extract_archive(blob, tmp_path / "out")
        assert (tmp_path / "out" / "build.yaml").read_bytes() == b"key: value"

    def test_oversized_zip_rejected_without_tar_fallback(self, tmp_path: Path):
        zf = _make_zip({"big.txt": b"0" * 10_000})
        blob = zf.fp.getvalue()  # type: ignore[union-attr]
        with pytest.raises(ValueError, match="uncompressed size too large"):
            extract_archive(blob, tmp_path / "out", max_uncompressed_bytes=100)
        assert not (tmp_path / "out" / "big.txt").exists()


class TestCheckTarSafeStopsEarly:
    """The caps must be enforced while reading the tar index, not after it: a
    tar's headers are only discovered by reading them, so checking a full
    getmembers() list would load every header of an oversized archive first."""

    @staticmethod
    def _open(names_and_sizes):
        infos = []
        payloads = {}
        for name, size in names_and_sizes:
            infos.append(tarfile.TarInfo(name))
            payloads[name] = b"x" * size
        blob = _make_tar(infos, payloads)
        return tarfile.open(fileobj=io.BytesIO(blob), mode="r:*")

    def test_entry_cap_stops_reading_headers(self):
        with self._open([(f"f{i}", 1) for i in range(100)]) as tar:
            with pytest.raises(ValueError, match="too many entries"):
                check_tar_safe(tar, max_entries=5)
            # Only the headers up to the first one past the cap were read.
            assert len(tar.members) == 6

    def test_size_cap_stops_reading_headers(self):
        with self._open([(f"f{i}", 60) for i in range(100)]) as tar:
            with pytest.raises(ValueError, match="uncompressed size too large"):
                check_tar_safe(tar, max_uncompressed_bytes=100)
            assert len(tar.members) == 2

    def test_within_caps_passes(self):
        with self._open([("a", 1), ("b", 1)]) as tar:
            check_tar_safe(tar, max_entries=2, max_uncompressed_bytes=2)


class TestCheckArchiveBytesSafe:
    """Submit-time check: the caps apply to a zip or tar without extracting it,
    so an oversized build is refused at POST /builds/ instead of being stored
    and failing later in the runner. Bytes that are neither zip nor tar are left
    for the runner to reject, as before."""

    def test_zip_within_caps(self):
        from gbcommon.utils.archive_safety import check_archive_bytes_safe

        blob = _make_zip({"build.yaml": b"x"}).fp.getvalue()  # type: ignore[union-attr]
        check_archive_bytes_safe(blob)

    def test_zip_over_size_cap(self):
        from gbcommon.utils.archive_safety import (
            ArchiveLimitError,
            check_archive_bytes_safe,
        )

        blob = _make_zip({"big": b"0" * 1000}).fp.getvalue()  # type: ignore[union-attr]
        with pytest.raises(ArchiveLimitError, match="uncompressed size too large"):
            check_archive_bytes_safe(blob, max_uncompressed_bytes=100)

    def test_tar_over_entry_cap(self):
        from gbcommon.utils.archive_safety import (
            ArchiveLimitError,
            check_archive_bytes_safe,
        )

        infos = [tarfile.TarInfo(f"f{i}") for i in range(10)]
        blob = _make_tar(infos, {i.name: b"x" for i in infos})
        with pytest.raises(ArchiveLimitError, match="too many entries"):
            check_archive_bytes_safe(blob, max_entries=5)

    def test_non_archive_bytes_pass_through(self):
        from gbcommon.utils.archive_safety import check_archive_bytes_safe

        check_archive_bytes_safe(b"test")

    def test_tar_check_reexported_from_gbserver(self):
        from gbcommon.utils import archive_safety
        from gbserver.utils import archive

        assert archive.check_tar_safe is archive_safety.check_tar_safe
