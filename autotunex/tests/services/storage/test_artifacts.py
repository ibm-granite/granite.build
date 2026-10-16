# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for the artifact-listing storage backends."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from autotunex.core.exceptions import ArtifactSourceUnavailableError, JobArtifactsNotFoundError
from autotunex.services.storage.artifacts import (
    FilesystemArtifactLister,
    HuggingFaceArtifactLister,
    OpenedArtifact,
)


async def _drain(opened: OpenedArtifact) -> bytes:
    return b"".join([chunk async for chunk in opened.stream])


async def test_filesystem_lists_files_recursively(tmp_path: Path) -> None:
    (tmp_path / "best_config.json").write_text("{}")
    nested = tmp_path / "weights"
    nested.mkdir()
    (nested / "adapter.safetensors").write_bytes(b"0123456789")

    assets = await FilesystemArtifactLister().list_files(location=str(tmp_path))

    by_name = {a.filename: a for a in assets}
    assert set(by_name) == {"best_config.json", "adapter.safetensors"}
    assert by_name["adapter.safetensors"].size == 10
    assert by_name["adapter.safetensors"].path == "weights/adapter.safetensors"


async def test_filesystem_missing_directory_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(JobArtifactsNotFoundError):
        await FilesystemArtifactLister().list_files(location=str(tmp_path / "does-not-exist"))


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses filesystem permissions")
async def test_filesystem_unreadable_directory_raises_source_unavailable(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "secret.txt").write_text("x")
    os.chmod(locked, 0o000)

    try:
        with pytest.raises(ArtifactSourceUnavailableError):
            await FilesystemArtifactLister().list_files(location=str(tmp_path))
    finally:
        os.chmod(locked, 0o755)


async def test_filesystem_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    assert await FilesystemArtifactLister().list_files(location=str(tmp_path)) == []


async def test_filesystem_open_file_streams_content_with_metadata(tmp_path: Path) -> None:
    nested = tmp_path / "weights"
    nested.mkdir()
    (nested / "adapter.safetensors").write_bytes(b"0123456789")

    opened = await FilesystemArtifactLister().open_file(
        location=str(tmp_path), path="weights/adapter.safetensors"
    )

    assert await _drain(opened) == b"0123456789"
    assert opened.size == 10
    assert opened.filename == "adapter.safetensors"
    assert opened.media_type == "application/octet-stream"


async def test_filesystem_open_file_guesses_media_type(tmp_path: Path) -> None:
    (tmp_path / "final_config.json").write_text("{}")

    opened = await FilesystemArtifactLister().open_file(
        location=str(tmp_path), path="final_config.json"
    )

    assert opened.media_type == "application/json"


async def test_filesystem_open_file_missing_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(JobArtifactsNotFoundError):
        await FilesystemArtifactLister().open_file(location=str(tmp_path), path="nope.txt")


async def test_filesystem_open_file_directory_raises_not_found(tmp_path: Path) -> None:
    (tmp_path / "checkpoints").mkdir()

    with pytest.raises(JobArtifactsNotFoundError):
        await FilesystemArtifactLister().open_file(location=str(tmp_path), path="checkpoints")


async def test_filesystem_open_file_rejects_path_traversal(tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("top secret")
    root = tmp_path / "root"
    root.mkdir()

    with pytest.raises(JobArtifactsNotFoundError):
        await FilesystemArtifactLister().open_file(location=str(root), path="../secret.txt")


async def test_filesystem_open_file_rejects_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(JobArtifactsNotFoundError):
        await FilesystemArtifactLister().open_file(location=str(tmp_path), path="/etc/hosts")


def _hf_lister(handler: Callable[[httpx.Request], httpx.Response]) -> HuggingFaceArtifactLister:
    lister = HuggingFaceArtifactLister(base_url="https://hf.example.com", token="tok")
    lister._client_factory = lambda: httpx.AsyncClient(  # type: ignore[method-assign]
        transport=httpx.MockTransport(handler)
    )
    return lister


async def test_huggingface_maps_file_entries() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization") or ""
        return httpx.Response(
            200,
            json=[
                {"type": "file", "path": "adapter.safetensors", "size": 42},
                {"type": "directory", "path": "checkpoints"},
                {"type": "file", "path": "checkpoints/config.json", "size": 7},
            ],
        )

    assets = await _hf_lister(handler).list_files(location="ibm-research/autotunex_x")

    assert {a.filename for a in assets} == {"adapter.safetensors", "config.json"}
    assert seen["auth"] == "Bearer tok"
    assert "/api/models/ibm-research/autotunex_x/tree/main" in seen["url"]


async def test_huggingface_404_raises_not_found() -> None:
    def handler_404(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    lister = _hf_lister(handler_404)

    with pytest.raises(JobArtifactsNotFoundError):
        await lister.list_files(location="ibm-research/missing")


async def test_huggingface_500_raises_source_unavailable() -> None:
    def handler_500(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    lister = _hf_lister(handler_500)

    with pytest.raises(ArtifactSourceUnavailableError):
        await lister.list_files(location="ibm-research/x")


async def test_huggingface_open_file_streams_from_resolve_url() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization") or ""
        return httpx.Response(
            200,
            content=b"safetensor-bytes",
            headers={"content-type": "application/octet-stream", "content-length": "16"},
        )

    opened = await _hf_lister(handler).open_file(
        location="org/repo", path="checkpoints/adapter.safetensors"
    )

    assert await _drain(opened) == b"safetensor-bytes"
    assert opened.filename == "adapter.safetensors"
    assert opened.size == 16
    assert "/org/repo/resolve/main/checkpoints/adapter.safetensors" in seen["url"]
    assert seen["auth"] == "Bearer tok"


async def test_huggingface_open_file_404_raises_not_found() -> None:
    def handler_404(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with pytest.raises(JobArtifactsNotFoundError):
        await _hf_lister(handler_404).open_file(location="org/repo", path="missing.bin")


async def test_huggingface_open_file_500_raises_source_unavailable() -> None:
    def handler_500(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with pytest.raises(ArtifactSourceUnavailableError):
        await _hf_lister(handler_500).open_file(location="org/repo", path="adapter.safetensors")


def _handler_fails_if_called(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"HTTP request must not be issued for a traversal path, got {request.url}")


async def test_huggingface_open_file_rejects_path_traversal() -> None:
    lister = _hf_lister(_handler_fails_if_called)

    with pytest.raises(JobArtifactsNotFoundError):
        await lister.open_file(
            location="org/repo", path="../../other-owner/other-repo/resolve/main/config.json"
        )


async def test_huggingface_open_file_rejects_absolute_path() -> None:
    lister = _hf_lister(_handler_fails_if_called)

    with pytest.raises(JobArtifactsNotFoundError):
        await lister.open_file(location="org/repo", path="/other-owner/other-repo/config.json")
