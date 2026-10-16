# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""List a job's downloadable output files from its artifact source.

The ``ArtifactLister`` seam extends the storage layer (open decision 3) with a
read-only "list the files at this location" capability used by the
result-report endpoint. Each backend takes a source-specific ``location`` — a
filesystem path, or an ``owner/repo`` id — and returns
:class:`~autotunex.models.asset.AssetSummary` records. Scheme parsing of the
stored ``artifact_uri`` happens once, in
:func:`autotunex.services.storage.registry.resolve_artifact_lister`, so a lister
never sees a raw URI.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from autotunex.core.exceptions import ArtifactSourceUnavailableError, JobArtifactsNotFoundError
from autotunex.core.logging import get_logger
from autotunex.models.asset import AssetSummary

logger = get_logger(__name__)

_HF_LIST_TIMEOUT_SECONDS = 15.0
_HF_DOWNLOAD_TIMEOUT_SECONDS = 300.0
_STREAM_CHUNK_BYTES = 1 << 20  # 1 MiB — bounds per-chunk memory when streaming
_DEFAULT_MEDIA_TYPE = "application/octet-stream"


@dataclass
class OpenedArtifact:
    """A single downloadable file opened for streaming.

    ``stream`` is a *single-use* byte iterator that owns and releases the
    underlying resource (an open file handle, or an HTTP response/client) when
    exhausted, so callers must iterate it exactly once.
    """

    filename: str
    media_type: str
    size: int | None
    stream: AsyncIterator[bytes]


class ArtifactLister(Protocol):
    """List and open the downloadable files at a resolved artifact ``location``."""

    async def list_files(self, *, location: str) -> list[AssetSummary]:
        """Return the files at ``location``.

        Raises:
            JobArtifactsNotFoundError: the location does not exist.
            ArtifactSourceUnavailableError: the location exists but is unreadable.
        """
        ...

    async def open_file(self, *, location: str, path: str) -> OpenedArtifact:
        """Open the file identified by ``path`` (relative to ``location``) for streaming.

        ``path`` is the relative path reported by :meth:`list_files` — keying on
        the path, not the base filename, is what lets two files that share a
        basename (e.g. several ``adapters.safetensors`` under different
        directories) be fetched unambiguously.

        Raises:
            JobArtifactsNotFoundError: no such file under ``location``.
            ArtifactSourceUnavailableError: the source exists but is unreadable.
        """
        ...


class FilesystemArtifactLister:
    """List files under a local directory (``file://`` and local-runner output).

    Satisfies :class:`ArtifactLister`. ``path`` on each returned asset is
    relative to the scanned root, so an absolute server path is never exposed.
    """

    async def list_files(self, *, location: str) -> list[AssetSummary]:
        """Recursively list regular files under ``location`` (off the event loop)."""
        return await asyncio.to_thread(self._scan, location)

    @staticmethod
    def _scan(location: str) -> list[AssetSummary]:
        root = Path(location)
        if not root.is_dir():
            logger.debug("Artifact directory does not exist: %s", location)
            raise JobArtifactsNotFoundError

        def _reraise(error: OSError) -> None:
            raise error

        summaries: list[AssetSummary] = []
        try:
            for dirpath, _dirnames, filenames in os.walk(root, onerror=_reraise):
                for name in filenames:
                    path = Path(dirpath) / name
                    if not path.is_file():
                        continue
                    stat_result = path.stat()
                    summaries.append(
                        AssetSummary(
                            filename=path.name,
                            size=stat_result.st_size,
                            modified=datetime.fromtimestamp(stat_result.st_mtime, tz=UTC),
                            path=str(path.relative_to(root)),
                        )
                    )
        except OSError as exc:
            logger.warning("Could not read artifact directory %s: %s", location, exc)
            raise ArtifactSourceUnavailableError from exc
        return sorted(summaries, key=lambda asset: asset.path or asset.filename)

    async def open_file(self, *, location: str, path: str) -> OpenedArtifact:
        """Open a file under ``location`` for streaming, guarding against traversal."""
        resolved = await asyncio.to_thread(self._resolve_within, location, path)
        media_type = mimetypes.guess_type(resolved.name)[0] or _DEFAULT_MEDIA_TYPE
        return OpenedArtifact(
            filename=resolved.name,
            media_type=media_type,
            size=resolved.stat().st_size,
            stream=_stream_file(resolved),
        )

    @staticmethod
    def _resolve_within(location: str, path: str) -> Path:
        """Resolve ``path`` under ``location``, rejecting anything that escapes it.

        An absolute ``path`` or one containing ``..`` that resolves outside the
        root is treated as "not found" (never a distinct, information-leaking
        error), as is a path that is not a regular file.
        """
        root = Path(location).resolve()
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            logger.debug("Rejected artifact path %r under %s", path, location)
            raise JobArtifactsNotFoundError
        return candidate


class HuggingFaceArtifactLister:
    """List a HuggingFace model repo's files via the Hub tree API.

    Satisfies :class:`ArtifactLister`. Uses ``httpx`` (no ``huggingface_hub``
    dependency), mirroring :mod:`autotunex.services.storage.hf_viewer`. The
    ``_client_factory`` seam lets tests inject an ``httpx.MockTransport``.
    """

    def __init__(self, *, base_url: str, token: str | None, revision: str = "main") -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._revision = revision

    def _client_factory(self) -> httpx.AsyncClient:
        # A short connect budget, but a generous read budget so a large model
        # checkpoint streaming through open_file is not aborted mid-download.
        # Listing is unaffected — its response is small and arrives at once.
        return httpx.AsyncClient(
            timeout=httpx.Timeout(_HF_LIST_TIMEOUT_SECONDS, read=_HF_DOWNLOAD_TIMEOUT_SECONDS)
        )

    async def list_files(self, *, location: str) -> list[AssetSummary]:
        """List files in the ``owner/repo`` model repo named by ``location``."""
        url = f"{self._base_url}/api/models/{location}/tree/{self._revision}"
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        try:
            async with self._client_factory() as client:
                response = await client.get(url, params={"recursive": "true"}, headers=headers)
        except httpx.HTTPError as exc:
            logger.warning("HF tree request failed for %s: %s", location, exc)
            raise ArtifactSourceUnavailableError from exc
        if response.status_code == httpx.codes.NOT_FOUND:
            logger.debug("HF repo tree not found: %s", location)
            raise JobArtifactsNotFoundError
        if response.status_code != httpx.codes.OK:
            logger.warning("HF tree returned HTTP %s for %s", response.status_code, location)
            raise ArtifactSourceUnavailableError
        try:
            entries = response.json()
        except ValueError as exc:
            logger.warning("HF tree returned malformed JSON for %s", location)
            raise ArtifactSourceUnavailableError from exc
        if not isinstance(entries, list):
            logger.warning("HF tree body was not a list for %s", location)
            raise ArtifactSourceUnavailableError
        return [
            self._to_summary(entry)
            for entry in entries
            if isinstance(entry, dict) and entry.get("type") == "file" and entry.get("path")
        ]

    async def open_file(self, *, location: str, path: str) -> OpenedArtifact:
        """Stream one repo file, proxying through the server so the token stays here.

        Follows the redirect the Hub's ``resolve`` endpoint issues to its CDN.
        The upstream status is checked eagerly (before returning), so a missing
        file or unreadable source surfaces as an exception rather than as a
        corrupt download body.
        """
        # Reject path traversal: the Hub's ``resolve`` endpoint is relative to
        # THIS repo, so a caller-supplied ``..`` or absolute ``path`` could escape
        # to another repo — and it would be fetched with the server's own token.
        # Treated as "not found" (never a distinct, information-leaking error),
        # mirroring FilesystemArtifactLister._resolve_within.
        if path.startswith("/") or ".." in path.replace("\\", "/").split("/"):
            logger.debug("Rejected HF artifact path %r for %s", path, location)
            raise JobArtifactsNotFoundError

        # Unlike the tree API (under /api/models), resolve lives at the repo root.
        url = f"{self._base_url}/{location}/resolve/{self._revision}/{path}"
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        client = self._client_factory()
        request = client.build_request("GET", url, headers=headers)
        try:
            response = await client.send(request, stream=True, follow_redirects=True)
        except httpx.HTTPError as exc:
            await client.aclose()
            logger.warning("HF resolve request failed for %s/%s: %s", location, path, exc)
            raise ArtifactSourceUnavailableError from exc
        if response.status_code == httpx.codes.NOT_FOUND:
            await response.aclose()
            await client.aclose()
            logger.debug("HF file not found: %s/%s", location, path)
            raise JobArtifactsNotFoundError
        if response.status_code != httpx.codes.OK:
            await response.aclose()
            await client.aclose()
            logger.warning(
                "HF resolve returned HTTP %s for %s/%s", response.status_code, location, path
            )
            raise ArtifactSourceUnavailableError
        raw_type = (response.headers.get("content-type") or "").split(";", 1)[0].strip()
        return OpenedArtifact(
            filename=path.rsplit("/", 1)[-1],
            media_type=raw_type or _DEFAULT_MEDIA_TYPE,
            size=_content_length(response.headers.get("content-length")),
            stream=_stream_response(response, client),
        )

    @staticmethod
    def _to_summary(entry: dict[str, Any]) -> AssetSummary:
        path = str(entry["path"])
        return AssetSummary(
            filename=path.rsplit("/", 1)[-1],
            size=int(entry.get("size") or 0),
            path=path,
            published=True,
        )


async def _stream_file(path: Path) -> AsyncIterator[bytes]:
    """Yield a local file's bytes in bounded chunks, reading off the event loop."""
    handle = await asyncio.to_thread(path.open, "rb")
    try:
        while True:
            chunk = await asyncio.to_thread(handle.read, _STREAM_CHUNK_BYTES)
            if not chunk:
                break
            yield chunk
    finally:
        await asyncio.to_thread(handle.close)


async def _stream_response(
    response: httpx.Response, client: httpx.AsyncClient
) -> AsyncIterator[bytes]:
    """Yield an HTTP response body, closing the response and client when done."""
    try:
        async for chunk in response.aiter_bytes(_STREAM_CHUNK_BYTES):
            yield chunk
    finally:
        await response.aclose()
        await client.aclose()


def _content_length(raw: str | None) -> int | None:
    """Parse a ``Content-Length`` header into an int, or ``None`` if absent/malformed."""
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None
