# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The result-report asset service.

Lists a job's downloadable output files and serves them for download. A
pipeline-populated ``output_artifacts`` column is served as-is when present
for *listing*; otherwise the job's artifact source (a granite.build TUNING
task's ``artifact_uri``, or a local run's output directory) is listed on
demand — no DB write, and no caching, so the endpoint always reflects the
current state of the source. Downloads (:meth:`AssetService.download_file` and
:meth:`AssetService.open_archive`) always resolve the physical source and
stream from it. Everything here is owner-scoped like the rest of the job read
path.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from autotunex.core.config import Settings
from autotunex.core.exceptions import JobArtifactsNotReadyError, JobNotFoundError
from autotunex.db.repositories.protocols import JobRepository
from autotunex.models.asset import AssetSummary
from autotunex.models.auth import Principal
from autotunex.models.common import DataScope
from autotunex.models.status import GbTaskType, RunStatus
from autotunex.services.scoping import resolve_owner_filter, sees_nothing
from autotunex.services.storage.archive import stream_zip
from autotunex.services.storage.artifacts import ArtifactLister, OpenedArtifact
from autotunex.services.storage.registry import resolve_artifact_lister

_UNSAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")


class AssetService:
    """List a job's output assets, owner-scoped, computed on read."""

    def __init__(
        self,
        *,
        job_repository: JobRepository,
        principal: Principal,
        settings: Settings,
        filesystem: ArtifactLister,
        huggingface: ArtifactLister,
    ) -> None:
        self._repository = job_repository
        self._principal = principal
        self._settings = settings
        self._filesystem = filesystem
        self._huggingface = huggingface

    async def list_assets(
        self, job_id: UUID, *, scope: DataScope = DataScope.OWN
    ) -> list[AssetSummary]:
        """Return the job's downloadable assets; 404 if the job is not visible.

        Serves a pipeline-populated ``output_artifacts`` column when present;
        otherwise lists the job's artifact source on demand (no DB write).

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
            JobNotFoundError: no such job, or it is not visible under ``scope``.
            JobArtifactsNotReadyError: the run has not produced artifacts yet.
            JobArtifactsNotFoundError: no artifact source could be located.
            ArtifactSourceUnavailableError: the source exists but is unreadable.
        """
        job = await self._get_visible_job(job_id, scope)

        prepopulated = self._map(job.output_artifacts)
        if prepopulated:
            return prepopulated

        lister, location = self._resolve_source(job)
        return await lister.list_files(location=location)

    async def download_file(
        self, job_id: UUID, *, path: str, scope: DataScope = DataScope.OWN
    ) -> OpenedArtifact:
        """Open one output file for streaming; 404 if the job/file is not visible.

        ``path`` is the relative path from :meth:`list_assets` (keying on the path,
        not the basename, disambiguates files that share a name across
        directories). Unlike :meth:`list_assets`, this always resolves the
        *physical* artifact source — bytes cannot be served from the metadata-only
        ``output_artifacts`` column.

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
            JobNotFoundError: no such job, or it is not visible under ``scope``.
            JobArtifactsNotReadyError: the run has not produced artifacts yet.
            JobArtifactsNotFoundError: no source, or no such file under it.
            ArtifactSourceUnavailableError: the source exists but is unreadable.
        """
        job = await self._get_visible_job(job_id, scope)
        lister, location = self._resolve_source(job)
        return await lister.open_file(location=location, path=path)

    async def open_archive(
        self, job_id: UUID, *, scope: DataScope = DataScope.OWN
    ) -> tuple[str, AsyncIterator[bytes]]:
        """Return ``(archive_filename, zip_byte_stream)`` for all of a job's assets.

        Resolves the physical source and lists it up front (so a not-ready/not-found
        job fails before any bytes are streamed), then returns a lazily-produced ZIP
        stream. Entry names are the assets' relative paths, preserving directory
        structure so same-named files stay distinct. Same errors as
        :meth:`download_file`.
        """
        job = await self._get_visible_job(job_id, scope)
        lister, location = self._resolve_source(job)
        files = await lister.list_files(location=location)
        paths = [asset.path or asset.filename for asset in files]
        archive_name = f"{_safe_stem(job.experiment_name)}_assets.zip"

        async def opener(path: str) -> OpenedArtifact:
            return await lister.open_file(location=location, path=path)

        return archive_name, stream_zip(paths, opener)

    async def _get_visible_job(self, job_id: UUID, scope: DataScope) -> Any:  # noqa: ANN401 — ORM row
        """Resolve the owner-scoped, visible job or raise ``JobNotFoundError``."""
        if sees_nothing(self._principal, scope):
            raise JobNotFoundError(job_id)
        owner_id = resolve_owner_filter(self._principal, scope)
        job = await self._repository.get(job_id, owner_id=owner_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job

    def _resolve_source(self, job: Any) -> tuple[ArtifactLister, str]:  # noqa: ANN401 — ORM row
        """Pick the (lister, location) for a job with no pre-populated column."""
        task = next((t for t in job.tasks if t.type == GbTaskType.TUNING), None)
        if task is not None:
            if not task.artifact_uri or not _build_succeeded(task.build_status):
                raise JobArtifactsNotReadyError(job.id, _task_status(task))
            return resolve_artifact_lister(
                task.artifact_uri, filesystem=self._filesystem, huggingface=self._huggingface
            )
        if job.status != RunStatus.COMPLETED:
            raise JobArtifactsNotReadyError(job.id, job.status)
        location = str(self._settings.local_output_dir / str(job.id) / "results")
        return self._filesystem, location

    @staticmethod
    def _map(artifacts: Any) -> list[AssetSummary]:  # noqa: ANN401 — genuinely-arbitrary JSON
        """Tolerantly map ``output_artifacts`` to a list of :class:`AssetSummary`.

        ``output_artifacts`` is free-form JSON written by the tuning pipeline, so
        its runtime shape is not enforced: pipeline rows may hold a bare list of
        file descriptors instead of a dict wrapper (both are declared on the ORM
        and on :class:`JobRead`). This tolerates ``list``, ``dict`` with a
        ``files`` or ``assets`` key, or ``None`` — never raising on an unexpected
        shape, since a coarse empty result is preferable to a 500 on the Results
        panel.
        """
        items: list[dict[str, Any]]
        if isinstance(artifacts, list):
            items = [a for a in artifacts if isinstance(a, dict)]
        elif isinstance(artifacts, dict):
            raw = artifacts.get("files") or artifacts.get("assets") or []
            items = [a for a in raw if isinstance(a, dict)] if isinstance(raw, list) else []
        else:
            items = []
        return [
            AssetSummary(
                filename=str(item.get("filename") or item.get("name") or ""),
                size=int(item.get("size") or item.get("file_size") or 0),
                modified=item.get("modified") or item.get("created"),
                path=item.get("path"),
                file_hash=item.get("file_hash"),
                published=item.get("published"),
            )
            for item in items
            if item.get("filename") or item.get("name")
        ]


def _build_succeeded(build_status: Any) -> bool:  # noqa: ANN401 — arbitrary JSON blob
    """True when a TUNING task's build reported ``details.status == 'success'``."""
    if not isinstance(build_status, dict):
        return False
    details = build_status.get("details")
    return isinstance(details, dict) and details.get("status") == "success"


def _task_status(task: Any) -> str:  # noqa: ANN401 — ORM row
    """A human-facing status for a not-ready TUNING task."""
    if isinstance(task.build_status, dict):
        details = task.build_status.get("details")
        if isinstance(details, dict) and details.get("status"):
            return str(details["status"])
    return str(task.status)


def _safe_stem(name: str | None) -> str:
    """Reduce an experiment name to a filesystem/header-safe archive stem."""
    stem = _UNSAFE_STEM.sub("_", (name or "").strip()).strip("._")
    return stem or "job"
