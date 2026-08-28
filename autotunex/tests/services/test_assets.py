# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for the result-report asset service."""

from __future__ import annotations

import io
import zipfile
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest

from autotunex.core.config import Settings
from autotunex.core.exceptions import (
    JobArtifactsNotFoundError,
    JobArtifactsNotReadyError,
    JobNotFoundError,
)
from autotunex.models.asset import AssetSummary
from autotunex.models.auth import Principal
from autotunex.models.status import GbTaskType, RunStatus
from autotunex.services.assets import AssetService
from autotunex.services.storage.artifacts import FilesystemArtifactLister, OpenedArtifact


class _Task:
    def __init__(self, *, artifact_uri: str | None, build_status: object) -> None:
        self.type = GbTaskType.TUNING
        self.artifact_uri = artifact_uri
        self.build_status = build_status
        self.status = RunStatus.RUNNING


class _Job:
    def __init__(
        self,
        *,
        output_artifacts: object = None,
        tasks: list[_Task] | None = None,
        status: RunStatus = RunStatus.RUNNING,
        experiment_name: str = "experiment",
    ) -> None:
        self.id = uuid4()
        self.output_artifacts = output_artifacts
        self.tasks = tasks or []
        self.status = status
        self.experiment_name = experiment_name


class _FakeJobRepo:
    def __init__(self, job: object | None) -> None:
        self._job = job

    async def get(self, job_id: object, *, owner_id: object) -> object | None:
        return self._job


class _StubLister:
    def __init__(self) -> None:
        self.seen: str | None = None

    async def list_files(self, *, location: str) -> list[AssetSummary]:
        self.seen = location
        return [AssetSummary(filename="model.safetensors", size=1)]

    async def open_file(self, *, location: str, path: str) -> OpenedArtifact:
        async def _stream() -> AsyncIterator[bytes]:
            yield b""

        return OpenedArtifact(
            filename=path, media_type="application/octet-stream", size=0, stream=_stream()
        )


def _principal() -> Principal:
    return Principal(email="u@example.com", provider="session", user_id=uuid4(), is_admin=False)


async def test_missing_job_raises_not_found() -> None:
    service = AssetService(
        job_repository=_FakeJobRepo(None),  # type: ignore[arg-type]
        principal=_principal(),
        settings=Settings(),
        filesystem=FilesystemArtifactLister(),
        huggingface=_StubLister(),
    )

    with pytest.raises(JobNotFoundError):
        await service.list_assets(uuid4())


async def test_prepopulated_column_short_circuits() -> None:
    job = _Job(output_artifacts=[{"filename": "results.csv", "size": 12}])
    hf = _StubLister()
    service = AssetService(
        job_repository=_FakeJobRepo(job),  # type: ignore[arg-type]
        principal=_principal(),
        settings=Settings(),
        filesystem=FilesystemArtifactLister(),
        huggingface=hf,
    )

    assets = await service.list_assets(job.id)

    assert [a.filename for a in assets] == ["results.csv"]
    assert hf.seen is None  # source not touched


async def test_prepopulated_column_maps_tolerant_shapes() -> None:
    job = _Job(
        output_artifacts=[
            {"filename": "results.csv", "size": 128, "modified": "2026-08-14T00:00:00+00:00"},
            {"name": "best_config.json", "file_size": 64},
        ]
    )
    service = AssetService(
        job_repository=_FakeJobRepo(job),  # type: ignore[arg-type]
        principal=_principal(),
        settings=Settings(),
        filesystem=FilesystemArtifactLister(),
        huggingface=_StubLister(),
    )

    assets = await service.list_assets(job.id)

    by_name = {a.filename for a in assets}
    assert by_name == {"results.csv", "best_config.json"}
    config = next(a for a in assets if a.filename == "best_config.json")
    assert config.size == 64  # size falls back to file_size, filename falls back to name


async def test_hf_task_lists_via_huggingface() -> None:
    job = _Job(
        tasks=[
            _Task(
                artifact_uri="hf://huggingface.co/models/org/repo",
                build_status={"details": {"status": "success"}},
            )
        ]
    )
    hf = _StubLister()
    service = AssetService(
        job_repository=_FakeJobRepo(job),  # type: ignore[arg-type]
        principal=_principal(),
        settings=Settings(),
        filesystem=FilesystemArtifactLister(),
        huggingface=hf,
    )

    assets = await service.list_assets(job.id)

    assert hf.seen == "org/repo"
    assert assets[0].filename == "model.safetensors"


async def test_build_not_success_raises_not_ready() -> None:
    job = _Job(
        tasks=[
            _Task(
                artifact_uri="hf://huggingface.co/models/org/repo",
                build_status={"details": {"status": "running"}},
            )
        ]
    )
    service = AssetService(
        job_repository=_FakeJobRepo(job),  # type: ignore[arg-type]
        principal=_principal(),
        settings=Settings(),
        filesystem=FilesystemArtifactLister(),
        huggingface=_StubLister(),
    )

    with pytest.raises(JobArtifactsNotReadyError):
        await service.list_assets(job.id)


async def test_local_completed_job_scans_output_dir(tmp_path: Path) -> None:
    job = _Job(status=RunStatus.COMPLETED)
    results = tmp_path / str(job.id) / "results"
    results.mkdir(parents=True)
    (results / "weights.zip").write_bytes(b"zzz")
    service = AssetService(
        job_repository=_FakeJobRepo(job),  # type: ignore[arg-type]
        principal=_principal(),
        settings=Settings(local_output_dir=tmp_path),
        filesystem=FilesystemArtifactLister(),
        huggingface=_StubLister(),
    )

    assets = await service.list_assets(job.id)

    assert [a.filename for a in assets] == ["weights.zip"]


async def test_local_incomplete_job_raises_not_ready() -> None:
    job = _Job(status=RunStatus.RUNNING)
    service = AssetService(
        job_repository=_FakeJobRepo(job),  # type: ignore[arg-type]
        principal=_principal(),
        settings=Settings(),
        filesystem=FilesystemArtifactLister(),
        huggingface=_StubLister(),
    )

    with pytest.raises(JobArtifactsNotReadyError):
        await service.list_assets(job.id)


async def test_local_completed_job_without_results_dir_raises_not_found(tmp_path: Path) -> None:
    job = _Job(status=RunStatus.COMPLETED)
    service = AssetService(
        job_repository=_FakeJobRepo(job),  # type: ignore[arg-type]
        principal=_principal(),
        settings=Settings(local_output_dir=tmp_path),
        filesystem=FilesystemArtifactLister(),
        huggingface=_StubLister(),
    )

    with pytest.raises(JobArtifactsNotFoundError):
        await service.list_assets(job.id)


def _local_service(job: _Job, tmp_path: Path) -> AssetService:
    return AssetService(
        job_repository=_FakeJobRepo(job),  # type: ignore[arg-type]
        principal=_principal(),
        settings=Settings(local_output_dir=tmp_path),
        filesystem=FilesystemArtifactLister(),
        huggingface=_StubLister(),
    )


async def test_download_file_streams_from_local_source_by_relative_path(tmp_path: Path) -> None:
    job = _Job(status=RunStatus.COMPLETED)
    weights = tmp_path / str(job.id) / "results" / "weights"
    weights.mkdir(parents=True)
    (weights / "adapter.safetensors").write_bytes(b"WEIGHTS")
    service = _local_service(job, tmp_path)

    opened = await service.download_file(job.id, path="weights/adapter.safetensors")

    assert b"".join([chunk async for chunk in opened.stream]) == b"WEIGHTS"
    assert opened.filename == "adapter.safetensors"


async def test_download_file_missing_job_raises_not_found() -> None:
    service = AssetService(
        job_repository=_FakeJobRepo(None),  # type: ignore[arg-type]
        principal=_principal(),
        settings=Settings(),
        filesystem=FilesystemArtifactLister(),
        huggingface=_StubLister(),
    )

    with pytest.raises(JobNotFoundError):
        await service.download_file(uuid4(), path="anything")


async def test_download_file_not_ready_raises_not_ready() -> None:
    job = _Job(status=RunStatus.RUNNING)
    service = AssetService(
        job_repository=_FakeJobRepo(job),  # type: ignore[arg-type]
        principal=_principal(),
        settings=Settings(),
        filesystem=FilesystemArtifactLister(),
        huggingface=_StubLister(),
    )

    with pytest.raises(JobArtifactsNotReadyError):
        await service.download_file(job.id, path="anything")


async def test_open_archive_names_zip_from_experiment_and_bundles_all_files(
    tmp_path: Path,
) -> None:
    job = _Job(status=RunStatus.COMPLETED, experiment_name="SmolLM2_lora-mac")
    results = tmp_path / str(job.id) / "results"
    (results / "run-a").mkdir(parents=True)
    (results / "run-b").mkdir(parents=True)
    (results / "final_config.json").write_text('{"a": 1}')
    (results / "run-a" / "adapters.safetensors").write_bytes(b"AAAA")
    (results / "run-b" / "adapters.safetensors").write_bytes(b"BBBBBB")
    service = _local_service(job, tmp_path)

    name, stream = await service.open_archive(job.id)
    blob = b"".join([chunk async for chunk in stream])

    assert name == "SmolLM2_lora-mac_assets.zip"
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        assert set(archive.namelist()) == {
            "final_config.json",
            "run-a/adapters.safetensors",
            "run-b/adapters.safetensors",
        }
        assert archive.read("run-b/adapters.safetensors") == b"BBBBBB"


async def test_open_archive_sanitizes_unsafe_experiment_name(tmp_path: Path) -> None:
    job = _Job(status=RunStatus.COMPLETED, experiment_name="a/b c:d")
    results = tmp_path / str(job.id) / "results"
    results.mkdir(parents=True)
    (results / "final_config.json").write_text("{}")
    service = _local_service(job, tmp_path)

    name, _stream = await service.open_archive(job.id)

    assert "/" not in name
    assert name.endswith("_assets.zip")
