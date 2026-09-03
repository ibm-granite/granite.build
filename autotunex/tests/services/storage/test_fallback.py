# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :class:`PreviewFallbackStorageBackend`.

The decorator augments only the preview path: when the primary backend returns no
rows in either split, it reads from the fallback; ``persist``/``delete`` delegate
to the primary. It must never raise from ``preview`` — a fallback failure degrades
to the primary's (empty) result so the metadata read still gets a non-null preview.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

from autotunex.models.dataset import DatasetPreview
from autotunex.services.storage.fallback import PreviewFallbackStorageBackend


class RecordingBackend:
    """A ``StorageBackend`` test double with a fixed preview and call counters."""

    def __init__(self, preview_result: DatasetPreview, *, raises: bool = False) -> None:
        self._preview_result = preview_result
        self._raises = raises
        self.preview_calls = 0
        self.persist_calls = 0
        self.delete_calls = 0

    async def persist(
        self,
        *,
        dataset_id: UUID,
        name: str,
        data_format: str,
        train: Path,
        validation: Path | None,
    ) -> tuple[UUID | None, str | None]:
        self.persist_calls += 1
        return (None, None)

    async def preview(
        self,
        *,
        dataset_id: UUID,
        name: str,
        data_format: str,
        artifact_url: str | None,
        rows: int,
    ) -> DatasetPreview:
        self.preview_calls += 1
        if self._raises:
            raise RuntimeError("backend down")
        return self._preview_result

    async def delete(self, *, dataset_id: UUID, name: str, artifact_url: str | None) -> None:
        self.delete_calls += 1


def _empty() -> DatasetPreview:
    return DatasetPreview(train=[], validation=[])


async def _preview(backend: PreviewFallbackStorageBackend) -> DatasetPreview:
    return await backend.preview(
        dataset_id=uuid4(),
        name="ds",
        data_format="jsonl",
        artifact_url="hf://huggingface.co/datasets/org/ds_abcd1234",
        rows=5,
    )


async def test_preview_returns_primary_and_skips_fallback_when_primary_has_rows() -> None:
    primary = RecordingBackend(DatasetPreview(train=[{"a": 1}], validation=[{"b": 2}]))
    fallback = RecordingBackend(DatasetPreview(train=[{"z": 9}], validation=[{"z": 9}]))
    backend = PreviewFallbackStorageBackend(primary=primary, fallback=fallback)

    result = await _preview(backend)

    assert result.train == [{"a": 1}]
    assert fallback.preview_calls == 0


async def test_preview_uses_fallback_when_primary_is_empty() -> None:
    primary = RecordingBackend(_empty())
    fallback = RecordingBackend(DatasetPreview(train=[{"local": 1}], validation=[{"local": 2}]))
    backend = PreviewFallbackStorageBackend(primary=primary, fallback=fallback)

    result = await _preview(backend)

    assert result.train == [{"local": 1}]
    assert result.validation == [{"local": 2}]
    assert fallback.preview_calls == 1


async def test_preview_returns_empty_when_both_are_empty() -> None:
    primary = RecordingBackend(_empty())
    fallback = RecordingBackend(_empty())
    backend = PreviewFallbackStorageBackend(primary=primary, fallback=fallback)

    result = await _preview(backend)

    assert result.train == []
    assert result.validation == []


async def test_preview_preserves_primary_not_ready_when_fallback_is_also_empty() -> None:
    # HF primary: viewer still precomputing -> empty + viewer_ready=False.
    # Local fallback: no local files for an HF-hosted dataset -> empty (+ default
    # viewer_ready=True). The primary's not-ready signal must survive rather than
    # be masked by the fallback's default, so the UI can still say "preview will
    # appear shortly" instead of showing a bare blank table.
    primary = RecordingBackend(DatasetPreview(train=[], validation=[], viewer_ready=False))
    fallback = RecordingBackend(_empty())
    backend = PreviewFallbackStorageBackend(primary=primary, fallback=fallback)

    result = await _preview(backend)

    assert result.train == []
    assert result.validation == []
    assert result.viewer_ready is False
    assert fallback.preview_calls == 1  # fallback is still consulted in case local has rows


async def test_preview_degrades_to_primary_when_fallback_raises() -> None:
    primary = RecordingBackend(_empty())
    fallback = RecordingBackend(_empty(), raises=True)
    backend = PreviewFallbackStorageBackend(primary=primary, fallback=fallback)

    result = await _preview(backend)  # must not raise

    assert result.train == []
    assert result.validation == []


async def test_preview_does_not_fall_back_on_partial_primary_result() -> None:
    # Train rows present but no validation split is a successful preview, not a
    # trigger — falling back here would mask the real (train-only) dataset.
    primary = RecordingBackend(DatasetPreview(train=[{"a": 1}], validation=[]))
    fallback = RecordingBackend(DatasetPreview(train=[{"z": 9}], validation=[{"z": 9}]))
    backend = PreviewFallbackStorageBackend(primary=primary, fallback=fallback)

    result = await _preview(backend)

    assert result.train == [{"a": 1}]
    assert result.validation == []
    assert fallback.preview_calls == 0


async def test_persist_and_delete_delegate_to_primary_only() -> None:
    primary = RecordingBackend(_empty())
    fallback = RecordingBackend(_empty())
    backend = PreviewFallbackStorageBackend(primary=primary, fallback=fallback)

    await backend.persist(
        dataset_id=uuid4(), name="ds", data_format="jsonl", train=Path("t"), validation=None
    )
    await backend.delete(dataset_id=uuid4(), name="ds", artifact_url=None)

    assert (primary.persist_calls, primary.delete_calls) == (1, 1)
    assert (fallback.persist_calls, fallback.delete_calls) == (0, 0)


async def test_preview_preserves_a_not_ready_signal_from_the_fallback() -> None:
    # The reversed composition (local primary, HF fallback) puts the viewer's
    # readiness signal in the *fallback*. Local storage always reports
    # viewer_ready=True, so returning the primary unconditionally when both are
    # empty would mask "still precomputing" behind a blank, ready-looking table —
    # the same defect commit 6752f089 fixed for the primary=HF direction.
    primary = RecordingBackend(_empty())
    fallback = RecordingBackend(DatasetPreview(train=[], validation=[], viewer_ready=False))
    backend = PreviewFallbackStorageBackend(primary=primary, fallback=fallback)

    result = await _preview(backend)

    assert result.viewer_ready is False
