# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""A preview-only fallback decorator over two storage backends.

When the primary backend cannot produce a preview — no rows in either split — the
decorator reads a bounded preview from a fallback backend instead. Only ``preview``
is augmented; ``persist`` and ``delete`` belong to the active backend and delegate
straight to the primary, so wrapping never changes where a dataset is written.

The decorator is deliberately **direction-agnostic**, and the registry composes it
both ways:

- ``primary=HuggingFace, fallback=local`` under the ``huggingface`` backend, so a
  dataset whose files also live on local disk (carried over from an earlier version
  of the app, or uploaded through the local path) still previews when the HF viewer
  cannot answer — no ``artifact_url``, viewer disabled/unavailable, or token missing.
- ``primary=local, fallback=HuggingFace`` whenever storage resolves to local, so a
  dataset row carrying an ``hf://`` locator still previews even though the local
  write path holds none of its files. Resolving to local storage is a *write*-path
  decision (``llmb artifact push`` is disabled in granite.build standalone mode);
  the HF dataset-viewer read is unaffected by it.

Either way the alternative is the UX's "Unable to load dataset" on a dataset that
is perfectly readable.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from autotunex.core.logging import get_logger
from autotunex.models.dataset import DatasetPreview
from autotunex.services.storage.base import StorageBackend

logger = get_logger(__name__)


class PreviewFallbackStorageBackend:
    """Wrap a ``primary`` backend, reading ``preview`` from ``fallback`` when empty.

    Satisfies :class:`~autotunex.services.storage.base.StorageBackend`
    structurally, so it is a drop-in wherever a ``StorageBackend`` is injected.
    """

    def __init__(self, *, primary: StorageBackend, fallback: StorageBackend) -> None:
        self._primary = primary
        self._fallback = fallback

    async def persist(
        self,
        *,
        dataset_id: UUID,
        name: str,
        data_format: str,
        train: Path,
        validation: Path | None,
    ) -> tuple[UUID | None, str | None]:
        """Delegate persistence to the primary backend (writes are not mirrored)."""
        return await self._primary.persist(
            dataset_id=dataset_id,
            name=name,
            data_format=data_format,
            train=train,
            validation=validation,
        )

    async def preview(
        self,
        *,
        dataset_id: UUID,
        name: str,
        data_format: str,
        artifact_url: str | None,
        rows: int,
    ) -> DatasetPreview:
        """Return the primary's preview, or the fallback's when the primary is empty.

        A preview counts as empty only when *both* splits have no rows; a partial
        preview (train rows, no validation split) is a success and is returned
        as-is. The fallback read is guarded so a failure degrades to the primary's
        empty result — this method never raises and always returns a
        :class:`~autotunex.models.dataset.DatasetPreview`.

        When the fallback is *also* empty, the result carrying
        ``viewer_ready=False`` wins, so "the HF viewer is still precomputing"
        survives instead of being masked by the local backend's default
        ``viewer_ready=True`` — otherwise a freshly-pushed, not-yet-indexed dataset
        would report "ready" with a blank table instead of "preview will appear
        shortly". Which side holds that signal depends on the composition (see the
        module docstring), so this cannot simply prefer the primary.
        """
        primary_result = await self._primary.preview(
            dataset_id=dataset_id,
            name=name,
            data_format=data_format,
            artifact_url=artifact_url,
            rows=rows,
        )
        if primary_result.train or primary_result.validation:
            return primary_result
        # Broad except is intentional: a preview must never fail the metadata read,
        # so a fallback failure degrades to the primary's (empty) result rather than
        # propagating. `BLE001` is not in this project's selected ruff rules.
        try:
            fallback_result = await self._fallback.preview(
                dataset_id=dataset_id,
                name=name,
                data_format=data_format,
                artifact_url=artifact_url,
                rows=rows,
            )
        except Exception:
            logger.warning(
                "Local preview fallback failed for dataset %s; returning empty preview.",
                dataset_id,
            )
            return primary_result
        if fallback_result.train or fallback_result.validation:
            return fallback_result
        # Both empty: keep whichever side reported a not-ready viewer, so a
        # freshly-pushed, not-yet-indexed HF repo still reports "preview will appear
        # shortly" instead of a blank table claiming to be ready. Which side that is
        # depends on the composition — HF is the primary under the `huggingface`
        # backend and the fallback under `local` — so this cannot assume the primary.
        if not fallback_result.viewer_ready and primary_result.viewer_ready:
            return fallback_result
        return primary_result

    async def delete(self, *, dataset_id: UUID, name: str, artifact_url: str | None) -> None:
        """Delegate deletion to the primary backend."""
        await self._primary.delete(dataset_id=dataset_id, name=name, artifact_url=artifact_url)
