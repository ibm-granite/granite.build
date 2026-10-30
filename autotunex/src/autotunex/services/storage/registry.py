# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Storage backend selection from settings (the ``ArtifactStore`` seam)."""

from __future__ import annotations

from urllib.parse import urlparse

from autotunex.core.config import Settings
from autotunex.core.exceptions import JobArtifactsNotFoundError
from autotunex.core.logging import get_logger
from autotunex.services.storage.artifacts import ArtifactLister
from autotunex.services.storage.base import StorageBackend
from autotunex.services.storage.fallback import PreviewFallbackStorageBackend
from autotunex.services.storage.hf_viewer import repo_id_from_artifact_url
from autotunex.services.storage.huggingface import HuggingFaceStorageBackend
from autotunex.services.storage.local import LocalStorageBackend

logger = get_logger(__name__)


def _huggingface(settings: Settings) -> HuggingFaceStorageBackend:
    return HuggingFaceStorageBackend(
        llmb_command=settings.llmb_command,
        hf_token_env=settings.hf_token_env,
        gb_token_env=settings.gb_token_env,
        hf_namespace=settings.hf_namespace,
        hf_preview_enabled=settings.hf_preview_enabled,
        hf_viewer_base_url=settings.hf_viewer_base_url,
        hf_viewer_timeout_seconds=settings.hf_viewer_timeout_seconds,
        tags=settings.gb_tags,
        push_timeout_seconds=settings.dataset_push_timeout_seconds,
    )


def _huggingface_with_local_fallback(settings: Settings) -> PreviewFallbackStorageBackend:
    """HuggingFace primary with a local-storage preview fallback.

    When the HF viewer cannot serve a preview (no ``artifact_url``, viewer
    disabled/unavailable, or token missing) the wrapper reads the preview from
    the same local directory the local backend uses, so datasets whose files also
    live on disk still render rather than showing "Unable to load dataset".
    """
    return PreviewFallbackStorageBackend(
        primary=_huggingface(settings),
        fallback=LocalStorageBackend(root=settings.dataset_storage_dir),
    )


def _local_with_hf_preview_fallback(
    settings: Settings, *, emit_file_uri: bool
) -> PreviewFallbackStorageBackend:
    """Local-storage primary with a HuggingFace preview fallback (the mirror above).

    Resolving to local storage is a *write*-path decision: ``llmb artifact push``
    is disabled under ``GB_ENVIRONMENT=STANDALONE``, and ``auto`` degrades when
    ``llmb`` or the tokens are missing. Reading a preview instead goes over the HF
    dataset-viewer HTTP API, which needs neither ``llmb`` nor the GB token — so the
    two decisions do not follow from each other, and collapsing them silently cost
    the preview of any dataset whose files live on HuggingFace.

    Wrapping keeps every write purely local (``persist``/``delete`` delegate to the
    primary, so the ``file://`` locator the bash build mounts is unchanged) while a
    dataset row that already carries an ``hf://`` locator — pushed before the
    deployment switched to standalone, or written by another deployment sharing the
    database — still previews rather than rendering "Unable to load dataset".
    """
    return PreviewFallbackStorageBackend(
        primary=LocalStorageBackend(root=settings.dataset_storage_dir, emit_file_uri=emit_file_uri),
        fallback=_huggingface(settings),
    )


def get_storage_backend(settings: Settings) -> StorageBackend:
    """Return the storage backend named by ``settings.resolved_dataset_storage``.

    That property is the single dataset-storage decision (``"local"``,
    ``"huggingface"`` or ``"auto"`` resolved against the deployment shape, ``llmb``
    and both token env vars); ``hf_import_available`` gates on the same value, so
    import is never open where this function would store with no usable locator.
    A forced ``huggingface`` with a missing token, or in the same-host bash
    standalone case, is already refused at settings validation
    (``Settings._validate_datasets``).

    In granite.build **standalone** mode ``llmb artifact push`` is disabled, so
    ``auto`` stores locally regardless of tokens. For the same-host local-bash build
    (``lsf_cluster`` unset) the local backend additionally emits a ``file://``
    locator that gbserver mounts as its ``dataset_files`` input; the remote
    LSF/SkyPilot build cannot read a local path, so no locator is emitted there (its
    dataset hosting is a separate, open concern).

    Every branch that resolves to local storage returns it wrapped with a
    HuggingFace **preview** fallback (``_local_with_hf_preview_fallback``), so the
    write-path decision above never costs the preview of a dataset row that already
    carries an ``hf://`` locator. Writes stay local either way.
    """
    storage = settings.resolved_dataset_storage
    if storage == "huggingface":
        return _huggingface_with_local_fallback(settings)
    emit_file_uri = storage == "local_file_uri"
    if settings.dataset_storage_backend == "auto":
        if settings.gb_environment == "standalone":
            logger.info(
                "dataset_storage_backend=auto with gb_environment=standalone: "
                "`llmb artifact push` is unavailable; using local storage%s.",
                " with a file:// locator" if emit_file_uri else "",
            )
        else:
            logger.info(
                "dataset_storage_backend=auto: llmb or %s/%s unavailable, using local storage.",
                settings.gb_token_env,
                settings.hf_token_env,
            )
    return _local_with_hf_preview_fallback(settings, emit_file_uri=emit_file_uri)


def resolve_artifact_lister(
    artifact_uri: str, *, filesystem: ArtifactLister, huggingface: ArtifactLister
) -> tuple[ArtifactLister, str]:
    """Return the ``(lister, location)`` for a stored ``artifact_uri``, by scheme.

    ``hf://`` yields the HuggingFace lister and the derived ``owner/repo`` id;
    ``file://`` yields the filesystem lister and the local path. An unrecognised
    scheme, or a value that yields no repo id / path, raises
    :class:`JobArtifactsNotFoundError`.
    """
    uri = artifact_uri.strip()
    if uri.startswith("hf://"):
        repo_id = repo_id_from_artifact_url(uri)
        if repo_id is None:
            raise JobArtifactsNotFoundError
        return huggingface, repo_id
    if uri.startswith("file://"):
        path = urlparse(uri).path
        if not path:
            raise JobArtifactsNotFoundError
        return filesystem, path
    raise JobArtifactsNotFoundError
