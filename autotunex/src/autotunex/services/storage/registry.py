# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Storage backend selection from settings (the ``ArtifactStore`` seam)."""

from __future__ import annotations

import os
import shutil
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


def _llmb_enabled(settings: Settings) -> bool:
    """Usable when ``llmb`` resolves and BOTH tokens are present.

    The HF push needs two credentials: the GB token authenticates the CLI
    (``llmb auth login``) and the HF token is the push destination (``--store hf``),
    so ``auto`` only chooses HuggingFace when both env vars are set.
    """
    return bool(
        shutil.which(settings.llmb_command)
        and os.environ.get(settings.gb_token_env)
        and os.environ.get(settings.hf_token_env)
    )


def get_storage_backend(settings: Settings) -> StorageBackend:
    """Return the storage backend chosen by ``dataset_storage_backend`` and env.

    ``"local"`` forces local storage. ``"huggingface"`` forces the HF push backend
    (wrapped with a local preview fallback); a forced ``huggingface`` with a
    missing token, or in the same-host bash standalone case
    (``gb_environment="standalone"`` without ``lsf_cluster``), is already refused at
    settings validation (``Settings._validate_datasets``) — the LSF/SkyPilot
    standalone variant keeps ``huggingface``. ``"auto"`` resolves to HuggingFace
    only when ``llmb`` and both tokens are available *and* the CLI is not in
    standalone mode, else local.

    In granite.build **standalone** mode ``llmb artifact push`` is disabled, so the
    HF push backend cannot run: storage falls back to local regardless of tokens.
    For the same-host local-bash build (``lsf_cluster`` unset) the local backend
    additionally emits a ``file://`` locator that gbserver mounts as its
    ``dataset_files`` input; the remote LSF/SkyPilot build cannot read a local
    path, so no locator is emitted there (its dataset hosting is a separate, open
    concern).
    """
    root = settings.dataset_storage_dir
    standalone = settings.gb_environment == "standalone"
    # Only the same-host bash build consumes the dataset off local disk.
    local_bash_standalone = standalone and not settings.lsf_cluster

    if settings.dataset_storage_backend == "local":
        return LocalStorageBackend(root=root, emit_file_uri=local_bash_standalone)
    if settings.dataset_storage_backend == "huggingface":
        # Validation refuses a forced `huggingface` only for the same-host bash
        # standalone case, so this branch is reached for non-standalone deployments
        # (where `llmb artifact push` works) and for the LSF/SkyPilot standalone
        # variant (remote cluster; its push limitation is a deferred non-goal).
        return _huggingface_with_local_fallback(settings)
    # auto:
    if standalone:
        logger.info(
            "dataset_storage_backend=auto with gb_environment=standalone: "
            "`llmb artifact push` is unavailable; using local storage%s.",
            " with a file:// locator" if local_bash_standalone else "",
        )
        return LocalStorageBackend(root=root, emit_file_uri=local_bash_standalone)
    if _llmb_enabled(settings):
        return _huggingface_with_local_fallback(settings)
    logger.info(
        "dataset_storage_backend=auto: llmb or %s/%s unavailable, using local storage.",
        settings.gb_token_env,
        settings.hf_token_env,
    )
    return LocalStorageBackend(root=root)


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
