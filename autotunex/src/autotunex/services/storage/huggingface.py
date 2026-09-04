# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace-backed dataset storage via the ``llmb artifact push`` CLI.

Ports the 2025 ``GBStorageBackend.persist``: authenticate the CLI, then push the
finalized train/validation files as a HuggingFace dataset artifact and record the
returned ``(artifact_id, artifact_url)``. The command mirrors the 2025 uploader's
push to HF, with the tag now driven by the ``gb_tags`` setting::

    llmb artifact push --from-local <dir> --artifact-name <name>_<id[:8]>
        --type dataset --store hf --certify-no-restrictions [--tag <gb_tags>]

The ``--tag`` argument is appended only when ``gb_tags`` is non-empty (it defaults
to ``autotunex``, preserving the tag this push historically carried).

Two credentials are involved and neither value is ever loaded into ``Settings``:

  - **GB_TOKEN** (env var named by ``gb_token_env``) authenticates the CLI via
    ``llmb auth login --token`` before the push — the same contract as the job
    launcher (``gb_service.login_gb``);
  - **HF_TOKEN** (env var named by ``hf_token_env``) is the HuggingFace
    destination credential; ``--store hf`` routes the push there and ``llmb``
    reads the token from the forwarded process environment.

The push runs inside ``asyncio.to_thread`` (the runner is already off the request
path, but this keeps the event loop free during a slow push). Tokens are never
logged.

``preview`` reads rows from the HuggingFace dataset viewer
(``datasets-server.huggingface.co``) via ``services.storage.hf_viewer``, deriving
the ``owner/repo`` from the dataset's stored ``hf://`` ``artifact_url``.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx

from autotunex.core.exceptions import DatasetPushFailedError, DatasetPushTimeoutError
from autotunex.core.logging import get_logger
from autotunex.models.dataset import DatasetPreview
from autotunex.services.storage.hf_viewer import (
    HFViewerUnavailable,
    discover_config,
    fetch_rows,
    repo_id_from_artifact_url,
)

logger = get_logger(__name__)

# ``artifact_id`` is a UUID; ``artifact_url`` is the pushed artifact locator.
# ``--store hf`` yields an ``hf://`` URI (an earlier internal push path returned
# ``lh://``); accept both, plus a plain HF https URL, taking the first match.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_URI_RE = re.compile(r"(?:hf://|lh://|https://huggingface\.co/)\S+")


def _rows_or_empty(result: list[dict[str, Any]] | BaseException) -> list[dict[str, Any]]:
    """Unwrap one ``asyncio.gather`` split result: rows on success, ``[]`` on error.

    A ``HFViewerUnavailable`` (viewer down or still precomputing) is logged at
    WARNING — its message names the repo, config, split, and HTTP status, so an
    empty preview is no longer silent — while any other exception is logged at
    DEBUG. Either way a failing split degrades to an empty list so the other split
    still shows.
    """
    if isinstance(result, BaseException):
        if isinstance(result, HFViewerUnavailable):
            logger.warning("HF dataset viewer unavailable for a split: %s", result)
        else:
            logger.debug("HF split fetch failed unexpectedly: %r", result)
        return []
    return result


class HuggingFaceStorageBackend:
    """Satisfies :class:`~autotunex.services.storage.base.StorageBackend`."""

    def __init__(
        self,
        *,
        llmb_command: str,
        hf_token_env: str,
        gb_token_env: str,
        hf_namespace: str | None,
        hf_preview_enabled: bool,
        hf_viewer_base_url: str,
        hf_viewer_timeout_seconds: float,
        tags: str,
        push_timeout_seconds: float,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._llmb_command = llmb_command
        self._hf_token_env = hf_token_env
        self._gb_token_env = gb_token_env
        self._hf_namespace = hf_namespace
        self._hf_preview_enabled = hf_preview_enabled
        self._hf_viewer_base_url = hf_viewer_base_url
        self._hf_viewer_timeout_seconds = hf_viewer_timeout_seconds
        # Value passed verbatim as `artifact push`'s --tag (single tag or a
        # comma-separated list; llmb splits it). Empty/whitespace omits the flag.
        self._tags = tags
        self._push_timeout_seconds = push_timeout_seconds
        self._http_client = http_client

    def _artifact_name(self, dataset_id: UUID, name: str) -> str:
        """Derive the artifact name ``[<namespace>/]<name>_<id[:8]>`` (2025 shape)."""
        base = f"{name}_{str(dataset_id)[:8]}"
        return f"{self._hf_namespace}/{base}" if self._hf_namespace else base

    async def persist(
        self,
        *,
        dataset_id: UUID,
        name: str,
        data_format: str,
        train: Path,
        validation: Path | None,
    ) -> tuple[UUID | None, str | None]:
        """Push the files to HuggingFace via ``llmb artifact push``; return artifact refs."""
        artifact_name = self._artifact_name(dataset_id, name)
        completed = await asyncio.to_thread(
            self._run_push, artifact_name, name, data_format, train, validation
        )
        artifact_id, artifact_url = self._parse_output(completed.stdout)
        if artifact_id is None and artifact_url is None:
            logger.error(
                "llmb push produced no parseable artifact id/url for %s; stdout=%r",
                dataset_id,
                completed.stdout,
            )
            raise DatasetPushFailedError
        return artifact_id, artifact_url

    def _staged_path(self, push_dir: Path, name: str, data_format: str, *, split: str) -> Path:
        """Build the staging path for one split, refusing to escape ``push_dir``.

        Mirrors ``LocalStorageBackend._path``'s containment guard (Task 9):
        ``name`` is expected to already be validated at the API boundary, but
        this is a second, independent layer against a traversal-bearing name
        reaching this method by some other route before ``shutil.copy2`` ever
        writes to the resolved path.
        """
        candidate = push_dir / f"{name}_{split}.{data_format}"
        base = push_dir.resolve()
        if not candidate.resolve().is_relative_to(base):
            raise ValueError("resolved staging path escapes its push directory")
        return candidate

    def _run_push(
        self,
        artifact_name: str,
        name: str,
        data_format: str,
        train: Path,
        validation: Path | None,
    ) -> subprocess.CompletedProcess[str]:
        """Stage the files into a clean dir, authenticate, and run ``llmb artifact push``.

        The prepared train/validation files can share a staging directory with the
        original upload and intermediate split files, so they are copied into a
        fresh temp directory under their canonical ``<name>_<split>.<ext>`` names
        (the names the trainer reads back). ``--from-local`` then pushes exactly
        those files and nothing else. The temp directory is always removed.

        Runs inside ``asyncio.to_thread``; ``env=dict(os.environ)`` forwards the HF
        token so ``llmb`` can read it for ``--store hf``.
        """
        push_dir = Path(tempfile.mkdtemp(prefix="llmb_push_"))
        try:
            shutil.copy2(train, self._staged_path(push_dir, name, data_format, split="train"))
            if validation is not None:
                shutil.copy2(
                    validation,
                    self._staged_path(push_dir, name, data_format, split="validation"),
                )
            self._authenticate()
            args = [
                self._llmb_command,
                "artifact",
                "push",
                "--from-local",
                str(push_dir),
                "--artifact-name",
                artifact_name,
                "--type",
                "dataset",
                "--store",
                "hf",
                "--certify-no-restrictions",
            ]
            # An empty/whitespace-only setting means "no tags": omit --tag rather
            # than pass an empty value the CLI would reject.
            if self._tags.strip():
                args += ["--tag", self._tags]
            try:
                return subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=self._push_timeout_seconds,
                    env=dict(os.environ),
                )
            except subprocess.TimeoutExpired:
                logger.error(
                    "`%s artifact push` timed out after %ss.",
                    self._llmb_command,
                    self._push_timeout_seconds,
                )
                # The push argv carries no token (HF token is in the env, not
                # argv), so this is defense-in-depth against a future argv
                # change rather than a live leak today; `from None` sets
                # __cause__ to None and __suppress_context__ to True (it does
                # NOT clear __context__ — CPython still records the in-flight
                # TimeoutExpired there), which together stop
                # traceback.format_exception/logger.exception from rendering
                # the argv-bearing exception at all.
                raise DatasetPushTimeoutError(self._push_timeout_seconds) from None
            except FileNotFoundError as exc:
                logger.error("llmb command %r not found for artifact push.", self._llmb_command)
                raise DatasetPushFailedError from exc
            except subprocess.CalledProcessError as exc:
                # The push argv carries no token (the HF token is in the env, not
                # argv), so logging stderr/exit here is safe; still, only a fixed
                # message is surfaced to the caller.
                logger.error(
                    "`%s artifact push` failed (exit %s): %s",
                    self._llmb_command,
                    exc.returncode,
                    (exc.stderr or "").strip(),
                )
                raise DatasetPushFailedError from exc
        finally:
            shutil.rmtree(push_dir, ignore_errors=True)

    def _authenticate(self) -> None:
        """Run ``llmb auth login --token <token>`` when the GB token env var is set.

        Ports ``gb_service.login_gb``. The token value is read from the environment
        (its var name is ``gb_token_env``) and never loaded into ``Settings`` or
        logged. When unset, authentication is skipped and ambient credentials are
        relied on. A failed login raises :class:`DatasetPushFailedError` (a safe,
        fixed message; the runner records it verbatim as ``status='error'``'s
        detail) — the raw stderr/exit is logged here, never surfaced, and the
        token is never in it — deliberately not ``check=True``, whose
        ``CalledProcessError`` repr embeds the full argv, including the token,
        into logs and tracebacks. The same reasoning applies to a timeout:
        ``subprocess.TimeoutExpired.cmd`` is the full argv (token included),
        so its handler raises with ``from None`` — setting ``__cause__`` to
        ``None`` and ``__suppress_context__`` to ``True`` — rather than a bare
        ``raise`` (or ``from exc``), either of which would leave the
        token-bearing exception rendered in a traceback, which is exactly
        what ``logger.exception`` in the runner would print.
        """
        token = os.environ.get(self._gb_token_env)
        if not token:
            logger.info(
                "%s not set; skipping `llmb auth login` (using ambient credentials).",
                self._gb_token_env,
            )
            return
        logger.info("Authenticating the llmb CLI via `%s auth login`.", self._llmb_command)
        try:
            result = subprocess.run(
                [self._llmb_command, "auth", "login", "--token", token],
                capture_output=True,
                text=True,
                timeout=self._push_timeout_seconds,
                env=dict(os.environ),
            )
        except subprocess.TimeoutExpired:
            logger.error(
                "`%s auth login` timed out after %ss.",
                self._llmb_command,
                self._push_timeout_seconds,
            )
            raise DatasetPushTimeoutError(self._push_timeout_seconds) from None
        except FileNotFoundError as exc:
            logger.error("llmb command %r not found for auth login.", self._llmb_command)
            raise DatasetPushFailedError from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no output"
            logger.error(
                "`%s auth login` failed (exit %s): %s",
                self._llmb_command,
                result.returncode,
                detail,
            )
            raise DatasetPushFailedError

    def _parse_output(self, stdout: str) -> tuple[UUID | None, str | None]:
        """Extract ``(artifact_id, artifact_url)`` from ``llmb`` stdout (2025 extract_uuid_uri)."""
        uuid_match = _UUID_RE.search(stdout)
        uri_match = _URI_RE.search(stdout)
        return (
            UUID(uuid_match.group()) if uuid_match else None,
            uri_match.group() if uri_match else None,
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
        """Best-effort bounded preview from the HuggingFace dataset viewer.

        Reads at most ``rows`` rows per split from ``datasets-server.huggingface.co``
        using the dataset's stored ``hf://`` ``artifact_url``. Every failure mode —
        disabled, an underivable or foreign locator, a missing token, or a viewer
        that is unavailable or still precomputing — degrades to an empty (or
        partial) preview. This method never raises: a preview must never fail a
        metadata read.
        """
        empty = DatasetPreview(train=[], validation=[])
        try:
            if not self._hf_preview_enabled:
                logger.debug("HF preview disabled; returning empty for %s.", dataset_id)
                return empty
            repo_id = repo_id_from_artifact_url(artifact_url)
            if repo_id is None:
                logger.debug("No hf:// locator for %s; returning empty preview.", dataset_id)
                return empty
            expected_suffix = f"_{str(dataset_id)[:8]}"
            if not repo_id.rsplit("/", 1)[-1].endswith(expected_suffix):
                logger.warning(
                    "HF locator %s does not belong to dataset %s; returning empty preview.",
                    repo_id,
                    dataset_id,
                )
                return empty
            token = os.environ.get(self._hf_token_env)
            if not token:
                logger.warning(
                    "%s not set; cannot read HF preview for %s.",
                    self._hf_token_env,
                    dataset_id,
                )
                return empty
            return await self._preview_from_viewer(repo_id, token, rows)
        except Exception:
            logger.exception("Unexpected HF preview failure for %s; returning empty.", dataset_id)
            return empty

    async def _preview_from_viewer(self, repo_id: str, token: str, rows: int) -> DatasetPreview:
        """Discover the config via ``/splits``, then fetch ``train``/``validation`` rows.

        ``/splits`` is queried first: it yields the repo's real config (subset)
        name and is the viewer's authoritative "is this dataset precomputed"
        signal. If it is unavailable — down, or still precomputing — the whole
        preview is reported ``viewer_ready=False`` and no ``/rows`` fetch is
        attempted (the rows cannot exist yet). Otherwise the two splits are fetched
        concurrently with the discovered config; a single failed split degrades to
        ``[]`` while the other still shows, and ``viewer_ready`` stays ``True``
        unless *both* splits were unavailable — a fully-empty preview caused by the
        viewer, not by a genuinely empty dataset.

        Uses the injected client when present (tests), else opens a short-lived one
        bounded by ``hf_viewer_timeout_seconds`` and closes it.
        """
        client = self._http_client
        owns_client = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=self._hf_viewer_timeout_seconds)
        try:
            try:
                config = await discover_config(
                    client, base_url=self._hf_viewer_base_url, repo_id=repo_id, token=token
                )
            except HFViewerUnavailable as exc:
                logger.warning(
                    "HF dataset viewer not ready for %s (config discovery): %s", repo_id, exc
                )
                return DatasetPreview(train=[], validation=[], viewer_ready=False)
            # Unpacked in two steps: mypy's overload resolution for
            # ``asyncio.gather(..., return_exceptions=True)`` cannot infer the
            # per-element ``T | BaseException`` type when the call target is a
            # tuple-unpacking assignment directly; a plain assignment followed by
            # unpacking resolves it cleanly (verified against mypy 2.3.0).
            results = await asyncio.gather(
                fetch_rows(
                    client,
                    base_url=self._hf_viewer_base_url,
                    repo_id=repo_id,
                    split="train",
                    limit=rows,
                    token=token,
                    config=config,
                ),
                fetch_rows(
                    client,
                    base_url=self._hf_viewer_base_url,
                    repo_id=repo_id,
                    split="validation",
                    limit=rows,
                    token=token,
                    config=config,
                ),
                return_exceptions=True,
            )
            train_result, validation_result = results
        finally:
            if owns_client:
                await client.aclose()
        both_unavailable = all(
            isinstance(result, HFViewerUnavailable) for result in (train_result, validation_result)
        )
        return DatasetPreview(
            train=_rows_or_empty(train_result),
            validation=_rows_or_empty(validation_result),
            viewer_ready=not both_unavailable,
        )

    async def delete(self, *, dataset_id: UUID, name: str, artifact_url: str | None) -> None:
        """Best-effort HF repo delete; failures are logged, never fatal to the DB delete."""
        logger.info("HF delete for %s is best-effort and not yet implemented.", dataset_id)
