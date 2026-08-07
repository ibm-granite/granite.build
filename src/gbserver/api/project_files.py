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

"""REST endpoints for browsing GPFS project folders (``/proj/{folder}``).

Three endpoints, mirroring the build-files API but rooted at a project folder
instead of a build root and authorized by POSIX group membership:
  - GET /projects/{folder}/files          — directory listing
  - GET /projects/{folder}/files/search   — recursive content grep
  - GET /projects/{folder}/file/download  — streamed file bytes / peek

The remote file-operation machinery is shared with build-files via
``remote_files_ops``; the genuinely new pieces are the group-membership
authorization (``project_files_paths``) and server-side tunnel selection
(there is no build to borrow ``space_name``/``environment_uri`` from).

SECURITY (authorize-before-read, no existence leak) — every handler follows
this exact order and touches NO folder data before authorization:

  1. cheap pure validation (pattern/peek args)          — no I/O
  2. validate_folder_name(folder)                        — ProjectAccessDenied on bad name
  3. open the service-identity tunnel (server-resolved space/env)
  4. authorize_project_access(...)                       — ONLY getent runs
  5. resolve the project root (first data-touch, POST-auth)
  6. validate_subpath + resolve_and_check_real_path
  7. delegate to run_search / run_list / peek_file / stream

Steps 2 and 4 raise the *same* 404 body, so a non-member cannot distinguish
"no such folder" from "you lack access"; a non-member never reaches step 5.
The tunnel opens for authorized and unauthorized requests alike and only ever
touches the fixed ``/proj`` base, so open-latency doesn't leak existence.
"""

from pathlib import PurePosixPath
from typing import AsyncIterator, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from gbserver.api.lsf_tunnel import open_lsf_tunnel
from gbserver.api.project_files_paths import (
    authorize_project_access,
    project_root,
    resolve_and_check_real_path,
    validate_folder_name,
    validate_subpath,
)
from gbserver.api.remote_files_ops import (
    FileEntry,
    GrepHit,
    _content_disposition,
    _reject_pattern_control_chars,
    _remote_stat,
    _stream_sftp_file,
    _validate_peek_args,
    peek_file,
    run_list,
    run_search,
)
from gbserver.types.constants import (
    PROJECT_FILES_DOWNLOAD_MAX_BYTES,
    PROJECT_FILES_GREP_MAX_CONTEXT,
    PROJECT_FILES_PEEK_MAX_LINES,
    PROJECTS_GPFS_BASE,
    PROJECTS_GPFS_ENVIRONMENT_URI,
    PROJECTS_GPFS_SPACE_NAME,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

projects_api = FastAPI()


# --------------------------------------------------------------------- helpers


def _gpfs_space_name() -> str:
    """Space whose Secret Manager holds the service SSH key (server-resolved)."""
    return PROJECTS_GPFS_SPACE_NAME


def _gpfs_environment_uri() -> str:
    """The well-known LSF ``environment_uri`` for GPFS browsing (server-resolved).

    This is NOT the dev/staging/prod axis (that's ``GB_ENVIRONMENT``); it's a
    ``space://…`` asset URI pointing at an LSF ``environment.yaml`` whose login
    nodes mount ``/proj``. Pinned per deployment via
    ``GBSERVER_PROJECTS_GPFS_ENVIRONMENT_URI``. If unset, the API is not
    configured for this deployment — 503 rather than guess.
    """
    if not PROJECTS_GPFS_ENVIRONMENT_URI:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "project-folder browsing is not configured for this deployment",
        )
    return PROJECTS_GPFS_ENVIRONMENT_URI


async def _resolve_project_paths(tunnel, folder: str, path: str):
    """Post-auth path resolution shared by all three handlers.

    Resolves the project root under the fixed ``/proj`` base (containment-
    checked), then the user ``path`` beneath it. MUST be called only after
    ``authorize_project_access`` has returned — it issues the first
    data-touching commands (``readlink -f``) against the folder.

    Returns ``(root, real)`` as ``PurePosixPath``.
    """
    base = PurePosixPath(PROJECTS_GPFS_BASE)
    # Canonicalize the project root itself (mirrors open_lsf_tunnel's
    # readlink -f of the workspace) and confirm it stays under /proj.
    root = await resolve_and_check_real_path(tunnel, base, project_root(folder))
    candidate = validate_subpath(root, path)
    real = await resolve_and_check_real_path(tunnel, root, candidate)
    return root, real


# ---------------------------------------------------------------- /files/search


@projects_api.get(
    "/{folder}/files/search",
    response_model=List[GrepHit],
)
async def search_project_files(
    request: Request,
    folder: str,
    pattern: str = Query(..., min_length=1, max_length=512),
    path: str = Query(".", min_length=1),
    ignore_case: bool = Query(False),
    regex: bool = Query(False),
    before: int = Query(0, ge=0, le=PROJECT_FILES_GREP_MAX_CONTEXT),
    after: int = Query(0, ge=0, le=PROJECT_FILES_GREP_MAX_CONTEXT),
    stat: bool = Query(False),
) -> List[GrepHit]:
    """Recursively grep for ``pattern`` under ``path`` in a project folder.

    Same semantics as the build-files search endpoint (literal ``grep -F`` by
    default, ``regex=true`` for extended regex, ``before``/``after`` context,
    ``stat=true`` size/mtime annotation). Access requires membership in the
    ``proj_{folder}`` POSIX group; non-members get an indistinguishable 404.
    """
    _reject_pattern_control_chars(pattern)
    folder = validate_folder_name(folder)

    async with open_lsf_tunnel(_gpfs_space_name(), _gpfs_environment_uri()) as (
        tunnel,
        _cfg,
    ):
        # Authorization first — ONLY getent runs; no folder data touched yet.
        await authorize_project_access(request, tunnel, folder)

        root, real = await _resolve_project_paths(tunnel, folder, path)

        logger.info(
            "[project-files] search folder=%s ignore_case=%s regex=%s "
            "before=%s after=%s stat=%s",
            folder,
            ignore_case,
            regex,
            before,
            after,
            stat,
        )

        return await run_search(
            tunnel,
            root,
            real,
            pattern=pattern,
            ignore_case=ignore_case,
            regex=regex,
            before=before,
            after=after,
            stat=stat,
        )


# ---------------------------------------------------------------------- /files


@projects_api.get(
    "/{folder}/files",
    response_model=Union[List[str], List[FileEntry]],
)
async def list_project_files(
    request: Request,
    folder: str,
    path: str = Query(".", min_length=1),
    recursive: bool = Query(False),
    pattern: Optional[str] = Query(None, min_length=1, max_length=256),
    regex: bool = Query(False),
    stat: bool = Query(False),
) -> Union[List[str], List[FileEntry]]:
    """List entries under ``path`` in a project folder, relative to its root.

    Same semantics as the build-files listing endpoint (``recursive``,
    ``pattern`` filter, ``regex``, ``stat`` FileEntry objects). Access requires
    membership in the ``proj_{folder}`` POSIX group; non-members get an
    indistinguishable 404.
    """
    if pattern is not None:
        _reject_pattern_control_chars(pattern)
    folder = validate_folder_name(folder)

    async with open_lsf_tunnel(_gpfs_space_name(), _gpfs_environment_uri()) as (
        tunnel,
        _cfg,
    ):
        # Authorization first — ONLY getent runs; no folder data touched yet.
        await authorize_project_access(request, tunnel, folder)

        root, real = await _resolve_project_paths(tunnel, folder, path)

        logger.info(
            "[project-files] list folder=%s recursive=%s filtered=%s "
            "regex=%s stat=%s",
            folder,
            recursive,
            pattern is not None,
            regex,
            stat,
        )

        return await run_list(
            tunnel,
            root,
            real,
            recursive=recursive,
            pattern=pattern,
            regex=regex,
            stat=stat,
        )


# ------------------------------------------------------------- /file/download


@projects_api.get("/{folder}/file/download")
async def download_project_file(
    request: Request,
    folder: str,
    path: str = Query(..., min_length=1),
    head: Optional[int] = Query(None, ge=1, le=PROJECT_FILES_PEEK_MAX_LINES),
    tail: Optional[int] = Query(None, ge=1, le=PROJECT_FILES_PEEK_MAX_LINES),
    range_: Optional[str] = Query(None, alias="range", pattern=r"^\d+-\d+$"),
) -> Response:
    """Download or peek at a file in a project folder.

    Same semantics as the build-files download endpoint: default streams the
    file as ``application/octet-stream`` (413 if it exceeds the configured
    cap); peek mode (exactly one of ``head``/``tail``/``range``) returns a
    bounded ``text/plain`` slice. Access requires membership in the
    ``proj_{folder}`` POSIX group; non-members get an indistinguishable 404.
    """
    peek = _validate_peek_args(head, tail, range_)
    folder = validate_folder_name(folder)

    if peek is not None:
        # Peek mode: bounded output, no streaming.
        async with open_lsf_tunnel(_gpfs_space_name(), _gpfs_environment_uri()) as (
            tunnel,
            _cfg,
        ):
            # Authorization first — ONLY getent runs; no folder data touched yet.
            await authorize_project_access(request, tunnel, folder)

            _root, real = await _resolve_project_paths(tunnel, folder, path)

            logger.info(
                "[project-files] peek folder=%s mode=%s args=%s",
                folder,
                peek[0],
                peek[1],
            )
            return await peek_file(tunnel, real, peek)

    # Tunnel lifecycle must outlive the streaming response body, so we open it
    # manually here and close it inside the body's finally on success or in the
    # except below if anything fails before we hand off to streaming.
    ctx = open_lsf_tunnel(_gpfs_space_name(), _gpfs_environment_uri())
    tunnel, _cfg = await ctx.__aenter__()
    try:
        # Authorization first — ONLY getent runs; no folder data touched yet.
        await authorize_project_access(request, tunnel, folder)

        _root, real = await _resolve_project_paths(tunnel, folder, path)

        size, is_dir = await _remote_stat(tunnel, real)
        if is_dir:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "download endpoint requires a file, not a directory",
            )
        if (
            PROJECT_FILES_DOWNLOAD_MAX_BYTES is not None
            and size > PROJECT_FILES_DOWNLOAD_MAX_BYTES
        ):
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"file exceeds download cap: size={size} "
                f"cap={PROJECT_FILES_DOWNLOAD_MAX_BYTES}",
            )

        logger.info(
            "[project-files] download folder=%s size=%d",
            folder,
            size,
        )

        filename = real.name or "download.bin"

        async def body() -> AsyncIterator[bytes]:
            try:
                async for chunk in _stream_sftp_file(tunnel, str(real), size):
                    yield chunk
            finally:
                await ctx.__aexit__(None, None, None)

        return StreamingResponse(
            body(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": _content_disposition(filename),
                "Content-Length": str(size),
            },
        )
    except BaseException:
        # Pre-stream failure: close the tunnel now.
        await ctx.__aexit__(None, None, None)
        raise


__all__ = ["projects_api"]
