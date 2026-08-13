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

"""REST endpoints for browsing folders on a supported environment's login nodes.

The feature is environment-specific file interaction, not a filesystem concept:
the ``{environment}`` segment selects a supported environment (today only
``bluevela``, whose LSF login nodes mount ``/proj``), and ``{folder}`` names a
directory under that environment's fixed base. Three endpoints, mirroring the
build-files API but rooted at an environment folder instead of a build root and
authorized by POSIX group membership:
  - GET /files/{environment}/{folder}/files          — directory listing
  - GET /files/{environment}/{folder}/files/search   — recursive content grep
  - GET /files/{environment}/{folder}/file/download   — streamed file bytes / peek

An unsupported ``{environment}`` is denied with the same uniform 404 as a
missing folder (no enumeration of supported environments); a *known* but
unconfigured environment returns 503.

The remote file-operation machinery is shared with build-files via
``remote_files_ops``; the genuinely new pieces are the environment registry +
group-membership authorization (``environment_files_paths``) and server-side
tunnel selection (there is no build to borrow ``space_name``/``environment_uri``
from — they come from the resolved environment config).

SECURITY (authorize-before-read, no existence leak) — every handler follows
this exact order and touches NO folder data before authorization:

  1. cheap pure validation (pattern/peek args)          — no I/O
  2. resolve_environment(environment)                    — AccessDenied on unsupported
  3. validate_folder_name(folder)                        — AccessDenied on bad name
  4. open the service-identity tunnel (from env config)
  5. authorize_folder_access(...)                        — ONLY getent runs
  6. resolve the folder root (first data-touch, POST-auth)
  7. validate_subpath + resolve_and_check_real_path
  8. delegate to run_search / run_list / peek_file / stream

Steps 2, 3 and 5 raise the *same* 404 body, so a caller cannot distinguish
"no such environment", "no such folder", and "you lack access"; a non-member
never reaches step 6. The tunnel opens for authorized and unauthorized requests
alike; on open it touches only the service workspace (``open_lsf_tunnel``
canonicalizes ``workspace_remote_dir``), never the folder, so open-latency
reveals nothing about the requested folder.
"""

from pathlib import PurePosixPath
from typing import AsyncIterator, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from gbserver.api.environment_files_paths import (
    authorize_folder_access,
    folder_root,
    resolve_and_check_real_path,
    resolve_environment,
    validate_folder_name,
    validate_subpath,
)
from gbserver.api.lsf_tunnel import open_lsf_tunnel
from gbserver.api.utils import translate_remote_file_errors
from gbserver.types.constants import (
    ENV_FILES_DOWNLOAD_MAX_BYTES,
    ENV_FILES_GREP_MAX_CONTEXT,
    ENV_FILES_PEEK_MAX_LINES,
    EnvironmentFilesConfig,
)
from gbserver.utils.logger import get_logger
from gbserver.utils.remote_files_ops import (
    FileEntry,
    GrepHit,
    content_disposition,
    peek_file,
    reject_pattern_control_chars,
    remote_stat,
    run_list,
    run_search,
    stream_sftp_file,
    validate_peek_args,
)

logger = get_logger(__name__)

files_api = FastAPI()


# --------------------------------------------------------------------- helpers


async def _resolve_folder_paths(
    tunnel, config: EnvironmentFilesConfig, folder: str, path: str
):
    """Post-auth path resolution shared by all three handlers.

    Resolves the folder root under the environment's fixed base (containment-
    checked), then the user ``path`` beneath it. MUST be called only after
    ``authorize_folder_access`` has returned — it issues the first
    data-touching commands (``readlink -f``) against the folder.

    Note the sandbox boundary is the *resolved* root: if the folder root is
    itself a symlink, the effective boundary becomes wherever it resolves to,
    not the literal ``gpfs_base/{folder}``. This mirrors what build-files does
    and what ``open_lsf_tunnel`` does for the workspace.

    Returns ``(root, real)`` as ``PurePosixPath``.
    """
    base = PurePosixPath(config.gpfs_base)
    # Canonicalize the folder root itself (mirrors open_lsf_tunnel's readlink -f
    # of the workspace) and confirm it stays under the environment base.
    root = await resolve_and_check_real_path(tunnel, base, folder_root(config, folder))
    candidate = validate_subpath(root, path)
    real = await resolve_and_check_real_path(tunnel, root, candidate)
    return root, real


# --------------------------------------------------------------- /files/search


@files_api.get(
    "/{environment}/{folder}/files/search",
    response_model=List[GrepHit],
)
async def search_environment_files(
    request: Request,
    environment: str,
    folder: str,
    pattern: str = Query(..., min_length=1, max_length=512),
    path: str = Query(".", min_length=1),
    ignore_case: bool = Query(False),
    regex: bool = Query(False),
    before: int = Query(0, ge=0, le=ENV_FILES_GREP_MAX_CONTEXT),
    after: int = Query(0, ge=0, le=ENV_FILES_GREP_MAX_CONTEXT),
    stat: bool = Query(False),
) -> List[GrepHit]:
    """Recursively grep for ``pattern`` under ``path`` in an environment folder.

    Same semantics as the build-files search endpoint (literal ``grep -F`` by
    default, ``regex=true`` for extended regex, ``before``/``after`` context,
    ``stat=true`` size/mtime annotation). Access requires membership in the
    ``proj_{folder}`` POSIX group; non-members get an indistinguishable 404.
    """
    with translate_remote_file_errors():
        reject_pattern_control_chars(pattern)
        config = resolve_environment(environment)
        folder = validate_folder_name(folder)

        async with open_lsf_tunnel(config.space_name, config.environment_uri) as (
            tunnel,
            _cfg,
        ):
            # Authorization first — ONLY getent runs; no folder data touched yet.
            await authorize_folder_access(request, tunnel, folder)

            root, real = await _resolve_folder_paths(tunnel, config, folder, path)

            logger.info(
                "[env-files] search environment=%s folder=%s ignore_case=%s regex=%s "
                "before=%s after=%s stat=%s",
                environment,
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


@files_api.get(
    "/{environment}/{folder}/files",
    response_model=Union[List[str], List[FileEntry]],
)
async def list_environment_files(
    request: Request,
    environment: str,
    folder: str,
    path: str = Query(".", min_length=1),
    recursive: bool = Query(False),
    pattern: Optional[str] = Query(None, min_length=1, max_length=256),
    regex: bool = Query(False),
    stat: bool = Query(False),
) -> Union[List[str], List[FileEntry]]:
    """List entries under ``path`` in an environment folder, relative to its root.

    Same semantics as the build-files listing endpoint (``recursive``,
    ``pattern`` filter, ``regex``, ``stat`` FileEntry objects). Access requires
    membership in the ``proj_{folder}`` POSIX group; non-members get an
    indistinguishable 404.
    """
    with translate_remote_file_errors():
        if pattern is not None:
            reject_pattern_control_chars(pattern)
        config = resolve_environment(environment)
        folder = validate_folder_name(folder)

        async with open_lsf_tunnel(config.space_name, config.environment_uri) as (
            tunnel,
            _cfg,
        ):
            # Authorization first — ONLY getent runs; no folder data touched yet.
            await authorize_folder_access(request, tunnel, folder)

            root, real = await _resolve_folder_paths(tunnel, config, folder, path)

            logger.info(
                "[env-files] list environment=%s folder=%s recursive=%s filtered=%s "
                "regex=%s stat=%s",
                environment,
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


@files_api.get("/{environment}/{folder}/file/download")
async def download_environment_file(
    request: Request,
    environment: str,
    folder: str,
    path: str = Query(..., min_length=1),
    head: Optional[int] = Query(None, ge=1, le=ENV_FILES_PEEK_MAX_LINES),
    tail: Optional[int] = Query(None, ge=1, le=ENV_FILES_PEEK_MAX_LINES),
    range_: Optional[str] = Query(None, alias="range", pattern=r"^\d+-\d+$"),
) -> Response:
    """Download or peek at a file in an environment folder.

    Same semantics as the build-files download endpoint: default streams the
    file as ``application/octet-stream`` (413 if it exceeds the configured
    cap); peek mode (exactly one of ``head``/``tail``/``range``) returns a
    bounded ``text/plain`` slice. Access requires membership in the
    ``proj_{folder}`` POSIX group; non-members get an indistinguishable 404.
    """
    with translate_remote_file_errors():
        peek = validate_peek_args(head, tail, range_)
    config = resolve_environment(environment)
    folder = validate_folder_name(folder)

    if peek is not None:
        # Peek mode: bounded output, no streaming.
        async with open_lsf_tunnel(config.space_name, config.environment_uri) as (
            tunnel,
            _cfg,
        ):
            # Authorization first — ONLY getent runs; no folder data touched yet.
            await authorize_folder_access(request, tunnel, folder)

            _root, real = await _resolve_folder_paths(tunnel, config, folder, path)

            logger.info(
                "[env-files] peek environment=%s folder=%s mode=%s args=%s",
                environment,
                folder,
                peek[0],
                peek[1],
            )
            with translate_remote_file_errors():
                text = await peek_file(tunnel, real, peek)
            return Response(content=text, media_type="text/plain; charset=utf-8")

    # Tunnel lifecycle must outlive the streaming response body, so we open it
    # manually here and close it inside the body's finally on success or in the
    # except below if anything fails before we hand off to streaming.
    ctx = open_lsf_tunnel(config.space_name, config.environment_uri)
    tunnel, _cfg = await ctx.__aenter__()
    try:
        # Authorization first — ONLY getent runs; no folder data touched yet.
        await authorize_folder_access(request, tunnel, folder)

        _root, real = await _resolve_folder_paths(tunnel, config, folder, path)

        with translate_remote_file_errors():
            size, is_dir = await remote_stat(tunnel, real)
        if is_dir:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "download endpoint requires a file, not a directory",
            )
        if (
            ENV_FILES_DOWNLOAD_MAX_BYTES is not None
            and size > ENV_FILES_DOWNLOAD_MAX_BYTES
        ):
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"file exceeds download cap: size={size} "
                f"cap={ENV_FILES_DOWNLOAD_MAX_BYTES}",
            )

        logger.info(
            "[env-files] download environment=%s folder=%s size=%d",
            environment,
            folder,
            size,
        )

        filename = real.name or "download.bin"

        # NOTE: unlike the search/list/peek paths, the streaming body below runs
        # AFTER the translate_remote_file_errors() wrapper has exited, so a
        # RemoteFileError raised inside it would NOT be translated to an HTTP
        # status. Today stream_sftp_file raises raw SFTP errors (not domain
        # RemoteFileError), so nothing needs translating here. If a future edit
        # makes the streaming path raise domain errors, wrap them explicitly —
        # they cannot rely on the outer context manager.
        async def body() -> AsyncIterator[bytes]:
            try:
                async for chunk in stream_sftp_file(tunnel, str(real), size):
                    yield chunk
            finally:
                await ctx.__aexit__(None, None, None)

        return StreamingResponse(
            body(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": content_disposition(filename),
                "Content-Length": str(size),
            },
        )
    except BaseException:
        # Pre-stream failure: close the tunnel now.
        await ctx.__aexit__(None, None, None)
        raise


__all__ = ["files_api"]
