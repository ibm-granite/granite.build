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

"""REST endpoints for inspecting an LSF build's remote-file outputs.

Three endpoints, registered on the shared builds_api:
  - GET /builds/{id}/files          — directory listing (optional substring filter)
  - GET /builds/{id}/files/search   — recursive content grep
  - GET /builds/{id}/file/download  — streamed file bytes (capped large)

Path resolution: ``path`` is relative to the build root
(``{workspace_remote_dir}/llm-build-{build_id}``).

Auth matches PUT /builds/{id}/update (owner or space/super admin).
Every user-supplied path passes through validate_subpath() and then
resolve_and_check_real_path() before it hits a shell or SFTP call — do
not bypass those helpers.

The remote file-operation machinery (grep/find/stat/peek/stream plumbing
and the ``GrepHit``/``FileEntry`` models) lives in ``remote_files_ops`` and
is shared with the project-files API; these handlers stay thin: resolve the
build, authorize, open the tunnel, resolve the path, and delegate.
"""

from datetime import datetime
from typing import AsyncIterator, List, Optional, Union, cast

from fastapi import HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from gbserver.api.build_files_paths import (
    authorize_build_access,
    lookup_build,
    resolve_and_check_real_path,
    validate_subpath,
)
from gbserver.api.builds import builds_api
from gbserver.api.lsf_tunnel import open_lsf_tunnel
from gbserver.api.utils import translate_remote_file_errors
from gbserver.environment.lsf_paths import build_remote_root_dir
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.constants import (
    BUILD_FILES_DOWNLOAD_MAX_BYTES,
    BUILD_FILES_GREP_MAX_CONTEXT,
    BUILD_FILES_PEEK_MAX_LINES,
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


# --------------------------------------------------------------------- helpers


def _pick_environment_uri(build: StoredBuild) -> str:
    """Return the most recent target run's environment_uri for this build.

    Build-root listings still need an SSH tunnel, which is keyed by
    environment_uri. We don't persist environment on the build, so we
    borrow it from any of its target runs.
    """
    storage: SingletonAdminStorage = get_admin_storage()
    target_runs = cast(
        list[StoredTargetRun],
        storage.target_storage.get_by_where({"build_id": build.uuid}),
    )
    if not target_runs:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"build {build.uuid!r} has no target runs to ssh through",
        )
    target = max(target_runs, key=lambda t: t.started_at or datetime.min)
    return target.environment_uri


# ---------------------------------------------------------------- /files/search


@builds_api.get(
    "/{build_id}/files/search",
    response_model=List[GrepHit],
)
async def search_files(
    request: Request,
    build_id: str,
    pattern: str = Query(..., min_length=1, max_length=512),
    path: str = Query(".", min_length=1),
    ignore_case: bool = Query(False),
    regex: bool = Query(False),
    before: int = Query(0, ge=0, le=BUILD_FILES_GREP_MAX_CONTEXT),
    after: int = Query(0, ge=0, le=BUILD_FILES_GREP_MAX_CONTEXT),
    stat: bool = Query(False),
) -> List[GrepHit]:
    """Recursively grep for ``pattern`` under ``path``.

    Defaults to literal substring match (``grep -F``). Set ``regex=true``
    to enable extended regex (``grep -E``). ``before``/``after`` add
    context lines — context entries are returned with ``is_match=false``.
    Skips binary files (``-I``). Caps total hits (matches + context) at
    ``BUILD_FILES_GREP_MAX_HITS`` and truncates each line's text to
    ``BUILD_FILES_GREP_LINE_MAX_BYTES`` bytes. With ``stat=true`` each
    hit's owning file is annotated with ``size`` and ``mtime``. Returns
    ``[]`` when the pattern doesn't match anything.
    """
    with translate_remote_file_errors():
        reject_pattern_control_chars(pattern)

        build = lookup_build(build_id)
        authorize_build_access(request, build)
        environment_uri = _pick_environment_uri(build)

        async with open_lsf_tunnel(build.space_name, environment_uri) as (
            tunnel,
            cfg,
        ):
            build_root = build_remote_root_dir(cfg.workspace_remote_dir, build.uuid)
            candidate = validate_subpath(build_root, path)
            real = await resolve_and_check_real_path(tunnel, build_root, candidate)

            logger.info(
                "[build-files] search build=%s ignore_case=%s regex=%s "
                "before=%s after=%s stat=%s",
                build_id,
                ignore_case,
                regex,
                before,
                after,
                stat,
            )

            return await run_search(
                tunnel,
                build_root,
                real,
                pattern=pattern,
                ignore_case=ignore_case,
                regex=regex,
                before=before,
                after=after,
                stat=stat,
            )


# ---------------------------------------------------------------------- /files


@builds_api.get(
    "/{build_id}/files",
    response_model=Union[List[str], List[FileEntry]],
)
async def list_files(
    request: Request,
    build_id: str,
    path: str = Query(".", min_length=1),
    recursive: bool = Query(False),
    pattern: Optional[str] = Query(None, min_length=1, max_length=256),
    regex: bool = Query(False),
    stat: bool = Query(False),
) -> Union[List[str], List[FileEntry]]:
    """List entries under the resolved path, returning paths relative to
    the build root, sorted lexicographically. Includes both files and
    directories (no trailing slash) and dotfiles.

    With ``recursive=true`` the subtree is walked (capped at
    ``BUILD_FILES_LIST_MAX_ENTRIES`` entries). Symlinks are listed as
    their own entries; their targets are not followed.

    With ``pattern`` set, the listing is filtered server-side by literal
    substring (``grep -F``), or extended regex when ``regex=true``
    (``grep -E``). Returns ``[]`` when the pattern doesn't match
    anything.

    With ``stat=true`` the response is a list of ``FileEntry`` objects
    (path, type, size, mtime) instead of bare path strings — this lets
    callers prioritize by recency/size and skip directories without a
    second round-trip. The pattern filter is applied to the path
    component in this mode.
    """
    with translate_remote_file_errors():
        if pattern is not None:
            reject_pattern_control_chars(pattern)

        build = lookup_build(build_id)
        authorize_build_access(request, build)
        environment_uri = _pick_environment_uri(build)

        async with open_lsf_tunnel(build.space_name, environment_uri) as (
            tunnel,
            cfg,
        ):
            build_root = build_remote_root_dir(cfg.workspace_remote_dir, build.uuid)
            candidate = validate_subpath(build_root, path)
            real = await resolve_and_check_real_path(tunnel, build_root, candidate)

            logger.info(
                "[build-files] list build=%s recursive=%s filtered=%s regex=%s stat=%s",
                build_id,
                recursive,
                pattern is not None,
                regex,
                stat,
            )
            logger.debug("[build-files] list real=%s build_root=%s", real, build_root)

            return await run_list(
                tunnel,
                build_root,
                real,
                recursive=recursive,
                pattern=pattern,
                regex=regex,
                stat=stat,
            )


# ------------------------------------------------------------- /file/download


@builds_api.get("/{build_id}/file/download")
async def download_file(
    request: Request,
    build_id: str,
    path: str = Query(..., min_length=1),
    head: Optional[int] = Query(None, ge=1, le=BUILD_FILES_PEEK_MAX_LINES),
    tail: Optional[int] = Query(None, ge=1, le=BUILD_FILES_PEEK_MAX_LINES),
    range_: Optional[str] = Query(None, alias="range", pattern=r"^\d+-\d+$"),
) -> Response:
    """Download or peek at a remote file.

    Default (no peek param): streams the file as
    ``application/octet-stream``. Rejects directories with 400. Downloads
    are uncapped by default; set ``GBSERVER_BUILD_FILES_DOWNLOAD_MAX_BYTES``
    to reject files larger than that many bytes with 413 before any bytes
    are streamed.

    Peek mode (set exactly one of ``head=N``, ``tail=N``, ``range=A-B``):
    returns ``text/plain; charset=utf-8`` with the requested slice of
    the file. Output bytes are capped at ``BUILD_FILES_PEEK_MAX_BYTES``
    (~256 KiB by default). The file size cap does **not** apply in peek
    mode — tailing the last 200 lines of a 50 GiB log is the use case.
    """
    with translate_remote_file_errors():
        peek = validate_peek_args(head, tail, range_)

    build = lookup_build(build_id)
    authorize_build_access(request, build)
    environment_uri = _pick_environment_uri(build)

    if peek is not None:
        # Peek mode: bounded output, no streaming.
        async with open_lsf_tunnel(build.space_name, environment_uri) as (
            tunnel,
            cfg,
        ):
            build_root = build_remote_root_dir(cfg.workspace_remote_dir, build.uuid)
            candidate = validate_subpath(build_root, path)
            real = await resolve_and_check_real_path(tunnel, build_root, candidate)

            logger.info(
                "[build-files] peek build=%s mode=%s args=%s",
                build_id,
                peek[0],
                peek[1],
            )
            with translate_remote_file_errors():
                text = await peek_file(tunnel, real, peek)
            return Response(content=text, media_type="text/plain; charset=utf-8")

    # Tunnel lifecycle must outlive the streaming response body, so we open
    # it manually here and close it inside the body's finally on success or
    # in the except below if anything fails before we hand off to streaming.
    ctx = open_lsf_tunnel(build.space_name, environment_uri)
    tunnel, cfg = await ctx.__aenter__()
    try:
        build_root = build_remote_root_dir(cfg.workspace_remote_dir, build.uuid)
        candidate = validate_subpath(build_root, path)
        real = await resolve_and_check_real_path(tunnel, build_root, candidate)

        with translate_remote_file_errors():
            size, is_dir = await remote_stat(tunnel, real)
        if is_dir:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "download endpoint requires a file, not a directory",
            )
        if (
            BUILD_FILES_DOWNLOAD_MAX_BYTES is not None
            and size > BUILD_FILES_DOWNLOAD_MAX_BYTES
        ):
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"file exceeds download cap: size={size} "
                f"cap={BUILD_FILES_DOWNLOAD_MAX_BYTES}",
            )

        logger.info(
            "[build-files] download build=%s size=%d",
            build_id,
            size,
        )

        filename = real.name or "download.bin"

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
