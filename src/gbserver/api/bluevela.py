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

"""REST endpoints for inspecting BlueVela/LSF remote-file outputs of a build.

Three endpoints, registered on the shared builds_api:
  - GET /builds/{id}/files          — single-level listing
  - GET /builds/{id}/file           — file contents (base64, capped small)
  - GET /builds/{id}/file/download  — streamed file bytes (capped large)

Path resolution: ``path`` is relative to the build root
(``{workspace_remote_dir}/llm-build-{build_id}``).

Auth matches PUT /builds/{id}/update (owner or space/super admin).
Every user-supplied path passes through validate_subpath() and then
resolve_and_check_real_path() before it hits a shell or SFTP call — do
not bypass those helpers.
"""

import base64
import shlex
from datetime import datetime
from pathlib import PurePosixPath
from typing import AsyncIterator, List, cast
from urllib.parse import quote

from fastapi import HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from gbserver.api.bluevela_paths import (
    authorize_build_access,
    lookup_build,
    resolve_and_check_real_path,
    validate_subpath,
)
from gbserver.api.bluevela_tunnel import open_bluevela_tunnel
from gbserver.api.builds import builds_api
from gbserver.environment.lsf_paths import build_remote_root_dir
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.constants import (
    BLUEVELA_CONTENT_MAX_BYTES,
    BLUEVELA_DOWNLOAD_MAX_BYTES,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

_FIND_MAX_ENTRIES = 10000


# --------------------------------------------------------------------- models


class FileContentResponse(BaseModel):
    path: str
    """Path relative to the build root."""
    content_base64: str


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


# ---------------------------------------------------------------------- /files


@builds_api.get(
    "/{build_id}/files",
    response_model=List[str],
)
async def list_files(
    request: Request,
    build_id: str,
    path: str = Query(".", min_length=1),
    recursive: bool = Query(False),
) -> List[str]:
    """List entries under the resolved path, returning paths relative to
    the build root, sorted lexicographically. Includes both files and
    directories (no trailing slash) and dotfiles.

    With ``recursive=true`` the subtree is walked (capped at
    ``_FIND_MAX_ENTRIES`` entries). Symlinks are listed as their own
    entries; their targets are not followed. Use ``GET /file`` to read a
    single symlink's target — that path goes through ``readlink -f`` and
    is rejected if it leaves the build root.
    """
    build = lookup_build(build_id)
    authorize_build_access(request, build)
    environment_uri = _pick_environment_uri(build)

    async with open_bluevela_tunnel(build.space_name, environment_uri) as (
        tunnel,
        cfg,
    ):
        build_root = build_remote_root_dir(cfg.workspace_remote_dir, build.uuid)
        candidate = validate_subpath(build_root, path)
        real = await resolve_and_check_real_path(tunnel, build_root, candidate)

        logger.info(
            "[bluevela] list build=%s recursive=%s",
            build_id,
            recursive,
        )
        logger.debug("[bluevela] list real=%s build_root=%s", real, build_root)

        quoted = shlex.quote(str(real))
        if recursive:
            # pipefail so find's rc propagates through head; without it
            # head's success would mask a failed find.
            cmd = (
                f"set -o pipefail; "
                f"find {quoted} -mindepth 1 "
                f"| head -n {_FIND_MAX_ENTRIES}"
            )
        else:
            cmd = f"ls -1A -- {quoted}"

        rc, stdout, stderr = await tunnel.run_remote(cmd, raise_on_error=False)
        if rc != 0:
            err = (stderr or "").lower()
            if "no such file" in err or "cannot access" in err:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "path not found")
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                f"listing failed: {stderr.strip() or 'unknown error'}",
            )

        lines = [ln for ln in (stdout or "").splitlines() if ln]
        if recursive:
            # find emits absolute paths.
            rels = [str(PurePosixPath(ln).relative_to(build_root)) for ln in lines]
        else:
            # ls -1A emits bare names rooted at `real`.
            rels = [str((real / name).relative_to(build_root)) for name in lines]
        rels.sort()
        return rels


# ----------------------------------------------------------------------- /file


@builds_api.get(
    "/{build_id}/file",
    response_model=FileContentResponse,
)
async def get_file(
    request: Request,
    build_id: str,
    path: str = Query(..., min_length=1),
) -> FileContentResponse:
    """Return a file's bytes base64-encoded, with its path relative to
    the build root.
    """
    build = lookup_build(build_id)
    authorize_build_access(request, build)
    environment_uri = _pick_environment_uri(build)

    async with open_bluevela_tunnel(build.space_name, environment_uri) as (
        tunnel,
        cfg,
    ):
        build_root = build_remote_root_dir(cfg.workspace_remote_dir, build.uuid)
        candidate = validate_subpath(build_root, path)
        real = await resolve_and_check_real_path(tunnel, build_root, candidate)

        size, is_dir = await _remote_stat(tunnel, real)
        if is_dir:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "file endpoint requires a file, not a directory",
            )
        if size > BLUEVELA_CONTENT_MAX_BYTES:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                {
                    "message": "file exceeds content cap",
                    "size": size,
                    "cap": BLUEVELA_CONTENT_MAX_BYTES,
                },
            )

        sftp = await tunnel.start_sftp()
        try:
            async with sftp.open(str(real), "rb") as fh:
                raw = await fh.read()
        finally:
            sftp.exit()

        return FileContentResponse(
            path=str(real.relative_to(build_root)),
            content_base64=base64.b64encode(raw).decode("ascii"),
        )


# ------------------------------------------------------------- /file/download


async def _stream_sftp_file(tunnel, remote_path: str) -> AsyncIterator[bytes]:
    """Yield chunks of a remote file via SFTP, closing the client on exit."""
    chunk_size = 256 * 1024
    sftp = await tunnel.start_sftp()
    try:
        async with sftp.open(remote_path, "rb") as fh:
            while True:
                chunk = await fh.read(chunk_size)
                if not chunk:
                    return
                yield chunk
    finally:
        sftp.exit()


def _content_disposition(filename: str) -> str:
    """RFC 5987 Content-Disposition value with an ASCII fallback + UTF-8 form."""
    ascii_fallback = (
        filename.encode("ascii", "replace").decode("ascii").replace('"', "_")
    ) or "download.bin"
    return (
        f'attachment; filename="{ascii_fallback}"; '
        f"filename*=UTF-8''{quote(filename, safe='')}"
    )


@builds_api.get("/{build_id}/file/download")
async def download_file(
    request: Request,
    build_id: str,
    path: str = Query(..., min_length=1),
) -> StreamingResponse:
    """Stream a remote file as application/octet-stream.

    ``path`` is relative to the build root. Rejects files larger than
    BLUEVELA_DOWNLOAD_MAX_BYTES with 413 before any bytes are streamed.
    """
    build = lookup_build(build_id)
    authorize_build_access(request, build)
    environment_uri = _pick_environment_uri(build)

    # Tunnel lifecycle must outlive the streaming response body, so we open
    # it manually here and close it inside the body's finally on success or
    # in the except below if anything fails before we hand off to streaming.
    ctx = open_bluevela_tunnel(build.space_name, environment_uri)
    tunnel, cfg = await ctx.__aenter__()
    try:
        build_root = build_remote_root_dir(cfg.workspace_remote_dir, build.uuid)
        candidate = validate_subpath(build_root, path)
        real = await resolve_and_check_real_path(tunnel, build_root, candidate)

        size, is_dir = await _remote_stat(tunnel, real)
        if is_dir:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "download endpoint requires a file, not a directory",
            )
        if size > BLUEVELA_DOWNLOAD_MAX_BYTES:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                {
                    "message": "file exceeds download cap",
                    "size": size,
                    "cap": BLUEVELA_DOWNLOAD_MAX_BYTES,
                },
            )

        logger.info(
            "[bluevela] download build=%s size=%d",
            build_id,
            size,
        )

        filename = real.name or "download.bin"

        async def body() -> AsyncIterator[bytes]:
            try:
                async for chunk in _stream_sftp_file(tunnel, str(real)):
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


async def _remote_stat(tunnel, target: PurePosixPath) -> tuple[int, bool]:
    """Return (size, is_dir) for `target`. 404 if missing, 500 otherwise."""
    cmd = f"stat -c '%s\t%F' -- {shlex.quote(str(target))}"  # literal TAB
    rc, stdout, stderr = await tunnel.run_remote(cmd, raise_on_error=False)
    if rc != 0:
        err = (stderr or "").strip().lower()
        if "no such file" in err or "cannot stat" in err:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "path not found")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"stat failed: {stderr.strip() or 'unknown error'}",
        )
    first = (stdout or "").splitlines()[0] if stdout else ""
    parts = first.split("\t")
    if len(parts) < 2:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"unexpected stat output: {first!r}",
        )
    try:
        size = int(parts[0])
    except ValueError as e:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"unexpected stat size: {parts[0]!r}",
        ) from e
    return size, parts[1].startswith("directory")
