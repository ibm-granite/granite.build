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

Three endpoints, all scoped to a single step-run:
  - GET /builds/{id}/files             — directory listing
  - GET /builds/{id}/files/content     — small text file, inline JSON
  - GET /builds/{id}/files/download    — any file, streamed via SFTP

Auth matches PUT /builds/{id}/update (owner or space/super admin).
Every user-supplied path passes through validate_subpath() before it hits
a shell or SFTP call — do not bypass that helper.

NOTE (in-flight steps): a running step's directory may be partially
written, so a listing taken mid-run is a snapshot and can disagree with
the next listing moments later. This is intentional — surfacing progress
is a feature.
"""

import shlex
from pathlib import PurePosixPath
from typing import AsyncIterator, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from gbserver.api.bluevela_paths import (
    _step_name_of,
    authorize_build_access,
    lookup_build_target_step,
    resolve_step_asset_dir,
    validate_subpath,
)
from gbserver.api.bluevela_tunnel import open_bluevela_tunnel
from gbserver.environment.lsf_paths import build_step_run_parent_dir
from gbserver.types.constants import (
    BLUEVELA_CONTENT_MAX_BYTES,
    BLUEVELA_DOWNLOAD_MAX_BYTES,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

bluevela_api = FastAPI()


# --------------------------------------------------------------------- models


class FileEntry(BaseModel):
    path: str
    """Path relative to the step's asset dir (empty string = the asset dir itself)."""
    size: int
    mtime_epoch: int
    is_dir: bool


class FileListResponse(BaseModel):
    asset_dir: str
    entries: List[FileEntry]


class FileContentResponse(BaseModel):
    path: str
    size: int
    content: str
    truncated: bool
    """Always false today; reserved for a future larger-cap mode."""


# --------------------------------------------------------------------- helpers


# stat format: name<TAB>size<TAB>mtime_epoch<TAB>type
# Use a literal TAB (0x09) inside the format. Some remote login shells
# (csh/tcsh) and some `stat` implementations don't expand `\t` escapes,
# leaving a backslash-t pair in the output that breaks our parser.
_STAT_FMT = "%n\t%s\t%Y\t%F"
_FIND_MAX_ENTRIES = 10000


def _parse_stat_line(line: str, asset_dir: PurePosixPath) -> Optional[FileEntry]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 4:
        return None
    abs_name, size_s, mtime_s, ftype = parts[0], parts[1], parts[2], parts[3]
    try:
        size = int(size_s)
        mtime = int(mtime_s)
    except ValueError:
        return None
    try:
        rel = str(PurePosixPath(abs_name).relative_to(asset_dir))
    except ValueError:
        # Shouldn't happen since we rooted the listing at asset_dir, but if
        # `stat` printed something outside (e.g. symlink target following),
        # drop it rather than risk leaking paths.
        return None
    if rel == ".":
        rel = ""
    return FileEntry(
        path=rel,
        size=size,
        mtime_epoch=mtime,
        is_dir=ftype.startswith("directory"),
    )


async def _remote_stat(tunnel, target: PurePosixPath) -> tuple[int, bool]:
    """Return (size, is_dir) for `target`. 404 if missing, 500 otherwise."""
    cmd = f"stat -c '%s\t%F' -- {shlex.quote(str(target))}"  # literal TAB
    rc, stdout, stderr = await tunnel.run_remote(cmd, raise_on_error=False)
    if rc != 0:
        err = (stderr or "").strip().lower()
        if "no such file" in err or "cannot stat" in err:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"path not found")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"stat failed: {stderr.strip() or 'unknown error'}",
        )
    first = (stdout or "").splitlines()[0] if stdout else ""
    parts = first.split("\t")
    if len(parts) < 2:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"unexpected stat output: {first!r}"
        )
    try:
        size = int(parts[0])
    except ValueError as e:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"unexpected stat size: {parts[0]!r}"
        ) from e
    is_dir = parts[1].startswith("directory")
    return size, is_dir


# --------------------------------------------------------------------- /files


@bluevela_api.get(
    "/builds/{build_id}/files",
    response_model=FileListResponse,
)
async def list_files(
    request: Request,
    build_id: str,
    target_name: str = Query(...),
    step_name: str = Query(...),
    path: Optional[str] = Query(None),
    recursive: bool = Query(False),
) -> FileListResponse:
    """List files under a step-run's asset dir.

    `path` is relative to the asset dir; a running step's dir may be
    partially written.
    """
    build, target, step = lookup_build_target_step(build_id, target_name, step_name)
    authorize_build_access(request, build)

    async with open_bluevela_tunnel(build.space_name, target.environment_uri) as (
        tunnel,
        cfg,
    ):
        asset_dir = await resolve_step_asset_dir(
            build=build,
            target=target,
            step=step,
            workspace_remote_dir=cfg.workspace_remote_dir,
            tunnel=tunnel,
        )
        target_path = validate_subpath(asset_dir, path)
        logger.info(
            "[bluevela] list build=%s target=%s step=%s recursive=%s",
            build_id,
            target_name,
            step_name,
            recursive,
        )
        logger.debug("[bluevela] list asset_dir=%s target=%s", asset_dir, target_path)

        quoted = shlex.quote(str(target_path))
        if recursive:
            # `find -exec stat` is portable across GNU and BSD find (some
            # cluster filesystems ship BSD-style find where `-printf %F`
            # returns the filesystem type, not the entry type — using stat
            # for formatting keeps output consistent.
            cmd = (
                f"find {quoted} -exec stat -c '{_STAT_FMT}' {{}} \\; "
                f"| head -n {_FIND_MAX_ENTRIES}"
            )
        else:
            # stat on the dir + its entries. -L would follow symlinks; we don't.
            cmd = (
                f"stat -c '{_STAT_FMT}' -- {quoted} 2>/dev/null; "
                f"stat -c '{_STAT_FMT}' -- {quoted}/* 2>/dev/null || true"
            )

        rc, stdout, stderr = await tunnel.run_remote(cmd, raise_on_error=False)
        if rc != 0 and not stdout:
            err = (stderr or "").lower()
            if "no such file" in err or "cannot stat" in err:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND, "path not found under asset dir"
                )
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                f"listing failed: {stderr.strip() or 'unknown error'}",
            )

        entries: List[FileEntry] = []
        for line in (stdout or "").splitlines():
            entry = _parse_stat_line(line, asset_dir)
            if entry is not None:
                entries.append(entry)

        return FileListResponse(asset_dir=str(asset_dir), entries=entries)


# ------------------------------------------------------------- /files/content


@bluevela_api.get(
    "/builds/{build_id}/files/content",
    response_model=FileContentResponse,
)
async def get_file_content(
    request: Request,
    build_id: str,
    target_name: str = Query(...),
    step_name: str = Query(...),
    path: str = Query(..., min_length=1),
) -> FileContentResponse:
    """Return a small text file inline.

    Rejects files larger than BLUEVELA_CONTENT_MAX_BYTES (413) and files
    that look binary (415). Decodes as UTF-8 with `replace` so one stray
    byte in an otherwise-text log doesn't fail the whole response.
    """
    build, target, step = lookup_build_target_step(build_id, target_name, step_name)
    authorize_build_access(request, build)

    async with open_bluevela_tunnel(build.space_name, target.environment_uri) as (
        tunnel,
        cfg,
    ):
        asset_dir = await resolve_step_asset_dir(
            build=build,
            target=target,
            step=step,
            workspace_remote_dir=cfg.workspace_remote_dir,
            tunnel=tunnel,
        )
        target_path = validate_subpath(asset_dir, path)

        size, is_dir = await _remote_stat(tunnel, target_path)
        if is_dir:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "content endpoint requires a file, not a directory",
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

        # Binary sniff on the first 8 KiB.
        sniff_cmd = (
            f"head -c 8192 -- {shlex.quote(str(target_path))} | od -An -vtu1 -N 8192"
        )
        rc_s, sniff_stdout, sniff_stderr = await tunnel.run_remote(
            sniff_cmd, raise_on_error=False
        )
        if rc_s != 0:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                f"content probe failed: {sniff_stderr.strip() or 'unknown error'}",
            )
        byte_values = []
        for tok in (sniff_stdout or "").split():
            try:
                byte_values.append(int(tok))
            except ValueError:
                continue
        if any(b == 0 for b in byte_values):
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                "file appears to be binary (null byte detected)",
            )

        # Read the full body via SFTP — cleaner than shell-quoting `cat`.
        sftp = await tunnel.start_sftp()
        try:
            async with sftp.open(str(target_path), "rb") as fh:
                raw = await fh.read()
        finally:
            sftp.exit()
        content = raw.decode("utf-8", errors="replace")
        return FileContentResponse(
            path=(
                str(target_path.relative_to(asset_dir))
                if target_path != asset_dir
                else ""
            ),
            size=size,
            content=content,
            truncated=False,
        )


# ------------------------------------------------------------ /files/download


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


@bluevela_api.get("/builds/{build_id}/files/download")
async def download_file(
    request: Request,
    build_id: str,
    target_name: str = Query(...),
    step_name: str = Query(...),
    path: str = Query(..., min_length=1),
) -> StreamingResponse:
    """Stream a remote file as application/octet-stream.

    Rejects files larger than BLUEVELA_DOWNLOAD_MAX_BYTES. Uses SFTP so
    filenames with shell-unfriendly characters work and backpressure flows
    end-to-end. Rate-limiting is out of scope for v1.
    """
    build, target, step = lookup_build_target_step(build_id, target_name, step_name)
    authorize_build_access(request, build)

    # Tunnel lifecycle must outlive the streaming response body, so we
    # open it here, stat, then hand the tunnel to the streaming generator
    # and close it when that generator finishes.
    ctx = open_bluevela_tunnel(build.space_name, target.environment_uri)
    tunnel, cfg = await ctx.__aenter__()
    try:
        asset_dir = await resolve_step_asset_dir(
            build=build,
            target=target,
            step=step,
            workspace_remote_dir=cfg.workspace_remote_dir,
            tunnel=tunnel,
        )
        target_path = validate_subpath(asset_dir, path)

        size, is_dir = await _remote_stat(tunnel, target_path)
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

        filename = target_path.name or "download.bin"

        async def body() -> AsyncIterator[bytes]:
            try:
                async for chunk in _stream_sftp_file(tunnel, str(target_path)):
                    yield chunk
            finally:
                await ctx.__aexit__(None, None, None)

        return StreamingResponse(
            body(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(size),
            },
        )
    except BaseException:
        # Pre-stream failure: close the tunnel now.
        await ctx.__aexit__(None, None, None)
        raise


# --------------------------------------------------------------- /files/debug


class StepRunDebugResponse(BaseModel):
    step_run_dir: str
    entries: List[FileEntry]


@bluevela_api.get(
    "/builds/{build_id}/files/debug",
    response_model=StepRunDebugResponse,
)
async def list_step_run_debug(
    request: Request,
    build_id: str,
    target_name: str = Query(...),
    step_name: str = Query(...),
) -> StepRunDebugResponse:
    """Recursively list the step-run dir (every launch-* sibling and its files).

    Intentionally non-traversable: pinned to the step-run dir so the latest
    launch being empty doesn't hide earlier ones. Caps results at
    _FIND_MAX_ENTRIES.
    """
    build, target, step = lookup_build_target_step(build_id, target_name, step_name)
    authorize_build_access(request, build)

    resolved_step_name = _step_name_of(step)
    if resolved_step_name == "":
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"step {step.uuid} has no step.name in its config",
        )

    async with open_bluevela_tunnel(build.space_name, target.environment_uri) as (
        tunnel,
        cfg,
    ):
        step_run_dir = build_step_run_parent_dir(
            workspace_remote_dir=cfg.workspace_remote_dir,
            build_id=build.uuid,
            target_name=target.name,
            targetrun_id=target.uuid,
            step_name=resolved_step_name,
            targetsteprun_id=step.uuid,
        )
        logger.info(
            "[bluevela] debug build=%s target=%s step=%s",
            build_id,
            target_name,
            step_name,
        )
        logger.debug("[bluevela] debug step_run_dir=%s", step_run_dir)

        quoted = shlex.quote(str(step_run_dir))
        # pipefail so find's rc propagates through head; without it, a
        # missing or unreadable dir produces empty stdout but rc=0 (head's
        # success). `find -exec stat` is used in lieu of `find -printf`
        # for portability across GNU and BSD find — see list_files.
        cmd = (
            f"set -o pipefail; "
            f"find {quoted} -exec stat -c '{_STAT_FMT}' {{}} \\; "
            f"| head -n {_FIND_MAX_ENTRIES}"
        )
        rc, stdout, stderr = await tunnel.run_remote(cmd, raise_on_error=False)
        if rc != 0:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"step-run dir not accessible: {stderr.strip() or 'unknown error'}",
            )

        entries: List[FileEntry] = []
        for line in (stdout or "").splitlines():
            entry = _parse_stat_line(line, step_run_dir)
            if entry is not None:
                entries.append(entry)

        return StepRunDebugResponse(
            step_run_dir=str(step_run_dir), entries=entries
        )


__all__ = ["bluevela_api"]
