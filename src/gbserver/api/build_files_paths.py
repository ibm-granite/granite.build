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

"""Path resolution and auth helpers for the build-files REST API.

Path math is delegated to gbserver.environment.lsf_paths so the REST API and
the LSF build runtime stay in sync.

SECURITY: every remote path that enters a shell command or SFTP call MUST
pass through validate_subpath(). Do not concatenate user-supplied segments
onto the build root yourself.
"""

import os
import shlex
from pathlib import PurePosixPath
from typing import Optional

from fastapi import HTTPException, Request, status

from gbserver.api.utils import confirm_space_write_access
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.storage.stored_build import StoredBuild
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def authorize_build_access(request: Request, build: StoredBuild) -> None:
    """Raise 401 if the requester is not the build's owner or a space/super admin.

    Wraps the shared confirm_space_write_access to keep auth parity with
    PUT /builds/{id}/update.
    """
    confirm_space_write_access(request, build.username, build.space_name)


def lookup_build(build_id: str) -> StoredBuild:
    """Load a StoredBuild by uuid; 404 if missing."""
    storage: SingletonAdminStorage = get_admin_storage()
    build = storage.build_storage.get_by_uuid(build_id)
    if build is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"build {build_id!r} not found")
    assert isinstance(build, StoredBuild)
    return build


def validate_subpath(
    build_root: PurePosixPath, user_path: Optional[str]
) -> PurePosixPath:
    """Resolve user_path relative to build_root, rejecting anything escaping it.

    Rejects: absolute paths, ``~``, ``..`` segments that escape the root,
    null bytes, backslashes. Returns a normalized absolute PurePosixPath
    that is guaranteed to be at or below build_root.
    """
    raw = (user_path or "").strip()
    if raw == "":
        return build_root

    if "\x00" in raw or "\\" in raw:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "path contains illegal characters"
        )
    if raw.startswith("/") or raw.startswith("~"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "path must be relative to the build root"
        )

    # os.path.normpath handles ../ collapsing on posix when run on any OS,
    # since our inputs are always posix-style. Re-join and re-check.
    normalized = os.path.normpath(raw)
    if normalized.startswith("..") or normalized == "..":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "path traversal above build root rejected"
        )
    candidate = build_root / normalized
    # Final defense: a normalized candidate must still be a descendant.
    try:
        candidate.relative_to(build_root)
    except ValueError as e:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "path escapes build root"
        ) from e
    return candidate


async def resolve_and_check_real_path(
    tunnel,
    build_root: PurePosixPath,
    candidate: PurePosixPath,
) -> PurePosixPath:
    """Resolve symlinks remotely and confirm the result stays under build_root.

    Returns 404 (without leaking existence) if readlink fails OR the resolved
    path escapes build_root.
    """
    cmd = f"readlink -f -- {shlex.quote(str(candidate))}"
    rc, stdout, _ = await tunnel.run_remote(cmd, raise_on_error=False)
    resolved = (stdout or "").strip()
    if rc != 0 or not resolved:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "path not found")
    real = PurePosixPath(resolved)
    if not real.is_relative_to(build_root):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "path not found")
    return real


__all__ = [
    "authorize_build_access",
    "lookup_build",
    "resolve_and_check_real_path",
    "validate_subpath",
]
