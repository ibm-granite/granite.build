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

"""Path resolution and auth helpers for the BlueVela remote-file REST API.

Path math is delegated to gbserver.environment.lsf_paths so the REST API and
the LSF build runtime stay in sync.

SECURITY: every remote path that enters a shell command or SFTP call MUST
pass through validate_subpath(). Do not concatenate user-supplied segments
onto the asset_dir yourself.
"""

import os
import shlex
from datetime import datetime
from pathlib import PurePosixPath
from typing import Optional, cast

from fastapi import HTTPException, Request, status

from gbserver.api.utils import confirm_space_write_access
from gbserver.environment.lsf_paths import build_step_run_parent_dir
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def authorize_build_access(request: Request, build: StoredBuild) -> None:
    """Raise 401 if the requester is not the build's owner or a space/super admin.

    Wraps the shared confirm_space_write_access to keep auth parity with
    PUT /builds/{id}/update.
    """
    confirm_space_write_access(request, build.username, build.space_name)


def lookup_build_target_step(
    build_id: str, target_name: str, step_name: str
) -> tuple[StoredBuild, StoredTargetRun, StoredStepRun]:
    """Resolve (build, target_run, step_run) by build_id + human names.

    Picks the most recent target run and its most recent step run matching
    the requested names. Raises 404 if any lookup fails.
    """
    storage: SingletonAdminStorage = get_admin_storage()
    build = storage.build_storage.get_by_uuid(build_id)
    if build is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"build {build_id!r} not found")
    assert isinstance(build, StoredBuild)

    target_runs = cast(
        list[StoredTargetRun],
        storage.target_storage.get_by_where(
            {"build_id": build_id, "name": target_name}
        ),
    )
    if not target_runs:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"target {target_name!r} not found on build {build_id}",
        )
    # Most recent target_run first; pending runs (no started_at) sort last.
    target = max(target_runs, key=lambda t: t.started_at or datetime.min)

    step_runs = cast(
        list[StoredStepRun],
        storage.step_storage.get_by_where({"target_id": target.uuid}),
    )
    matching_steps = [s for s in step_runs if _step_name_of(s) == step_name]
    if not matching_steps:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"step {step_name!r} not found on target {target_name!r}",
        )
    step = max(matching_steps, key=lambda s: s.started_at or datetime.min)
    return build, target, step


def _step_name_of(step: StoredStepRun) -> str:
    """Pull the human step name out of a StoredStepRun.

    Prefers config["step"]["name"]; falls back to the last non-empty path
    segment of definition_uri (e.g. "space://steps/tuning" -> "tuning",
    "file:///app/src/gbserver/builtins/steps/lhpull" -> "lhpull"). The
    fallback covers older step_run rows where config wasn't persisted.
    """
    try:
        name = step.config.get("step", {}).get("name", "")
    except AttributeError:
        name = ""
    if isinstance(name, str) and name:
        return name
    uri = step.definition_uri or ""
    # Strip query/fragment, then take the last non-empty path segment.
    path = uri.split("?", 1)[0].split("#", 1)[0]
    for segment in reversed(path.split("/")):
        if segment:
            return segment
    return ""


async def resolve_step_asset_dir(
    *,
    build: StoredBuild,
    target: StoredTargetRun,
    step: StoredStepRun,
    workspace_remote_dir: str,
    tunnel,
) -> PurePosixPath:
    """Return the absolute remote POSIX path of a step-launch's asset dir.

    Resolution: ``ls -1t`` the step-run parent dir and pick the newest
    ``launch-*`` entry. We do not persist launch_id on StoredStepRun, so
    mtime ordering is the source of truth. Concurrent overlapping
    relaunches are an accepted edge case.
    """
    # If per-launch selection is needed later, accept a launch_id arg here
    # and route through gbserver.environment.lsf_paths.build_remote_asset_dir.
    step_name = _step_name_of(step)
    if step_name == "":
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"step {step.uuid} has no step.name in its config",
        )

    parent = build_step_run_parent_dir(
        workspace_remote_dir=workspace_remote_dir,
        build_id=build.uuid,
        target_name=target.name,
        targetrun_id=target.uuid,
        step_name=step_name,
        targetsteprun_id=step.uuid,
    )
    cmd = f"ls -1t -- {shlex.quote(str(parent))}"
    rc, stdout, stderr = await tunnel.run_remote(cmd, raise_on_error=False)
    if rc != 0:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"step-run dir not accessible: {stderr.strip() or 'unknown error'}",
        )
    entries = [e for e in (stdout or "").splitlines() if e.startswith("launch-")]
    if not entries:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"no launch directories found under step-run {step.uuid}",
        )
    return parent / entries[0]


def validate_subpath(
    asset_dir: PurePosixPath, user_path: Optional[str]
) -> PurePosixPath:
    """Resolve user_path relative to asset_dir, rejecting anything escaping it.

    Rejects: absolute paths, ``~``, ``..`` segments that escape the root,
    null bytes, backslashes. Returns a normalized absolute PurePosixPath
    that is guaranteed to be at or below asset_dir.
    """
    raw = (user_path or "").strip()
    if raw == "":
        return asset_dir

    if "\x00" in raw or "\\" in raw:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "path contains illegal characters"
        )
    if raw.startswith("/") or raw.startswith("~"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "path must be relative to the step asset dir"
        )

    # os.path.normpath handles ../ collapsing on posix when run on any OS,
    # since our inputs are always posix-style. Re-join and re-check.
    normalized = os.path.normpath(raw)
    if normalized.startswith("..") or normalized == "..":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "path traversal above asset dir rejected"
        )
    candidate = asset_dir / normalized
    # Final defense: a normalized candidate must still be a descendant.
    try:
        candidate.relative_to(asset_dir)
    except ValueError as e:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "path escapes asset dir"
        ) from e
    return candidate


__all__ = [
    "authorize_build_access",
    "lookup_build_target_step",
    "resolve_step_asset_dir",
    "validate_subpath",
]
