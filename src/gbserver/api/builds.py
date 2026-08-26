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

import base64
import io
import zipfile
from enum import StrEnum, auto
from typing import Annotated, Dict, List, Optional, Self, Tuple, cast

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator

from gbserver.api.build_files_paths import authorize_build_read_access
from gbserver.api.utils import (
    NO_ACCESSIBLE_SPACE,
    ListAppendOrSet,
    apply_tag_update,
    confirm_space_write_access,
    get_query_control,
    get_row_filter,
    has_space_write_access,
    is_space_admin,
    is_super_admin,
    scope_space_name_filter,
    split_tags,
)
from gbserver.buildrunner.validation import BuildValidation
from gbserver.buildwatcher.buildwatcher import BuildWatcher
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.build_storage import IStoredBuildStorage
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.storage.space_storage import IStoredSpaceStorage
from gbserver.storage.stored_build import StoredBuild, reopen_finished_build
from gbserver.storage.stored_event import StoredEvent
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.api.builds import BuildValidateRequestType
from gbserver.types.auth import User
from gbserver.types.status import Status
from gbserver.types.validation import GBValidationErrors
from gbserver.utils.archive import check_zip_safe
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

builds_api = FastAPI()


class BuildSubmitRequest(BaseModel):
    """
    A build submission request.

        build_archive: The base64 encoded zip file of the build directory
        space_name: The name of the space to use. Mutually exclusive with space_uri
        username: The name of the user submitting the build
        targets: A list of targets to run (must be a subset of the targets in the build.yaml)
        description:
        tags: a list of string tokens.
    """

    name: str = "build-submitted-via-api"
    build_archive: str
    space_name: str
    username: str
    targets: Optional[List[str]] = None
    description: Optional[str] = ""
    tags: Optional[List[str]] = None

    @model_validator(mode="after")
    def validate_space(self: Self) -> Self:
        if self.build_archive == "":
            raise ValueError("build_archive cannot be empty")
        if self.space_name == "":
            raise ValueError("space_name cannot be empty")
        if self.username == "":
            raise ValueError("username cannot be empty")
        return self


class BuildRestartRequest(BaseModel):
    """
    A build restart request.

    Restarts a previously-executed build: the **same** build is re-opened and a
    fresh build runner re-runs it, skipping targets that already succeeded and
    re-running the rest. The build definition, space, and targets are already on
    the build, so only its id is required.

        build_id: uuid of the finished build to restart.
    """

    build_id: str

    @model_validator(mode="after")
    def validate_build_id(self: Self) -> Self:
        if self.build_id == "":
            raise ValueError("build_id cannot be empty")
        return self


class BuildSubmitResponse(BaseModel):
    """Response to a build submission."""

    build_id: str


class BuildRestartResponse(BaseModel):
    """Response to a build restart.

    build_id: uuid of the restarted build. Restart reuses the same build id, so
        this equals the requested build_id.
    """

    build_id: str


class BuildValidateRequest(BaseModel):
    """
    A build validation request.

        build_archive: The base64 encoded zip file of the build directory
        validation_type: The type of validation to perform - static/dynamic
        space_name: The name of the space to use. Mutually exclusive with space_uri
        space_uri: The URI of the space to use. Mutually exclusive with space_name
        username: The name of the user submitting the build
        targets: A list of targets to run (must be a subset of the targets in the build.yaml)
    """

    build_archive: str
    validation_type: BuildValidateRequestType = BuildValidateRequestType.STATIC
    space_name: str = ""
    space_uri: str = ""
    username: str = ""
    targets: Optional[List[str]] = None

    @model_validator(mode="after")
    def validate_space(self: Self) -> Self:
        # if self.validation_type is not BuildValidateRequestType.STATIC:
        #     raise ValueError("only static validation is supported right now")
        if self.build_archive == "":
            raise ValueError("build_archive cannot be empty")
        if self.space_name == "" and self.space_uri == "":
            raise ValueError("must specify either the space name or URI")
        if self.username == "":
            raise ValueError("username cannot be empty")
        return self


class ListBuildResponse(BaseModel):
    builds: list[StoredBuild]


class CountBuildsResponse(BaseModel):
    count: int


class BuildEventsResponse(BaseModel):
    build_id: str
    events: list[StoredEvent]


class GetBuildResponse(BaseModel):
    build: StoredBuild


class TargetRecord(BaseModel):
    target: StoredTargetRun
    steps: list[StoredStepRun]
    input_artifacts: list[ArtifactRegistration] = []
    output_artifacts: list[ArtifactRegistration] = []


class BuildStatus(BaseModel):
    build: StoredBuild
    target_runs: list[TargetRecord]


class BuildStatusResponse(BaseModel):
    status: BuildStatus


class CancelBuildResponse(BaseModel):
    canceled: StoredBuild


# Needed to list all
@builds_api.get("/")
def list_builds(
    request: Request,
    name: str = "",
    space_name: str = "",
    source_uri: str = "",
    username: str = "",  # needed
    tag: Annotated[
        list[str] | None, Query()
    ] = [],  # Specified as multiple tag=v1&tag=v2 in URI
    status: Annotated[
        list[str] | None, Query()
    ] = [],  # Specified as multiple status=RUNNING&status=PENDING in URI
    sort: Annotated[
        list[str] | None, Query()
    ] = [],  # Specified as 1 or more sort=<column>[]:(asc|desc)]
    page_index: int = -1,
    page_size: int = 0,
) -> ListBuildResponse:
    scoped_space_name = scope_space_name_filter(request, space_name)
    if scoped_space_name is NO_ACCESSIBLE_SPACE:
        return ListBuildResponse(builds=[])
    row_filter = get_row_filter(
        name=name,
        space_name=scoped_space_name,
        source_uri=source_uri,
        username=username,
        tags=tag,
        status=status,
    )
    query_control = get_query_control(sort, page_index, page_size)

    storage = get_admin_storage()
    build_storage = storage.build_storage
    item_list = cast(
        List[StoredBuild],
        build_storage.get_by_where(where=row_filter, query_control=query_control),
    )

    # Due to the increasing volume, the list API is too slow unless build_archive is excluded in the response
    def remove_build_archive(build: StoredBuild) -> StoredBuild:
        build.build_archive = ""
        return build

    item_list = [remove_build_archive(b) for b in item_list]

    resp = ListBuildResponse(builds=item_list)
    return resp


@builds_api.get("/count")
def count_builds(
    request: Request,
    name: str = "",
    space_name: str = "",
    source_uri: str = "",
    username: str = "",
    tag: Annotated[list[str] | None, Query()] = [],
    status: Annotated[list[str] | None, Query()] = [],
) -> CountBuildsResponse:
    """Return the number of builds matching the filter criteria."""
    scoped_space_name = scope_space_name_filter(request, space_name)
    if scoped_space_name is NO_ACCESSIBLE_SPACE:
        return CountBuildsResponse(count=0)
    row_filter = get_row_filter(
        name=name,
        space_name=scoped_space_name,
        source_uri=source_uri,
        username=username,
        tags=tag,
        status=status,
    )
    storage = get_admin_storage()
    build_storage = storage.build_storage
    count = build_storage.count(where=row_filter)
    return CountBuildsResponse(count=count)


@builds_api.post("/")
def submit_build(request: Request, req: BuildSubmitRequest) -> BuildSubmitResponse:
    # gather space information
    storage = get_admin_storage()
    space_storage: IStoredSpaceStorage = storage.space_storage
    build_storage: IStoredBuildStorage = storage.build_storage
    space_name = req.space_name
    stored_space = space_storage.get_by_name(space_name)
    if stored_space is None:
        err_no_space = f"Space {space_name} not found in space storage"
        logger.error("%s", err_no_space)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=err_no_space)

    # Protect system tags
    sys_tags, _ = split_tags(req.tags)
    if len(sys_tags) > 0 and not is_super_admin(request):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    # req.username is the identity the build will run under and whose per-user
    # secrets get injected into it — bind it to the caller unless the caller
    # is a space/super admin explicitly impersonating another user, the same
    # gate PUT /builds/{id}/update already applies to build.username.
    confirm_space_write_access(
        request, username_on_target=req.username, space_name=stored_space.name
    )

    stored_build = StoredBuild.create(
        name=req.name,
        space_name=stored_space.name,
        source_uri="",
        username=req.username,
        build_archive=req.build_archive,
        status=Status.SUBMITTED,
        targets=req.targets,
        description=req.description,
        tags=req.tags,
    )

    result = build_storage.add(stored_build)
    logger.info("stored build with id: %s", result)

    return BuildSubmitResponse(
        build_id=stored_build.uuid,
    )


@builds_api.post("/restart")
def restart_build(request: Request, req: BuildRestartRequest) -> BuildRestartResponse:
    """Restart a previously-executed build in a fresh runner.

    Re-opens the **same** build (reusing its build id) by flipping its finished
    status back to SUBMITTED, so the BuildWatcher re-dispatches it onto a fresh
    runner that skips targets which already succeeded and re-runs the rest. The
    build must be finished — restarting a build that is still active (there may be
    a live runner) is rejected.
    """
    storage = get_admin_storage()
    build_storage: IStoredBuildStorage = storage.build_storage
    space_storage: IStoredSpaceStorage = storage.space_storage

    # Every "you may not see this build" path (missing build, missing space, no
    # write access) must return the SAME 404: otherwise a caller lacking access
    # could tell a real build id (401) from a nonexistent one (404) and enumerate
    # ids across spaces they cannot reach. Authorize before disclosing the build's
    # existence or (below) its liveness.
    not_found = HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Build {req.build_id} not found",
    )
    build = build_storage.get_by_uuid(req.build_id)
    if not isinstance(build, StoredBuild):
        raise not_found

    # Authorize before disclosing the build's liveness (the is_finished 409 below).
    stored_space = space_storage.get_by_name(build.space_name)
    if stored_space is None:
        raise not_found
    has_access, _ = has_space_write_access(
        request, username_on_target=build.username, space_name=stored_space.name
    )
    if not has_access:
        raise not_found

    # Only a finished build can be restarted: re-opening a build with a live runner
    # would attach a second runner to it. There is no runner-liveness table; the
    # build status is the signal.
    if not build.status.is_finished():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Build {build.uuid} has status {build.status}; only a finished "
                "build (FAILED, INVALID, or CANCELLED) can be restarted"
            ),
        )

    # A fully-succeeded build has nothing to restart: every target already
    # succeeded, so target reuse would skip all of them and the fresh runner would
    # do no work. Reject it rather than re-open a completed build. (reopen's atomic
    # guard also rejects SUCCESS, covering a build that succeeds after this check.)
    if build.status == Status.SUCCESS:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Build {build.uuid} already succeeded; a SUCCESS build cannot be "
                "restarted (all targets have already completed)"
            ),
        )

    # Atomic guarded flip finished -> SUBMITTED; None means the build stopped being
    # finished between the check above and the write (a concurrent restart or a
    # runner that just picked it up) — reject rather than double-dispatch.
    reopened = reopen_finished_build(build_storage, build)
    if reopened is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Build {build.uuid} cannot be restarted: its status changed "
                "concurrently"
            ),
        )
    logger.info("re-opened build %s for restart (was %s)", build.uuid, build.status)
    return BuildRestartResponse(build_id=reopened.uuid)


@builds_api.post("/validate")
def validate_build(request: Request, req: BuildValidateRequest) -> JSONResponse:
    # req.username drives real per-user secret resolution inside Space (see
    # buildrunner/validation.py -> build/space.py), the same as submit_build's
    # req.username does — bind it to the caller the same way.
    if req.space_name:
        storage = get_admin_storage()
        stored_space = storage.space_storage.get_by_name(req.space_name)
        if stored_space is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Space {req.space_name} not found in space storage",
            )
        confirm_space_write_access(
            request, username_on_target=req.username, space_name=stored_space.name
        )
    else:
        # space_uri bypasses space storage entirely (validate_build_archive
        # builds a Space directly from the URI), so there is no stored space
        # to check admin-ness against. Validating as someone else here can
        # only be gated on being a caller-independent (super) admin.
        user_id = request.state.data["user"].login
        if req.username != user_id and not is_super_admin(request):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"User {user_id} cannot validate a build as {req.username}",
            )

    errors = BuildValidation.validate_build_archive(
        build_archive=req.build_archive,
        username=req.username,
        targets=req.targets,
        space_or_name=req.space_name,
        space_uri=req.space_uri,
        validation_type=req.validation_type,
    )
    status_code = (
        status.HTTP_200_OK
        if errors.is_valid()
        else status.HTTP_422_UNPROCESSABLE_CONTENT
    )
    return JSONResponse(
        content=errors.model_dump(),
        status_code=status_code,
    )


@builds_api.get("/tags")
def list_build_tags(
    request: Request,
    name: str = "",
    space_name: str = "",
    source_uri: str = "",
    username: str = "",  # needed
) -> List[str]:
    """Return the sort list of unique tag strings for the builds that match the condition."""
    # In this version, it simply pulls all the builds and programatically takes a unique
    builds_response = list_builds(
        request,
        name=name,
        space_name=space_name,
        source_uri=source_uri,
        username=username,
    )
    tags = set()  # type: ignore[var-annotated]
    for build in builds_response.builds:
        tags.update(build.tags)  # type: ignore[arg-type]
    unique_tags = list(tags)
    unique_tags.sort()
    return unique_tags


@builds_api.get("/{build_id}")
def read_build(request: Request, build_id: str) -> GetBuildResponse:
    storage: SingletonAdminStorage = get_admin_storage()
    build_storage = storage.build_storage
    item = build_storage.get_by_uuid(build_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="build not found!"
        )
    assert isinstance(item, StoredBuild), f"invalid item: {item}"
    authorize_build_read_access(request, item)
    resp = GetBuildResponse(build=item)
    return resp


@builds_api.get("/{build_id}/archive")
def get_build_archive(request: Request, build_id: str) -> Dict[str, Dict[str, str]]:
    """Decode the build's ZIP archive and return its files as a dict.

    Returns ``{"files": {"path/in/zip": "file contents", ...}}``.
    Used by the frontend Definition tab to display build.yaml and related files.
    """
    storage: SingletonAdminStorage = get_admin_storage()
    build = storage.build_storage.get_by_uuid(build_id)
    if build is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="build not found!"
        )
    assert isinstance(build, StoredBuild), f"invalid item: {build}"
    authorize_build_read_access(request, build)
    if not build.build_archive:
        return {"files": {}}
    raw = base64.b64decode(build.build_archive)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        try:
            check_zip_safe(zf)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e)
            )
        files = {name: zf.read(name).decode(errors="replace") for name in zf.namelist()}
    return {"files": files}


def __get_artifacts(
    storage: SingletonAdminStorage, target: StoredTargetRun
) -> Tuple[List[ArtifactRegistration], List[ArtifactRegistration]]:
    input_uuids = list(target.input_artifacts.values())
    output_uuids = []
    for _, uuids in target.output_artifacts.items():
        output_uuids.extend(uuids)
    uuid_list = input_uuids + output_uuids
    input_artifacts = []
    output_artifacts = []
    if len(uuid_list) > 0:
        artifacts = storage.artifact_registry.get_by_uuid(uuid_list)
        assert isinstance(artifacts, list), f"invalid artifacts: {artifacts}"
        assert len(uuid_list) == len(
            artifacts
        ), f"unequal lengths uuid_list: {uuid_list} artifacts: {artifacts}"
        for i, a in enumerate(artifacts):
            if i < len(input_uuids):
                input_artifacts.append(a)
            else:
                output_artifacts.append(a)
    return input_artifacts, output_artifacts


def __build_target_records(
    storage: SingletonAdminStorage, build_id: str
) -> List[TargetRecord]:
    """Assemble the per-target records (steps + artifacts) for one build."""
    row_filter = get_row_filter(build_id=build_id)
    target_runs = cast(
        List[StoredTargetRun], storage.target_storage.get_by_where(row_filter)
    )
    target_records = []
    for target in target_runs:
        input_artifacts, output_artifacts = __get_artifacts(storage, target)
        steps = cast(
            list[StoredStepRun],
            storage.step_storage.get_by_where({"target_id": target.uuid}),
        )
        record = TargetRecord(
            target=target,
            steps=steps,
            input_artifacts=input_artifacts,
            output_artifacts=output_artifacts,
        )
        target_records.append(record)
    return target_records


@builds_api.get("/{build_id}/status", response_model=BuildStatusResponse)
def get_build_status(request: Request, build_id: str) -> BuildStatusResponse:
    storage: SingletonAdminStorage = get_admin_storage()
    build = storage.build_storage.get_by_uuid(build_id)
    if build is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="build not found!"
        )
    assert isinstance(build, StoredBuild)
    authorize_build_read_access(request, build)
    build.build_archive = ""
    build_status = BuildStatus(
        build=build, target_runs=__build_target_records(storage, build_id)
    )
    return BuildStatusResponse(status=build_status)


@builds_api.get("/{build_id}/status2", response_model=BuildStatusResponse)
def get_build_status2(request: Request, build_id: str) -> BuildStatusResponse:
    # Retained as a backward-compatible alias of the primary /status endpoint.
    return get_build_status(request, build_id)


@builds_api.get("/{build_id}/events")
def get_buildevents(request: Request, build_id: str):
    storage: SingletonAdminStorage = get_admin_storage()
    build = storage.build_storage.get_by_uuid(build_id)
    if build is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="build not found!"
        )
    assert isinstance(build, StoredBuild)
    authorize_build_read_access(request, build)

    row_filter = get_row_filter(build_id=build_id)
    events = cast(List[StoredEvent], storage.event_storage.get_by_where(row_filter))
    # TODO sort order may be more preferrable by the index. Currently StoredEvent doesn't capture the index column
    resp = BuildEventsResponse(
        build_id=build_id,
        events=sorted(events, key=lambda event: event.build_event.timestamp),
    )
    return resp


# @builds_api.get("/{build_id}/logs")
# async def get_build_status(build_id: str) -> dict:
#     item = await storage.get_by_uuid(build_id)
#     if item is None:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND, detail="build not found!"
#         )
#     build_logs = await item.get_logs()
#     return {"logs": build_logs}

# def is_space_admin(space_name:str, username:str):
#     #TODO: This needs implementation when we support space administrators.
#     # Also, move to the utils module.
#     return True


@builds_api.delete("/{build_id}")
# Sync endpoint (like submit/restart/validate): FastAPI runs it in a threadpool so
# its blocking work stays off the event loop. request_cancellation reads storage
# and, for a FAILED build, calls has_retries_remaining() -> get_build_config(),
# which can extract the build archive from disk — must not run on the loop.
def cancel_build(build_id: str, request: Request) -> CancelBuildResponse:
    storage: SingletonAdminStorage = get_admin_storage()
    build = storage.build_storage.get_by_uuid(build_id)
    if build is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="build not found!"
        )
    assert isinstance(build, StoredBuild)
    user: User = request.state.data["user"]
    if user is None:  # Should never hit this
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="user not found"
        )
    elif (
        user.login != build.username
        and not is_space_admin(request, build.space_name)
        and not is_super_admin(request)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="You are not the owner or admin of this build.",
        )

    build = request_cancellation(storage.build_storage, build)

    build.build_archive = ""
    response: CancelBuildResponse = CancelBuildResponse(canceled=build)
    return response


def request_cancellation(
    build_storage: IStoredBuildStorage, build: StoredBuild
) -> StoredBuild:
    """Apply a cancellation request to a build.

    With in-place retry there is a single build id across attempts: an in-flight
    (possibly retrying) build is RUNNING and is set to CANCEL_REQUESTED; a
    SUBMITTED/PENDING build is set to CANCELLED. A build that is FAILED but still
    has retries remaining sits in a brief pre-retry window and is set directly to
    CANCELLED. A genuinely finished build (SUCCESS/CANCELLED, or a FAILED build
    whose retries are exhausted) is not cancellable.

    Args:
        build_storage: storage used to read/update builds.
        build: the build the cancellation was requested for.

    Returns:
        The build whose status was updated, or the unchanged build if no update
        was needed.

    Raises:
        HTTPException: 412 if the build is not cancellable; 409 if the status
            changed concurrently.
    """
    current_status = build.status
    # A build sits in FAILED for a short window between a failed attempt and the
    # retry loop flipping it back to RUNNING. If it still has retries remaining it
    # is effectively in-flight, so a cancel landing in that window must be honored
    # rather than rejected as "already finished".
    failed_but_retrying = (
        current_status == Status.FAILED and build.has_retries_remaining()
    )
    if not current_status.is_cancellable() and not failed_but_retrying:
        raise HTTPException(
            status_code=status.HTTP_412_PRECONDITION_FAILED,
            detail=f"Build {build.uuid} has status {current_status} and therefore can not be canceled.",
        )
    # A RUNNING build (including one the retry loop re-ran in place) is in-flight
    # and must be stopped by the runner.
    if current_status == Status.RUNNING:
        target_status: Optional[Status] = Status.CANCEL_REQUESTED
    elif current_status in (Status.SUBMITTED, Status.PENDING):
        target_status = Status.CANCELLED
    elif failed_but_retrying:
        # The failed attempt already tore down; there is no running work to stop.
        # Marking it CANCELLED both records the cancel and prevents the pending
        # retry (the runner's _should_retry only retries a FAILED build). The
        # should_update guard below yields 409 if the runner already advanced it to
        # RUNNING, prompting the client to retry into the CANCEL_REQUESTED path.
        target_status = Status.CANCELLED
    else:  # CANCEL_REQUESTED (already requested) or anything else: no update
        target_status = None

    if target_status is None:
        return build

    # Use a callback to ensure the status hasn't changed (fixes race condition).
    updated_build = build_storage.update_fields(
        build.uuid,
        {"status": target_status},
        should_update=lambda item: item.status == current_status,
    )
    if updated_build is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Build {build.uuid} status changed during update.  Try again?.",
        )
    return updated_build


class BuildUpdateRequest(BaseModel):
    description: Optional[str] = None
    tags: Optional[ListAppendOrSet] = None


class BuildUpdateResponse(BaseModel):
    build: StoredBuild


@builds_api.put("/{build_id}/update")
def update_build(
    request: Request, build_id: str, update: BuildUpdateRequest
) -> BuildUpdateResponse:
    read_resp = read_build(request, build_id)
    if read_resp is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Can not find build with id {build_id}!",
        )

    assert isinstance(read_resp, GetBuildResponse)
    build = read_resp.build
    assert isinstance(build, StoredBuild)

    # Make sure the user (owner or admin) has access to the build
    confirm_space_write_access(
        request=request, username_on_target=build.username, space_name=build.space_name
    )

    updates = {}
    if update.description is not None:
        build.description = update.description
        updates["description"] = update.description
    if update.tags:
        is_super = is_super_admin(request)
        apply_tag_update(build, update.tags, is_super)
        updates["tags"] = build.tags  # type: ignore[assignment]

    # Store the update
    if len(updates) > 0:
        storage = get_admin_storage().build_storage
        build = storage.update_fields(build.uuid, updates)  # type: ignore[assignment]

    build.build_archive = ""  # Don't send back the (large) archive.
    resp = BuildUpdateResponse(build=build)
    return resp
