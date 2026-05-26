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

from enum import StrEnum, auto
from typing import Annotated, List, Optional, Self, Tuple, cast

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator

from gbserver.api.utils import (
    ListAppendOrSet,
    apply_tag_update,
    confirm_space_write_access,
    get_query_control,
    get_row_filter,
    is_space_admin,
    is_super_admin,
    split_tags,
)
from gbserver.buildwatcher.buildwatcher import BuildWatcher
from gbserver.buildwatcher.validation import BuildValidation
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.build_storage import IStoredBuildStorage
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.storage.space_storage import IStoredSpaceStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_event import StoredEvent
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.api.builds import BuildValidateRequestType
from gbserver.types.auth import User
from gbserver.types.constants import GBSERVER_WEBHOOKS_ALLOW_HTTP, GBSERVER_WEBHOOKS_ENABLED
from gbserver.types.status import Status
from gbserver.types.validation import GBValidationErrors
from gbserver.utils.logger import get_logger
from gbserver.webhooks.models import WEBHOOK_MIN_FREQUENCY, StoredWebhookSubscription
from gbserver.webhooks.sql_storage import create_webhook_storage
from gbserver.webhooks.url_validator import WebhookURLError, validate_webhook_url

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
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    webhook_event_types: Optional[List[str]] = None
    webhook_frequency: Optional[int] = None

    @model_validator(mode="after")
    def validate_space(self: Self) -> Self:
        if self.build_archive == "":
            raise ValueError("build_archive cannot be empty")
        if self.space_name == "":
            raise ValueError("space_name cannot be empty")
        if self.username == "":
            raise ValueError("username cannot be empty")
        return self


class BuildSubmitResponse(BaseModel):
    """Response to a build submission."""

    build_id: str
    webhook_subscription_id: Optional[str] = None


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


class TargetRecord2(BaseModel):
    target: StoredTargetRun
    steps: list[StoredStepRun]
    input_artifacts: list[ArtifactRegistration] = []
    output_artifacts: list[ArtifactRegistration] = []


class BuildStatus2(BaseModel):
    build: StoredBuild
    target_runs: list[TargetRecord2]


class BuildStatusResponse2(BaseModel):
    status: BuildStatus2


class CancelBuildResponse(BaseModel):
    canceled: StoredBuild


# Needed to list all
@builds_api.get("/")
def list_builds(
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
    row_filter = get_row_filter(
        name=name,
        space_name=space_name,
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
    name: str = "",
    space_name: str = "",
    source_uri: str = "",
    username: str = "",
    tag: Annotated[list[str] | None, Query()] = [],
    status: Annotated[list[str] | None, Query()] = [],
) -> CountBuildsResponse:
    """Return the number of builds matching the filter criteria."""
    row_filter = get_row_filter(
        name=name,
        space_name=space_name,
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

    # Auto-create webhook subscription BEFORE enqueuing the build
    # (ensures subscription exists when BuildRunner starts processing events)
    webhook_subscription_id = _create_webhook_subscription(req, stored_build)

    result = build_storage.add(stored_build)
    logger.info("stored build with id: %s", result)

    return BuildSubmitResponse(
        build_id=stored_build.uuid,
        webhook_subscription_id=webhook_subscription_id,
    )


def _create_webhook_subscription(
    req: BuildSubmitRequest, stored_build: StoredBuild
) -> Optional[str]:
    """Create a webhook subscription for a build if webhook_url is provided.

    This is a best-effort operation: if subscription creation fails, the
    build is still considered successfully submitted.

    Args:
        req: The original build submission request containing webhook fields.
        stored_build: The persisted build record.

    Returns:
        The UUID of the created subscription, or None if not created.
    """
    if not req.webhook_url:
        return None

    try:
        if not GBSERVER_WEBHOOKS_ENABLED:
            return None

        # Validate URL (best-effort — don't fail build submission)
        allow_http = GBSERVER_WEBHOOKS_ALLOW_HTTP
        try:
            validate_webhook_url(req.webhook_url, allow_http=allow_http)
        except WebhookURLError as e:
            logger.warning(
                "Webhook URL validation failed for build %s: %s", stored_build.uuid, e
            )
            return None

        # Require a meaningful secret for HMAC authentication
        if not req.webhook_secret or len(req.webhook_secret) < 8:
            logger.warning(
                "Webhook secret too short (min 8 chars) for build %s, skipping subscription",
                stored_build.uuid,
            )
            return None

        webhook_storage = create_webhook_storage()
        subscription = StoredWebhookSubscription(
            space_name=stored_build.space_name,
            build_id=None,
            build_filter=stored_build.uuid,
            scope="space",
            status="active",
            webhook_url=req.webhook_url,
            secret=req.webhook_secret,
            event_types=req.webhook_event_types or ["*"],
            frequency=max(
                req.webhook_frequency or WEBHOOK_MIN_FREQUENCY,
                WEBHOOK_MIN_FREQUENCY,
            ),
            created_by=req.username,
        )
        webhook_storage.add(subscription)
        logger.info(
            "Auto-created webhook subscription %s for build %s",
            subscription.uuid,
            stored_build.uuid,
        )
        return subscription.uuid
    except Exception as e:
        logger.warning("Failed to create webhook subscription: %s", e)
        return None


@builds_api.post("/validate")
def validate_build(req: BuildValidateRequest) -> JSONResponse:
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
    name: str = "",
    space_name: str = "",
    source_uri: str = "",
    username: str = "",  # needed
) -> List[str]:
    """Return the sort list of unique tag strings for the builds that match the condition."""
    # In this version, it simply pulls all the builds and programatically takes a unique
    builds_response = list_builds(
        name=name, space_name=space_name, source_uri=source_uri, username=username
    )
    tags = set()  # type: ignore[var-annotated]
    for build in builds_response.builds:
        tags.update(build.tags)  # type: ignore[arg-type]
    unique_tags = list(tags)
    unique_tags.sort()
    return unique_tags


@builds_api.get("/{build_id}")
def read_build(build_id: str) -> GetBuildResponse:
    storage: SingletonAdminStorage = get_admin_storage()
    build_storage = storage.build_storage
    item = build_storage.get_by_uuid(build_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="build not found!"
        )
    assert isinstance(item, StoredBuild), f"invalid item: {item}"
    resp = GetBuildResponse(build=item)
    return resp


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


@builds_api.get("/{build_id}/status", response_model=BuildStatusResponse2)
def get_build_status(build_id: str) -> BuildStatusResponse2:
    return get_build_status2(build_id)


@builds_api.get("/{build_id}/status2", response_model=BuildStatusResponse2)
def get_build_status2(build_id: str) -> BuildStatusResponse2:
    storage: SingletonAdminStorage = get_admin_storage()
    build = storage.build_storage.get_by_uuid(build_id)
    if build is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="build not found!"
        )
    assert isinstance(build, StoredBuild)
    build.build_archive = ""
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
        record = TargetRecord2(
            target=target,
            steps=steps,
            input_artifacts=input_artifacts,
            output_artifacts=output_artifacts,
        )
        target_records.append(record)
    build_status = BuildStatus2(build=build, target_runs=target_records)
    resp = BuildStatusResponse2(status=build_status)
    return resp


@builds_api.get("/{build_id}/events")
def get_buildevents(build_id: str):
    storage: SingletonAdminStorage = get_admin_storage()
    build = storage.build_storage.get_by_uuid(build_id)
    if build is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="build not found!"
        )
    assert isinstance(build, StoredBuild)

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
async def cancel_build(build_id: str, request: Request) -> CancelBuildResponse:
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

    # Determine the target status based on current status
    current_status = build.status
    if not current_status.is_cancellable():
        raise HTTPException(
            status_code=status.HTTP_412_PRECONDITION_FAILED,
            detail=f"Build {build.uuid} has status {current_status} and therefore can not be canceled.",
        )
    elif current_status == Status.RUNNING:
        target_status = Status.CANCEL_REQUESTED
    elif current_status == Status.SUBMITTED or current_status == Status.PENDING:
        target_status = Status.CANCELLED
    elif current_status == Status.CANCEL_REQUESTED:
        target_status = None  # Already requested, no update needed
    else:
        target_status = None

    if target_status is not None:
        # Use callback to ensure status hasn't changed (fixes race condition)
        updated_build = storage.build_storage.update_fields(
            build.uuid,
            {"status": target_status},
            should_update=lambda item: item.status == current_status,
        )
        if updated_build is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Build {build.uuid} status changed during update.  Try again?.",
            )
        build = updated_build

    build.build_archive = ""
    response: CancelBuildResponse = CancelBuildResponse(canceled=build)
    return response


class BuildUpdateRequest(BaseModel):
    description: Optional[str] = None
    tags: Optional[ListAppendOrSet] = None


class BuildUpdateResponse(BaseModel):
    build: StoredBuild


@builds_api.put("/{build_id}/update")
def update_build(
    request: Request, build_id: str, update: BuildUpdateRequest
) -> BuildUpdateResponse:
    read_resp = read_build(build_id)
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
