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

from typing import Annotated, Any, List, Optional, Tuple, Union, cast

from fastapi import FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from gbcommon.uri.hf import HfType, HfURI
from gbcommon.uri.lh import LhURI
from gbcommon.uri.uri import URI
from gbserver.api.utils import (
    ListAppendOrSet,
    apply_tag_update,
    confirm_space_write_access,
    get_row_filter,
    is_super_admin,
    split_tags,
)
from gbserver.lineage.jobstats import get_lineage_store
from gbserver.storage.artifact_registration import (
    ArtifactRegistration,
    ArtifactRegistrationStatus,
)
from gbserver.storage.artifact_registry import (
    ChecksumConflictException,
    IArtifactRegistry,
)
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.types.artifact import ArtifactType
from gbserver.types.constants import (
    ENV_URI_SCHEME,
    GB_ENVIRONMENT,
    LAKEHOUSE_ENVIRONMENT,
)

artifacts_api = FastAPI()


class RegisterArtifactResponse(BaseModel):
    registered: ArtifactRegistration


class ArtifactDestination(BaseModel):
    namespace: str
    type: str
    table_name: str


class ArtifactRegistrationRequest(BaseModel):
    location: ArtifactDestination
    artifact: ArtifactRegistration


# @artifacts_api.post("/lh")
# def register_lakehouse_artifact(
#     request: ArtifactRegistrationRequest,
# ) -> RegisterArtifactResponse:
#     """Deprecated in favor of lh/model, lh/table and lh/dataset endpoints"""
#     location = request.location
#     uri = LhURI._get_uri_from_name(uri_suffix=location.table_name, lh_type=LhType(location.type.lower()), namespace=location.namespace)
#     request.artifact.uri = uri
#     return register_artifact(request.artifact)


class BaseArtifactRequest(BaseModel):
    space_name: str
    username: str
    namespace: str
    table_name: str
    name: str = ""
    """ Optional and will be set to tablename if not set"""
    lh_env: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    certified_no_restrictions: bool = False
    origin_uris: Optional[List[str]] = None
    description: str = ""
    checksum: Optional[str] = None
    status: str = str(ArtifactRegistrationStatus.SUCCESS)


def _get_artifact_registrations(
    artifact_uris: List[str], space_name: str
) -> List[ArtifactRegistration]:
    """Get the list of artifact registrations with the given uris.

    Args:
        artifact_uris (list[str]): list of registered artifact uris

    Raises:
        HTTPException: If any of the given uris were not found.

    Returns:
        _type_: list of ArtifactRegistration
    """
    if artifact_uris is None or len(artifact_uris) == 0:
        return []
    artifact_storage: IArtifactRegistry = get_admin_storage().artifact_registry
    # assert isinstance(artifact_storage,LhArtifactRegistry)
    artifacts: List[ArtifactRegistration] = []
    for artifact_uri in artifact_uris:
        artifact = artifact_storage.get_by_uri(uri=artifact_uri, space_name=space_name)
        assert isinstance(
            artifact, (type(None), ArtifactRegistration)
        ), f"invalid artifact: {artifact}"
        if artifact is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No artifact registration for the uri: '{artifact_uri}' in the space '{space_name}'",
            )
        artifacts.append(artifact)
    return artifacts


def __get_registerable_uris(uris: Optional[list[str]]) -> Tuple[list[str], list[str]]:
    if uris is None:
        return [], []
    non_reg_uris = []
    reg_uris = []
    for item_uri in uris:
        # Ignore origin uris that are from the local file system (e.g on BlueVela or other perhaps)
        if item_uri.startswith(ENV_URI_SCHEME + "://"):
            non_reg_uris.append(item_uri)
        else:
            reg_uris.append(item_uri)
    return reg_uris, non_reg_uris


def _create_and_register_artifact(
    request: Request,
    artifact_request: BaseArtifactRequest,
    uri: str,
    type: ArtifactType,
) -> RegisterArtifactResponse:
    if artifact_request.name == "":
        artifact_request.name = getattr(artifact_request, "table_name", "")

    art_status = _convert_status_str(artifact_request.status)

    artifact = ArtifactRegistration(
        uri=uri,
        space_name=artifact_request.space_name,
        username=artifact_request.username,
        name=artifact_request.name,
        tags=artifact_request.tags,
        type=type,
        origin_uris=artifact_request.origin_uris,
        certified_no_restrictions=artifact_request.certified_no_restrictions,
        description=artifact_request.description,
        checksum=artifact_request.checksum or "",
        status=art_status,
    )

    # Get the set of origin uris that are registerable in the artifact register.  Not to include env:// uris from BlueVela, etc.
    reg_uris, non_reg_uris = __get_registerable_uris(artifact_request.origin_uris)
    if len(reg_uris) > 0:
        # When the artifact has a set of source data used to create it,
        #   1) the artifact need not be certified with no restrictions since it was created with artifacts
        #       that presumabily have no restrictions since they are already registered.
        #   2) create a JobStats to represent that lineage.
        input_artifacts = _get_artifact_registrations(
            artifact_uris=reg_uris,
            space_name=artifact_request.space_name,
        )
        if len(input_artifacts) > 0:
            jobstats_storage = get_lineage_store()
            jobstats_storage.add_jobstats_for_original_artifact(
                artifact, input_artifacts
            )
    elif len(reg_uris) == 0 and not artifact_request.certified_no_restrictions:
        # If no origins, then we don't know how the artifact was created and the user must certify that it has no restrictions.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="When registering an artifact, you must certify that it has no restrictions",
        )
    elif len(non_reg_uris) > 0 and not artifact_request.certified_no_restrictions:
        # Non-registered origin uris (e.g. env://...), then the user must specify that these have no restrictions.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="When registering an artifact, you must certify that it has no restrictions",
        )

    resp = register_artifact(request, artifact)

    return resp


class ArtifactTableRequest(BaseArtifactRequest):
    pass


@artifacts_api.post("/lh/table")
def register_lakehouse_table(
    request: Request,
    artifact_request: ArtifactTableRequest,
) -> RegisterArtifactResponse:
    """Register a whole Lakehouse table in the artifact registry"""
    lh_env = artifact_request.lh_env or LAKEHOUSE_ENVIRONMENT.lower()
    if GB_ENVIRONMENT == "PROD" and lh_env != "prod":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Non-production artifacts are not allowed!",
        )
    uri = LhURI.get_table_uri(
        table_name=artifact_request.table_name,
        namespace=artifact_request.namespace,
        lh_env=lh_env,
    )
    return _create_and_register_artifact(
        request, artifact_request, uri, ArtifactType.TABLE
    )


class ArtifactDatasetRequest(BaseArtifactRequest):
    dataset_name: str


@artifacts_api.post("/lh/dataset")
def register_lakehouse_dataset(
    request: Request,
    artifact_request: ArtifactDatasetRequest,
) -> RegisterArtifactResponse:
    lh_env = artifact_request.lh_env or LAKEHOUSE_ENVIRONMENT.lower()
    uri = LhURI.get_dataset_uri(
        namespace=artifact_request.namespace,
        table_name=artifact_request.table_name,
        dataset_name=artifact_request.dataset_name,
        lh_env=lh_env,
    )
    return _create_and_register_artifact(
        request, artifact_request, uri, ArtifactType.DATASET
    )


class ArtifactModelRequest(BaseArtifactRequest):
    model_label: str
    model_revision: str = ""


@artifacts_api.post("/lh/model")
def register_lakehouse_model(
    request: Request,
    artifact_request: ArtifactModelRequest,
) -> RegisterArtifactResponse:
    lh_env = artifact_request.lh_env or LAKEHOUSE_ENVIRONMENT.lower()
    uri = LhURI.get_model_uri(
        namespace=artifact_request.namespace,
        table_name=artifact_request.table_name,
        model_label=artifact_request.model_label,
        model_revision=artifact_request.model_revision,
        lh_env=lh_env,
    )
    return _create_and_register_artifact(
        request, artifact_request, uri, ArtifactType.MODEL
    )


class ArtifactFilesetlRequest(BaseArtifactRequest):
    fileset_label: str
    fileset_version: str = ""


@artifacts_api.post("/lh/fileset")
def register_lakehouse_fileset(
    request: Request,
    artifact_request: ArtifactFilesetlRequest,
) -> RegisterArtifactResponse:
    if not artifact_request.fileset_version:
        artifact_request.fileset_version = ""
    lh_env = artifact_request.lh_env or LAKEHOUSE_ENVIRONMENT.lower()
    uri = LhURI.get_fileset_uri(
        namespace=artifact_request.namespace,
        table_name=artifact_request.table_name,
        fileset_label=artifact_request.fileset_label,
        fileset_version=artifact_request.fileset_version,
        lh_env=lh_env,
    )
    return _create_and_register_artifact(
        request, artifact_request, uri, ArtifactType.FILESET
    )


def register_artifact(
    request: Request,
    artifact: ArtifactRegistration,
) -> RegisterArtifactResponse:
    new_artifact = artifact

    # Make sure we can't register non-prod uris in the PROD environment
    if GB_ENVIRONMENT == "PROD":
        uri_obj = URI.get_uri(new_artifact.uri)
        if not hasattr(uri_obj, "is_prod") or not uri_obj.is_prod():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Non-production artifacts are not allowed!",
            )

    # Protect system tags
    sys_tags, _ = split_tags(new_artifact.tags)
    if len(sys_tags) > 0 and not is_super_admin(request):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    # TODO: should we also make sure they have access to the space
    storage = get_admin_storage().artifact_registry

    try:
        storage.add(new_artifact)
    except ChecksumConflictException as exc:
        conflict_uri = exc.existing_artifact.uri
        decoded = decode_uri(conflict_uri).model_dump()
        decoded["uuid"] = exc.existing_artifact.uuid
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=decoded)
    except (
        ValueError
    ) as exc:  # Other duplication not from the DB, UUID or [uri,space_name] for example.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception:  # On URI conflicts
        artifact_id = new_artifact.uuid
        item = storage.get_by_uuid(artifact_id)
        if item is not None:  # Duplicate UUIDs, but see above
            # Now that artifact_storage.add() checks for duplicate uuids, I don't think we get here.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="There is already an artifact with that ID!",
            )
        else:  # For SQL implementation we can get here on URI conflict
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"There is already an artifact with the uri {artifact.uri} in space {artifact.space_name}!",
            )

    resp = RegisterArtifactResponse(registered=new_artifact)

    return resp


class ChangeArchiveResponse(BaseModel):
    artifact: ArtifactRegistration
    was_archived: bool


def set_archive_bit(artifact_id: str, is_archived: bool) -> ChangeArchiveResponse:
    storage = get_admin_storage().artifact_registry
    item = storage.get_by_uuid(artifact_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found!"
        )
    assert isinstance(item, ArtifactRegistration)
    was_archived = item.is_archived
    if was_archived != is_archived:
        item.is_archived = is_archived
        updated = storage.update_fields(item.uuid, {"is_archived": is_archived})
        if updated is not None:
            item = updated
    resp = ChangeArchiveResponse(was_archived=was_archived, artifact=item)
    return resp


@artifacts_api.put("/{artifact_id}/archive")
def archive_artifact(artifact_id: str) -> ChangeArchiveResponse:
    return set_archive_bit(artifact_id, True)


@artifacts_api.put("/{artifact_id}/unarchive")
def unarchive_artifact(artifact_id: str) -> ChangeArchiveResponse:
    return set_archive_bit(artifact_id, False)


class DecodedURIResponse(BaseModel):
    uri: str
    namespace: str
    table_name: str
    type: str
    model_label: Optional[str] = None
    model_revision: Optional[str] = None
    fileset_label: Optional[str] = None
    fileset_version: Optional[str] = None
    dataset_name: Optional[str] = None


class DecodedHfURIResponse(BaseModel):
    uri: str
    host: str
    owner: str
    repo: str
    revision: str
    hf_type: Optional[str] = None
    organization: Optional[str] = None
    resource_group_id: Optional[str] = None


def __get_lh_decoded_uri_response(lh_uri: LhURI) -> DecodedURIResponse:
    metadata = lh_uri.get_metadata()
    response = DecodedURIResponse(**metadata)
    return response


def __get_hf_decoded_uri_response(hf_uri: HfURI) -> DecodedHfURIResponse:
    metadata = hf_uri.get_metadata()
    if metadata.get("hf_type") is not None:
        metadata["hf_type"] = str(metadata["hf_type"])
    response = DecodedHfURIResponse(**metadata)
    return response


@artifacts_api.get("/decode")
def decode_uri(
    uri: Optional[str] = None, id: Optional[str] = None
) -> Union[DecodedURIResponse, DecodedHfURIResponse]:
    if uri is None and id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="One of uri or id must be provided!",
        )
    elif uri is not None and id is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only one of uri or id can be provided!",
        )
    elif id is not None:
        artifact_storage = get_admin_storage().artifact_registry
        artifact = artifact_storage.get_by_uuid(id)
        assert isinstance(
            artifact, ArtifactRegistration
        ), f"invalid artifact: {artifact}"
        if artifact is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Artifact with id {id} not found",
            )
        uri = artifact.uri
    assert uri is not None, "uri is None"
    uriobj = URI.get_uri(uri)
    if isinstance(uriobj, LhURI):
        return __get_lh_decoded_uri_response(uriobj)
    elif isinstance(uriobj, HfURI):
        return __get_hf_decoded_uri_response(uriobj)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Received unsupported uri {uri}",
        )


class GetArtifactResponse(BaseModel):
    artifact: ArtifactRegistration


class ListArtifactsResponse(BaseModel):
    artifacts: list[ArtifactRegistration]


# Good to be after other GETs, since it might match others otherwise.
@artifacts_api.get("/")
def list_artifacts(
    uri: str = "",
    username: str = "",
    build_id: str = "",
    space_name: str = "",
    is_archived: Optional[bool] = None,
    checksum: Optional[str] = None,
    tag: Annotated[
        list[str] | None, Query()
    ] = [],  # Specified as multiple tag=v1&tag=v2 in URI
) -> ListArtifactsResponse:

    row_filter = get_row_filter(
        uri=uri,
        username=username,
        created_by_build_id=build_id,
        space_name=space_name,
        is_archived=is_archived,
        checksum=checksum,
        tags=tag,
    )
    storage = get_admin_storage()
    items = cast(
        List[ArtifactRegistration], storage.artifact_registry.get_by_where(row_filter)
    )
    resp = ListArtifactsResponse(artifacts=items)
    return resp


@artifacts_api.get("/tags")
def list_artifact_tags(
    uri: str = "",
    username: str = "",
    build_id: str = "",
    space_name: str = "",
) -> List[str]:
    """Return the sort list of unique tag strings for the aartifacts that match the condition."""
    # In this version, it simply pulls all the artifacts and programatically takes a unique
    artifacts_response = list_artifacts(
        uri=uri, username=username, build_id=build_id, space_name=space_name
    )
    tags: set[str] = set()
    for artifact in artifacts_response.artifacts:
        if artifact.tags:
            tags.update(artifact.tags)
    unique_tags = list(tags)
    unique_tags.sort()
    return unique_tags


# Needs to be after /tags and /decode GET, since it will match others otherwise.
@artifacts_api.get("/{artifact_id}")
def read_artifact(artifact_id: str) -> GetArtifactResponse:
    storage = get_admin_storage().artifact_registry
    item = storage.get_by_uuid(artifact_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found!"
        )
    assert isinstance(item, ArtifactRegistration)
    resp = GetArtifactResponse(artifact=item)
    return resp


class ArtifactUpdateRequest(BaseModel):
    description: Optional[str] = None
    tags: Optional[ListAppendOrSet] = None
    status: Optional[str] = None


class ArtifactUpdateResponse(BaseModel):
    artifact: ArtifactRegistration


def _convert_status_str(status_str: str) -> ArtifactRegistrationStatus:
    try:
        art_status = ArtifactRegistrationStatus[status_str.upper()]
        return art_status
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not convert artifact status value {status_str}",
        )


@artifacts_api.put("/{artifact_id}/update")
def update_artifact(
    request: Request, artifact_id: str, update: ArtifactUpdateRequest
) -> ArtifactUpdateResponse:
    read_resp = read_artifact(artifact_id)
    if read_resp is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Can not find artifact with id {artifact_id}!",
        )

    assert isinstance(read_resp, GetArtifactResponse)
    artifact = read_resp.artifact
    assert isinstance(artifact, ArtifactRegistration)

    # Make sure the user (owner or admin) has access to the artifact
    confirm_space_write_access(
        request=request,
        username_on_target=artifact.username,
        space_name=artifact.space_name,
    )

    updates: dict[str, Any] = {}
    if update.status:
        updates["status"] = _convert_status_str(update.status)
    if update.description is not None:
        updates["description"] = update.description
    if update.tags:
        is_super = is_super_admin(request)
        apply_tag_update(artifact, update.tags, is_super)
        updates["tags"] = artifact.tags

    if len(updates) > 0:
        # Store the update
        storage = get_admin_storage().artifact_registry
        updated = storage.update_fields(artifact.uuid, updates)
        if updated is not None:
            artifact = updated

    resp = ArtifactUpdateResponse(artifact=artifact)
    return resp


# HuggingFace Registration Endpoints


class HFModelRegistrationRequest(BaseArtifactRequest):
    model_id: str = ""
    organization: str = ""
    revision: Optional[str] = None
    namespace: str = ""
    table_name: str = ""


class HFDatasetRegistrationRequest(BaseArtifactRequest):
    dataset_id: str = ""
    organization: str = ""
    revision: Optional[str] = None
    namespace: str = ""
    table_name: str = ""


@artifacts_api.post("/hf/model")
def register_hf_model(
    request: Request, artifact_request: HFModelRegistrationRequest
) -> RegisterArtifactResponse:
    """Register a HuggingFace model."""
    hf_uri = HfURI.from_parts(
        owner=artifact_request.organization,
        repo=artifact_request.model_id,
        hf_type=HfType.MODEL,
        revision=artifact_request.revision or "main",
    )
    uri_str = hf_uri.get_uristr(hf_uri)

    if artifact_request.name == "":
        artifact_request.name = artifact_request.model_id

    return _create_and_register_artifact(
        request, artifact_request, uri_str, ArtifactType.MODEL
    )


@artifacts_api.post("/hf/dataset")
def register_hf_dataset(
    request: Request,
    artifact_request: HFDatasetRegistrationRequest,
) -> RegisterArtifactResponse:
    """Register a HuggingFace dataset."""
    hf_uri = HfURI.from_parts(
        owner=artifact_request.organization,
        repo=artifact_request.dataset_id,
        hf_type=HfType.DATASET,
        revision=artifact_request.revision or "main",
    )
    uri_str = hf_uri.get_uristr(hf_uri)

    if artifact_request.name == "":
        artifact_request.name = artifact_request.dataset_id

    return _create_and_register_artifact(
        request, artifact_request, uri_str, ArtifactType.DATASET
    )


class ArtifactBucketRequest(BaseArtifactRequest):
    bucket_id: str = ""
    organization: str = ""
    revision: Optional[str] = None
    namespace: str = ""
    table_name: str = ""


@artifacts_api.post("/hf/bucket")
def register_hf_bucket(
    request: Request,
    artifact_request: ArtifactBucketRequest,
) -> RegisterArtifactResponse:
    """Register a HuggingFace bucket."""
    hf_uri = HfURI.from_parts(
        owner=artifact_request.organization,
        repo=artifact_request.bucket_id,
        hf_type=HfType.BUCKET,
        revision=artifact_request.revision or "main",
    )
    uri_str = hf_uri.get_uristr(hf_uri)

    if artifact_request.name == "":
        artifact_request.name = artifact_request.bucket_id

    return _create_and_register_artifact(
        request, artifact_request, uri_str, ArtifactType.BUCKET
    )
