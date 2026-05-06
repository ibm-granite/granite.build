import logging
from typing import Any, List, Optional

import requests
from fastapi import HTTPException
from requests.exceptions import ConnectionError

from gbcli.utils.gbconstants import (
    GBSERVER_SPACES_API,
    VPN_CONNECTION_ERROR_MESSAGE,
    is_standalone,
)

logger = logging.getLogger(__name__)


def gb_server_request(
    user_token: str,
    url: str,
    http_method: str,
    body,
    params,
    timeout: Optional[int] = None,
):
    """
    helper function to make calls to gbserver

    raises an http exception if 400 <= response.status_code < 600
    """

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {user_token}",
    }

    response = None

    match http_method:
        case "get":
            response = requests.get(
                url, headers=headers, json=body, params=params, timeout=timeout
            )
        case "post":
            response = requests.post(
                url, headers=headers, json=body, params=params, timeout=timeout
            )
        case "put":
            response = requests.put(
                url, headers=headers, json=body, params=params, timeout=timeout
            )
        case "delete":
            response = requests.delete(
                url, headers=headers, json=body, params=params, timeout=timeout
            )
        case "patch":
            response = requests.patch(
                url, headers=headers, json=body, params=params, timeout=timeout
            )
        case _:
            response = requests.get(
                url, headers=headers, json=body, params=params, timeout=timeout
            )

    if 400 <= response.status_code < 600:
        detail = (
            response.json().get("detail")
            if response.json().get("detail")
            else response.json().get("error", "")
        )
        raise HTTPException(status_code=response.status_code, detail=detail)

    data_obj = response.json()

    return data_obj


def get_server_version(user_token: str, gbserver_instance: str) -> Any:
    gbserver_root_api = f"{gbserver_instance}/api/v1"

    return gb_server_request(
        user_token=user_token,
        url=gbserver_root_api,
        http_method="get",
        body=None,
        params=None,
        timeout=10,
    )


def get_build_status(user_token: str, build_id: str, gbserver_api: str) -> Any:
    build_url = f"{gbserver_api}{build_id}/status"

    return gb_server_request(
        user_token=user_token,
        url=build_url,
        http_method="get",
        body=None,
        params=None,
    )


def get_artifact_body(
    namespace: str,
    table: str,
    artifact_name: str,
    username: str,
    space_name: str,
    type: str,
    label: str,
    revision: str,
    dataset_name: str,
    lh_env: str,
    version: str,
    tags: list[str],
    status: str,
    description: str,
    checksum: str,
    origin_uris: Optional[list[str]] = None,
    certified_no_restrictions: bool = False,
):
    if type == "table":
        body = {
            "space_name": space_name,
            "username": username,
            "namespace": namespace,
            "table_name": table,
            "description": description,
            "checksum": checksum,
            "name": artifact_name,
            "lh_env": lh_env,
            "tags": tags,
            "status": status,
            "origin_uris": origin_uris,
            "certified_no_restrictions": certified_no_restrictions,
        }

    elif type == "dataset":
        body = {
            "space_name": space_name,
            "username": username,
            "namespace": namespace,
            "table_name": table,
            "name": artifact_name,
            "lh_env": lh_env,
            "tags": tags,
            "status": status,
            "dataset_name": dataset_name if dataset_name else artifact_name,
            "description": description,
            "checksum": checksum,
            "origin_uris": origin_uris,
            "certified_no_restrictions": certified_no_restrictions,
        }

    elif type == "model":
        body = {
            "space_name": space_name,
            "username": username,
            "namespace": namespace,
            "table_name": table,
            "name": artifact_name,
            "lh_env": lh_env,
            "tags": tags,
            "status": status,
            "model_label": label,
            "model_revision": revision,
            "description": description,
            "checksum": checksum,
            "origin_uris": origin_uris,
            "certified_no_restrictions": certified_no_restrictions,
        }
    elif type == "fileset":
        body = {
            "space_name": space_name,
            "username": username,
            "namespace": namespace,
            "table_name": table,
            "name": artifact_name,
            "lh_env": lh_env,
            "tags": tags,
            "status": status,
            "fileset_label": label,
            "fileset_version": version,
            "description": description,
            "checksum": checksum,
            "origin_uris": origin_uris,
            "certified_no_restrictions": certified_no_restrictions,
        }
    return body


def get_build_status_with_targets_runs(
    user_token: str, build_id: str, gbserver_api: str
) -> Any:
    build_url = f"{gbserver_api}{build_id}/status2"

    return gb_server_request(
        user_token=user_token,
        url=build_url,
        http_method="get",
        body=None,
        params=None,
    )


def register_artifact(
    namespace: str,
    table: str,
    artifact_name: str,
    username: str,
    space_name: str,
    server_api: str,
    user_token: str,
    type: str,
    label: str,
    revision: str,
    dataset_name: str,
    lh_env: str,
    version: str,
    description: str,
    checksum: str,
    tags: list[str],
    status: str,
    origin_uris: Optional[list[str]] = None,
    certified_no_restrictions: bool = False,
):
    if server_api and user_token:
        push_url = f"{server_api}lh/{type}"

        new_artifact = get_artifact_body(
            namespace=namespace,
            table=table,
            artifact_name=artifact_name,
            username=username,
            space_name=space_name,
            type=type,
            label=label,
            revision=revision,
            dataset_name=dataset_name,
            lh_env=lh_env,
            version=version,
            tags=tags,
            status=status,
            description=description,
            checksum=checksum,
            origin_uris=origin_uris,
            certified_no_restrictions=certified_no_restrictions,
        )

        try:
            # response = gbserver_post(user_token, push_url, new_artifact)
            response = gb_server_request(
                user_token=user_token,
                url=push_url,
                http_method="post",
                body=new_artifact,
                params=None,
            )
        except HTTPException as e:
            if e.status_code == 409:
                raise ValueError(f"{e.detail}")
            else:
                if "UniqueViolation" in e.detail:
                    raise ValueError(
                        f"Artifact may already exist; registration was not completed."
                    )
                raise ValueError(
                    f"gbserver returned '{e.status_code} {e.detail}' for url: {push_url}."
                )

        if response.get("registered") != None:
            obj = response.get("registered")
            return {"uuid": obj.get("uuid"), "uri": obj.get("uri")}
        else:
            raise Exception(
                f"There was a problem registering the artifact type: {type}."
            )


def get_builds(
    token: str,
    server_build_url: str,
    username: Optional[str] = None,
    source_uri: Optional[str] = None,
    space_name: Optional[str] = None,
    tag: Optional[list[str]] = None,
    status: Optional[list[str]] = None,
    sort: Optional[list[str]] = None,
    page_index: Optional[int] = None,
    page_size: Optional[int] = None,
) -> Any:
    params = {}
    if username:
        params["username"] = username
    if source_uri:
        params["source_uri"] = source_uri
    if space_name:
        params["space_name"] = space_name
    if tag:
        params["tag"] = tag
    if status:
        params["status"] = [s.upper() for s in status]
    if sort:
        params["sort"] = sort
    if page_index != None:
        params["page_index"] = page_index
    if page_size != None:
        params["page_size"] = page_size
    return gb_server_request(
        user_token=token,
        url=server_build_url,
        http_method="get",
        body=None,
        params=params,
    )


def get_build(build_id: str, token: str, gbserver_api: str) -> Any:
    build_details_url = f"{gbserver_api}{build_id}"
    return gb_server_request(
        user_token=token,
        url=build_details_url,
        http_method="get",
        body=None,
        params=None,
    )


def get_build_lineage(build_id: str, token: str, gbserver_api: str) -> Any:
    build_lineage_url = f"{gbserver_api}build/{build_id}"

    a= gb_server_request(
        user_token=token,
        url=build_lineage_url,
        http_method="get",
        body=None,
        params=None,
    )
    return a


def get_build_events(build_id: str, token: str, gbserver_api: str) -> Any:
    build_events_url = f"{gbserver_api}{build_id}/events"

    return gb_server_request(
        user_token=token,
        url=build_events_url,
        http_method="get",
        body=None,
        params=None,
    )


def cancel_build(build_id: str, token: str, gbserver_api: str) -> Any:
    delete_build_url = f"{gbserver_api}{build_id}"

    return gb_server_request(
        user_token=token,
        url=delete_build_url,
        http_method="delete",
        body=None,
        params=None,
    )


def get_artifacts(
    token: str,
    server_artifact_url: str,
    username: str,
    build_id: str,
    space_name: str,
    checksum: str = None,
    tag: list[str] = None,
):
    params = {
        "username": username,
        "build_id": build_id,
        "space_name": space_name,
        "checksum": checksum,
        "tag": tag,
    }

    return gb_server_request(
        user_token=token,
        url=server_artifact_url,
        http_method="get",
        body=None,
        params=params,
    )


def get_remote_spaces(token: str, callback=None):
    """
    returns spaces available for user from gbserver
    """
    if callback:
        callback(callback_event="fetching_spaces", callback_args={"steps": 1})

    url = f"{GBSERVER_SPACES_API}spaces_for_user"
    response = make_gbserver_call(lambda: gbserver_get(token, url), callback)

    if not response or "spaces" not in response:
        raise Exception("Error getting spaces from GBServer.")

    return response["spaces"]


def get_space_members(token: str, space_name: str) -> Any:
    url = f"{GBSERVER_SPACES_API}{space_name}/members"
    return gb_server_request(
        user_token=token,
        url=url,
        http_method="get",
        body=None,
        params=None,
    )


def add_space_member(token: str, space_name: str, username: str, role: str) -> Any:
    url = f"{GBSERVER_SPACES_API}{space_name}/members"
    return gb_server_request(
        user_token=token,
        url=url,
        http_method="post",
        body={"username": username, "role": role},
        params=None,
    )


def update_space_member(token: str, space_name: str, username: str, role: str) -> Any:
    url = f"{GBSERVER_SPACES_API}{space_name}/members/{username}"
    return gb_server_request(
        user_token=token,
        url=url,
        http_method="patch",
        body={"role": role},
        params=None,
    )


def delete_space_member(token: str, space_name: str, username: str) -> Any:
    url = f"{GBSERVER_SPACES_API}{space_name}/members/{username}"
    return gb_server_request(
        user_token=token,
        url=url,
        http_method="delete",
        body=None,
        params=None,
    )


def get_secrets(
    token: str,
    server_build_url: str,
    personal: bool,
    space_name: str,
    secret_name: Optional[str] = None,
) -> Any:
    secret_scope = "user_secrets" if personal else f"space_secrets/{space_name}"
    if secret_name:
        secrets_url = f"{server_build_url}{secret_scope}/{secret_name}"
    else:
        secrets_url = f"{server_build_url}{secret_scope}"

    return gb_server_request(
        user_token=token,
        url=secrets_url,
        http_method="get",
        body=None,
        params=None,
    )


def create_space_secret(
    token: str,
    server_build_url: str,
    personal: bool,
    space_name: str,
    secret_name: str,
    secret_value: str,
) -> Any:
    secret_scope = "user_secrets" if personal else f"space_secrets/{space_name}"
    secrets_url = f"{server_build_url}{secret_scope}"

    secret_obj = {
        "secret_name": secret_name,
        "secret_value": secret_value,
        "encoding": "base64",
    }

    return gb_server_request(
        user_token=token,
        url=secrets_url,
        http_method="post",
        body=secret_obj,
        params=None,
    )


def update_space_secret(
    token: str,
    server_build_url: str,
    personal: bool,
    space_name: str,
    secret_name: str,
    secret_value: str,
) -> Any:
    secret_scope = "user_secrets" if personal else f"space_secrets/{space_name}"
    secrets_url = f"{server_build_url}{secret_scope}/{secret_name}"

    secret_obj = {
        "secret_value": secret_value,
        "encoding": "base64",
    }

    return gb_server_request(
        user_token=token,
        url=secrets_url,
        http_method="put",
        body=secret_obj,
        params=None,
    )


def delete_space_secret(
    token: str,
    server_build_url: str,
    personal: bool,
    space_name: str,
    secret_name: Optional[str] = None,
) -> Any:
    secret_scope = "user_secrets" if personal else f"space_secrets/{space_name}"
    secrets_url = f"{server_build_url}{secret_scope}/{secret_name}"

    return gb_server_request(
        user_token=token,
        url=secrets_url,
        http_method="delete",
        body=None,
        params=None,
    )


def gbserver_put(token: str, url: str, payload: Any):
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    response = requests.put(url, headers=headers, json=payload)
    if 400 <= response.status_code < 600:
        raise HTTPException(
            status_code=response.status_code, detail=response.json().get("detail", "")
        )

    data_obj = response.json()

    return data_obj


def gbserver_post(token: str, url: str, payload: Any):
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    response = requests.post(url, headers=headers, json=payload)
    if 400 <= response.status_code < 600:
        raise HTTPException(
            status_code=response.status_code, detail=response.json().get("detail", "")
        )

    data_obj = response.json()

    return data_obj


def gbserver_get(token: str, url: str):
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    response = requests.get(url, headers=headers)

    if 400 <= response.status_code < 600:
        raise HTTPException(
            status_code=response.status_code, detail=response.json().get("detail", "")
        )

    data_obj = response.json()

    return data_obj


def archive_artifact(token: str, artifact_uuid: str, server_api: str):
    url = f"{server_api}{artifact_uuid}/archive"

    return gb_server_request(
        user_token=token,
        url=url,
        http_method="put",
        body=None,
        params=None,
    )


def unarchive_artifact(token: str, artifact_uuid: str, server_api: str):

    url = f"{server_api}{artifact_uuid}/unarchive"

    return gb_server_request(
        user_token=token,
        url=url,
        http_method="put",
        body=None,
        params=None,
    )


def validate_build(
    token: str,
    gbserver_api: str,
    build_archive: str,
    space_name: Optional[str] = None,
    username: Optional[str] = None,
    validation_type: str = "static",
) -> Any:
    """'validation_type' can be 'static' or 'dynamic'"""
    validate_url = f"{gbserver_api}validate"

    body = {
        "build_archive": build_archive,
        "space_name": space_name,
        "username": username,
        "validation_type": validation_type,
    }

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }

    response = requests.post(url=validate_url, headers=headers, json=body)
    # if response.status_code == 400:
    #     raise HTTPException(
    #         status_code=response.status_code, detail=response.json().get("errors", "")
    #     )
    # elif 400 < response.status_code < 600:
    #     raise HTTPException(
    #         status_code=response.status_code, detail=response.json().get("detail", "")
    #     )

    if response.status_code >= 400 and response.status_code != 422:
        response.raise_for_status()

    resp = response.json()

    return resp


def submit_build(
    token: str,
    gbserver_api: str,
    build_name: str,
    build_archive: str,
    space_name: Optional[str] = None,
    username: Optional[str] = None,
    targets: Optional[List[str]] = [],
    tags: Optional[list[str]] = [],
    description: Optional[str] = None,
) -> Any:

    body = {
        "name": build_name,
        "build_archive": build_archive,
        "space_name": space_name,
        "username": username,
        "targets": targets,
        "tags": tags,
        "description": description,
    }

    return gb_server_request(
        user_token=token,
        url=gbserver_api,
        http_method="post",
        body=body,
        params=None,
    )


def make_gbserver_call(gbserver_call, callback=None, final_command=None):
    try:
        result = gbserver_call()
    except HTTPException as e:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "steps": 1,
                    "reason": f"gbserver returned '{e.status_code} {e.detail}'.",
                },
            )
        return None
    except ConnectionError:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "steps": 1,
                    "reason": f"gbserver connection error. {VPN_CONNECTION_ERROR_MESSAGE}",
                },
            )
        return None
    finally:
        if final_command is not None:
            final_command()
    return result


def update_artifact_gserver(
    artifact_id: str,
    server_api: str,
    user_token: str,
    description: str = None,
    tags: Optional[list[str]] = None,
    status: str = None,
    append: bool = False,
):
    if server_api and user_token:
        put_url = f"{server_api}{artifact_id}/update"
        try:

            body = {}

            if description is not None:
                body["description"] = description

            # Add tags if not None (allow empty list to clear tags)
            if tags is not None:
                body["tags"] = {"append" if append else "set": tags}

            if status:
                body["status"] = status

            if not body:
                raise Exception(f"There was not values to update.")

            response = gbserver_put(user_token, put_url, body)
            return response["artifact"]

        except HTTPException as e:
            if e.status_code == 409:
                raise ValueError(e.detail)
            else:

                raise ValueError(
                    f"gbserver returned '{e.status_code} {e.detail}' for url: {put_url}."
                )


def update_build_gserver(
    build_id: str,
    server_api: str,
    user_token: str,
    description: str = None,
    tags: Optional[list[str]] = None,
    append: bool = False,
):
    if server_api and user_token:
        put_url = f"{server_api}{build_id}/update"
        try:

            body = {}

            if description is not None:
                body["description"] = description

            # Add tags if not None (allow empty list to clear tags)
            if tags is not None:
                body["tags"] = {"append" if append else "set": tags}

            if not body:
                raise Exception(f"There was not values to update.")

            response = gbserver_put(user_token, put_url, body)
            return response["build"]

        except HTTPException as e:
            if e.status_code == 409:
                raise ValueError(e.detail)
            else:

                raise ValueError(
                    f"gbserver returned '{e.status_code} {e.detail}' for url: {put_url}."
                )


def get_builds_count(
    token: str,
    server_build_url: str,
    username: Optional[str] = None,
    source_uri: Optional[str] = None,
    space_name: Optional[str] = None,
    tag: Optional[list[str]] = None,
    status: Optional[list[str]] = None,
) -> Any:
    count_url = f"{server_build_url}count"
    params = {}
    if username:
        params["username"] = username
    if source_uri:
        params["source_uri"] = source_uri
    if space_name:
        params["space_name"] = space_name
    if tag:
        params["tag"] = tag
    if status:
        params["status"] = [s.upper() for s in status]

    return gb_server_request(
        user_token=token,
        url=count_url,
        http_method="get",
        body=None,
        params=params,
    )
