import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import numpy as np
import yaml

from gbcli.services.service_build import get_build_id_from_url
from gbcli.utils.gbconstants import (
    GB_DMF_LOADER_SIZE_LIMIT,
    GB_DMF_USE_CLASSIC_LOADER,
    GBSERVER_ARTIFACT_API,
    HF_ORGANIZATION_DEFAULT,
    HF_REVISION_DEFAULT,
    LAKEHOUSE_FILESET_SHARED_TABLE_NAME,
    LAKEHOUSE_FILESET_TABLE_NAME,
    LAKEHOUSE_MODEL_SHARED_TABLE,
    LAKEHOUSE_MODEL_TABLE,
    LAKEHOUSE_NAMESPACE,
    REVISION_DEFAULT,
    SPACE_DEFAULT_NAME,
    USER_NOT_LOGGED_IN_ERROR_MESSAGE,
    gb_environment_config,
)
from gbcli.utils.gbserver import (
    archive_artifact,
    gb_server_request,
    gbserver_get,
    get_artifacts,
    make_gbserver_call,
    register_artifact,
    unarchive_artifact,
    update_artifact_gserver,
)
from gbcli.utils.gh_auth import get_user
from gbcli.utils.hf_registry import HFRegistry
from gbcli.utils.lh_auth import getLH
from gbcli.utils.lh_dataset import dataset_info_lh
from gbcli.utils.lh_fileset import checkFileset, createFileset, pullFileset
from gbcli.utils.lh_model import copyModel, createModel, getModel, pullModel
from gbcli.utils.lh_table import (
    convert_to_df,
    createTableDataset,
    createTableFromFile,
    hasNullValues,
    preprocess_df,
    table_lh,
)
from gbcli.utils.spaceutil import resolve_space, user_is_space_admin
from gbcli.utils.utils import (
    find_duplicates,
    format_artifact_tags,
    parse_artifact_identifier,
    read_lines,
    remove_suffix,
)

if TYPE_CHECKING:
    from lakehouse.core import CopyAssetStatus


class ArtifactCopyResult(TypedDict):
    copy_response: Any  # CopyAssetStatus at runtime when lakehouse is installed
    target_table: str


class ArtifactURIError(Exception):
    """Custom exception for artifact uri fetch errors."""

    pass


def upload_to_lh(
    github_token: str,
    lh_token: str,
    path_name: str,
    artifact_name: str,
    type: str,
    label: str,
    size: str,
    variant: str,
    model_type: str,
    version: str,
    space: Optional[str],
    table_name: Optional[str] = None,
    namespace: Optional[str] = None,
    callback=None,
):
    lh = getLH(lh_token)

    global_space = resolve_space(github_token, space, callback=callback)
    namespace = (
        global_space.get("lakehouse_namespace") if namespace is None else namespace
    )
    if namespace == None:
        raise Exception(
            f"Error: Lakehouse namespace of space='{space}' does not exist. Please check the 'llmb space list --all' or check if it is a valid space name and try again."
        )
    public = global_space.get("name") == "public"

    if type == "model":
        return upload_model_lh(
            lh=lh,
            path_name=path_name,
            model_label=label if label is not None else artifact_name,
            size=size,
            variant=variant,
            model_type=model_type,
            namespace=namespace,
            table_name=(
                LAKEHOUSE_MODEL_SHARED_TABLE if public else LAKEHOUSE_MODEL_TABLE
            ),
            revision=REVISION_DEFAULT,
            disable_aspera=GB_DMF_USE_CLASSIC_LOADER,
        )

    elif type == "dataset" or type == "table":
        return upload_file_lh(
            lh=lh,
            path_name=path_name,
            type=type,
            namespace=namespace,
            table_name=table_name,
            public=public,
            callback=callback,
        )
    elif type == "fileset":
        return upload_fileset_lh(
            lh=lh,
            path_name=path_name,
            file_label=label if label is not None else artifact_name,
            version=version,
            namespace=namespace,
            table_name=(
                LAKEHOUSE_FILESET_SHARED_TABLE_NAME
                if public
                else LAKEHOUSE_FILESET_TABLE_NAME
            ),
            disable_aspera=GB_DMF_USE_CLASSIC_LOADER,
        )


def upload_file_lh(
    lh,
    path_name: str,
    type: str,
    namespace: str,
    table_name: Optional[str] = None,
    public: bool = False,
    callback=None,
):
    file_path = Path(path_name)
    # Check if file exists
    if file_path.exists() and file_path.is_file():
        # Get file extension
        file_extension = file_path.suffix

        if file_extension in [".jsonl", ".csv"]:
            table = createTableFromFile(
                lh=lh,
                filepath=file_path,
                namespace=namespace,
                table_name=table_name,
                public=public,
            )
        else:
            file_size = os.path.getsize(file_path)
            if file_size > GB_DMF_LOADER_SIZE_LIMIT:
                raise Exception(
                    f"Error: use .jsonl or .csv to upload table contents larger than {round(GB_DMF_LOADER_SIZE_LIMIT / (1024 ** 3), 2)} GB. Other formats require more memory and are slower."
                )

            df = file_to_dataframe(path_name=file_path, callback=callback)
            table = createTableDataset(lh, df, namespace, table_name, type, public)

        return table

    else:
        raise Exception(
            f"Error: The file '{file_path.absolute().as_posix()}' does not exist. Please check the path and try again."
        )


def file_to_dataframe(path_name: str, callback=None):
    file_path = Path(path_name)
    # Check if file exists
    if file_path.exists() and file_path.is_file():
        # Get file extension
        file_extension = file_path.suffix
        df = convert_to_df(file_path, file_extension)
        df.columns = df.columns.str.replace(r"[. ]", "_", regex=True).str.lower()
        if hasNullValues(df):
            reason = f"⚠️ Warning: The data contains null values that must be removed before loading. Auto-correction will be applied, which may result in unexpected changes."
            if callback is not None:
                callback(
                    callback_event="remove-empty-values",
                    callback_args={"steps": 1, "reason": reason},
                )
                df = preprocess_df(df)

        return df

    else:
        raise Exception(
            f"Error: The file '{file_path.absolute().as_posix()}' does not exist. Please check the path and try again."
        )


def upload_model_lh(
    lh,
    path_name: str,
    model_label: str,
    size: str,
    variant: str,
    model_type: str,
    namespace: str,
    table_name: str = LAKEHOUSE_MODEL_SHARED_TABLE,
    revision: str = REVISION_DEFAULT,
    disable_aspera: bool = GB_DMF_USE_CLASSIC_LOADER,
):
    file_path = Path(path_name)

    # Check if file exists
    if file_path.exists():
        model_team = createModel(
            lh=lh,
            path_name=file_path,
            namespace=namespace,
            table_name=table_name,
            model_label=model_label,
            size=size,
            variant=variant,
            model_type=model_type,
            revision=revision,
            disable_aspera=disable_aspera,
        )
        return model_team

    else:
        raise Exception(
            f"Error: The directory '{path_name}' does not exist. Please check the path and try again."
        )


def upload_fileset_lh(
    lh,
    path_name: str,
    file_label: str,
    version: str,
    namespace: str,
    table_name: str,
    disable_aspera: bool = GB_DMF_USE_CLASSIC_LOADER,
):
    local_path = Path(path_name)

    # Check if path exists
    if local_path.exists():
        fileset = createFileset(
            lh=lh,
            path_name=local_path.resolve().as_posix(),
            namespace=namespace,
            table_name=table_name,
            file_label=file_label,
            version=version,
            disable_aspera=disable_aspera,
        )

        return fileset

    else:
        raise Exception(
            f"Error: The path '{path_name}' does not exist. Please check the path and try again."
        )


def check_fileset_lh(
    github_token: str,
    lh_token: str,
    namespace: str,
    fileset_name: str,
    table_name: str,
    version: str,
    space: str = SPACE_DEFAULT_NAME,
    callback=None,
):
    lh = getLH(lh_token)
    if namespace == None:
        global_space = resolve_space(github_token, space, callback=callback)
        namespace = global_space.get("lakehouse_namespace")

    return checkFileset(lh, namespace, fileset_name, table_name, version)


def upload_to_hf(
    hf_token: str,
    path_name: str,
    artifact_name: str,
    type: str,
    hf_organization: Optional[str] = None,
    resource_group_id: Optional[str] = None,
    private: bool = False,
    callback=None,
):
    org = hf_organization or HF_ORGANIZATION_DEFAULT
    if not org:
        raise Exception(
            "Error: No HuggingFace organization configured. Use --hf-organization or set hf_organization in the environment config."
        )
    repo_id = f"{org}/{artifact_name}"
    artifact_type_map = {
        "model": "model",
        "fileset": "bucket",
        "table": "dataset",
        "bucket": "bucket",
    }
    hf_type = artifact_type_map.get(type, "dataset")
    registry = HFRegistry(
        hf_token=hf_token,
        resource_group_id=resource_group_id,
        organization=org,
    )
    return registry.upload_artifact(
        local_path=path_name,
        repo_id=repo_id,
        artifact_type=hf_type,
        private=private,
    )


def get_artifact(github_token: str, artifact_id: str, callback=None):
    try:
        server_api = GBSERVER_ARTIFACT_API
        url = f"{server_api}{artifact_id}"
        response = make_gbserver_call(
            lambda: gbserver_get(github_token, url),
            callback,
        )
        return response["artifact"]

    except Exception as e:
        raise Exception(f"Error downloading file: {e}")


def get_artifact_uri(github_token: str, artifact_uri: str, callback=None):
    try:
        server_api = GBSERVER_ARTIFACT_API
        url = f"{server_api}?uri={artifact_uri}"
        response = make_gbserver_call(
            lambda: gbserver_get(github_token, url),
            callback,
        )
        return response["artifacts"][0]

    except Exception as e:
        if len(response["artifacts"]) != 1:
            raise ArtifactURIError(
                f"Error downloading file: URI has no matching artifacts."
            )
        raise Exception(f"Error downloading file: {e}")


def lookup_hf_resource_group_id(
    github_token: str,
    space_name: str,
    organization: str,
    callback=None,
) -> Optional[str]:
    """Resolve an HF Enterprise resource group id via gbserver."""
    if not space_name or not organization:
        return None
    url = f"{GBSERVER_ARTIFACT_API}hf/resource-group"
    try:
        response = gb_server_request(
            user_token=github_token,
            url=url,
            http_method="get",
            body=None,
            params={"space_name": space_name, "organization": organization},
        )
    except Exception:
        return None
    return response.get("resource_group_id") if response else None


def get_model_lh(
    github_token: str,
    lh_token: str,
    namespace: str,
    table_name: str,
    model_label: str,
    revision: str,
    space: str = SPACE_DEFAULT_NAME,
    callback=None,
):
    lh = getLH(lh_token)
    if namespace == None:
        global_space = resolve_space(github_token, space, callback=callback)
        namespace = global_space.get("lakehouse_namespace")

    return getModel(lh, namespace, table_name, model_label, revision)


def get_dataset_lh(
    github_token: str,
    lh_token: str,
    dataset_name: str,
    namespace: str,
    space: str = SPACE_DEFAULT_NAME,
    callback=None,
):
    lh = getLH(lh_token)
    if namespace == None:
        global_space = resolve_space(github_token, space, callback=callback)
        namespace = global_space.get("lakehouse_namespace")

    return dataset_info_lh(lh, dataset_name, namespace)


def get_table_lh(
    github_token: str,
    lh_token: str,
    namespace: str,
    table_name: str,
    space: str = SPACE_DEFAULT_NAME,
    callback=None,
):
    lh = getLH(lh_token)
    if namespace == None:
        global_space = resolve_space(github_token, space, callback=callback)
        namespace = global_space.get("lakehouse_namespace")

    return table_lh(lh, namespace, table_name)


def download_model_lh(
    github_token: str,
    lh_token: str,
    namespace: str,
    table_name: str,
    model_label: str,
    revision: str,
    directory: str,
    space: str = SPACE_DEFAULT_NAME,
    callback=None,
):
    lh = getLH(lh_token)
    if namespace == None:
        global_space = resolve_space(github_token, space, callback=callback)
        namespace = global_space.get("lakehouse_namespace")

    pullModel(
        lh,
        directory,
        namespace,
        model_label,
        table_name,
        revision,
        disable_aspera=GB_DMF_USE_CLASSIC_LOADER,
    )


def download_fileset_lh(
    github_token: str,
    lh_token: str,
    namespace: str,
    table_name: str,
    label: str,
    version: str,
    directory: str,
    space: str = SPACE_DEFAULT_NAME,
    callback=None,
):
    lh = getLH(lh_token)
    if namespace == None:
        global_space = resolve_space(github_token, space, callback=callback)
        namespace = global_space.get("lakehouse_namespace")

    pullFileset(
        lh,
        directory,
        namespace,
        label,
        table_name,
        version,
        disable_aspera=GB_DMF_USE_CLASSIC_LOADER,
    )


def download_table_lh(
    github_token: str,
    lh_token: str,
    namespace: str,
    table_name: str,
    directory: str,
    format: str,
    space: str = SPACE_DEFAULT_NAME,
    callback=None,
):
    lh = getLH(lh_token)
    if namespace == None:
        global_space = resolve_space(github_token, space, callback=callback)
        namespace = global_space.get("lakehouse_namespace")

    try:
        valid_format = ["json", "jsonl", "csv", "parquet"]
        if format in valid_format:
            # Create the directory if it does not exist

            table = table_lh(lh, namespace, table_name)

            metadata = table.metadata()

            if metadata.size_bytes > GB_DMF_LOADER_SIZE_LIMIT and format != "parquet":
                raise Exception(
                    f"\n❌ Error: use parquet format to pull table contents larger than {round(GB_DMF_LOADER_SIZE_LIMIT / (1024 ** 3), 2)} GB. Other formats require more memory and are slower."
                )

            output_path = f"{directory}/{table_name}.{format}"
            match format:
                case "parquet":
                    os.makedirs(f"{directory}/{table_name}", exist_ok=True)
                    table.pull(
                        table_dir=f"{directory}",
                        use_aspera=not GB_DMF_USE_CLASSIC_LOADER,
                        force_download=True,
                    )
                    return
                case "jsonl":
                    output = table.to_json()

                    if output and output_path:
                        # Parse the JSON string into a Python list
                        json_array = json.loads(output)
                        with open(output_path, "w", encoding="utf-8") as file:
                            for obj in json_array:
                                file.write(
                                    json.dumps(obj) + "\n"
                                )  # Convert dict to JSON string and write to file
                        return
                case "csv":
                    output = table.to_pandas().to_csv(
                        encoding="utf-8", index=False, header=True, quoting=2
                    )
                    if output and output_path:
                        with open(output_path, "w", encoding="utf-8") as file:
                            file.write(output)
                    return
                case _:
                    # json is default option
                    output = table.to_json()
                    if output and output_path:
                        with open(output_path, "w", encoding="utf-8") as file:
                            file.write(output)
                    return

        else:
            raise Exception(
                f"Error: Invalid format, please select one of the following formats: {str(valid_format)}"
            )
    except Exception as e:
        raise Exception(f"Error downloading file: {e}")


def download_hf_artifact(
    hf_token: str,
    repo_id: str,
    artifact_type: str,
    directory: str,
    revision: str = HF_REVISION_DEFAULT,
    callback=None,
):
    """Download a HuggingFace artifact (model, dataset, or space) to a local directory.

    This function uses the HFRegistry to download artifacts from HuggingFace Hub.
    Supports models, datasets, and spaces.

    Args:
        hf_token: HuggingFace API token for authentication.
        repo_id: Repository ID in format 'organization/repo-name'.
        artifact_type: Type of artifact ('model', 'dataset', or 'space').
        directory: Local directory where files will be downloaded.
        revision: Git revision/branch name (default: "main").
        callback: Optional callback for progress reporting.

    Returns:
        Dictionary with download information:
        - repo_id: Repository ID
        - artifact_type: Type of artifact
        - download_dir: Where files were downloaded
        - revision: Revision used
        - file_count: Number of files downloaded
        - total_size: Total size in bytes

    Raises:
        ValueError: If artifact_type is invalid or repo_id format is wrong.
        RuntimeError: If download fails.
        FileNotFoundError: If directory cannot be created.
    """
    # Validate artifact type
    valid_types = ["model", "dataset", "space", "bucket"]
    if artifact_type not in valid_types:
        raise ValueError(
            f"Invalid artifact_type: '{artifact_type}'. Must be one of {valid_types}"
        )

    try:
        # Initialize registry
        registry = HFRegistry(hf_token=hf_token)

        # Download the artifact
        result = registry.download_artifact(
            repo_id=repo_id,
            artifact_type=artifact_type,
            download_dir=directory,
            revision=revision,
        )

        return result

    except Exception as e:
        raise Exception(f"Error downloading HuggingFace artifact: {e}")


def check_artifact_existence(
    github_token: str,
    lh_token: str,
    type: str,
    space: Optional[str] = None,
    namespace: Optional[str] = None,
    table: Optional[str] = None,
    dataset: Optional[str] = None,
    label: Optional[str] = None,
    revision: Optional[str] = None,
    version: Optional[str] = None,
    callback=None,
):
    if space is None:
        global_space = resolve_space(github_token, space, callback)
        space = global_space.get("name")

    if namespace == None:
        global_space = resolve_space(github_token, space, callback)
        namespace = global_space.get("lakehouse_namespace")

    match type:
        case "model":
            model_obj = get_model_lh(
                lh_token,
                namespace,
                table,
                label,
                revision,
                space,
            )

            if not model_obj:
                return {
                    "success": False,
                    "error": f"\n❌ Model '{label}.{revision}' does not exist in {namespace}.{table}! Please verify artifact location, label, and revision.",
                }

        case "dataset":
            try:
                get_dataset_lh(
                    lh_token,
                    dataset,
                    namespace,
                    space,
                )
            except Exception:
                return {
                    "success": False,
                    "error": f"\n❌ Dataset '{dataset}' does not exist in {namespace}.{table}! Please verify artifact name and location.",
                }

        case "fileset":
            exists = check_fileset_lh(
                lh_token,
                namespace,
                label,
                table,
                version,
                space,
            )
            if not exists:
                return {
                    "success": False,
                    "error": f"\n❌ Fileset '{label}' version '{version}' does not exist in {namespace}.{table}! Please verify fileset.",
                }

        case "table":
            try:
                get_table_lh(
                    lh_token,
                    namespace,
                    table,
                    space,
                )
            except Exception:
                return {
                    "success": False,
                    "error": f"\n❌ Table '{namespace}.{table}' does not exist! Please verify artifact location.",
                }

    return {"success": True}


def register_artifact_hf(
    github_token: str,
    artifact_name: str,
    type: str,
    description: str,
    checksum: str,
    tags: list[str],
    status: str,
    revision: str,
    space_name: Optional[str],
    env: str,
    origin_uris: Optional[list[str]] = None,
    certified_no_restrictions: bool = False,
    hf_organization: Optional[str] = None,
    resource_group_id: Optional[str] = None,
    server_api: str = None,
):
    """Register an artifact to HuggingFace store via /hf/{type} endpoint."""
    if not server_api or not github_token:
        raise Exception("Missing server API or authentication token.")

    username = get_user(github_token).login
    push_url = f"{server_api}hf/{type}"

    if space_name is None:
        space_name = "public"

    if hf_organization is None:
        hf_organization = HF_ORGANIZATION_DEFAULT

    payload = {
        "space_name": space_name,
        "username": username,
        "organization": hf_organization,
        "name": artifact_name,
        "env": env,
        "revision": revision,
        "certified_no_restrictions": certified_no_restrictions,
        "origin_uris": origin_uris,
        "description": description,
        "checksum": checksum,
        "status": status,
        "tags": format_artifact_tags(tags),
    }

    if type == "model":
        payload["model_id"] = artifact_name
    elif type == "bucket":
        payload["bucket_id"] = artifact_name
    else:
        payload["dataset_id"] = artifact_name
    try:
        response = gb_server_request(
            user_token=github_token,
            url=push_url,
            http_method="post",
            body=payload,
            params=None,
        )
    except Exception as e:
        error_detail = str(e)
        if "409" in error_detail:
            raise ValueError(f"{error_detail}")
        else:
            if "UniqueViolation" in error_detail:
                raise ValueError(
                    f"Artifact may already exist; registration was not completed."
                )
            raise ValueError(
                f"gbserver returned error for url: {push_url}. {error_detail}"
            )

    if response.get("registered") != None:
        obj = response.get("registered")
        return {"uuid": obj.get("uuid"), "uri": obj.get("uri")}
    else:
        raise Exception(f"There was a problem registering the artifact type: {type}.")


def register_artifact_gbserver_hf(
    github_token: str,
    artifact_name: str,
    type: str,
    description: str,
    checksum: str,
    tags: list[str],
    status: str,
    revision: Optional[str] = None,
    env: Optional[str] = None,
    origin_uris: Optional[list[str]] = None,
    certified_no_restrictions: bool = False,
    hf_organization: Optional[str] = None,
    resource_group_id: Optional[str] = None,
):
    """Register artifact to HF store without space validation."""
    if not github_token:
        raise Exception(USER_NOT_LOGGED_IN_ERROR_MESSAGE)

    env = env if env else str(gb_environment_config()["lakehouse_environment"]).lower()

    if revision is None:
        revision = HF_REVISION_DEFAULT

    server_api = GBSERVER_ARTIFACT_API

    return register_artifact_hf(
        github_token=github_token,
        artifact_name=artifact_name,
        type=type,
        description=description,
        checksum=checksum,
        tags=tags,
        status=status,
        revision=revision,
        space_name="public",
        env=env,
        hf_organization=hf_organization,
        origin_uris=origin_uris,
        certified_no_restrictions=certified_no_restrictions,
        resource_group_id=resource_group_id,
        server_api=server_api,
    )


def register_artifact_gbserver(
    github_token: str,
    artifact_name: str,
    type: str,
    label: str,
    description: str,
    checksum: str,
    tags: list[str],
    status: str,
    revision: Optional[str] = None,
    version: Optional[str] = None,
    space: Optional[str] = None,
    namespace: Optional[str] = None,
    table: Optional[str] = None,
    dataset_name: Optional[str] = None,
    lh_env: Optional[str] = None,
    origin_uris: Optional[list[str]] = None,
    certified_no_restrictions: bool = False,
    callback=None,
):

    global_space = resolve_space(github_token, space, callback=callback)
    space_name = global_space.get("name")
    namespace_lh = global_space.get("lakehouse_namespace")
    if namespace == None:
        namespace = namespace_lh
    public = global_space.get("name") == "public"

    username = get_user(github_token).login
    if not username or not github_token:
        raise Exception(USER_NOT_LOGGED_IN_ERROR_MESSAGE)

    lh_env = (
        lh_env
        if lh_env
        else str(gb_environment_config()["lakehouse_environment"]).lower()
    )

    if namespace is None:
        namespace = LAKEHOUSE_NAMESPACE

    if table is None:
        if type == "fileset":
            table = (
                LAKEHOUSE_FILESET_SHARED_TABLE_NAME
                if public
                else LAKEHOUSE_FILESET_TABLE_NAME
            )
        elif type == "model":
            table = LAKEHOUSE_MODEL_SHARED_TABLE if public else LAKEHOUSE_MODEL_TABLE
        else:
            table = artifact_name

    if revision is None:
        revision = REVISION_DEFAULT

    server_api = GBSERVER_ARTIFACT_API

    artifact_obj = register_artifact(
        namespace,
        table,
        artifact_name,
        username,
        space_name,
        server_api,
        github_token,
        type,
        label,
        revision,
        dataset_name,
        lh_env,
        version,
        description,
        checksum,
        tags,
        status,
        origin_uris,
        certified_no_restrictions,
    )
    return artifact_obj


def artifact_lineage(token: str, artifact_name: str):
    from lakehouse import LakehouseLineage

    lh = getLH(token)
    lineage_df = LakehouseLineage(lh=lh).get_lineage(name=artifact_name)

    return (
        lineage_df.replace({np.nan: "None"})
        .sort_values(by="job_started_at")
        .to_dict("records")
    )


def artifact_lineage_hf(
    github_token: str, artifact_uri: str, artifact_type: Optional[str] = None
):
    from gbcli.utils.gbconstants import GBSERVER_LINEAGE_API

    url = f"{GBSERVER_LINEAGE_API}artifact"
    body: dict[str, Any] = {
        "artifact_url": artifact_uri,
        "max_depth": 10,
        "direction": "both",
    }
    if artifact_type:
        body["artifact_type"] = artifact_type
    response = gb_server_request(
        user_token=github_token,
        url=url,
        http_method="post",
        body=body,
        params=None,
    )
    return response


def artifact_list(
    github_token: str,
    list_all=False,
    show_archived=False,
    show_pending=False,
    build_id=None,
    id_format=None,
    space=None,
    all_spaces=False,
    username=None,
    checksum=None,
    tags=None,
    callback=None,
):
    if all_spaces:
        space_default = None
        space_default_name = None
        space_org = None
        space_name = None

    else:
        s = resolve_space(github_token, space, callback)
        space_default = s["git_repo_uri"] if s is not None else None
        space_default_name = s["name"] if s is not None else None
        if not space_default:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={
                        "reason": f"Space {space} not found in available spaces."
                    },
                )
            return None

        space_org, space_name = space_default.split("/")[-2:]
        space_name = remove_suffix(space_name, ".git")

    if username == "default":
        username = get_user(github_token).login

    if build_id is not None and id_format not in ["url", "uuid"]:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"❌ Build ID formatted incorrectly. Please try again with a valid build ID."
                },
            )
        return None

    if id_format == "url":
        build_id_from_url = get_build_id_from_url(github_token, build_id, callback)
        build_id = build_id_from_url[0]["uuid"]

    if callback is not None:
        callback(
            callback_event="listing_artifacts",
            callback_args={
                "steps": 10,
                "space": space_default_name if space_default_name else "public",
                "space_name": f"{space_org}/{space_name}",
                "build_id": build_id if id_format == "url" else None,
            },
        )

    gbserver_username = None if list_all else username
    gbserver_artifact_filter = gb_environment_config()["feature_flags"][
        "gbserver_artifact_filter"
    ]
    if gbserver_artifact_filter:
        # -u option with not username and not all, show all current user
        # -u option with username and not all, show specified user artifacts
        # no -u option given all current user artifacts + tag=sys-official

        user_artifacts = []
        gbserver_artifacts = []
        gbserver_username = (
            get_user(github_token).login
            if username == None and not list_all
            else username
        )

        user_artifacts = make_gbserver_call(
            lambda: get_artifacts(
                github_token,
                GBSERVER_ARTIFACT_API,
                gbserver_username,
                build_id,
                space_name=space_default_name,
                checksum=checksum,
                tag=tags,
            )["artifacts"],
            callback,
        )

        unique_artifacts = user_artifacts
        if not list_all and username == None:

            if tags is None:
                tags = []
            tags.append("sys-official")

            gbserver_artifacts = make_gbserver_call(
                lambda: get_artifacts(
                    github_token,
                    GBSERVER_ARTIFACT_API,
                    None,
                    build_id,
                    space_name=space_default_name,
                    checksum=checksum,
                    tag=tags,
                )["artifacts"],
                callback,
            )

        unique_by_id = {
            item["uuid"]: item for item in (user_artifacts + gbserver_artifacts)
        }
        unique_artifacts = list(unique_by_id.values())

        if callback is not None:
            callback(
                callback_event="listed_artifacts",
                callback_args={
                    "steps": 90,
                    "space": space_default_name if space_default_name else "public",
                    "space_name": f"{space_org}/{space_name}",
                    "build_id": build_id if id_format == "url" else None,
                },
            )
        unique_artifacts.sort(key=lambda x: x["created_at"])

        filtered_artifacts = format_artifact_tags(unique_artifacts)
    else:
        # ------FILTER in CLI This will be deleted with the feature flag------
        gbserver_artifacts = make_gbserver_call(
            lambda: get_artifacts(
                github_token,
                GBSERVER_ARTIFACT_API,
                None,
                build_id,
                space_name=space_default_name,
            )["artifacts"],
            callback,
        )

        if callback is not None:
            callback(
                callback_event="listed_artifacts",
                callback_args={
                    "steps": 90,
                    "space": space_default_name if space_default_name else "public",
                    "space_name": f"{space_org}/{space_name}",
                    "build_id": build_id if id_format == "url" else None,
                },
            )

        filtered_artifacts = gbserver_artifacts
        if username == None and not list_all:
            # format tags to json
            formatted_artifacts = format_artifact_tags(gbserver_artifacts)

            # only return current user artifacts and official ones
            filtered_artifacts = [
                a
                for a in formatted_artifacts
                if a["username"] == get_user(github_token).login
            ]
        if checksum:
            # filter by the checksum
            filtered_artifacts = [
                a for a in filtered_artifacts if a["checksum"] == checksum
            ]
        if tags:
            # filter the tags archived
            filtered_artifacts = [
                a for a in filtered_artifacts if all(tag in a["tags"] for tag in tags)
            ]

        # -------END FILTER in CLI This will be deleted with the feature flag------

    if not show_archived:
        # remove archived
        filtered_artifacts = [
            a for a in filtered_artifacts if a["is_archived"] is False
        ]

    if not show_pending:
        # remove pending
        filtered_artifacts = [
            a for a in filtered_artifacts if a.get("status", "success") == "success"
        ]

    return filtered_artifacts


def artifact_archive(
    github_token: str, artifact_uuid: str, archive: bool, callback=None
):
    username = get_user(github_token).login
    if not username or not github_token:
        raise Exception(USER_NOT_LOGGED_IN_ERROR_MESSAGE)

    server_api = GBSERVER_ARTIFACT_API

    def get_resp():
        if archive:
            return archive_artifact(github_token, artifact_uuid, server_api)
        else:
            return unarchive_artifact(github_token, artifact_uuid, server_api)

    resp = make_gbserver_call(
        lambda: get_resp(),
        callback,
    )
    return resp


def save_origin(file_path, artifact):
    # Write to .yaml
    artifact_id = artifact["uuid"]
    artifact_uri = artifact["uri"]
    data = {"artifact_id": artifact_id, "artifact_uri": artifact_uri}
    with open(file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def validate_origins(
    github_token: str,
    artifact_name: str,
    local_origins: list,
    origin: list,
    origin_list: str,
    certify_no_restrictions: bool,
    callback=None,
):
    origin_list_values = read_lines(origin_list) if origin_list != None else None
    origin_list_provided = origin_list_values and len(origin_list_values) > 0
    origin_provided = origin and len(origin) > 0
    local_origin_found = local_origins and len(local_origins) > 0
    origin_values = []
    artifacts = []
    uuid_to_uri = {}
    valid_uris = set()
    if origin_list_provided or origin_provided or local_origin_found:
        artifacts = artifact_list(
            github_token, show_pending=True, all_spaces=True, callback=callback
        )
        valid_uris = set(item["uri"] for item in artifacts)
        uuid_to_uri = {item["uuid"]: item["uri"] for item in artifacts}

    if certify_no_restrictions and origin_list_provided:
        raise Exception(
            f"❌Error: --certify-no-restrictions and --origin-list were provided. Only one can be provided. "
        )
    if certify_no_restrictions and origin_provided:
        raise Exception(
            f"❌Error: --certify-no-restrictions and --origin were provided. Only one can be provided. "
        )
    if origin_list_provided and origin_provided:
        raise Exception(
            f"❌Error: --origin and --origin-list were provided. Only one can be provided. "
        )

    if local_origin_found:
        artifact_id = local_origins[0]["artifact_id"]
        if origin_list_provided or origin_provided:
            origin_cmd = "--origin-list" if origin_list_provided else "--origin"
            if callback is not None:
                callback(
                    callback_event="origin-file",
                    callback_args={
                        "steps": 1,
                        "reason": f"A '.origin' file was found but values from '{origin_cmd}' will be used. ",
                    },
                )

            origin_values = (origin or []) + (origin_list_values or [])
        elif uuid_to_uri.get(artifact_id) != local_origins[0]["artifact_uri"]:
            raise Exception(
                f"❌Error: An 'origin' file was found. The provided artifact_id does not match the expected artifact_id retrieved from reverse lookup using the artifact_uri."
            )
        else:
            if callback is not None:
                callback(
                    callback_event="origin-file",
                    callback_args={
                        "steps": 1,
                        "reason": f"""🚨 New Requirement: To track artifacts from models with restricted use, you must provide the origin information.
📝 An "origin" file was found. Artifact '{artifact_id}' will be used as origin. """,
                    },
                )

            origin_values = [item["artifact_uri"] for item in local_origins]
    elif origin_list_provided:
        origin_values = origin_list_values
    elif origin_provided:
        origin_values = origin

    if len(origin_values) > 0:
        normalized_origins = []
        for value in origin_values:
            format = parse_artifact_identifier(value)
            uri = uuid_to_uri[value] if format == "uuid" else value
            normalized_origins.append(uri)

        for uri in normalized_origins:
            if uri not in valid_uris:
                raise ValueError(f"Invalid artifact URI: {uri}")

            duplicated = find_duplicates(normalized_origins)
            if len(duplicated) > 0:
                raise ValueError(
                    f"Multiple URIs point to the same item name '{artifact_name}'"
                )

        return normalized_origins


def artifact_copy(
    github_token: str,
    lh_token: str,
    source_namespace: str,
    source_table: str,
    space_to: str,
    model: str,
    revision: str,
    callback=None,
) -> ArtifactCopyResult:
    lh = getLH(lh_token)
    global_space = resolve_space(github_token, space_to, callback=callback)
    target_namespace = global_space.get("lakehouse_namespace")
    if target_namespace == None:
        raise Exception(
            f"Error: Lakehouse namespace of space='{space_to}' does not exist. Please check the 'llmb space list --all' or check if it is a valid space name and try again."
        )
    public = global_space.get("name") == "public"
    target_table = LAKEHOUSE_MODEL_SHARED_TABLE if public else LAKEHOUSE_MODEL_TABLE

    if target_namespace == source_namespace:
        raise Exception(
            f"Error:  The origin and target Lakehouse's namespaces cannot be the same."
        )
    copy_response = copyModel(
        lh,
        source_namespace,
        target_namespace,
        source_table,
        target_table,
        model,
        revision,
        None,
    )
    return {
        "copy_response": copy_response,
        "target_table": target_table,
    }


def update_artifact(
    github_token: str,
    artifact_id: str,
    tags: Optional[list[str]] = None,
    description: str = None,
    status: str = None,
    append: bool = False,
    isUpdate: bool = False,
    callback=None,
):
    # Validate that append is not used with empty tags
    if append and tags is not None and len(tags) == 0:
        raise ValueError("--append cannot be used with empty tags")

    username = get_user(github_token).login
    if not username or not github_token:
        raise Exception(USER_NOT_LOGGED_IN_ERROR_MESSAGE)

    # Check admin permission when updating status
    if status and isUpdate:
        # Fetch artifact to get space_name
        artifact = get_artifact(github_token, artifact_id, callback)
        if not artifact:
            if callback:
                callback(
                    callback_event="error",
                    callback_args={"reason": "Artifact not found"},
                )
            return None

        # Check if user is admin in the artifact's space
        if not user_is_space_admin(github_token, artifact.get("space_name"), callback):
            if callback:
                callback(
                    callback_event="error",
                    callback_args={
                        "reason": "Only an admin for this space can update artifact status"
                    },
                )
            return None

    gbserver_artifact = update_artifact_gserver(
        artifact_id=artifact_id,
        server_api=GBSERVER_ARTIFACT_API,
        user_token=github_token,
        description=description,
        tags=tags,
        status=status,
        append=append,
    )

    return gbserver_artifact
