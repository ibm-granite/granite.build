import difflib
import io
import itertools
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import zipfile
from base64 import b64decode, b64encode
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import yaml
from fastapi import HTTPException
from pydantic import BaseModel
from requests import HTTPError
from requests.exceptions import ConnectionError

from gbcli.utils.buildutil import (
    apply_parameters,
    get_yaml_diff,
    process_build_validation_response,
)
from gbcli.utils.cli_config import get_local_build_cache
from gbcli.utils.gbconstants import (
    ASSETS_REPO_URL,
    BUILD_FILENAME,
    BUILD_LOG_DEFAULT_QUERY_RANGE,
    BUILD_LOG_FOLLOW_SLEEP_TIME,
    BUILD_LOGALL_PAGE_SIZE,
    BUILD_PARAMETERS_FILE,
    BUILD_RUN_FILE,
    BUILD_RUN_YAML_KEY,
    CURRENT_BUILD_YAML_VERSION,
    CURRENT_BUILD_YAML_VERSION_KEY,
    GBSERVER_BUILD_API,
    GBSERVER_LINEAGE_API,
    SPACE_REPO_BUILD_FOLDER,
    TEMPLATES_REPO_FOLDER,
    USER_NOT_LOGGED_IN_ERROR_MESSAGE,
    VPN_CONNECTION_ERROR_MESSAGE,
    gb_environment,
    gb_environment_config,
    is_standalone,
)
from gbcli.utils.gbcredentials import GBCredentials
from gbcli.utils.gbserver import (
    cancel_build,
    get_build,
    get_build_events,
    get_build_lineage,
    get_build_status_with_targets_runs,
    get_builds,
    get_builds_count,
    make_gbserver_call,
    submit_build,
    update_build_gserver,
    validate_build,
)
from gbcli.utils.gh_auth import get_user
from gbcli.utils.gh_clone import (  # get_prs,
    clone_github_repo,
    delete_repo_subscription,
    get_pr_comments,
    get_repo_subscription,
    ignore_repo_subscription,
)
from gbcli.utils.lh_auth import getLH
from gbcli.utils.log_query import run_logquery
from gbcli.utils.spaceutil import (
    get_spaces,
    resolve_space,
    user_is_space_admin,
)
from gbcli.utils.utils import (
    change_timestamp_by_days,
    check_current_timestamp,
    convert_milliseconds_to_seconds,
    convert_seconds_to_milliseconds,
    create_if_not_dir_local_build_cache,
    generate_unique_id,
    get_current_epoch,
    remove_suffix,
)
from gbcommon.types.buildconfig import BuildConfig
from gbcommon.types.constants import BUILD_YAML_BASE_KEYS

logger = logging.getLogger(__name__)


class BuildResponse(BaseModel):
    """A single build returned by the build REST API."""

    build_archive: str
    # created_time: datetime
    # name: str
    # source_uri: str
    # space_name: str
    # status: str
    # targets: None
    # updated_time: datetime
    # username: str
    uuid: str


def ignore_git_global_config():
    os.environ["GIT_CONFIG_GLOBAL"] = ""
    os.environ["GIT_CONFIG_SYSTEM"] = ""


def build_init(
    github_token: str,
    build_name: str,
    filename: Optional[str] = "",
    space: Optional[str] = None,
    from_build: Optional[str] = None,
    from_template: Optional[str] = None,
    template_repo: Optional[str] = None,
    id_format: Optional[str] = None,
    callback=None,
) -> Tuple[bool, str]:

    # validate command options
    if from_build and from_template:
        raise Exception(
            f"❌ Error: --from-build and --from-template were provided. Only one can be provided. "
        )

    if not build_name and (not filename or filename == ""):
        raise Exception(
            f"❌ Error: Please specify a 'BUILD_NAME' or specify a filepath via -f option"
        )

    cache_path = get_local_build_cache()

    # check if local build folder already exists, fail gracefully
    if build_name and os.path.isdir(build_name):
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"Error: New build has not been created, the folder {build_name} already exists."
                },
            )
        return
    elif build_name and os.path.isfile(build_name):
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"Error: New build has not been created, the file with the same name {build_name} already exists."
                },
            )
        return

    if build_name:
        build_dir = Path(os.path.join(os.getcwd(), build_name))
        shutil.rmtree(build_dir, ignore_errors=True)
    else:
        build_dir = Path(os.getcwd())

    resolved_space = None
    if space:
        resolved_space = resolve_space(github_token, space, callback)
        if not resolved_space:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={
                        "reason": f"Space '{space}' provided could not be resolved. Please run 'llmb space list --all --refresh' to see your latest available spaces."
                    },
                )
            return None
    if not resolved_space:
        # get default space if none is provided
        resolved_space = resolve_space(github_token, "default", callback)
        if not resolved_space:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={
                        "reason": f"Space default not found in available spaces."
                    },
                )
            return None

    if from_build:
        # from a build
        logger.info("Using a previous build '%s' to create a new build", from_build)

        build_id = from_build
        get_build_from_remote(
            github_token,
            build_id,
            callback,
            id_format,
            resolved_space,
            filename,
            build_dir,
        )

    else:
        # from a template
        branch_name = gb_environment_config()["branch_space"]
        if space:
            template_repo = resolved_space.get("git_repo_uri")
        elif template_repo != None:
            if "github.ibm.com" not in template_repo:
                if callback is not None:
                    callback(
                        callback_event="error",
                        callback_args={
                            "reason": f"--template-repo value is invalid, make sure you are providing a GitHub repository URL."
                        },
                    )
                return
            elif "@" in template_repo:
                template_repo_split = template_repo.split("@", 1)
                template_repo = template_repo_split[0]
                if template_repo_split[1] != "":
                    branch_name = template_repo_split[1]
        else:
            template_repo = ASSETS_REPO_URL
            branch_name = gb_environment_config()["branch_assets"]

        if callback:
            time.sleep(1)
            callback(callback_event="preparing_contents", callback_args={"steps": 1})

        clone_github_repo(
            github_token,
            template_repo,
            branch_name,
            cache_path,
            TEMPLATES_REPO_FOLDER,
            from_template,
            update_bar=callback,
        )

        template_path = os.path.join(cache_path, TEMPLATES_REPO_FOLDER, from_template)
        if os.path.exists(template_path):
            if not filename or filename == "":
                shutil.copytree(template_path, build_dir)
            else:
                # check if we need to create a folder
                folder_path = os.path.dirname(filename)
                if not os.path.isdir(folder_path) and folder_path and folder_path != "":
                    os.makedirs(folder_path)
                shutil.copy(os.path.join(template_path, "build.yaml"), filename)

        else:
            raise FileNotFoundError(
                f"Template {from_template} not found in {template_repo}."
            )

    return True, (
        from_build
        if from_build
        else f"{from_template} in {template_repo}@{branch_name}"
    )


def build_start(
    github_token: str,
    quiet: bool,
    filename: str = "",
    space: Optional[str] = None,
    params: Optional[List[str]] = [],
    skip_validation=False,
    parameters_path: Optional[str] = None,
    targets: Optional[tuple[str, ...]] = (),
    description: str = "",
    tags: list[str] = [],
    callback=None,
    validation_type: str = "static",
) -> str:

    gbserver_build_update = gb_environment_config()["feature_flags"][
        "gbserver_build_update"
    ]
    if gbserver_build_update == False:
        description = None
        if len(tags) > 0:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={
                        "reason": "build start using --tag or --tags is not available in this environment yet"
                    },
                )
            return

    build_file_path = False
    if filename and os.path.exists(filename):
        build_file_path = filename
    elif os.path.exists(os.path.join(os.getcwd(), "build.yaml")):
        build_file_path = os.path.join(os.getcwd(), "build.yaml")
    elif os.path.exists(os.path.join(os.getcwd(), "build.yml")):
        build_file_path = os.path.join(os.getcwd(), "build.yml")

    if not build_file_path:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"Build yaml file could not be found. Specify a valid file path via -f option or be in the same current working directory as 'build.yaml' file"
                },
            )
        return

    ignore_git_global_config()

    if is_standalone():
        creds = GBCredentials()
        user_name = creds.get("login", section="user.gbserver") or os.environ.get(
            "GBSERVER_API_USER", "standalone"
        )
    else:
        user_name = get_user(github_token).login
    build_archive = None

    if not space:
        if is_standalone():
            space = "standalone"
        else:
            space = "default"
    global_space = resolve_space(github_token, space, callback)
    if not global_space:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"Space {space} not found in available spaces."
                },
            )
        return
    space_repo = global_space.get("git_repo_uri")
    if not space_repo:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"Space {space} not found in available spaces."
                },
            )
        return

    if callback and not quiet:
        callback(callback_event="preparing_contents", callback_args={"steps": 1})

    if filename:
        filename_split = os.path.split(filename)[-1]
        suffix = f".{filename_split.split('.', 1)[-1]}" if "." in filename_split else ""
        build_name = remove_suffix(filename_split, suffix)
    else:
        build_name = os.path.split(os.getcwd())[-1]
    branch_name = f"{build_name}-{generate_unique_id()}"

    experiment_folder = prepare_build_local_contents(
        build_file_path, branch_name, filename
    )

    if callback and not quiet:
        callback(callback_event="prepared_contents", callback_args={"steps": 100})

    if os.path.exists(BUILD_RUN_FILE):
        if callback is not None and not quiet:
            callback(
                callback_event="warning",
                callback_args={
                    "reason": f"User-created {BUILD_RUN_FILE} file in build folder will not be included in build submission PR."
                },
            )

    if skip_validation:
        if callback and not quiet:
            callback(callback_event="skip__pr_validation", callback_args={"steps": 100})
        try:
            parameters_helper(
                quiet,
                parameters_path,
                build_file_path,
                experiment_folder,
                params,
                callback,
            )
        except Exception as e:
            if callback is not None:
                callback(callback_event="clear", callback_args={})
                callback(
                    callback_event="error",
                    callback_args={"reason": f"Error applying build parameters: {e}."},
                )
            return None

        if len(targets) > 0:
            run_yaml_path = os.path.join(experiment_folder, BUILD_RUN_FILE)
            run_dict = {BUILD_RUN_YAML_KEY: {}}
            for target in targets:
                run_dict[BUILD_RUN_YAML_KEY][target] = None

            with open(run_yaml_path, "w", encoding="utf-8") as f:
                run_yaml = yaml.safe_dump(run_dict).replace("null", "")
                f.write(run_yaml)
    else:
        try:
            validate_helper(
                github_token,
                quiet,
                experiment_folder,
                branch_name,
                build_file_path,
                space,
                params,
                parameters_path,
                targets,
                callback,
                validation_type=validation_type,
            )
        except Exception as e:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={"reason": f"Error validating build contents: {e}."},
                )
            return None

    zip_buffer = io.BytesIO()
    build_archive = create_build_folder_archive(experiment_folder, zip_buffer)
    zip_buffer.close()

    if callback and not quiet:
        callback(callback_event="submitting_pr", callback_args={"steps": 1})

    logger.debug(f"Submitting build {branch_name} to gbserver...")
    build_name = extract_build_name(experiment_folder, filename)
    gbserver_build = make_gbserver_call(
        lambda: submit_build(
            github_token,
            GBSERVER_BUILD_API,
            build_name,
            build_archive,
            global_space.get("name"),
            user_name,
            list(targets),
            tags,
            description,
        ),
        callback,
    )

    if callback is not None and not quiet:
        callback(
            callback_event="submitted_pr",
            callback_args={
                "steps": 100,
            },
        )

    return gbserver_build["build_id"]


def build_validate(
    github_token: str,
    quiet: bool,
    filename: Optional[str] = "",
    space: Optional[str] = None,
    params: Optional[List[str]] = [],
    parameters_path: Optional[str] = None,
    targets: Optional[tuple[str, ...]] = (),
    callback=None,
    validation_type: str = "static",
) -> Optional[Tuple]:
    try:
        ignore_git_global_config()

        build_name = os.path.split(os.getcwd())[-1]
        branch_name = f"{build_name}-{generate_unique_id()}"
        build_file_path = False
        if filename and os.path.exists(filename):
            build_file_path = filename
        elif os.path.exists(os.path.join(os.getcwd(), "build.yaml")):
            build_file_path = os.path.join(os.getcwd(), "build.yaml")
        elif os.path.exists(os.path.join(os.getcwd(), "build.yml")):
            build_file_path = os.path.join(os.getcwd(), "build.yml")

        if not build_file_path:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={
                        "reason": f"Build yaml file could not be found. Specify a valid file path via -f option or be in the same current working directory as 'build.yaml' file"
                    },
                )
            return

        experiment_folder = prepare_build_local_contents(
            build_file_path, branch_name, filename
        )
        return validate_helper(
            github_token,
            quiet,
            experiment_folder,
            branch_name,
            build_file_path,
            space,
            params,
            parameters_path,
            targets,
            callback,
            validation_type=validation_type,
        )
    except Exception as e:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={"reason": f"Error validating build contents: {e}"},
            )
        return False


def build_cancel(
    github_token: str,
    build_id: str,
    id_format: str,
    space: Optional[str] = None,
    callback=None,
) -> Any:
    user_name = get_user(github_token).login

    if not space:
        space = "default"
    global_space = resolve_space(github_token, space, callback)
    space_repo = global_space.get("git_repo_uri")
    if not space_repo:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"Space {space} not found in available spaces."
                },
            )
        return

    if id_format == "url":
        build_id_from_url = get_build_id_from_url(github_token, build_id, callback)
        build_id = build_id_from_url[0]["uuid"]

    if callback is not None:
        callback(
            callback_event="obtaining_build",
            callback_args={"steps": 1, "build_id": build_id},
        )

    build_info = make_gbserver_call(
        lambda: get_build(build_id, github_token, GBSERVER_BUILD_API),
        callback,
    )

    if callback is not None:
        callback(
            callback_event="obtained_build",
            callback_args={"steps": 100, "build_id": build_id},
        )

    if build_info and build_info["build"]["username"] != user_name:
        if not user_is_space_admin(github_token, build_info["build"]["space_name"]):
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={
                        "reason": f"Build {build_id} was created by a different user. Make sure you are cancelling builds of your own."
                    },
                )
            return None
        else:
            logger.debug(
                f"Proceeding cancellation request as user is an admin in space {build_info['build']['space_name']}."
            )

    if build_info and build_info["build"]["space_name"] != global_space.get("name"):
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"Build {build_id} was created in a different space: {build_info['build']['space_name']}. Try running the command again using the --space option."
                },
            )
        return None

    if callback is not None:
        callback(
            callback_event="cancelling_build",
            callback_args={"steps": 1, "build_id": build_id},
        )

    canceled_build = make_gbserver_call(
        lambda: cancel_build(build_id, github_token, GBSERVER_BUILD_API),
        callback,
    )

    if callback is not None:
        callback(
            callback_event="canceled_build",
            callback_args={"steps": 100, "build_id": build_id},
        )

    return canceled_build


def build_lineage_lh(
    github_token: str, token: str, build_id: str, id_format: str, callback=None
):
    from lakehouse import LakehouseLineage

    if not github_token:
        raise Exception(USER_NOT_LOGGED_IN_ERROR_MESSAGE)

    resolved_space = resolve_space(github_token, "default", callback)

    if id_format == "url":
        build_id_from_url = get_build_id_from_url(
            github_token, build_id, callback, resolved_space
        )
        build_id = build_id_from_url[0]["uuid"]
    else:
        try:
            id_check = get_build(build_id, github_token, GBSERVER_BUILD_API)
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

        resolved_space_name = resolved_space.get("name")
        available_spaces = [
            space.get("name") for space in get_spaces(github_token, callback)
        ]

        id_check_space_name = id_check["build"]["space_name"]
        build_space = (
            id_check_space_name if id_check_space_name in available_spaces else None
        )

        check_build_space_boundary(
            build_id,
            resolved_space_name,
            id_check["build"],
            build_space,
            callback,
        )

    # Get build status
    build_obj = make_gbserver_call(
        lambda: get_build(build_id, github_token, GBSERVER_BUILD_API),
        callback,
    )["build"]

    if callback is not None:
        callback(callback_event="fetching_build_lineage_lh", callback_args={})

    stop_event = threading.Event()
    # Start spinner in a separate thread
    spinner_thread = threading.Thread(
        target=spinner_running, args=(stop_event, callback, "build_lineage_spinner_lh")
    )
    spinner_thread.start()

    lh = getLH(token)
    lineage_df = LakehouseLineage(lh=lh).get_lineage_by_column(
        column_name="release_id", column_values=[build_id]
    )

    stop_event.set()  # Stop spinner thread
    spinner_thread.join()  # Ensure spinner stops

    return (
        build_obj["status"],
        (
            lineage_df.replace({np.nan: "None"})
            .sort_values(by="job_started_at")
            .to_dict("records")
        ),
    )


def build_lineage_gbserver(
    github_token: str, build_id: str, id_format: str, callback=None
):
    if not github_token:
        raise Exception(USER_NOT_LOGGED_IN_ERROR_MESSAGE)

    resolved_space = resolve_space(github_token, "default", callback)

    if id_format == "url":
        build_id_from_url = get_build_id_from_url(
            github_token, build_id, callback, resolved_space
        )
        build_id = build_id_from_url[0]["uuid"]

    # Get build status
    build_obj = make_gbserver_call(
        lambda: get_build(build_id, github_token, GBSERVER_BUILD_API),
        callback,
    )["build"]

    if callback is not None:
        callback(callback_event="fetching_build_lineage_gbserver", callback_args={})

    stop_event = threading.Event()
    # Start spinner in a separate thread
    spinner_thread = threading.Thread(
        target=spinner_running, args=(stop_event, callback, "build_lineage_spinner")
    )
    spinner_thread.start()

    lineage = make_gbserver_call(
        lambda: get_build_lineage(
            build_id,
            github_token,
            GBSERVER_LINEAGE_API,
        ),
        callback,
    )

    def format_obj_name(obj):
        facets = obj.get("facets", {})
        artifact_type = facets.get("artifact_type", "").lower()
        name = obj.get("name", "")
        namespace = obj.get("namespace", "")
        uri = obj.get("uri", namespace)
        match artifact_type:
            case "model":
                return name + "|" + uri
            case "fileset":
                return name + "|" + uri
            case "dataset":
                return name + "|" + uri
            case _:
                return name if name else uri

    def infer_artifact_type(dataset):
        facets = dataset.get("facets", {})
        atype = facets.get("artifact_type", "")
        if atype and atype != "UNDEFINED":
            return atype
        uri = dataset.get("uri", "") or dataset.get("namespace", "")
        if "/datasets/" in uri:
            return "DATASET"
        if uri.startswith("hf://"):
            return "MODEL"
        return atype

    def format_lineage(lineage_data):
        jobs = []
        for step_object in lineage_data["targets"]:
            for target_key, lineage_records in step_object.items():
                for record in lineage_records:
                    run_facets = record.get("run", {}).get("facets", {})
                    job_details = run_facets.get("job_details", {})
                    tags = run_facets.get("tags", {})

                    sources = record.get("inputs", [])
                    targets = record.get("outputs", [])

                    if sources and targets:
                        for source in sources:
                            for target in targets:
                                jobs.append(
                                    {
                                        "release_id": job_details.get("release_id", ""),
                                        "category": job_details.get("category", ""),
                                        "job_name": record.get("job", {}).get(
                                            "name", ""
                                        ),
                                        "job_id": job_details.get("job_id", ""),
                                        "job_type": job_details.get("job_type", ""),
                                        "job_started_at": job_details.get(
                                            "job_started_at", ""
                                        ),
                                        "job_completed_at": job_details.get(
                                            "job_completed_at", ""
                                        ),
                                        "job_status": job_details.get("job_status", ""),
                                        "owner": tags.get("username", ""),
                                        "source": format_obj_name(source),
                                        "source_type": infer_artifact_type(source),
                                        "source_object": source,
                                        "target": format_obj_name(target),
                                        "target_type": infer_artifact_type(target),
                                        "target_object": target,
                                        "source_code_details": run_facets.get(
                                            "source_code", {}
                                        ),
                                        "job_input_params": run_facets.get(
                                            "job_input_params", {}
                                        ),
                                        "execution_stats": run_facets.get(
                                            "execution_stats", {}
                                        ),
                                        "job_output_stats": job_details.get(
                                            "job_output_stats", {}
                                        ),
                                    }
                                )

        return jobs

    stop_event.set()  # Stop spinner thread
    spinner_thread.join()  # Ensure spinner stops

    return build_obj["status"], format_lineage(lineage)


def build_list(
    github_token: str,
    list_all: bool,
    show_done: bool,
    show_all: bool,
    all_spaces: bool,
    space: Optional[str] = None,
    username: Optional[str] = None,
    tags: list[str] = None,
    page_index: Optional[int] = None,
    page_size: Optional[int] = None,
    callback=None,
) -> Any | None:
    username = get_user(github_token).login if username == None else username
    if all_spaces:
        used_all_spaces = True
        space_default = None
        space_default_name = None
        space_org = None
        space_name = None
    else:
        used_all_spaces = False
        if all_spaces:
            if callback is not None:
                callback(callback_event="all_spaces_warning", callback_args={})

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

    # total_prs_progress = 0
    # prs_progress = 1

    if callback is not None:
        callback(
            callback_event="listing_builds",
            callback_args={
                "steps": 1,
                "space": space_default_name if space_default_name else "public",
                "space_name": f"{space_org}/{space_name}",
                "used_all_spaces": used_all_spaces,
            },
        )

    gbserver_username = None if list_all else username

    gbserver_build_update = gb_environment_config()["feature_flags"][
        "gbserver_build_update"
    ]
    if gbserver_build_update == False:
        if len(tags) > 0:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={
                        "reason": "build list using --tag or --tags filter is not available in this environment yet"
                    },
                )
            return
    status = None
    if not show_done and not show_all:
        status = ["pending", "running", "submitted"]
    sort = ["created_time:desc"]
    gbserver_builds = make_gbserver_call(
        lambda: get_builds(
            github_token,
            GBSERVER_BUILD_API,
            gbserver_username,
            space_name=space_default_name,
            tag=tags,
            status=status,
            sort=sort,
            page_index=page_index,
            page_size=page_size,
        ),
        callback,
    )
    gbserver_builds_count = make_gbserver_call(
        lambda: get_builds_count(
            github_token,
            GBSERVER_BUILD_API,
            gbserver_username,
            space_name=space_default_name,
            tag=tags,
            status=status,
        ),
        callback,
    )
    if callback is not None:
        callback(
            callback_event="listed_builds",
            callback_args={
                "steps": 100,
                "space": space_default_name if space_default_name else "public",
                "space_name": f"{space_org}/{space_name}",
                "used_all_spaces": used_all_spaces,
            },
        )

    prs = []
    if callback is not None:
        callback(
            callback_event="processing_builds",
            callback_args={
                "steps": 1,
                "space": space_default_name if space_default_name else "public",
                "space_name": f"{space_org}/{space_name}",
                "used_all_spaces": used_all_spaces,
            },
        )
    # TODO: cleanup after confirmed no longer needed
    # else:
    #     if callback is not None:
    #         callback(
    #             callback_event="listing_prs",
    #             callback_args={
    #                 "steps": prs_progress,
    #                 "space_name": (
    #                     remove_suffix(space_default, ".git") if space_default else None
    #                 ),
    #                 "used_all_spaces": used_all_spaces,
    #             },
    #         )
    #         total_prs_progress = total_prs_progress + prs_progress
    #     try:
    #         prs, next_page = get_prs(github_token, space_org, space_name, "open")
    #     except HTTPError as e:
    #         if callback is not None:
    #             callback(
    #                 callback_event="error",
    #                 callback_args={
    #                     "reason": f"Error obtaining additional information. {e.response.status_code} {e.response.reason} for {space_default}.",
    #                     "used_all_spaces": used_all_spaces,
    #                 },
    #             )
    #         return None

    #     if callback is not None:
    #         prs_progress = 30 if next_page else 70
    #         callback(
    #             callback_event="listing_prs",
    #             callback_args={
    #                 "steps": prs_progress,
    #                 "space_name": (
    #                     remove_suffix(space_default, ".git") if space_default else None
    #                 ),
    #                 "used_all_spaces": used_all_spaces,
    #             },
    #         )
    #         total_prs_progress = total_prs_progress + prs_progress

    #     while next_page != None and next_page.get("url") != None:
    #         next_page_prs, next_page = get_prs(
    #             github_token, space_org, space_name, next_page_url=next_page.get("url")
    #         )
    #         prs = prs + next_page_prs

    #         if callback is not None:
    #             prs_progress = 10 if next_page else (70 - prs_progress)
    #             callback(
    #                 callback_event="listing_prs",
    #                 callback_args={
    #                     "steps": prs_progress,
    #                     "space_name": (
    #                         remove_suffix(space_default, ".git")
    #                         if space_default
    #                         else None
    #                     ),
    #                     "used_all_spaces": used_all_spaces,
    #                 },
    #             )
    #             total_prs_progress = total_prs_progress + prs_progress

    #     if not list_all:
    #         prs = [p for p in prs if p["user"]["login"] == username]

    # TODO: phase out merge_pr_info function
    builds = merge_pr_info(prs, gbserver_builds["builds"])

    build_list = []

    for b in builds:
        build_obj = {
            "build_id": b.get("uuid", ""),
            "name": b.get("name", ""),
            "user": b.get("username", ""),
            "start_time": (
                b.get("created_time", "") + ".000000Z"
                if len(b.get("created_time", "")) == 19
                else b.get("created_time", "")
            ),
            "status": (
                str(b.get("status", "")).capitalize()
                if b.get("status", "") != ""
                else "Under review"
            ),
            "description": b.get("description", ""),
            "tags": b.get("tags", ""),
        }

        if all_spaces:
            build_obj["space_name"] = b.get("space_name", space_default_name)

        build_list.append(build_obj)

    if callback is not None:
        callback(
            callback_event="processed_builds",
            callback_args={
                "steps": 100,
                "space": space_default_name if space_default_name else "public",
                "space_name": f"{space_org}/{space_name}",
                "used_all_spaces": used_all_spaces,
            },
        )

    count = gbserver_builds_count["count"] if gbserver_builds_count else 0
    return {"items": build_list, "count": count}


def build_log(
    github_token: str,
    id_format: str,
    start_epoch: Optional[int] = None,
    end_epoch: Optional[int] = None,
    page_size: Optional[int] = None,
    page_index: Optional[int] = None,
    stream: Optional[str] = None,
    text: Optional[str] = None,
    sort: Optional[str] = None,
    build_id: Optional[str] = None,
    build_step_id: Optional[str] = None,
    build_step_name: Optional[str] = None,
    runner: Optional[bool] = False,
    follow: Optional[bool] = False,
    all: Optional[bool] = False,
    skip_id_check: Optional[bool] = False,
    callback=None,
):
    if not github_token:
        raise Exception(USER_NOT_LOGGED_IN_ERROR_MESSAGE)

    resolved_space = (
        None if skip_id_check else resolve_space(github_token, "default", callback)
    )
    if id_format == "url":
        build_id_from_url = get_build_id_from_url(
            github_token, build_id, callback, resolved_space
        )
        build_id = build_id_from_url[0]["uuid"]
    elif not skip_id_check:
        try:
            id_check = get_build(build_id, github_token, GBSERVER_BUILD_API)
        except HTTPException as e:
            if e.status_code == 404:
                id_check = None
            else:
                if callback is not None:
                    callback(
                        callback_event="error",
                        callback_args={
                            "steps": 1,
                            "reason": f"gbserver returned '{e.status_code} {e.detail}'.",
                        },
                    )
                return None

        if not id_check and user_is_space_admin(github_token, "public", callback):
            if callback is not None:
                callback(
                    callback_event="warning",
                    callback_args={
                        "reason": f"Build ID {build_id} is unknown. Making log query by admin access."
                    },
                )
        elif not id_check and not user_is_space_admin(github_token, "public", callback):
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={"reason": f"Build ID {build_id} is not found."},
                )
            return None
        elif id_check:
            resolved_space_name = resolved_space.get("name")
            available_spaces = [
                space.get("name") for space in get_spaces(github_token, callback)
            ]

            id_check_space_name = id_check["build"]["space_name"]
            build_space = (
                id_check_space_name if id_check_space_name in available_spaces else None
            )

            check_build_space_boundary(
                build_id,
                resolved_space_name,
                id_check["build"],
                build_space,
                callback,
            )

    current_epoch = get_current_epoch()
    if start_epoch == None:
        start_epoch = change_timestamp_by_days(
            current_epoch, BUILD_LOG_DEFAULT_QUERY_RANGE
        )
    if end_epoch == None:
        end_epoch = current_epoch
    if all:
        page_size = BUILD_LOGALL_PAGE_SIZE
    elif page_size == None:
        page_size = 50

    if all or follow:
        page_index = 0
        displayed_logs_ids = []
        next_timestamp = start_epoch
        continue_logquery = True
        is_current_timestamp = False
    else:
        if page_index == None:
            page_index = 0

    if callback:
        callback(
            callback_event="querying_log",
            callback_args={
                "start_epoch": start_epoch,
                "end_epoch": end_epoch if end_epoch else round(time.time()),
            },
        )

    application_name = (
        gb_environment_config()["server_log_application_name"] if runner else None
    )

    if not all or follow:
        if all:
            sort = "asc"
        elif sort == None:
            sort = "desc"
        response = run_logquery(
            github_token,
            start_epoch,
            end_epoch,
            page_size,
            page_index,
            application_name,
            stream,
            text,
            sort,
            build_id,
            build_step_id,
            build_step_name,
            runner,
            callback,
        )

        logs = response["logs"]

        if follow:
            if logs != None:
                # query is successful
                if len(logs) > 0:
                    next_timestamp = convert_milliseconds_to_seconds(
                        logs[len(logs) - 1]["timestamp"]
                    )
                    if sort == "desc":
                        logs.reverse()
                    if callback:
                        callback(
                            callback_event="display_logs",
                            callback_args={"logs": logs},
                        )
                    displayed_logs_ids = [log["logId"] for log in logs]
        else:
            return logs

    if all or follow:
        sort = "asc"
        while continue_logquery or follow:
            if follow:
                time.sleep(BUILD_LOG_FOLLOW_SLEEP_TIME)

            end_epoch, is_current_timestamp = check_current_timestamp(
                change_timestamp_by_days(
                    next_timestamp, BUILD_LOG_DEFAULT_QUERY_RANGE, True
                )
            )

            response = run_logquery(
                github_token,
                next_timestamp,
                end_epoch,
                page_size,
                page_index,
                application_name,
                stream,
                text,
                sort,
                build_id,
                build_step_id,
                build_step_name,
                runner,
                callback,
            )

            if response["logs"]:
                logs = [
                    log
                    for log in response["logs"]
                    if (not displayed_logs_ids)
                    or (displayed_logs_ids and (log["logId"] not in displayed_logs_ids))
                ]
                if logs and len(logs) > 0:
                    timestamps = []
                    for log in logs:
                        timestamp = convert_milliseconds_to_seconds(log["timestamp"])
                        if timestamp not in timestamps:
                            timestamps.append(timestamp)
                    next_timestamp = timestamps[len(timestamps) - 2]
                    displayed_logs_ids = displayed_logs_ids + [
                        log["logId"] for log in logs
                    ]
                    if callback:
                        callback(
                            callback_event="display_logs",
                            callback_args={"logs": logs},
                        )
            if not response["logs"] or not logs or len(logs) == 0:
                if not follow and is_current_timestamp:
                    continue_logquery = False
                else:
                    next_timestamp, is_current_timestamp = check_current_timestamp(
                        change_timestamp_by_days(
                            next_timestamp, BUILD_LOG_DEFAULT_QUERY_RANGE, True
                        ),
                        True,
                    )

        return displayed_logs_ids


def build_status(
    github_token: str,
    build_id: str,
    quiet: bool,
    id_format: str,
    show_events: bool,
    fetch_pr: bool,
    result_format: str,
    callback=None,
) -> List[Any]:
    gbserver_build_events = gb_environment_config()["feature_flags"][
        "gbserver_build_events"
    ]

    global_space = resolve_space(github_token, "default", callback)
    space_default = global_space.get("git_repo_uri")
    if not space_default:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"Space default not found in available spaces."
                },
            )
        return None

    if id_format == "url":
        build_id_from_url = get_build_id_from_url(
            github_token, build_id, callback, global_space
        )
        if not build_id_from_url:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={
                        "reason": f"No builds were found with URL {build_id}."
                    },
                )
            return None, None, None, f"No builds were found with URL {build_id}."
        build_id = build_id_from_url[0]["uuid"]

    stop_event = threading.Event()
    # Start spinner in a separate thread
    spinner_thread = threading.Thread(
        target=spinner_running, args=(stop_event, callback, "fetching_build_status")
    )
    spinner_thread.start()

    def stop_spinner():
        stop_event.set()  # Stop spinner thread
        spinner_thread.join()  # Ensure spinner stops

    # TODO: use global_space to properly scope the build ID
    build_status = make_gbserver_call(
        lambda: get_build_status_with_targets_runs(
            github_token, build_id, GBSERVER_BUILD_API
        )["status"],
        callback,
        stop_spinner,
    )

    if id_format != "url":
        resolved_space_name = global_space.get("name")
        available_spaces = [
            space.get("name") for space in get_spaces(github_token, callback)
        ]

        gbserver_space_name = build_status["build"]["space_name"]
        build_space = (
            gbserver_space_name if gbserver_space_name in available_spaces else None
        )

        check_build_space_boundary(
            build_id,
            resolved_space_name,
            build_status["build"],
            build_space,
            callback,
        )

    if callback is not None:
        callback(
            callback_event="fetched_build_status", callback_args={"build_id": build_id}
        )
        if not quiet:
            callback(
                callback_event="processing_status_artifacts", callback_args={"steps": 1}
            )

    targets = (
        process_target_runs(build_status["target_runs"])
        if result_format == "plain"
        else process_target_runs_to_json(build_status.get("target_runs"))
    )

    if callback is not None:
        if not quiet:
            callback(
                callback_event="processed_status_artifacts", callback_args={"steps": 99}
            )

    pr_uri = build_status["build"]["source_uri"]
    if gbserver_build_events and not fetch_pr and show_events:
        logger.debug("Fetching build events from gbserver...")
        build_history = get_gbserver_build_events(build_id, github_token, callback)

    elif pr_uri and show_events:
        logger.debug("Fetching build events from build PR...")
        space_org, space_name = pr_uri.split("/")[3:5]
        pull_number = pr_uri.split("/")[-1]

        comments = get_pr_events(
            github_token,
            space_org,
            space_name,
            pull_number,
            callback,
        )

        build_history = [
            {
                "time": comment["created_at"],
                "description": comment["body"].replace("`", ""),
            }
            for comment in comments
        ]
    else:
        build_history = []
        if callback is not None:
            callback(callback_event="skip_additional_info", callback_args={"steps": 1})

    build_details = {
        "build_id": build_status["build"]["uuid"],
        "name": build_status["build"]["name"],
        "started_at": build_status["build"]["created_time"],
        "updated_at": build_status["build"]["updated_time"],
        "status": build_status["build"]["status"],
        "source_pr": build_status["build"]["source_uri"],
        "description": build_status["build"]["description"],
    }

    return build_details, targets, build_history, None


def build_describe(
    github_token: str,
    filename: str,
    format: str,
    raw: bool,
    build_id: Optional[str] = None,
    id_format: Optional[str] = None,
    space: Optional[str] = None,
    callback=None,
) -> str:
    downloaded_build_file_path = False
    build = None
    # if build_id is supplied, fetch build.yaml from server
    if build_id and len(build_id) > 0:

        resolved_space = None
        if space:
            resolved_space = resolve_space(github_token, space, callback)
            if not resolved_space:
                if callback is not None:
                    callback(
                        callback_event="error",
                        callback_args={
                            "reason": f"Space '{space}' provided could not be resolved. Please run 'llmb space list --all --refresh' to see your latest available spaces."
                        },
                    )
                return None
        if not resolved_space:
            # get default space if none is provided
            resolved_space = resolve_space(github_token, "default", callback)
            if not resolved_space:
                if callback is not None:
                    callback(
                        callback_event="error",
                        callback_args={
                            "reason": f"Space default not found in available spaces."
                        },
                    )
                return None

        build_dir = Path(os.getcwd())

        # create temporary output file, to be deleted after generating targets
        output_filename = f"build-{generate_unique_id()}.yaml"

        downloaded_build_file_path = os.path.join(os.getcwd(), output_filename)

        build = get_build_from_remote(
            github_token,
            build_id,
            callback,
            id_format,
            resolved_space,
            output_filename,
            build_dir,
        )
    build_file_path = False
    if downloaded_build_file_path and os.path.exists(downloaded_build_file_path):
        build_file_path = downloaded_build_file_path
    elif filename and os.path.exists(filename):
        build_file_path = filename
    elif os.path.exists(os.path.join(os.getcwd(), "build.yaml")):
        build_file_path = os.path.join(os.getcwd(), "build.yaml")
    elif os.path.exists(os.path.join(os.getcwd(), "build.yml")):
        build_file_path = os.path.join(os.getcwd(), "build.yml")

    if not build_file_path:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"Build yaml file could not be found. Specify a valid file path via -f option or be in the same current working directory as 'build.yaml' file"
                },
            )
        return None

    build_yaml_path = Path(build_file_path)
    describe_build_yaml_path = os.path.join(
        build_yaml_path.cwd(), f"describe_{BUILD_FILENAME}"
    )

    try:
        if raw:
            with open(build_yaml_path, "r", encoding="utf-8") as f:
                # build_yaml_dict = yaml.safe_load(f)
                file_str = f.read()
            # return yaml.dump(build_yaml_dict, indent=2)
            return file_str
        else:
            targets = describe_build_yaml(
                build_yaml_path, describe_build_yaml_path, format
            )
            return targets, build
    except Exception as e:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"The {BUILD_FILENAME} at {build_yaml_path} is invalid, error: {e}."
                },
            )
        return None
    finally:
        if describe_build_yaml_path and os.path.exists(describe_build_yaml_path):
            os.remove(describe_build_yaml_path)
        if downloaded_build_file_path and os.path.exists(downloaded_build_file_path):
            os.remove(downloaded_build_file_path)


def build_diff(
    github_token: str,
    build_id_1: str,
    id_format_1: str,
    build_id_2: Optional[str] = None,
    id_format_2: Optional[str] = None,
    space: Optional[str] = None,
    callback=None,
) -> Tuple[str, str, List[Any]]:
    if not space:
        space = "default"
    global_space = resolve_space(github_token, space, callback)
    space_repo = global_space.get("git_repo_uri")
    if not space_repo:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"Space {space} not found in available spaces."
                },
            )
        return

    if not build_id_2:
        build_yaml_paths = get_build_yaml_paths(os.getcwd())
        if len(build_yaml_paths) == 0:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={
                        "reason": f"Expected at least one {BUILD_FILENAME}."
                    },
                )
            return None

        local_build_filename = build_yaml_paths[0]
        with open(local_build_filename, "r", encoding="utf-8") as f1:
            local_build_file_lines = f1.readlines()
    else:
        build_url_2 = ""
        if id_format_2 == "url":
            build_url_2 = build_id_2
            build_id_2_from_url = get_build_id_from_url(
                github_token,
                build_id_2,
                callback,
                global_space,
            )
            build_id_2 = build_id_2_from_url[0]["uuid"]

        if id_format_2 != "filename":
            build_2_filename, build_2_file_contents = get_remote_build_file_contents(
                build_id_2,
                build_url_2,
                github_token,
                callback,
                (global_space if not build_url_2 else None),
            )
            build_2_file_lines = io.StringIO(build_2_file_contents).readlines()

            if callback is not None and id_format_1 != "filename":
                callback(callback_event="break_line", callback_args={})
        else:
            build_2_filename, build_2_file_lines = get_local_build_file_contents(
                build_id_2, callback
            )

    build_url_1 = ""
    if id_format_1 == "url":
        build_url_1 = build_id_1
        build_id_1_from_url = get_build_id_from_url(
            github_token, build_id_1, callback, global_space
        )
        build_id_1 = build_id_1_from_url[0]["uuid"]

    if id_format_1 != "filename":
        build_1_filename, build_1_file_contents = get_remote_build_file_contents(
            build_id_1,
            build_url_1,
            github_token,
            callback,
            (global_space if not build_url_1 else None),
        )
        build_1_file_lines = io.StringIO(build_1_file_contents).readlines()
    else:
        build_1_filename, build_1_file_lines = get_local_build_file_contents(
            build_id_1, callback
        )

    diff_1_filename = build_1_filename if build_id_2 else local_build_filename
    diff_1_lines = build_1_file_lines if build_id_2 else local_build_file_lines
    diff_2_filename = build_2_filename if build_id_2 else build_1_filename
    diff_2_lines = build_2_file_lines if build_id_2 else build_1_file_lines

    diff_output = difflib.unified_diff(
        diff_1_lines,
        diff_2_lines,
        fromfile=diff_1_filename,
        tofile=diff_2_filename,
        lineterm="",
    )

    return diff_1_filename, diff_2_filename, list(diff_output)


def build_monitor(
    github_token: str,
    build_id: str,
    show_events: bool,
    fetch_pr: bool,
    id_format: Optional[str] = None,
    callback=None,
) -> Tuple[Any, List[Any], List[Any], List[Any]]:
    gbserver_build_events = gb_environment_config()["feature_flags"][
        "gbserver_build_events"
    ]

    resolved_space = resolve_space(github_token, "default", callback)

    if id_format == "url":
        build_id_from_url = get_build_id_from_url(
            github_token, build_id, callback, resolved_space
        )
        build_id = build_id_from_url[0]["uuid"]
    else:
        try:
            id_check = get_build(build_id, github_token, GBSERVER_BUILD_API)
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

        resolved_space_name = resolved_space.get("name")
        available_spaces = [
            space.get("name") for space in get_spaces(github_token, callback)
        ]

        id_check_space_name = id_check["build"]["space_name"]
        build_space = (
            id_check_space_name if id_check_space_name in available_spaces else None
        )

        check_build_space_boundary(
            build_id,
            resolved_space_name,
            id_check["build"],
            build_space,
            callback,
        )

    monitor_output = None
    next_timestamp = None
    previous_logs = []
    start_epoch = change_timestamp_by_days(
        get_current_epoch(), BUILD_LOG_DEFAULT_QUERY_RANGE
    )
    response = None

    while not monitor_output:
        stop_event = threading.Event()
        # Start spinner in a separate thread
        spinner_thread = threading.Thread(
            target=spinner_running, args=(stop_event, callback, "fetching_build_status")
        )
        spinner_thread.start()

        def stop_spinner():
            stop_event.set()  # Stop spinner thread
            spinner_thread.join()  # Ensure spinner stops

        # TODO: use global_space to properly scope the build ID
        status_obj = make_gbserver_call(
            lambda: get_build_status_with_targets_runs(
                github_token, build_id, GBSERVER_BUILD_API
            )["status"],
            callback,
            stop_spinner,
        )

        status = status_obj["build"]["status"]
        if status == "pending" or status == "running" or status == "submitted":
            end_epoch = get_current_epoch()
            if callback:
                if not response:
                    callback(
                        callback_event="fetched_build_status",
                        callback_args={"build_status": status},
                    )
                if not next_timestamp:
                    callback(
                        callback_event="querying_log_range",
                        callback_args={
                            "start_epoch": (
                                start_epoch
                                if not next_timestamp
                                else int(next_timestamp)
                            ),
                            "end_epoch": end_epoch,
                        },
                    )
                callback(
                    callback_event="querying_log_server",
                    callback_args={"new_line": True if not next_timestamp else False},
                )

            response = run_logquery(
                github_token,
                (start_epoch if not next_timestamp else int(next_timestamp)),
                end_epoch,
                50,
                0,
                "granite-build",
                sort="desc",
                build_id=build_id,
            )

            if response == None:
                callback(
                    callback_event="error",
                    callback_args={"reason": f"No response from log server"},
                )
                return None

            if response.get("Error") != None:
                if response.get("Error") == "Connection error":
                    callback(
                        callback_event="error",
                        callback_args={
                            "reason": f"Logs server connection error. {VPN_CONNECTION_ERROR_MESSAGE}"
                        },
                    )
                else:
                    callback(
                        callback_event="error",
                        callback_args={
                            "reason": f"Logs server returns '{response['Error']}'"
                        },
                    )
                return None

            if response["status"] != 200:
                callback(
                    callback_event="error",
                    callback_args={"reason": f"Query fails from log server"},
                )
                return None

            if response["logs"] != None:
                logs = (
                    [
                        log
                        for log in response["logs"]
                        if log["timestamp"]
                        >= convert_seconds_to_milliseconds(next_timestamp)
                    ]
                    if next_timestamp
                    else response["logs"]
                )
                # query is successful
                if len(logs) > 0:
                    next_timestamp = convert_milliseconds_to_seconds(
                        logs[0]["timestamp"]
                    )
                    logs.reverse()
                    if callback:
                        callback(
                            callback_event="display_logs",
                            callback_args={
                                "logs": logs,
                                "previous_logs": previous_logs,
                            },
                        )
                    previous_logs = [log["logId"] for log in logs]

        else:
            monitor_output = status_obj["build"]
            targets = process_target_runs(status_obj["target_runs"])

            pr_uri = status_obj["build"]["source_uri"]
            if show_events:
                info_stop_event = threading.Event()
                # Start spinner in a separate thread
                info_spinner_thread = threading.Thread(
                    target=spinner_running,
                    args=(info_stop_event, callback, "fetching_additional_info"),
                )
                info_spinner_thread.start()

                if gbserver_build_events and not fetch_pr:
                    logger.debug("Fetching build events from gbserver...")
                    build_history = get_gbserver_build_events(build_id, github_token)

                elif pr_uri:
                    logger.debug("Fetching build events from build PR...")
                    space_org, space_name = pr_uri.split("/")[3:5]
                    pull_number = pr_uri.split("/")[-1]

                    comments = get_pr_events(
                        github_token,
                        space_org,
                        space_name,
                        pull_number,
                    )

                    build_history = [
                        {
                            "time": comment["created_at"],
                            "description": comment["body"].replace("`", ""),
                        }
                        for comment in comments
                    ]

                info_stop_event.set()  # Stop spinner thread
                info_spinner_thread.join()  # Ensure spinner stops
            else:
                build_history = []

            build_details = {
                "build_id": status_obj["build"]["uuid"],
                "name": status_obj["build"]["name"],
                "started_at": status_obj["build"]["created_time"],
                "updated_at": status_obj["build"]["updated_time"],
                "status": status_obj["build"]["status"],
                "source_pr": status_obj["build"]["source_uri"],
            }

        time.sleep(BUILD_LOG_FOLLOW_SLEEP_TIME)
    return build_details, targets, build_history


def build_notification(
    github_token: str,
    status: Optional[str] = None,
    space: Optional[str] = None,
    callback=None,
) -> Tuple[str, str]:
    if not space:
        space = "default"
    global_space = resolve_space(github_token, space, callback)
    space_repo = global_space.get("git_repo_uri")
    if not space_repo:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"Space {space} not found in available spaces."
                },
            )
        return

    space_org, space_name = space_repo.split("/")[3:]
    space_name = remove_suffix(space_name, ".git")
    space_output_message = f'"{global_space.get("name")}" ({space_org}/{space_name})'

    if not status:
        try:
            notification = get_repo_subscription(github_token, space_org, space_name)
        except HTTPError as e:
            if e.response.status_code == 404:
                return True, space_output_message
            else:
                if callback is not None:
                    callback(
                        callback_event="error",
                        callback_args={"reason": f"{e}"},
                    )
                return None
    else:
        try:
            if status == "on":
                delete_repo_subscription(github_token, space_org, space_name)
                return True, space_output_message
            else:
                notification = ignore_repo_subscription(
                    github_token, space_org, space_name
                )
        except HTTPError as e:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={"reason": f"{str(e)}"},
                )
            return None

    return notification["subscribed"], space_output_message


def process_target_runs(target_runs: List[Any]) -> Tuple[List[Any], List[Any]]:
    target_runs = sorted(
        target_runs,
        key=lambda x: datetime.fromisoformat(x["target"]["started_at"]),
    )
    targets = {}
    for target_run in target_runs:
        target = target_run["target"]
        input_artifacts = []
        output_artifacts = []
        steps = []

        for input_artifact in target_run["input_artifacts"]:
            input_artifacts.append(
                {
                    "artifact_id": input_artifact.get("uuid", ""),
                    "uri": input_artifact.get("uri", ""),
                }
            )
        for output_artifact in target_run["output_artifacts"]:
            output_artifacts.append(
                {
                    "artifact_id": output_artifact.get("uuid", ""),
                    "uri": output_artifact.get("uri", ""),
                }
            )
        for step in target_run["steps"]:
            steps.append(
                {
                    "step_id": step.get("uuid", ""),
                    "uri": step.get("definition_uri", ""),
                    "status": step.get("status", ""),
                    "started_at": step.get("started_at", ""),
                }
            )

        targets[f"{target['name']} ({target['uuid']})"] = {
            "status": target["status"],
            "input_artifacts": input_artifacts,
            "output_artifacts": output_artifacts,
            "steps": sorted(
                steps,
                key=lambda x: datetime.fromisoformat(x["started_at"]),
            ),
        }

    return targets


def process_target_runs_to_json(target_runs: List[Any]) -> List[Any]:
    target_runs = sorted(
        target_runs,
        key=lambda x: datetime.fromisoformat(x["target"]["started_at"]),
    )

    targets = []
    for target_run in target_runs:
        target = target_run.get("target")
        input_artifacts = []
        output_artifacts = []
        steps = []

        for input_artifact in target_run["input_artifacts"]:
            input_artifacts.append(
                {
                    "artifact_id": input_artifact.get("uuid", ""),
                    "uri": input_artifact.get("uri", ""),
                }
            )
        for output_artifact in target_run["output_artifacts"]:
            output_artifacts.append(
                {
                    "artifact_id": output_artifact.get("uuid", ""),
                    "uri": output_artifact.get("uri", ""),
                }
            )
        for step in target_run["steps"]:
            steps.append(
                {
                    "step_id": step.get("uuid", ""),
                    "uri": step.get("definition_uri", ""),
                    "status": step.get("status", ""),
                    "started_at": step.get("started_at", ""),
                }
            )

        targets.append(
            {
                "target_name": target.get("name"),
                "build_id": target.get("build_id"),
                "target_id": target.get("uuid"),
                "status": target.get("status"),
                "input_artifacts": input_artifacts,
                "output_artifacts": output_artifacts,
                "steps": sorted(
                    steps,
                    key=lambda x: datetime.fromisoformat(x["started_at"]),
                ),
            }
        )

    return targets


def get_local_build_file_contents(build_id: str, callback=None):
    if os.path.exists(build_id):
        with open(build_id, "r", encoding="utf-8") as build_file:
            build_file_lines = build_file.readlines()
        build_filename = os.path.abspath(build_id)
    else:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"Provided file path {build_id} doesn't exist."
                },
            )
        return None

    return build_filename, build_file_lines


def get_remote_build_file_contents(
    build_id: str,
    build_url: str,
    github_token: str,
    callback=None,
    resolved_space=None,
) -> Tuple[str, str]:
    stop_event = threading.Event()
    # Start spinner in a separate thread
    spinner_thread = threading.Thread(
        target=spinner_running,
        args=(
            stop_event,
            callback,
            "fetching_build_file",
            (build_id if build_url == "" else build_url),
        ),
    )
    spinner_thread.start()

    try:
        build_details = get_build(build_id, github_token, GBSERVER_BUILD_API)
    except Exception as e:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={"reason": f"{e}"},
            )
        return None
    finally:
        stop_event.set()  # Stop spinner thread
        spinner_thread.join()  # Ensure spinner stops

    if resolved_space:
        resolved_space_name = resolved_space.get("name")
        available_spaces = [
            space.get("name") for space in get_spaces(github_token, callback)
        ]

        gbserver_build_space_name = build_details["build"]["space_name"]
        build_space = (
            gbserver_build_space_name
            if gbserver_build_space_name in available_spaces
            else None
        )

        check_build_space_boundary(
            build_id,
            resolved_space_name,
            build_details["build"],
            build_space,
            callback,
        )

    remote_build_file_contents = None
    remote_build_file_name = ""

    build_archive_bytes = b64decode(build_details["build"]["build_archive"])
    with zipfile.ZipFile(io.BytesIO(build_archive_bytes), "r") as zip_ref:
        for file in zip_ref.namelist():
            if file == BUILD_FILENAME:
                remote_build_file_name = (
                    f"{file} ({build_id if build_url == '' else build_url})"
                )
                with zip_ref.open(file, "r") as build_file:
                    remote_build_file_contents = build_file.read().decode("utf-8")

    if not remote_build_file_contents:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"Expected at least one {BUILD_FILENAME} in build submission PR."
                },
            )
        return None

    return remote_build_file_name, remote_build_file_contents


def merge_pr_info(gh_pulls: List[Any], gbserver_pulls: List[Any]) -> List[Any]:
    gbserver_builds = [b for b in gbserver_pulls if b["username"] != "gb-local-user"]
    for pr_gh in gh_pulls:
        pr_gh_gbserver = {
            "uuid": "",
            "name": pr_gh["title"],
            "created_time": pr_gh["created_at"],
            "updated_time": pr_gh["updated_at"],
            "status": "",
            "source_uri": pr_gh["html_url"],
            "username": pr_gh["user"]["login"],
            "tags": [],
            "description": "",
        }
        gbserver_builds.append(pr_gh_gbserver)

    return gbserver_builds


def get_params_from_file(file_path: str) -> dict:
    if os.path.exists(file_path) and os.stat(file_path).st_size > 0:
        with open(file_path, "r", encoding="utf-8") as f:
            params_from_file = yaml.safe_load(f)
        if not isinstance(params_from_file, dict):
            raise Exception(
                "parameters defined in parameters.yaml must be written in key-value pairs."
            )
        return params_from_file
    else:
        return {}


def apply_build_parameters(
    build_folder_path: str, params, params_from_file, callback=None
) -> Tuple[Any, str]:
    build_yaml_dict = None
    build_yaml_path = None

    build_yaml_paths = get_build_yaml_paths(build_folder_path)
    if len(build_yaml_paths) > 0:
        build_yaml_path = build_yaml_paths[0]
        if os.path.isfile(build_yaml_path):
            with open(build_yaml_path, "r", encoding="utf-8") as f:
                # Apply variable replacements here
                yaml_converted = apply_parameters(
                    f.read(), params, params_from_file, build_folder_path
                )
                build_yaml_dict = yaml.safe_load(yaml_converted)
                f.close()
                with open(build_yaml_path, "w", encoding="utf-8") as f2:
                    f2.write(yaml_converted)
        else:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={
                        "reason": f"Expected {build_yaml_path} to be a file."
                    },
                )
            return None
    else:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={"reason": f"Expected at least one {BUILD_FILENAME}."},
            )
        return None
    return build_yaml_dict, Path(build_yaml_path)


def get_build_yaml_dict_inner(build_yaml_dict: dict) -> dict:
    assert isinstance(
        build_yaml_dict, dict
    ), f"invalid build_yaml_dict: {build_yaml_dict}"
    build_yaml_dict_inner = None
    for k in BUILD_YAML_BASE_KEYS:
        if k in build_yaml_dict:
            build_yaml_dict_inner = build_yaml_dict[k]
            break
    if build_yaml_dict_inner is None:
        raise ValueError(
            f"The build.yaml is invalid. The top-level key must be among: {BUILD_YAML_BASE_KEYS}"
        )
    assert isinstance(
        build_yaml_dict_inner, dict
    ), f"invalid build_yaml_dict_inner: {build_yaml_dict_inner}"
    return build_yaml_dict_inner


def validate_build_content(build_yaml_dict: Any, build_yaml_path: str) -> str:
    try:
        assert isinstance(build_yaml_dict, dict)
        build_yaml_dict_inner = get_build_yaml_dict_inner(build_yaml_dict)
        version = build_yaml_dict_inner.get(CURRENT_BUILD_YAML_VERSION_KEY, "")
        if version == "":
            pass
            # TODO: for this version we're allowing a missing version number
            # return f"The {BUILD_FILENAME} version missing, expected {CURRENT_BUILD_YAML_VERSION}."
        elif version != CURRENT_BUILD_YAML_VERSION:
            return f"The {BUILD_FILENAME} version doesn't match, actual {version} expected {CURRENT_BUILD_YAML_VERSION}."
        BuildConfig.from_yaml(path=Path(build_yaml_path))
    except AssertionError:
        return f"The {BUILD_FILENAME} at {build_yaml_path} is invalid, error: file content must be in YAML format."
    except Exception as e:
        if "Invalid parameter" in str(e):
            error_message = e
        else:
            error_message = (
                f"The {BUILD_FILENAME} at {build_yaml_path} is invalid, error: {e}."
            )
        return error_message


def get_build_yaml_paths(build_folder_path: str):
    build_yaml_paths = []
    for file in glob(f"{build_folder_path}/*", recursive=True):
        if file.split("/")[-1] == BUILD_FILENAME:
            build_yaml_paths.append(file)

    return build_yaml_paths


def process_targets(targets: tuple, build_folder_path: str):
    build_yaml_paths = get_build_yaml_paths(build_folder_path)
    if len(build_yaml_paths) < 0:
        raise Exception(f"Expected at least one {BUILD_FILENAME}.")

    build_yaml_path = build_yaml_paths[0]
    run_yaml_path = os.path.join(build_folder_path, BUILD_RUN_FILE)

    run_dict = {BUILD_RUN_YAML_KEY: {}}
    with open(build_yaml_path, "r", encoding="utf-8") as f:
        build_yaml_dict = yaml.safe_load(f)

    for target in targets:
        build_yaml_dict_inner = get_build_yaml_dict_inner(build_yaml_dict)
        if "targets" not in build_yaml_dict_inner:
            raise ValueError(f"Error: 'targets' were not found in the build.yaml.")
        if target not in build_yaml_dict_inner["targets"]:
            raise Exception(f"Error: target {target} not found in build.yaml.")
        else:
            if targets.count(target) > 1:
                raise Exception(f"Error: {target} was specified multiple times.")
            else:
                run_dict[BUILD_RUN_YAML_KEY][target] = None

    with open(run_yaml_path, "w", encoding="utf-8") as f:
        run_yaml = yaml.safe_dump(run_dict).replace("null", "")
        f.write(run_yaml)


def spinner_running(stop_event, callback=None, callback_event=None, build_id=None):
    """Displays a spinner until `stop_event` is set."""
    spinner = itertools.cycle(["-", "\\", "|", "/"])  # Spinner characters
    while not stop_event.is_set():
        if callback and callback_event:
            args_obj = (
                {"spinner": next(spinner)}
                if not build_id
                else {"build_id": build_id, "spinner": next(spinner)}
            )
            callback(callback_event=callback_event, callback_args=args_obj)
            time.sleep(0.1)


def describe_build_yaml(
    build_yaml_path: str, describe_build_yaml_path: str, format: str
) -> List[Any]:
    with open(build_yaml_path, "r", encoding="utf-8") as f:
        yaml_converted = re.sub(
            # fmt: off
            r"\$\$\{[\s\S\}]*?\}",
            # fmt: on
            lambda x: x.group().replace("$${", "").replace("}", ""),
            f.read(),
        )
        f.close()
        with open(describe_build_yaml_path, "w", encoding="utf-8") as f2:
            f2.write(yaml_converted)

    build_config = BuildConfig.from_yaml(path=Path(describe_build_yaml_path))

    targets = []
    for target in build_config.targets:
        inputs = (
            build_config.targets[target].inputs
            if build_config.targets[target].inputs
            else []
        )
        outputs = (
            build_config.targets[target].outputs
            if build_config.targets[target].outputs
            else []
        )
        steps = (
            build_config.targets[target].steps
            if build_config.targets[target].steps
            else []
        )

        target_obj = {
            "target_name": target,
            "environment_uri": build_config.targets[target].environment_uri,
            "inputs": (
                [
                    {
                        "name": input,
                        "uri": (
                            inputs[input].uri
                            if inputs[input].uri
                            else inputs[input].binding
                        ),
                    }
                    for input in inputs
                ]
            ),
            "outputs": (
                [{"name": output, "uri": outputs[output].uri} for output in outputs]
            ),
            "steps": (
                [{"uri": step.step_uri, "config": step.config} for step in steps]
                if format != "simple"
                else [{"uri": step.step_uri} for step in steps]
            ),
        }
        targets.append(target_obj)
    return targets


def ignore_build_files(directory, contents):
    return [
        item
        for item in contents
        if item != BUILD_FILENAME and item != BUILD_PARAMETERS_FILE
    ]


def get_build_id_from_url(
    github_token: str, build_url: str, callback=None, resolved_space=None
) -> list:
    build_url_split = urlsplit(build_url)
    repo_path = "/".join(build_url_split[2].split("/")[:-2])
    space_from_build_url = urlunsplit(
        (build_url_split[0], build_url_split[1], repo_path, "", "")
    )

    build_space = None
    spaces = get_spaces(github_token, callback)
    for space in spaces:
        if str(space.get("git_repo_uri")).strip("/") == space_from_build_url.strip("/"):
            build_space = space.get("name")

    if not build_space:
        if callback is not None:
            callback(
                callback_event="warning",
                callback_args={
                    "reason": f"The build URL pattern doesn't match any space repo. Make sure that your GB_ENVIRONMENT value ({gb_environment()}) is correct to look up the right space."
                },
            )

    if callback is not None:
        callback(
            callback_event="fetching_build_id",
            callback_args={"steps": 1, "source_uri": build_url},
        )
    build_from_url = make_gbserver_call(
        lambda: get_builds(github_token, GBSERVER_BUILD_API, source_uri=build_url)[
            "builds"
        ],
        callback,
    )

    logger.debug(f"Found {len(build_from_url)} builds with 'source_uri = {build_url}'.")
    if len(build_from_url) == 0:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={"reason": f"No builds were found with URL {build_url}."},
            )
        return None

    build_id = build_from_url[0]["uuid"]
    if callback is not None:
        callback(
            callback_event="fetched_build_id",
            callback_args={"steps": 100, "build_id": build_id, "source_uri": build_url},
        )

    if resolved_space:
        resolved_space_name = resolved_space.get("name")
        check_build_space_boundary(
            build_id, resolved_space_name, build_from_url[0], build_space, callback
        )

    return build_from_url


def get_build_url_from_build_id(user_token: str, build_id: str, callback=None) -> list:
    if callback is not None:
        callback(
            callback_event="fetching_build_url",
            callback_args={"steps": 1, "build_id": build_id},
        )
    build_url = make_gbserver_call(
        lambda: get_build(build_id, user_token, GBSERVER_BUILD_API)["build"][
            "source_uri"
        ],
        callback,
    )

    if not build_url:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={"reason": f"No build was found with ID {build_id}."},
            )
        return None
    else:
        if callback is not None:
            callback(
                callback_event="fetched_build_url",
                callback_args={"steps": 100, "build_id": build_id},
            )

    return build_url


def create_build_archive(build_path: Path, zip_buffer, build_filename: str) -> str:
    with open(build_path, "r") as build_path_file:
        build_path_file_read = build_path_file.read()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(build_filename, build_path_file_read)

    build_archive = b64encode(zip_buffer.getvalue()).decode("utf-8")

    return build_archive


def create_build_folder_archive(build_path: Path, zip_buffer) -> str:
    build_files = os.listdir(build_path)
    build_files_contents = {}
    for build_file in build_files:
        with open(os.path.join(build_path, build_file), "r") as f:
            build_file_read = f.read()
            build_files_contents[build_file] = build_file_read

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for build_file in build_files_contents:
            zipf.writestr(build_file, build_files_contents[build_file])

    build_archive = b64encode(zip_buffer.getvalue()).decode("utf-8")

    return build_archive


def get_gbserver_build_events(
    build_id: str, github_token: str, callback=None
) -> List[Any]:
    if callback is not None:
        callback(callback_event="fetching_additional_info", callback_args={"steps": 1})

    build_events = make_gbserver_call(
        lambda: get_build_events(build_id, github_token, GBSERVER_BUILD_API)["events"],
        callback,
    )

    build_history = [
        {
            "time": event["build_event"]["timestamp"],
            "description": event["build_event"]["payload"]
            .get("msg", "")
            .replace("`", ""),
        }
        for event in build_events
        if event["build_event"]["payload"].get("msg", "") != ""
    ]

    if callback is not None:
        callback(callback_event="fetched_additional_info", callback_args={"steps": 100})

    return build_history


def get_pr_events(
    github_token: str,
    space_org: str,
    space_name: str,
    pull_number: int,
    callback=None,
) -> List[Any]:
    if callback is not None:
        callback(callback_event="fetching_additional_info", callback_args={"steps": 1})

    comments = []
    comments, next_page = get_pr_comments(
        github_token, space_org, space_name, pull_number
    )

    comments_progress = 50
    progress = 0
    if callback is not None:
        progress = 20 if next_page else 50
        callback(
            callback_event=(
                "fetching_additional_info" if next_page else "fetched_additional_info"
            ),
            callback_args={"steps": progress},
        )
        comments_progress = comments_progress + progress

    while next_page != None and next_page.get("url") != None:
        next_page_comments, next_page = get_pr_comments(
            github_token,
            space_org,
            space_name,
            pull_number,
            next_page_url=next_page.get("url"),
        )
        comments = comments + next_page_comments
        if callback is not None:
            progress = 10 if next_page else (100 - comments_progress)
            callback(
                callback_event=(
                    "fetching_additional_info"
                    if next_page
                    else "fetched_additional_info"
                ),
                callback_args={"steps": progress},
            )
            comments_progress = comments_progress + progress

    return comments


def prepare_build_local_contents(
    build_file_path: str,
    branch_name: str,
    filename: Optional[str] = "",
) -> str:
    # check to see if cache exists, if not create
    cache_path = create_if_not_dir_local_build_cache()

    experiments_folder = os.path.join(cache_path, SPACE_REPO_BUILD_FOLDER)
    if not os.path.exists(experiments_folder):
        os.mkdir(experiments_folder)

    experiment_folder = tempfile.mkdtemp(
        prefix=f"{branch_name}_", dir=experiments_folder
    )
    experiment_build_file_path = os.path.join(experiment_folder, "build.yaml")
    # TODO: do not copy entire folder, lets copy only needed files
    if not filename or filename == "":
        shutil.copytree(
            os.getcwd(),
            experiment_folder,
            ignore=ignore_build_files,
            dirs_exist_ok=True,
        )
    else:
        os.makedirs(experiment_folder, exist_ok=True)
        shutil.copy(build_file_path, experiment_build_file_path)

    return experiment_folder


def parameters_helper(
    quiet: bool,
    parameters_path: str,
    build_file_path: str,
    experiment_folder: str,
    params: Optional[List[str]] = [],
    callback=None,
    callback_event=None,
) -> Tuple[str, str]:
    if not parameters_path:
        build_path = "/".join(build_file_path.split("/")[:-1])
        parameters_path = os.path.join(build_path, BUILD_PARAMETERS_FILE)
    logger.debug(f"Searching for parameters file {parameters_path}...")
    try:
        params_from_file = get_params_from_file(parameters_path)
    except Exception as e:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={"reason": f"Error parsing file contents: {e}"},
            )
        return

    if callback and callback_event and not quiet:
        callback(callback_event=callback_event, callback_args={"steps": 20})

    build_yaml_dict, build_yaml_path = apply_build_parameters(
        experiment_folder, params, params_from_file, callback
    )

    if callback and callback_event and not quiet:
        callback(callback_event=callback_event, callback_args={"steps": 20})

    return build_yaml_dict, build_yaml_path


def validate_helper(
    github_token: str,
    quiet: bool,
    experiment_folder: str,
    branch_name: str,
    build_file_path: str,
    space: Optional[str] = None,
    params: Optional[List[str]] = [],
    parameters_path: Optional[str] = None,
    targets: Optional[tuple[str, ...]] = (),
    callback=None,
    validation_type: str = "static",
) -> Optional[Tuple]:
    """logic originally in buld_start, refactored to be used by build_start and build_validate"""
    user_name = get_user(github_token).login

    if not space:
        space = "default"
    global_space = resolve_space(github_token, space, callback)
    space_repo = global_space.get("git_repo_uri")
    if not space_repo:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"Space {space} not found in available spaces."
                },
            )
        return

    if callback and not quiet:
        callback(callback_event="validating_pr", callback_args={"steps": 1})

    build_yaml_dict, build_yaml_path = parameters_helper(
        quiet,
        parameters_path,
        build_file_path,
        experiment_folder,
        params,
        callback,
        callback_event="validating_pr",
    )

    logger.debug("Calling gbserver validate endpoint..")
    # allocate memory to store build archive
    zip_buffer = io.BytesIO()
    build_archive = create_build_archive(build_yaml_path, zip_buffer, BUILD_FILENAME)

    try:
        validate_response = validate_build(
            github_token,
            GBSERVER_BUILD_API,
            build_archive,
            global_space.get("name"),
            user_name,
            validation_type=validation_type,
        )
    except HTTPError as e:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={"reason": f"{e}"},
            )
        return None
    except HTTPException as e:
        if callback is not None:
            if 400 < e.status_code < 600:
                callback(
                    callback_event="error",
                    callback_args={
                        "reason": f"gbserver returned '{e.status_code} {e.detail}'."
                    },
                )
        return None
    except ConnectionError:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"gbserver connection error. {VPN_CONNECTION_ERROR_MESSAGE}"
                },
            )
        return None
    finally:
        # clean up memory allocated to build_archive
        zip_buffer.close()

    validations = process_build_validation_response(validate_response)
    for validation in validations:
        updated_yaml_dict = get_yaml_diff(build_yaml_dict, validation)

        if updated_yaml_dict:
            updated_yaml = yaml.safe_dump(updated_yaml_dict)
            validation["updated_yaml"] = updated_yaml

    if len(validations) > 0:
        has_errors = len(validate_response.get("errors", [])) > 0
        if callback is not None:
            callback(
                callback_event="validation",
                callback_args={
                    "reformatted_original_yaml": yaml.safe_dump(build_yaml_dict),
                    "validations": validations,
                    "build_path": build_file_path,
                    "has_errors": has_errors,
                },
            )
        if has_errors:
            return build_archive, validate_response

    if callback and not quiet:
        callback(callback_event="validated_pr", callback_args={"steps": 60})

    if len(targets) > 0:
        if callback and not quiet:
            callback(callback_event="validating_targets", callback_args={"steps": 1})
        try:
            process_targets(targets, experiment_folder)
        except Exception as e:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={"reason": e},
                )
            return build_archive, validate_response

        if callback and not quiet:
            callback(callback_event="validated_targets", callback_args={"steps": 100})

    return build_archive, validate_response


def get_build_from_remote(
    github_token,
    build_id,
    callback,
    id_format,
    resolved_space,
    filename,
    build_dir,
):
    """
    common function used by both init and describe to pull down build.yaml for an existing build
    """
    if id_format == "url":
        build_id_from_url = get_build_id_from_url(
            github_token, build_id, callback, resolved_space
        )
        build_id = build_id_from_url[0]["uuid"]
    response = make_gbserver_call(
        lambda: get_build(build_id, github_token, GBSERVER_BUILD_API),
        callback,
    )

    assert isinstance(response, dict) and "build" in response
    _gbserver_build = response["build"]
    if id_format != "url":
        # check current space vs incoming build space to make sure they are the same
        available_spaces = [
            space.get("name") for space in get_spaces(github_token, callback)
        ]
        build_space = (
            _gbserver_build.get("space_name")
            if _gbserver_build.get("space_name") in available_spaces
            else None
        )
        resolved_space_name = resolved_space.get("name") if resolved_space else None
        check_build_space_boundary(
            build_id, resolved_space_name, _gbserver_build, build_space, callback
        )

    assert isinstance(_gbserver_build, dict)
    gbserver_build = BuildResponse.model_validate(_gbserver_build)
    if gbserver_build.build_archive == "":
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"gbserver returned an empty 'build_archive' field for build id {build_id}."
                },
            )
        return None
    build_archive_bytes = b64decode(gbserver_build.build_archive)

    if not filename or filename == "":
        with zipfile.ZipFile(io.BytesIO(build_archive_bytes), "r") as zip_ref:
            zip_ref.extractall(build_dir)
    else:
        with zipfile.ZipFile(io.BytesIO(build_archive_bytes), "r") as zip_ref:
            build_copy_path = tempfile.TemporaryDirectory(prefix="build").name
            zip_ref.extractall(build_copy_path)
            # check if we need to create a folder
            folder_path = os.path.dirname(filename)
            if not os.path.isdir(folder_path) and folder_path and folder_path != "":
                os.makedirs(folder_path)
            shutil.copy(os.path.join(build_copy_path, "build.yaml"), filename)
    return response


def extract_build_name(experiment_folder: str, filename: str):
    build_yaml_path = os.path.join(experiment_folder, BUILD_FILENAME)
    try:
        with open(build_yaml_path, "r", encoding="utf-8") as f:
            build_yaml_dict = yaml.safe_load(f)
    except Exception as e:
        raise Exception(f"Error: {build_yaml_path} can't be parsed: {str(e)}")

    if not filename:
        filename = BUILD_FILENAME

    build_yaml_dict_inner = get_build_yaml_dict_inner(build_yaml_dict)
    if build_yaml_dict_inner == {} or build_yaml_dict_inner.get("name", "") == "":
        raise Exception(
            f"Error: 'name' field not found in provided {filename}. Build name is required to start a build."
        )

    return build_yaml_dict_inner["name"]


def update_build(
    github_token: str,
    build_id: str,
    tags: Optional[list[str]] = None,
    description: str = None,
    append: bool = False,
    callback=None,
):
    # Validate that append is not used with empty tags
    if append and tags is not None and len(tags) == 0:
        raise ValueError("--append cannot be used with empty tags")

    gbserver_build_update = gb_environment_config()["feature_flags"][
        "gbserver_build_update"
    ]
    if gbserver_build_update == False:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": "build update is not available in this environment yet"
                },
            )

    username = get_user(github_token).login
    if not username or not github_token:
        raise Exception(USER_NOT_LOGGED_IN_ERROR_MESSAGE)

    gbserver_artifact = update_build_gserver(
        build_id=build_id,
        server_api=GBSERVER_BUILD_API,
        user_token=github_token,
        description=description,
        tags=tags,
        append=append,
    )

    return gbserver_artifact


def fetch_build(
    github_token: str,
    build_id: str,
    id_format: str,
    callback=None,
):
    """
    fetch an existing build
    """
    if id_format == "url":
        build_id_from_url = get_build_id_from_url(github_token, build_id, callback)
        build_id = build_id_from_url[0]["uuid"]
    response = make_gbserver_call(
        lambda: get_build(build_id, github_token, GBSERVER_BUILD_API),
        callback,
    )

    assert isinstance(response, dict) and "build" in response

    return response["build"]


def check_build_space_boundary(
    github_token: str,
    build_id: str,
    resolved_space_name: str,
    gbserver_build: Any,
    build_space: Optional[str],
    callback=None,
):
    space_not_available = (
        not build_space and user_is_space_admin(github_token, "public", callback)
    ) or (build_space and resolved_space_name and resolved_space_name != build_space)
    if space_not_available:
        if callback is not None:
            callback(
                callback_event="warning",
                callback_args={
                    "reason": f"Build {build_id} is available in a different space: '{gbserver_build.get('space_name')}'.",
                },
            )
    elif not build_space and not user_is_space_admin(github_token, "public", callback):
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"Build {build_id} was not found within space '{resolved_space_name}'.",
                },
            )
        return None
