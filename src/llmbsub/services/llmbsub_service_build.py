import io
import logging
import os
from typing import List, Optional

import yaml

from gbcli.services.service_build import (
    create_build_folder_archive,
    extract_build_name,
    parameters_helper,
    prepare_build_local_contents,
    validate_helper,
)
from gbcli.utils.gbconstants import (
    BUILD_RUN_FILE,
    BUILD_RUN_YAML_KEY,
    GBSERVER_BUILD_API,
)
from gbcli.utils.gbcredentials import GBCredentials, get_user_token
from gbcli.utils.gbserver import make_gbserver_call, submit_build
from gbcli.utils.spaceutil import resolve_space
from gbcli.utils.utils import generate_unique_id, remove_suffix

logger = logging.getLogger(__name__)


def llmbsub_build_start(
    quiet: bool,
    filename: str = "",
    space: Optional[str] = None,
    params: Optional[List[str]] = [],
    skip_validation=False,
    parameters_path: Optional[str] = None,
    targets: Optional[tuple[str, ...]] = (),
    message: str = "",
    callback=None,
    skip_space_resolution=False,
    validation_type: str = "static",
) -> str:
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

    credentials = GBCredentials()
    user_token = credentials.get("token", section="user.github")
    user_name = credentials.get("login", section="user.github")
    build_archive = None

    if not space:
        space = "default"
    if not skip_space_resolution:
        global_space = resolve_space(space, callback)
        if not global_space:
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
                get_user_token(),
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

    build_name = extract_build_name(experiment_folder, filename)
    logger.debug(f"Submitting build {build_name} to gbserver...")
    gbserver_build = make_gbserver_call(
        lambda: submit_build(
            user_token,
            GBSERVER_BUILD_API,
            build_name,
            build_archive,
            global_space.get("name") if not skip_space_resolution else space,
            user_name,
            list(targets),
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
