import logging
import os
from pathlib import Path
from typing import Any, List, Optional

from requests import HTTPError

from gbcli.utils.gbconstants import (
    ASSETS_REPO_URL,
    STEP_FILENAME,
    STEP_README_FILENAME,
    STEPS_REPO_FOLDER,
)
from gbcli.utils.gh_clone import download_repo_file, list_repo_tree
from gbcli.utils.spaceutil import resolve_space
from gbcli.utils.utils import parse_markdown_str, remove_suffix
from gbcommon.types.constants import DEFAULT_GH_DOMAIN
from gbcommon.types.gbenvconfig import gb_environment_config
from gbcommon.types.stepconfig import (
    StepConfig,
    StepEnvironmentTypeConfig,
    StepLauncherConfig,
    StepMonitorConfig,
)

logger = logging.getLogger(__name__)


def list_steps(
    github_token: str,
    step_repo: Optional[str] = None,
    space: Optional[str] = None,
    callback=None,
) -> List[Any]:

    branch_name = gb_environment_config().space_config_branch_name
    if space:
        s = resolve_space(github_token, space, callback)
        step_repo = s["git_repo_uri"] if s is not None else None
        if not step_repo:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={
                        "steps": 1,
                        "reason": f"Space {space} not found in available spaces.",
                    },
                )
            return None
    elif step_repo != None:
        if "@" in step_repo:
            step_repo_split = step_repo.split("@", 1)
            step_repo = step_repo_split[0]
            if step_repo_split[1] != "":
                branch_name = step_repo_split[1]
    else:
        step_repo = ASSETS_REPO_URL
        branch_name = gb_environment_config().branch_assets

    if callback is not None:
        callback(
            callback_event="listing_steps",
            callback_args={
                "steps": 1,
                "space": space if space else "public",
                "steps_repo": step_repo,
            },
        )

    step_repo_split = step_repo.split("/")
    if len(step_repo_split) < 2:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "steps": 1,
                    "reason": f"--step-repo value is invalid, make sure you are providing a GitHub repository URL.",
                },
            )

    steps_org, steps_name = step_repo_split[-2:]
    steps_name = remove_suffix(steps_name, ".git")

    try:
        repo_tree = list_repo_tree(github_token, steps_org, steps_name, branch_name)
    except HTTPError as e:
        if callback is not None:
            if e.response.status_code == 404:
                error_message = f"{step_repo} not found"
            else:
                error_message = (
                    f"{e.response.status_code} {e.response.reason} for {step_repo}"
                )

            callback(
                callback_event="error",
                callback_args={
                    "steps": 1,
                    "reason": f"Error obtaining steps. {error_message}.",
                },
            )
        return None

    if callback is not None:
        callback(
            callback_event="listing_steps",
            callback_args={
                "steps": 70,
                "space": space if space else "public",
                "steps_repo": step_repo,
            },
        )

    steps = []
    for file in repo_tree["tree"]:
        file_path = str(file["path"]).split("/")
        if (
            str(file["path"]).startswith(f"{STEPS_REPO_FOLDER}/")
            and file["type"] == "tree"
            and len(file_path) == 2
        ):
            steps.append(
                {
                    "step_name": file_path[1],
                    "description": f"https://{DEFAULT_GH_DOMAIN}/{steps_org}/{steps_name}/tree/{branch_name}/{file['path']}",
                }
            )

    if callback is not None:
        callback(
            callback_event="listed_steps",
            callback_args={
                "steps": 100,
                "space": space if space else "public",
                "steps_repo": step_repo,
            },
        )

    return steps


def describe_step(
    github_token: str,
    step_name: str,
    step_repo: Optional[str] = None,
    space: Optional[str] = None,
    callback=None,
) -> Any:
    branch_name = gb_environment_config().space_config_branch_name
    if space:
        s = resolve_space(github_token, space, callback)
        step_repo = s["git_repo_uri"] if s is not None else None
        if not step_repo:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={
                        "steps": 1,
                        "reason": f"Space {space} not found in available spaces.",
                    },
                )
            return None
    elif step_repo != None:
        if "@" in step_repo:
            step_repo_split = step_repo.split("@", 1)
            step_repo = step_repo_split[0]
            if step_repo_split[1] != "":
                branch_name = step_repo_split[1]
    else:
        step_repo = ASSETS_REPO_URL
        branch_name = gb_environment_config().branch_assets

    steps_org, steps_name = step_repo.split("/")[3:]
    steps_name = remove_suffix(steps_name, ".git")

    step_yaml_path = (
        f"{STEPS_REPO_FOLDER}/{step_name}/{STEP_FILENAME}?ref={branch_name}"
    )
    step_yaml_cache = f"describe_{step_name}_{STEP_FILENAME}"

    step_readme_path = (
        f"{STEPS_REPO_FOLDER}/{step_name}/{STEP_README_FILENAME}?ref={branch_name}"
    )

    if callback is not None:
        callback(
            callback_event="processing_step_content",
            callback_args={
                "steps": 1,
                "steps_repo": f"{step_repo}?branch={branch_name}",
            },
        )

    try:
        readme_content = download_repo_file(
            github_token, steps_org, steps_name, step_readme_path
        )
    except HTTPError:
        readme_content = ""

    try:
        yaml_content = download_repo_file(
            github_token, steps_org, steps_name, step_yaml_path
        )

        if callback is not None:
            callback(
                callback_event="processing_step_content",
                callback_args={
                    "steps": 35,
                    "steps_repo": f"{step_repo}?branch={branch_name}",
                },
            )

        with open(step_yaml_cache, "w", encoding="utf-8") as cache_file:
            cache_file.write(yaml_content)

        step_config = StepConfig.from_yaml(path=Path(step_yaml_cache))

        if callback is not None:
            callback(
                callback_event="processing_step_content",
                callback_args={
                    "steps": 14,
                    "steps_repo": f"{step_repo}?branch={branch_name}",
                },
            )

        step_obj = {
            "name": step_config.name,
            "type": step_config.type,
            "config": [
                {config: step_config.config[config]} for config in step_config.config
            ],
            "environment_configs": (
                [
                    {
                        env_config: parse_env_config(
                            step_config.environment_configs[env_config]
                        )
                    }
                    for env_config in step_config.environment_configs
                ]
            ),
            "readme": parse_markdown_str(readme_content),
        }

        if callback is not None:
            callback(
                callback_event="processed_step_content",
                callback_args={
                    "steps": 50,
                    "steps_repo": f"{step_repo}?branch={branch_name}",
                },
            )

        return step_obj
    except HTTPError as e:
        if callback is not None:
            if e.response.status_code == 404:
                error_message = f"{step_name} not found in {step_repo}"
            else:
                error_message = (
                    f"{e.response.status_code} {e.response.reason} for {step_repo}"
                )

            callback(
                callback_event="error",
                callback_args={
                    "steps": 1,
                    "reason": f"Error obtaining step content. {error_message}.",
                },
            )
        return None
    except Exception as e:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "steps": 1,
                    "reason": f"The {STEP_FILENAME} at {step_yaml_path} is invalid, error: {e}.",
                },
            )
        return None
    finally:
        if os.path.exists(step_yaml_cache):
            os.remove(step_yaml_cache)


def parse_env_config(env_config: StepEnvironmentTypeConfig) -> dict:
    env_config_obj = {
        "launchers": [
            {launcher: parse_launcher(env_config.launchers[launcher])}
            for launcher in env_config.launchers
        ],
        "monitors": [
            {monitor: parse_monitor(env_config.monitors[monitor])}
            for monitor in env_config.monitors
        ],
    }
    return env_config_obj


def parse_launcher(launcher: StepLauncherConfig) -> dict:
    launcher_obj = {
        "type": launcher.type,
        "monitors": launcher.monitors,
        "config": launcher.config,
    }

    return launcher_obj


def parse_monitor(monitor: StepMonitorConfig) -> dict:
    monitor_obj = {
        "type": monitor.type,
        "config": monitor.config,
    }

    return monitor_obj
