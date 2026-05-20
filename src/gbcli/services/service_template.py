import logging
import os
from typing import Any, List, Optional

from requests import HTTPError

from gbcli.services.service_build import describe_build_yaml
from gbcli.utils.gbconstants import (
    ASSETS_REPO_URL,
    BUILD_FILENAME,
    TEMPLATES_REPO_FOLDER,
    gb_environment_config,
)
from gbcli.utils.gh_clone import download_repo_file, list_repo_tree
from gbcli.utils.spaceutil import resolve_space
from gbcli.utils.utils import remove_suffix
from gbcommon.types.constants import DEFAULT_GH_DOMAIN

logger = logging.getLogger(__name__)


def list_templates(
    github_token: str,
    space: Optional[str] = None,
    template_repo: Optional[str] = None,
    callback=None,
) -> List[Any]:
    if space:
        global_space = resolve_space(github_token, space, callback)
        if not global_space:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={
                        "steps": 1,
                        "reason": f"Space {space} not found in available spaces.",
                    },
                )
            return None
        template_repo = global_space.get("git_repo_uri")
        branch_name = gb_environment_config()["branch_space"]
    elif template_repo != None:
        branch_name = gb_environment_config()["branch_space"]
        if "@" in template_repo:
            template_repo_split = template_repo.split("@", 1)
            template_repo = template_repo_split[0]
            if template_repo_split[1] != "":
                branch_name = template_repo_split[1]
    else:
        template_repo = ASSETS_REPO_URL
        branch_name = gb_environment_config()["branch_assets"]

    if callback is not None:
        callback(
            callback_event="listing_templates",
            callback_args={"steps": 1, "assets_repo": template_repo},
        )

    assets_org, assets_name = template_repo.split("/")[3:]
    assets_name = remove_suffix(assets_name, ".git")

    repo_tree = list_repo_tree(github_token, assets_org, assets_name, branch_name)

    if callback is not None:
        callback(
            callback_event="listing_templates",
            callback_args={"steps": 70, "assets_repo": template_repo},
        )

    templates = []
    for file in repo_tree["tree"]:
        file_path = str(file["path"]).split("/")
        if (
            str(file["path"]).startswith(f"{TEMPLATES_REPO_FOLDER}/")
            and file["type"] == "tree"
            and len(file_path) == 2
        ):
            templates.append(
                {
                    "template_name": file_path[1],
                    "description": f"https://{DEFAULT_GH_DOMAIN}/{assets_org}/{assets_name}/tree/{branch_name}/{file['path']}",
                }
            )

    if callback is not None:
        callback(
            callback_event="listed_templates",
            callback_args={"steps": 100, "assets_repo": template_repo},
        )

    return templates


def describe_template(
    github_token: str,
    template_name: str,
    format: str,
    space: Optional[str] = None,
    template_repo: Optional[str] = None,
    callback=None,
) -> List[Any]:
    if space:
        global_space = resolve_space(github_token, space, callback)
        if not global_space:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={
                        "steps": 1,
                        "reason": f"Space {space} not found in available spaces.",
                    },
                )
            return None
        template_repo = global_space.get("git_repo_uri")
        branch_name = gb_environment_config()["branch_space"]
    elif template_repo != None:
        branch_name = gb_environment_config()["branch_space"]
        if "@" in template_repo:
            template_repo_split = template_repo.split("@", 1)
            template_repo = template_repo_split[0]
            if template_repo_split[1] != "":
                branch_name = template_repo_split[1]
    else:
        template_repo = ASSETS_REPO_URL
        branch_name = gb_environment_config()["branch_assets"]

    assets_org, assets_name = template_repo.split("/")[3:]
    assets_name = remove_suffix(assets_name, ".git")

    template_yaml_path = (
        f"{TEMPLATES_REPO_FOLDER}/{template_name}/{BUILD_FILENAME}?ref={branch_name}"
    )
    template_yaml_cache = f"describe_{template_name}_{BUILD_FILENAME}"

    if callback is not None:
        callback(
            callback_event="processing_template_content",
            callback_args={
                "steps": 1,
                "assets_repo": f"{remove_suffix(template_repo, '.git')}?branch={branch_name}",
            },
        )

    try:
        content = download_repo_file(
            github_token, assets_org, assets_name, template_yaml_path
        )

        if callback is not None:
            callback(
                callback_event="processing_template_content",
                callback_args={
                    "steps": 45,
                    "assets_repo": f"{remove_suffix(template_repo, '.git')}?branch={branch_name}",
                },
            )

        with open(template_yaml_cache, "w", encoding="utf-8") as cache_file:
            cache_file.write(content)

        targets = describe_build_yaml(template_yaml_cache, template_yaml_cache, format)

        if callback is not None:
            callback(
                callback_event="processed_template_content",
                callback_args={
                    "steps": 54,
                    "assets_repo": f"{remove_suffix(template_repo, '.git')}?branch={branch_name}",
                },
            )

        return targets
    except HTTPError as e:
        if callback is not None:
            if e.response.status_code == 404:
                error_message = f"{template_name} not found in {template_repo}"
            else:
                error_message = (
                    f"{e.response.status_code} {e.response.reason} for {template_repo}"
                )

            callback(
                callback_event="error",
                callback_args={
                    "steps": 1,
                    "reason": f"Error obtaining template content. {error_message}.",
                },
            )
        return None
    except Exception as e:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "steps": 1,
                    "reason": f"The {BUILD_FILENAME} at {template_yaml_path} is invalid, error: {e}.",
                },
            )
        return None
    finally:
        if os.path.exists(template_yaml_cache):
            os.remove(template_yaml_cache)
