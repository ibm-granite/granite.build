import logging
import os
import shutil

from gbcli.utils.cli_config import (
    get_local_build_cache,
    get_local_gb_config,
)
from gbcli.utils.gh_auth import get_user
from gbcli.utils.gh_clone import get_forks
from gbcli.utils.spaceutil import resolve_space
from gbcli.utils.utils import remove_suffix

logger = logging.getLogger(__name__)


def remove_config():
    """
    locate config - remove if it exists
    """
    config_path = os.path.abspath(os.path.join(get_local_gb_config(), "config"))
    if os.path.exists(config_path):
        os.remove(config_path)
        return str(config_path)
    else:
        return str(config_path)


def remove_credentials():
    """
    locate credential - remove if it exists
    """
    credentials_path = os.path.abspath(
        os.path.join(get_local_gb_config(), "credentials")
    )
    if os.path.exists(credentials_path):
        os.remove(credentials_path)
        return str(credentials_path)
    else:
        return str(credentials_path)


def remove_local_cache():
    """
    locate local cache - remove if it exists
    """

    cache_path = get_local_build_cache()
    if cache_path.is_dir():
        # future TODO: do not ignore errors, handle and communicate to the user
        shutil.rmtree(cache_path, ignore_errors=True)
        return str(cache_path)
    else:
        # return f"Error: Local cache at '{cache_path}' does not exist or is not a folder"
        # It's okay to treat the case when the path doesn't exist as a success
        return str(cache_path)


def remove_user_fork_from_default(github_token: str, callback=None):
    """
    find user repository forked from default space
    """
    user_name = get_user(github_token).login
    global_space = resolve_space(github_token, "default", callback=callback)
    space_repo = global_space.get("git_repo_uri")
    if not space_repo:
        return f"Error: Space 'default' not found in available spaces."

    space_org, space_name = space_repo.split("/")[3:]
    space_name = remove_suffix(space_name, ".git")

    fork_name = None
    space_forks, next_page = get_forks(github_token, space_org, space_name)
    while next_page != None and next_page.get("url") != None:
        next_page_forks, next_page = get_forks(
            github_token,
            space_org,
            space_name,
            next_page_url=next_page.get("url"),
        )
        space_forks = space_forks + next_page_forks

    for fork in space_forks:
        if fork["owner"]["login"] == user_name:
            fork_name = fork["full_name"]

    if not fork_name:
        return f"Error: No forks found for default space '{space_name}' that were created by you"

    return fork_name
