import logging
from typing import Any, List

from gbcli.utils.gbserver import add_space_member as _add_space_member
from gbcli.utils.gbserver import delete_space_member as _delete_space_member
from gbcli.utils.gbserver import (
    get_remote_spaces,
)
from gbcli.utils.gbserver import get_space_members as _get_space_members
from gbcli.utils.gbserver import (
    make_gbserver_call,
)
from gbcli.utils.gbserver import update_space_member as _update_space_member
from gbcli.utils.spaceutil import (
    get_profile,
    get_spaces,
    resolve_space,
    save_new_spaces,
    save_profile,
)
from gbcli.utils.utils import find_space_by_name, map_build_spaces_to_user_spaces

logger = logging.getLogger(__name__)


def list_spaces(
    github_token: str, all: bool = False, refresh: bool = False, callback=None
) -> List[Any]:
    if refresh:
        logger.info("fetching user spaces from GBSERVER")
        remote_spaces = get_remote_spaces(github_token, callback)
        save_new_spaces(remote_spaces, callback)

    if all:
        spaces = get_spaces(github_token, callback)

        spaces_list = [
            {
                "name": space.get("name"),
                "git_repo_uri": space.get("git_repo_uri"),
                "lakehouse_namespace": space.get("lakehouse_namespace"),
                "is_admin": space.get("is_admin"),
            }
            for space in spaces
        ]
    else:
        spaces = get_spaces(github_token, callback)
        profile = get_profile()

        spaces_list = map_build_spaces_to_user_spaces(spaces, profile)
    if callback is not None:
        callback(callback_event="complete", callback_args={"steps": 100})

    return spaces_list


def set_space(
    github_token: str,
    space_name: str = None,
    default_space: bool = False,
    callback=None,
    name=None,
):
    user_spaces = get_spaces(github_token, callback)

    if not user_spaces:
        raise Exception(
            f"Error: unable to find any spaces. Try running space list --all --refresh to see latest available spaces "
        )

    # check if space_name is in user's available spaces
    space = find_space_by_name(user_spaces, space_name)
    if not space:
        raise Exception(
            f"Error: unable to find space with name '{space_name}'. Try running space list --all --refresh to see latest available spaces "
        )

    # New space set command behavior- when it's time to clean up the old behavior, get rid of the above all
    if name:
        save_profile(name, space_name)
    if default_space:
        save_profile("default", space_name)
    # Note that "space set" without "--default" or "--name" makes no effect
    return


SPACE_MEMBERSHIP_ADMIN_ERROR = "You must be an admin of this space to manage members."


def _resolve_and_check_admin(github_token, space, callback):
    """Resolve the space and verify the user is an admin. Returns (space_info, space_name) or (None, None)."""
    space_info = resolve_space(github_token, space, callback)
    if space_info is None:
        return None, None
    if not space_info.get("is_admin"):
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={"reason": SPACE_MEMBERSHIP_ADMIN_ERROR},
            )
        return None, None
    return space_info, space_info["name"]


def list_space_members(github_token: str, space=None, callback=None):
    space_info, space_name = _resolve_and_check_admin(github_token, space, callback)
    if space_info is None:
        return None
    result = make_gbserver_call(
        lambda: _get_space_members(github_token, space_name), callback
    )
    return result


def add_space_member(
    github_token: str, space=None, username=None, role=None, callback=None
):
    space_info, space_name = _resolve_and_check_admin(github_token, space, callback)
    if space_info is None:
        return None
    result = make_gbserver_call(
        lambda: _add_space_member(github_token, space_name, username, role), callback
    )
    return result


def update_space_member(
    github_token: str, space=None, username=None, role=None, callback=None
):
    space_info, space_name = _resolve_and_check_admin(github_token, space, callback)
    if space_info is None:
        return None
    result = make_gbserver_call(
        lambda: _update_space_member(github_token, space_name, username, role), callback
    )
    return result


def delete_space_member(github_token: str, space=None, username=None, callback=None):
    space_info, space_name = _resolve_and_check_admin(github_token, space, callback)
    if space_info is None:
        return None
    result = make_gbserver_call(
        lambda: _delete_space_member(github_token, space_name, username), callback
    )
    return result
