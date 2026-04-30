"""Service tag module."""

import logging
from typing import Literal, Optional

from gbcli.utils.gbconstants import GBSERVER_ARTIFACT_API, GBSERVER_BUILD_API
from gbcli.utils.gbserver import gb_server_request
from gbcli.utils.spaceutil import resolve_space

logger = logging.getLogger(__name__)


def get_tags(
    github_token: str,
    resource_type: Literal["artifacts", "builds"],
    username: Optional[str] = None,
    space: Optional[str] = None,
    callback=None,
) -> list[str]:
    """
    Get list of unique tags for a resource type (artifacts or builds).

    Args:
        github_token: GitHub user token for authentication
        resource_type: Either "artifacts" or "builds"
        username: Filter by username
        space: Filter by space name
        callback: Optional callback for logging/events

    Returns:
        List of tag strings as returned by the API
    """
    # Resolve space
    space_name = None
    s = resolve_space(github_token, space, callback)
    if s is None:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={"reason": f"Space {space} not found in available spaces."},
            )
        raise Exception(f"Space {space} not found in available spaces.")
    space_name = s.get("name")

    # Select the appropriate API endpoint based on resource type
    if resource_type == "artifacts":
        api_url = f"{GBSERVER_ARTIFACT_API}tags"
    elif resource_type == "builds":
        api_url = f"{GBSERVER_BUILD_API}tags"
    else:
        raise ValueError(f"Invalid resource_type: {resource_type}. Must be 'artifacts' or 'builds'")

    # Build parameters, filtering out None values
    params = {}
    if username and username != "default":
        params["username"] = username
    if space_name:
        params["space_name"] = space_name

    # Make API call
    try:
        response = gb_server_request(
            user_token=github_token,
            url=api_url,
            http_method="get",
            body=None,
            params=params if params else None,
        )

        # Handle response - should be a list of tags
        if isinstance(response, list):
            return response
        elif isinstance(response, dict) and "tags" in response:
            return response.get("tags", [])
        else:
            return response if response else []
    except Exception as e:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={"reason": str(e)},
            )
        raise


def artifact_tag_list(
    github_token: str,
    username: Optional[str] = None,
    space: Optional[str] = None,
    callback=None,
) -> list[str]:
    """
    Get list of unique tags for artifacts.

    Args:
        github_token: GitHub user token for authentication
        username: Filter by username
        space: Filter by space name
        callback: Optional callback for logging/events

    Returns:
        List of tag strings as returned by the API
    """
    return get_tags(github_token, "artifacts", username, space, callback)


def build_tag_list(
    github_token: str,
    username: Optional[str] = None,
    space: Optional[str] = None,
    callback=None,
) -> list[str]:
    """
    Get list of unique tags for builds.

    Args:
        github_token: GitHub user token for authentication
        username: Filter by username
        space: Filter by space name
        callback: Optional callback for logging/events

    Returns:
        List of tag strings as returned by the API
    """
    return get_tags(github_token, "builds", username, space, callback)
