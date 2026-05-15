"""
User space access control functions.

This module provides wrapper functions for space access control.
The actual implementation is delegated to the ISpaceAccessManager interface,
which allows for different implementations (e.g., storage-based, mock for testing).

For new code, consider using get_space_access_manager() directly to access
the interface, rather than these wrapper functions.
"""

from typing import Optional, Union

from fastapi.responses import JSONResponse

from gbserver.spaces.space_access_manager import (
    SpaceAccessInfo,
    get_space_access_manager,
)
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.types.constants import PUBLIC_SPACE_NAME


def user_spaces_list(username: str) -> list[dict]:
    """Get list of spaces that the user has access to.

    Args:
        username: User email address.

    Returns:
        List of dicts with keys: uuid, name, git_repo_uri, is_admin
        Always includes the "public" space (with is_admin=False) if it exists
        in storage and the user doesn't already have an explicit membership.
    """
    manager = get_space_access_manager()
    spaces = manager.get_user_spaces_with_access(username)

    # Ensure the "public" space is always present in the response.
    # If the user already has an explicit membership (admin or member),
    # that entry takes precedence and we don't add a duplicate.
    has_public = any(s.space.name == PUBLIC_SPACE_NAME for s in spaces)
    if not has_public:
        public_space = get_admin_storage().space_storage.get_by_name(PUBLIC_SPACE_NAME)
        if public_space is not None:
            spaces.append(SpaceAccessInfo(space=public_space, is_admin=False))

    # Convert SpaceAccessInfo objects to dicts
    # Flatten the StoredSpace attributes into the dict
    return [
        {
            "uuid": space.space.uuid,
            "name": space.space.name,
            "git_repo_uri": space.space.git_repo_uri,
            "is_admin": space.is_admin,
        }
        for space in spaces
    ]


def space_admin_check(
    username: str,
    space_name: str = PUBLIC_SPACE_NAME,
) -> bool:
    """Check if user is an admin of the specified space.

    Args:
        username: User email address.
        space_name: Name of the space to check. Defaults to PUBLIC_SPACE_NAME.

    Returns:
        True if user is an admin of the space, False otherwise
    """
    manager = get_space_access_manager()
    return manager.is_space_admin(username, space_name)


def space_access_check(
    username: str,
    space_name: str,
) -> bool:
    """Check if user has write access to the specified space.

    Args:
        username: User email address.
        space_name: Name of the space to check.

    Returns:
        True if user has write access to the space, False otherwise
    """
    manager = get_space_access_manager()
    return manager.has_space_access(username, space_name)


def build_id_access_check(
    username: str,
    build_id: str,
) -> Union[bool, JSONResponse]:
    """Check if user has access to the specified build.

    This checks access based on the space that the build belongs to.

    Args:
        username: User email address.
        build_id: UUID of the build to check.

    Returns:
        True if user has access, False if no access,
        or JSONResponse with 404 error if build not found
    """
    manager = get_space_access_manager()
    return manager.has_build_access(username, build_id)
