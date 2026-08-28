#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Abstract interface for space access control management.

This module defines the interface for checking user permissions on spaces,
allowing for multiple implementations (e.g., storage-based, mock for testing).
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

from fastapi.responses import JSONResponse
from pydantic import BaseModel

from gbserver.storage.stored_space import StoredSpace


class SpaceAccessInfo(BaseModel):
    """Information about a space that a user has access to.

    Attributes:
        space: The StoredSpace object containing space details
        is_admin: Whether the user has admin privileges on this space
    """

    space: StoredSpace
    is_admin: bool


class ISpaceAccessManager(ABC):
    """Abstract interface for managing space access control.

    This interface defines the contract for checking user permissions on spaces.
    Implementations should provide concrete logic for determining access based on
    their specific authorization backend.
    """

    @abstractmethod
    def get_user_spaces_with_access(self, username: str) -> list[SpaceAccessInfo]:
        """Get list of spaces that the user has access to.

        Used by scope_space_name_filter() (api/utils.py) to authorize every
        list/count/tags endpoint -- implementations must propagate storage
        errors rather than swallowing them into an empty list. A caller-side
        empty accessible-space set is treated as "deny all"; masking a real
        outage as that would silently convert a 5xx into a wrong, misleading
        "no access" result instead of a visible failure.

        Must also include the public space in the result whenever a
        StoredSpace row for it exists, even if the user has no explicit
        membership row for it (with is_admin=False unless they hold a real
        admin membership) -- callers (e.g. user_spaces_list()) rely on this
        and do not re-check it themselves.

        Args:
            username: User email address

        Returns:
            List of SpaceAccessInfo objects for spaces the user can access,
            with is_admin flag indicating admin status for each space. Empty
            list if the user has no accessible spaces (including no public
            space row) -- NOT on error; see above.

        Raises:
            Exception: propagates uncaught storage/DB errors.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_user_spaces_with_access()"
        )

    @abstractmethod
    def is_space_admin(self, username: str, space_name: str) -> bool:
        """Check if user is an admin of the specified space.

        Args:
            username: User email address
            space_name: Name of the space to check

        Returns:
            True if user is an admin of the space, False otherwise
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement is_space_admin()"
        )

    @abstractmethod
    def has_space_access(self, username: str, space_name: str) -> bool:
        """Check if user has access (any role, not just write/admin) to the
        specified space. Implementations should grant the public space
        unconditionally, even without a membership row -- this is a
        single-object convenience check, deliberately looser than
        get_user_spaces_with_access()'s row-gated listing contract above; see
        StorageSpaceAccessManager.has_space_access()'s docstring for why the
        two are allowed to diverge.

        Args:
            username: User email address
            space_name: Name of the space to check

        Returns:
            True if user has any access to the space, False otherwise
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement has_space_access()"
        )

    @abstractmethod
    def has_build_access(
        self, username: str, build_id: str
    ) -> Union[bool, JSONResponse]:
        """Check if user has access to the specified build.

        This checks access based on the space that the build belongs to.

        Args:
            username: User email address
            build_id: UUID of the build to check

        Returns:
            True if user has access, False if no access,
            or JSONResponse with 404 error if build not found
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement has_build_access()"
        )


# Instance management: _override_manager is set by set_space_access_manager()
# for testing / standalone mode; _storage_manager caches the StorageSpaceAccessManager.
_override_manager: Optional[ISpaceAccessManager] = None
_storage_manager: Optional["StorageSpaceAccessManager"] = None  # type: ignore[name-defined]


def get_space_access_manager() -> ISpaceAccessManager:
    """Get a space access manager instance.

    When an override has been set via :func:`set_space_access_manager` (e.g.
    for testing or standalone mode), that instance is returned unconditionally.

    Otherwise a cached :class:`StorageSpaceAccessManager` singleton is returned.

    Returns:
        An ISpaceAccessManager instance
    """
    global _storage_manager

    if _override_manager is not None:
        return _override_manager

    if _storage_manager is None:
        from gbserver.spaces.storage_space_access_manager import (
            StorageSpaceAccessManager,
        )

        _storage_manager = StorageSpaceAccessManager()
    return _storage_manager


def set_space_access_manager(manager: ISpaceAccessManager) -> None:
    """Set the global space access manager override.

    This is primarily used for testing to inject mock implementations,
    or by standalone mode to use :class:`StandaloneSpaceAccessManager`.

    Pass ``None`` to clear the override and revert to feature-flag-based
    selection.

    Args:
        manager: The ISpaceAccessManager instance to use globally
    """
    global _override_manager, _storage_manager
    _override_manager = manager
    _storage_manager = None
