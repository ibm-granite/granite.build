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

"""Server-side resolution of HuggingFace Enterprise resource group ids.

HF Enterprise access control keys repository/bucket creation on a resource
group *id* (an internal HF identifier). There is no non-admin HF API to map a
resource group *name* to its id, so a name/space lookup requires an admin-scoped
token. To avoid depending on that token everywhere, the resolved id is cached on
the ``gb_spaces`` row for the space (``StoredSpace.resource_group_id``).

This module owns the resolution order used by every server call site:

1. Deduce/normalize the space name (caller supplies it).
2. Read the cached ``resource_group_id`` off the ``StoredSpace`` row, if any.
3. Otherwise fall back to the direct HF API lookup
   (:meth:`HfURI.resolve_resource_group_id_for_org`). On success, write the id
   back onto the space row (only when a row exists) so subsequent lookups are
   cheap and work without an admin token.

``gbcommon.uri.hf`` stays storage-agnostic: it only ever *receives* a resolved
id. The table read/write lives here.
"""

from typing import Optional

from gbcommon.uri.hf import HF_HOST, HfURI
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_space_resource_group_id(
    space_name: Optional[str],
    organization: str,
    token: Optional[str],
    resource_group_name: Optional[str] = None,
    host: str = HF_HOST,
) -> Optional[str]:
    """Resolve the HF resource group id for a space, table-first with HF fallback.

    This function deliberately does not accept an explicit ``resource_group_id``.
    Callers with a user/config-pinned id must use it verbatim and must NOT route
    it through here: the id resolved from the space is what gets cached (written
    back onto the space row), and a caller-pinned id may intentionally differ
    from the space's default group. Only names/spaces are resolved and cached.

    Args:
        space_name: GB space name. Used both to look up the cached id on the
            ``gb_spaces`` row and (via the HF fallback) to derive the resource
            group name. May be ``None`` if only ``resource_group_name`` is known,
            in which case there is no row to cache against.
        organization: HF organization namespace.
        token: HF auth token used for the fallback HF API lookup. Typically the
            server functional/admin token from ``get_hf_token()``.
        resource_group_name: Explicit resource group name, if known independently
            of the space name. Passed through to the HF fallback for cross-check.
        host: HF host (defaults to ``huggingface.co``).

    Returns:
        The resolved resource group id, or ``None`` when nothing resolves.

    Raises:
        ValueError: propagated from :meth:`HfURI.resolve_resource_group_id_for_org`
            when the provided inputs disagree.
    """
    space_storage = get_admin_storage().space_storage
    space = None
    if space_name:
        space = space_storage.get_by_name(space_name)
        if space is not None and space.resource_group_id:
            logger.info(
                "Using cached resource group id '%s' for space '%s'",
                space.resource_group_id,
                space_name,
            )
            return space.resource_group_id

    # Fallback: query the HF API (requires an admin-scoped token).
    resolved_id = HfURI.resolve_resource_group_id_for_org(
        token=token,
        organization=organization,
        resource_group_name=resource_group_name,
        space_name=space_name,
        host=host,
    )

    # Write the resolved id back onto the space row so future lookups are cheap
    # and don't require an admin token. Only cache when a row exists; we never
    # create a space row here.
    if resolved_id and space is not None:
        space.resource_group_id = resolved_id
        space_storage.update(space)
        logger.info(
            "Cached resource group id '%s' onto space '%s'",
            resolved_id,
            space_name,
        )

    return resolved_id
