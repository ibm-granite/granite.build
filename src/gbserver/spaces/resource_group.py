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
token. To avoid depending on that token everywhere, the id of the space's
*default* resource group is cached on the ``gb_spaces`` row
(``StoredSpace.hf_default_resource_group_id``).

The cache holds ONLY the space's default group — the ``gbspace-<space>`` group
derived from the space name by :meth:`HfURI.space_name_to_resource_group_name`.
A request that names a *different* (non-default) group must never read or write
the cache, or it would silently receive/poison the default id. So this module:

1. Computes the space-derived default resource group name from ``space_name``.
2. Decides whether the request targets that default (no explicit
   ``resource_group_name``, or one equal to the derived default name).
3. Default request: read the cached id off the ``StoredSpace`` row; on a miss,
   fall back to the HF API and write the resolved id back (only when a row
   exists) so later default lookups are cheap and need no admin token.
4. Non-default request: bypass the cache entirely — resolve via the HF API
   (which cross-checks name vs. id and raises on mismatch) and never write back.

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
        space_name: GB space name. Used to look up the cached default-group id on
            the ``gb_spaces`` row and (via the HF fallback) to derive the resource
            group name. May be ``None`` if only ``resource_group_name`` is known,
            in which case there is no row to cache against.
        organization: HF organization namespace.
        token: HF auth token used for the fallback HF API lookup. Typically the
            server functional/admin token from ``get_hf_token()``.
        resource_group_name: Explicit resource group name, if the caller wants a
            specific group. When it differs from the space's derived default name,
            the cache is bypassed and the id is resolved (and cross-checked) via
            the HF API without being cached.
        host: HF host (defaults to ``huggingface.co``).

    Returns:
        The resolved resource group id, or ``None`` when nothing resolves.

    Raises:
        ValueError: propagated from :meth:`HfURI.resolve_resource_group_id_for_org`
            when the provided inputs disagree (e.g. an explicit
            ``resource_group_name`` whose resolved id contradicts a supplied id).
    """
    # The cache represents ONLY the space's default group. A request targets the
    # default when it supplies no explicit name, or a name equal to the derived
    # default; otherwise the cache must not be consulted or updated.
    derived_default_name = (
        HfURI.space_name_to_resource_group_name(space_name) if space_name else None
    )
    is_default_request = (
        not resource_group_name or resource_group_name == derived_default_name
    )

    space_storage = get_admin_storage().space_storage
    space = None
    if space_name and is_default_request:
        space = space_storage.get_by_name(space_name)
        if space is not None and space.hf_default_resource_group_id:
            logger.info(
                "Using cached default resource group id '%s' for space '%s'",
                space.hf_default_resource_group_id,
                space_name,
            )
            return space.hf_default_resource_group_id

    # Fallback: query the HF API (requires an admin-scoped token). For a
    # non-default name this also cross-checks the name and raises on mismatch.
    resolved_id = HfURI.resolve_resource_group_id_for_org(
        token=token,
        organization=organization,
        resource_group_name=resource_group_name,
        space_name=space_name,
        host=host,
    )

    # Write back only the DEFAULT group's id, and only when a row exists. Never
    # cache a non-default group's id (it would be served for later default
    # lookups) and never create a space row here.
    if resolved_id and space is not None and is_default_request:
        space.hf_default_resource_group_id = resolved_id
        space_storage.update(space)
        logger.info(
            "Cached default resource group id '%s' onto space '%s'",
            resolved_id,
            space_name,
        )

    return resolved_id
