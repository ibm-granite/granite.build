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

from typing import TYPE_CHECKING, Optional, Tuple

from gbcommon.types.gbenvconfig import parse_boolean
from gbcommon.uri.hf import HF_HOST, HfURI
from gbcommon.utils.hf_utils import is_enterprise_hf_org
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from gbserver.asset.hfstore import Hfstore
    from gbserver.types.buildconfig import BuildTargetOutputConfig
    from gbserver.types.environmentconfig import StorePush

logger = get_logger(__name__)

# Key in a store_push ``config.hf`` block that opts an Enterprise org out of
# resource groups. Consumed here and stripped before the config reaches a
# worker step template.
USE_RESOURCE_GROUP_KEY = "use_resource_group"


class HfPushConfigError(ValueError):
    """A push config asks for something the target organization cannot honor.

    Raised for a *configuration* mistake that resolution cannot work around: a
    resource group pinned for a non-Enterprise org, or a pin combined with
    ``use_resource_group: false`` at the same level. Distinct from a resolution
    *miss* (a non-admin token that cannot read the org's resource groups), which
    is expected on the standalone path and must not abort a best-effort push.

    Subclasses ``ValueError`` so existing ``except ValueError`` callers keep
    working; callers that need to tell the two apart catch this type instead.
    """


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


def _non_enterprise_rg_error(organization: str, pinned: str) -> str:
    """Build the error message for a resource group pinned on a non-Enterprise org."""
    return (
        f"Resource group '{pinned}' was configured for HuggingFace organization "
        f"'{organization}', but '{organization}' is not an HF Enterprise "
        "organization. Resource groups apply only to Enterprise organizations. "
        "Remove store_push.config.hf.resource_group_id / resource_group_name, "
        f"or add '{organization}' to enterprise_organizations in the hf asset "
        "store's store.yaml."
    )


def _merge_hf_levels(levels: Tuple[dict, dict]) -> dict:
    """Merge already-parsed ``hf`` config levels, lowest priority first."""
    merged: dict = {}
    for level in levels:
        # A yaml null means "not set here", so it must not erase a lower level.
        merged.update({k: v for k, v in level.items() if v is not None})
    return merged


def _hf_push_config_levels(
    storepush_config: Optional["StorePush"] = None,
    output_config: Optional["BuildTargetOutputConfig"] = None,
) -> Tuple[dict, dict]:
    """Return the ``hf`` config from each level separately, lowest priority first.

    ``(environment_level, output_level)``. Kept distinct from the merged view so
    a per-output setting can be told apart from one inherited from the
    environment — see the ``use_resource_group`` handling in
    :func:`resolve_hfpush_resource_group_id`.
    """

    def _hf(config: Optional[dict]) -> dict:
        hf_cfg = (config or {}).get("hf") or {}
        return hf_cfg if isinstance(hf_cfg, dict) else {}

    env_level = _hf(storepush_config.config) if storepush_config is not None else {}
    output_level = (
        _hf(output_config.store_push.config)
        if output_config is not None and output_config.store_push is not None
        else {}
    )
    return env_level, output_level


def _private_from_hf_cfg(hf_cfg: dict) -> bool:
    """Apply the ``private`` default to an already-merged ``hf`` config block.

    The single definition of the rule, shared by
    :func:`resolve_hfpush_resource_group_id` (which has the merged block in hand)
    and :func:`resolve_hfpush_private` (which merges it first).
    """
    return parse_boolean(hf_cfg.get("private"), True)


def _level_pin(level: dict) -> Optional[str]:
    """Return the resource group pinned at one config level, if any."""
    return level.get("resource_group_id") or level.get("resource_group_name") or None


def resolve_hfpush_private(
    storepush_config: Optional["StorePush"] = None,
    output_config: Optional["BuildTargetOutputConfig"] = None,
) -> bool:
    """Resolve the ``private`` flag for an HF push from the merged push config.

    Artifacts are private by default: HuggingFace's own ``create_repo`` default is
    PUBLIC, so an unset/omitted value must resolve to ``True`` here to keep a user
    from unintentionally publishing a model. Only an explicit falsy value
    (``false``/``no``/``off``/``0``, quoted or not) opts into a public repo.

    Split out of :func:`resolve_hfpush_resource_group_id` so a caller that cannot
    classify the org (no ``Hfstore``, hence no Enterprise org list) can still honor
    the flag without attempting resource group resolution.

    Args:
        storepush_config: Environment-level ``store_push`` (environment.yaml).
        output_config: Per-output config whose ``store_push`` (build.yaml)
            overrides the environment level.

    Returns:
        ``True`` for a private repo (the default), ``False`` only when explicitly
        configured public.
    """
    hf_cfg = _merge_hf_levels(_hf_push_config_levels(storepush_config, output_config))
    return _private_from_hf_cfg(hf_cfg)


def sanitize_hf_step_overlay(hf_cfg: dict) -> dict:
    """Drop keys that must never reach a worker step's ``hfpush_config``.

    ``use_resource_group`` is consumed during resolution (it opts an Enterprise
    org out of resource groups); leaking it verbatim into the emitted step
    config would hand the LSF/Helm/SkyPilot templates a key they do not
    understand.

    Args:
        hf_cfg: An ``hf`` config dict from a push configuration.

    Returns:
        A copy without the resolution-only keys.
    """
    return {k: v for k, v in (hf_cfg or {}).items() if k != USE_RESOURCE_GROUP_KEY}


def apply_hf_step_overlay(
    hfpush_config: dict, hf_cfg: dict, resource_group_id: Optional[str]
) -> None:
    """Overlay the raw ``hf`` push config onto a built step config, in place.

    Shared by the k8s and skypilot launchers, which build an ``hfpush_config``
    with :meth:`Hfstore.build_hfpush_step_config` and then fold the remaining
    ``hf`` keys from the merged push config over it. Two invariants the callers
    must not get subtly wrong, kept here so they cannot drift between the two
    environments:

    - ``use_resource_group`` is stripped (:func:`sanitize_hf_step_overlay`); it
      is consumed during resolution, not by the worker template.
    - the resolved ``resource_group_id`` is re-asserted *after* the overlay, so
      a stray pinned-but-skipped id in the raw config cannot be resurrected.

    Args:
        hfpush_config: The step config dict; its ``hf`` sub-dict is mutated.
        hf_cfg: The raw merged ``hf`` push config to overlay.
        resource_group_id: The resolved id (or ``None``) to re-assert last.
    """
    hfpush_config["hf"].update(sanitize_hf_step_overlay(hf_cfg))
    hfpush_config["hf"]["resource_group_id"] = resource_group_id


def resolve_hfpush_resource_group_id(
    hfuri: HfURI,
    assetstore: "Hfstore",
    space_name: Optional[str],
    storepush_config: Optional["StorePush"] = None,
    output_config: Optional["BuildTargetOutputConfig"] = None,
) -> Tuple[Optional[str], bool, dict]:
    """Resolve the HF resource group id for a push, honoring the Enterprise split.

    Resource groups exist only in HF Enterprise organizations. Which orgs are
    Enterprise is configuration-driven (``enterprise_organizations`` in the hf
    asset store's ``store.yaml``) because no non-admin HF API distinguishes an
    Enterprise org from an individual user namespace.

    For a non-Enterprise org this skips resource group resolution entirely: no
    HF API call, no space lookup, and nothing cached.

    Args:
        hfuri: Target HuggingFace URI; its owner is the organization.
        assetstore: The ``Hfstore`` supplying the token and the Enterprise list.
        space_name: GB space name used to derive the default resource group.
        storepush_config: Environment-level push configuration (lower priority).
        output_config: Per-output build.yaml configuration (higher priority).

    Returns:
        A ``(resource_group_id, private, hf_config)`` tuple, where
        ``resource_group_id`` is ``None`` when no resource group applies and
        ``hf_config`` is the merged ``hf`` settings for logging/overlay use.

    Raises:
        ValueError: If a resource group is pinned for a non-Enterprise org, or
            if ``use_resource_group: false`` is combined with a pinned group.
    """
    levels = _hf_push_config_levels(storepush_config, output_config)
    env_level, output_level = levels
    hf_cfg = _merge_hf_levels(levels)
    resource_group_id = hf_cfg.get("resource_group_id") or None
    resource_group_name = hf_cfg.get("resource_group_name") or None
    # parse_boolean, not .get(key, default): a yaml null is a *present* key, so
    # .get's default would not apply and a None reaching a worker template
    # stringifies as "None". parse_boolean also folds the quoted forms
    # ("false"/"no"/"off"/"0") onto False, so `private: "false"` means what it
    # says instead of being truthy as a non-empty string.
    private = _private_from_hf_cfg(hf_cfg)
    use_resource_group = parse_boolean(hf_cfg.get(USE_RESOURCE_GROUP_KEY), True)

    organization = hfuri.get_owner()
    enterprise = is_enterprise_hf_org(
        organization, assetstore.get_enterprise_organizations()
    )
    pinned = resource_group_id or resource_group_name

    if not enterprise:
        if pinned:
            raise HfPushConfigError(_non_enterprise_rg_error(organization, pinned))
        logger.info(
            "HuggingFace organization '%s' is not an Enterprise org; "
            "skipping resource group resolution",
            organization,
        )
        return None, private, hf_cfg

    if not use_resource_group:
        # Same-level opt-out plus pin is contradictory; across levels the higher
        # one wins, per the precedence in docs/builds/hf-push.md.
        for level in (output_level, env_level):
            if not parse_boolean(
                level.get(USE_RESOURCE_GROUP_KEY), True
            ) and _level_pin(level):
                raise HfPushConfigError(
                    f"'{USE_RESOURCE_GROUP_KEY}: false' cannot be combined with "
                    f"an explicit resource group ('{_level_pin(level)}') in the "
                    f"same push config for organization '{organization}'. Remove "
                    "one of them."
                )
        output_pin = _level_pin(output_level)
        if output_pin and parse_boolean(output_level.get(USE_RESOURCE_GROUP_KEY), True):
            # build.yaml outranks environment.yaml, so a pin here re-enables
            # resource groups over an inherited opt-out.
            logger.info(
                "output-level resource group '%s' overrides the inherited "
                "'%s: false' for organization '%s'",
                output_pin,
                USE_RESOURCE_GROUP_KEY,
                organization,
            )
        else:
            if pinned:
                logger.info(
                    "'%s: false' overrides the inherited resource group '%s' for "
                    "organization '%s'",
                    USE_RESOURCE_GROUP_KEY,
                    pinned,
                    organization,
                )
            logger.info(
                "'%s: false' configured for organization '%s'; pushing without a "
                "resource group",
                USE_RESOURCE_GROUP_KEY,
                organization,
            )
            return None, private, hf_cfg

    if resource_group_id:
        # A caller-pinned id is used verbatim and never routed through
        # resolve_space_resource_group_id, whose cache represents only the
        # space's default group.
        return resource_group_id, private, hf_cfg

    resolved_id = resolve_space_resource_group_id(
        space_name=space_name,
        organization=organization,
        token=assetstore.resolve_token(hfuri),
        resource_group_name=resource_group_name,
        host=hfuri.get_host(),
    )
    return resolved_id, private, hf_cfg
