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

"""Server-side resolution of a HuggingFace push's configuration.

The single home for turning the ``public``/``store_push`` config written in
build.yaml and store.yaml/environment.yaml into the concrete values a push needs:

- the public→private flip (:func:`_private_from_hf_cfg`) — the one boundary
  between granite.build's surface ``public`` and the HF-API ``private``;
- the cross-level config merge and the load-time output guard
  (:func:`validate_output_push`);
- Enterprise resource group id resolution (below).

## Resource group resolution

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
from gbcommon.uri.hf import HF_HOST, HF_URI_SCHEME, HfURI
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

# The granite.build surface key at both exposed interfaces (build.yaml and
# store.yaml/environment.yaml). Flipped to the HF-API ``private`` at one place,
# :func:`_private_from_hf_cfg`; everything below that boundary speaks ``private``.
PUBLIC_KEY = "public"


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


def _extract_hf(config: Optional[dict], top_public: object = None) -> dict:
    """The ``hf`` block for one config level, with a top-level ``public`` folded in.

    ``public`` may be written two ways: the ergonomic output-level top ``public``,
    or the store-namespaced ``config.hf.public`` (the only form at the environment
    level, where there is no output field). ``top_public`` carries the output's
    top-level field (``None`` at the environment level); it is folded into the
    returned ``hf`` block so downstream reads one location. Setting both on the
    same output with conflicting values raises; equal values collapse. A yaml null
    is "unset" on either form. The retired ``private`` key raises (see
    :func:`_reject_retired_private`).
    """
    config = config or {}
    hf_cfg = config.get("hf") or {}
    hf_cfg = dict(hf_cfg) if isinstance(hf_cfg, dict) else {}
    _reject_retired_private(hf_cfg)
    hf_public = hf_cfg.get(PUBLIC_KEY)
    if (
        hf_public is not None
        and top_public is not None
        and _differ(hf_public, top_public)
    ):
        raise HfPushConfigError(
            f"conflicting public settings for one push "
            f"('config.hf.{PUBLIC_KEY}: {hf_public}' vs "
            f"'public: {top_public}') — keep one."
        )
    if hf_public is None and top_public is not None:
        hf_cfg[PUBLIC_KEY] = top_public
    return hf_cfg


def _reject_retired_private(hf_cfg: dict) -> None:
    """Reject the retired ``private`` key with a message pointing to ``public``.

    ``private`` was replaced by ``public`` (inverted, default False). An old
    ``config.hf.private`` reaching here would otherwise be ignored and silently
    make the repo private, so fail loudly instead. Raised at load time via
    :func:`validate_output_push` and defensively at resolve time.
    """
    if "private" in hf_cfg:
        raise HfPushConfigError(
            "`store_push.config.hf.private` is no longer supported; use `public` "
            f"(inverted, default False) instead — e.g. `private: false` → "
            f"`public: true`. Got `private: {hf_cfg['private']}`."
        )


def _hf_push_config_levels(
    storepush_config: Optional["StorePush"] = None,
    output_config: Optional["BuildTargetOutputConfig"] = None,
) -> Tuple[dict, dict]:
    """Return the ``hf`` config from each level separately, lowest priority first.

    ``(environment_level, output_level)``. Kept distinct from the merged view so
    a per-output setting can be told apart from one inherited from the
    environment — see ``use_resource_group`` in
    :func:`resolve_hfpush_resource_group_id`.
    """
    env_level = (
        _extract_hf(storepush_config.config) if storepush_config is not None else {}
    )
    output_level = {}
    if output_config is not None:
        push = output_config.store_push
        output_level = _extract_hf(
            push.config if push is not None else None,
            top_public=output_config.public,
        )
    return env_level, output_level


def _differ(a: object, b: object) -> bool:
    """Whether two ``public`` values disagree in meaning (strict, like the flip)."""
    return _is_public(a) != _is_public(b)


def _is_public(value: object) -> bool:
    """Resolve a surface ``public`` value, failing closed (a typo → private)."""
    return parse_boolean(value, strict=True)


def _private_from_hf_cfg(hf_cfg: dict) -> bool:
    """Flip the surface ``public`` flag to the internal ``private`` bool.

    THE public→private boundary: the one place granite.build's ``public`` (default
    False) becomes the HF-API ``private`` (default True) used everywhere below.
    Fails closed via :func:`_is_public` — an unset/null/typo value stays private,
    so no ``.get(default)``/``hasKey`` gymnastics are needed (unlike the old
    ``private`` key, whose safe default was True).
    """
    return not _is_public(hf_cfg.get(PUBLIC_KEY))


def _uri_scheme(uri: Optional[str]) -> Optional[str]:
    """Scheme of a (possibly Jinja-templated) URI by prefix, or ``None``.

    An output ``uri`` is a template at load time (``hf:///org/repo-{{ x }}``), so
    split on ``://`` rather than parsing — both ``hf://`` and ``hf:///`` match.
    """
    if not uri or "://" not in uri:
        return None
    return uri.split("://", 1)[0].lower() or None


def validate_output_push(
    output_name: str, output_config: "BuildTargetOutputConfig"
) -> Optional[str]:
    """Validate an output's HuggingFace push config at load time; else ``None``.

    Fails fast with the output named on:

    - non-``hf://`` guard: ``public`` and any ``store_push.config.hf.*`` key are
      HuggingFace-only (no other store reads ``store_push``), so on an ``lh://``/
      ``env://``/``file://``/``cos://`` output they are a misconfiguration that
      would otherwise be silently ignored.
    - the retired ``config.hf.private`` key, and a same-level ``public`` conflict:
      both caught by folding the forms via :func:`_hf_push_config_levels` (the
      single place those rules live).

    Returns an error string for the generic build validator to collect. Kept here
    so ``buildconfig`` stays generic.
    """
    labels = []
    if getattr(output_config, PUBLIC_KEY, None) is not None:
        labels.append(f"`{PUBLIC_KEY}`")
    push = output_config.store_push
    cfg = push.config if push is not None else None
    if isinstance(cfg, dict) and isinstance(cfg.get("hf"), dict) and cfg["hf"]:
        labels.append("`store_push.config.hf.*`")
    if not labels:
        return None
    if _uri_scheme(output_config.uri) != HF_URI_SCHEME:
        return (
            f"Output `{output_name}`: {', '.join(labels)} is a HuggingFace push "
            f"option, only valid on an hf:// output; got uri '{output_config.uri}'."
        )
    try:
        _hf_push_config_levels(output_config=output_config)
    except HfPushConfigError as e:
        return f"Output `{output_name}`: {e}"
    return None


def _level_pin(level: dict) -> Optional[str]:
    """Return the resource group pinned at one config level, if any."""
    return level.get("resource_group_id") or level.get("resource_group_name") or None


def resolve_hfpush_private(
    storepush_config: Optional["StorePush"] = None,
    output_config: Optional["BuildTargetOutputConfig"] = None,
) -> bool:
    """Resolve the internal ``private`` flag for an HF push from the push config.

    Artifacts are private by default. The surface flag is ``public`` (default
    False); this returns the flipped internal ``private`` via the
    :func:`_private_from_hf_cfg` boundary, so an unset/omitted ``public`` yields a
    private repo and only an explicit truthy ``public`` (``true``/``yes``/``1``,
    quoted or not) opts into a public one. HuggingFace's own ``create_repo``
    defaults to PUBLIC, which is exactly why the safe granite.build default is
    ``public: false``.

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


_RESOLUTION_ONLY_HF_KEYS = frozenset({USE_RESOURCE_GROUP_KEY, PUBLIC_KEY})


def sanitize_hf_step_overlay(hf_cfg: dict) -> dict:
    """Drop keys that must never reach a worker step's ``hfpush_config``.

    Both keys are consumed during *resolution*, not by the worker template, so
    leaking them verbatim into the emitted step config would hand the
    LSF/Helm/SkyPilot templates keys they do not understand:

    - ``use_resource_group`` opts an Enterprise org out of resource groups.
    - ``public`` is the surface flag flipped to the flat ``private`` step key by
      :func:`_private_from_hf_cfg`; the templates read ``hfpush_config.private``,
      never ``hf.public``, so a leftover ``hf.public`` would be dead weight.

    Args:
        hf_cfg: An ``hf`` config dict from a push configuration.

    Returns:
        A copy without the resolution-only keys.
    """
    return {
        k: v for k, v in (hf_cfg or {}).items() if k not in _RESOLUTION_ONLY_HF_KEYS
    }


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
    # The public→private flip. `private` below is the HF-API vocabulary; the
    # surface key is `public` (default False → private), so the safe default is
    # the flag's zero value and no unset/null special-casing is needed here.
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
