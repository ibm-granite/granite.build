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

"""Unit tests for ``resolve_space_resource_group_id`` (table-first + HF fallback).

Storage and the HF API are mocked; no live Hub calls are made.
"""

from unittest.mock import MagicMock, patch

import pytest

from gbcommon.uri.hf import HfURI
from gbserver.spaces.hf_push_config import (
    apply_hf_step_overlay,
    resolve_hfpush_resource_group_id,
    resolve_space_resource_group_id,
    sanitize_hf_step_overlay,
)
from gbserver.storage.stored_space import StoredSpace


def _make_space(name="public", hf_default_resource_group_id=None):
    return StoredSpace(
        name=name,
        git_repo_uri="http://example/repo",
        lakehouse_namespace="lh",
        hf_default_resource_group_id=hf_default_resource_group_id,
    )


class TestResolveSpaceResourceGroupId:
    def test_returns_cached_id_without_hf_call(self):
        """A space row with a cached id short-circuits the HF lookup."""
        space = _make_space(hf_default_resource_group_id="cached-id")
        storage = MagicMock()
        storage.space_storage.get_by_name.return_value = space
        admin = MagicMock(return_value=storage)

        with (
            patch("gbserver.spaces.hf_push_config.get_admin_storage", admin),
            patch(
                "gbserver.spaces.hf_push_config.HfURI.resolve_resource_group_id_for_org"
            ) as mock_hf,
        ):
            result = resolve_space_resource_group_id(
                space_name="public",
                organization="ibm-research",
                token="tok",
            )

        assert result == "cached-id"
        mock_hf.assert_not_called()
        storage.space_storage.update.assert_not_called()

    def test_falls_back_to_hf_and_writes_back(self):
        """A row with no cached id triggers the HF lookup and a write-back."""
        space = _make_space(hf_default_resource_group_id=None)
        storage = MagicMock()
        storage.space_storage.get_by_name.return_value = space

        with (
            patch(
                "gbserver.spaces.hf_push_config.get_admin_storage",
                return_value=storage,
            ),
            patch(
                "gbserver.spaces.hf_push_config.HfURI.resolve_resource_group_id_for_org",
                return_value="resolved-id",
            ) as mock_hf,
        ):
            result = resolve_space_resource_group_id(
                space_name="public",
                organization="ibm-research",
                token="tok",
            )

        assert result == "resolved-id"
        mock_hf.assert_called_once()
        # The resolved id is written back onto the (same) space object.
        assert space.hf_default_resource_group_id == "resolved-id"
        storage.space_storage.update.assert_called_once_with(space)

    def test_no_row_falls_back_without_write_back(self):
        """When no space row exists, resolve via HF but do not persist."""
        storage = MagicMock()
        storage.space_storage.get_by_name.return_value = None

        with (
            patch(
                "gbserver.spaces.hf_push_config.get_admin_storage",
                return_value=storage,
            ),
            patch(
                "gbserver.spaces.hf_push_config.HfURI.resolve_resource_group_id_for_org",
                return_value="resolved-id",
            ),
        ):
            result = resolve_space_resource_group_id(
                space_name="unknown-space",
                organization="ibm-research",
                token="tok",
            )

        assert result == "resolved-id"
        storage.space_storage.update.assert_not_called()

    def test_unresolved_returns_none_no_write_back(self):
        """A failed HF lookup returns None and does not write back."""
        space = _make_space(hf_default_resource_group_id=None)
        storage = MagicMock()
        storage.space_storage.get_by_name.return_value = space

        with (
            patch(
                "gbserver.spaces.hf_push_config.get_admin_storage",
                return_value=storage,
            ),
            patch(
                "gbserver.spaces.hf_push_config.HfURI.resolve_resource_group_id_for_org",
                return_value=None,
            ),
        ):
            result = resolve_space_resource_group_id(
                space_name="public",
                organization="ibm-research",
                token="tok",
            )

        assert result is None
        storage.space_storage.update.assert_not_called()

    def test_explicit_non_default_name_bypasses_cache(self):
        """An explicit non-default resource_group_name ignores the cached default id.

        The cache holds ONLY the space's default group. When a caller requests a
        different group by name, the helper must NOT return (or overwrite) the
        cached default id: it resolves via the HF API and does not write back.
        """
        # Row has a cached DEFAULT id, but the request names a different group.
        space = _make_space(hf_default_resource_group_id="default-cached-id")
        storage = MagicMock()
        storage.space_storage.get_by_name.return_value = space

        with (
            patch(
                "gbserver.spaces.hf_push_config.get_admin_storage",
                return_value=storage,
            ),
            patch(
                "gbserver.spaces.hf_push_config.HfURI.space_name_to_resource_group_name",
                return_value="gbspace-public",
            ),
            patch(
                "gbserver.spaces.hf_push_config.HfURI.resolve_resource_group_id_for_org",
                return_value="non-default-id",
            ) as mock_hf,
        ):
            result = resolve_space_resource_group_id(
                space_name="public",
                organization="ibm-research",
                token="tok",
                resource_group_name="some-other-group",
            )

        # Resolved via HF (the explicit group), NOT the cached default id.
        assert result == "non-default-id"
        mock_hf.assert_called_once()
        # The cached default id is untouched (no poisoning).
        assert space.hf_default_resource_group_id == "default-cached-id"
        storage.space_storage.update.assert_not_called()

    def test_explicit_default_name_uses_cache(self):
        """An explicit name equal to the derived default still hits the cache."""
        space = _make_space(hf_default_resource_group_id="default-cached-id")
        storage = MagicMock()
        storage.space_storage.get_by_name.return_value = space

        with (
            patch(
                "gbserver.spaces.hf_push_config.get_admin_storage",
                return_value=storage,
            ),
            patch(
                "gbserver.spaces.hf_push_config.HfURI.space_name_to_resource_group_name",
                return_value="gbspace-public",
            ),
            patch(
                "gbserver.spaces.hf_push_config.HfURI.resolve_resource_group_id_for_org"
            ) as mock_hf,
        ):
            result = resolve_space_resource_group_id(
                space_name="public",
                organization="ibm-research",
                token="tok",
                resource_group_name="gbspace-public",
            )

        assert result == "default-cached-id"
        mock_hf.assert_not_called()
        storage.space_storage.update.assert_not_called()


def _make_assetstore(enterprise_orgs, token="tok"):
    """Hfstore double exposing only what the resolver reads."""
    store = MagicMock()
    store.get_enterprise_organizations.return_value = enterprise_orgs
    store.resolve_token.return_value = token
    return store


def _make_hfuri(owner="ibm-research", repo="my-model"):
    return HfURI.from_parts(owner=owner, repo=repo)


def _output_config(hf_cfg):
    """BuildTargetOutputConfig-alike carrying a store_push hf block."""
    cfg = MagicMock()
    cfg.public = None  # no top-level public (a real output defaults it to None)
    cfg.store_push = MagicMock()
    cfg.store_push.config = {"hf": hf_cfg}
    return cfg


def _storepush_config(hf_cfg):
    cfg = MagicMock()
    cfg.config = {"hf": hf_cfg}
    return cfg


class TestResolveHfpushResourceGroupIdNonEnterprise:
    """A non-Enterprise org must skip resource group resolution entirely."""

    def test_non_enterprise_skips_resolution(self):
        with patch(
            "gbserver.spaces.hf_push_config.resolve_space_resource_group_id"
        ) as mock_resolve:
            rg_id, private, _ = resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="my-user"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
            )

        assert rg_id is None
        assert private is True
        mock_resolve.assert_not_called()

    def test_non_enterprise_with_pinned_id_raises(self):
        with pytest.raises(ValueError, match="not an HF Enterprise organization"):
            resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="my-user"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
                output_config=_output_config({"resource_group_id": "rg-123"}),
            )

    def test_non_enterprise_with_pinned_name_raises(self):
        with pytest.raises(ValueError, match="not an HF Enterprise organization"):
            resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="my-user"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
                output_config=_output_config({"resource_group_name": "gbspace-public"}),
            )

    def test_absent_enterprise_list_keeps_legacy_behavior(self):
        """None (key absent) => every org is Enterprise, so resolution still runs."""
        with patch(
            "gbserver.spaces.hf_push_config.resolve_space_resource_group_id",
            return_value="resolved-id",
        ) as mock_resolve:
            rg_id, _, _ = resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="some-random-user"),
                assetstore=_make_assetstore(None),
                space_name="public",
            )

        assert rg_id == "resolved-id"
        mock_resolve.assert_called_once()


class TestResolveHfpushResourceGroupIdEnterprise:
    def test_enterprise_resolves_via_space(self):
        with patch(
            "gbserver.spaces.hf_push_config.resolve_space_resource_group_id",
            return_value="space-id",
        ) as mock_resolve:
            rg_id, _, _ = resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="ibm-research"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
            )

        assert rg_id == "space-id"
        assert mock_resolve.call_args.kwargs["organization"] == "ibm-research"
        assert mock_resolve.call_args.kwargs["space_name"] == "public"

    def test_pinned_id_used_verbatim_without_resolver(self):
        with patch(
            "gbserver.spaces.hf_push_config.resolve_space_resource_group_id"
        ) as mock_resolve:
            rg_id, _, _ = resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="ibm-research"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
                output_config=_output_config({"resource_group_id": "pinned-id"}),
            )

        assert rg_id == "pinned-id"
        mock_resolve.assert_not_called()

    def test_use_resource_group_false_opts_out(self):
        """An Enterprise org can opt out with use_resource_group: false."""
        with patch(
            "gbserver.spaces.hf_push_config.resolve_space_resource_group_id"
        ) as mock_resolve:
            rg_id, _, _ = resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="ibm-research"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
                output_config=_output_config({"use_resource_group": False}),
            )

        assert rg_id is None
        mock_resolve.assert_not_called()

    def test_use_resource_group_false_with_pinned_group_raises(self):
        with pytest.raises(ValueError, match="cannot be combined"):
            resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="ibm-research"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
                output_config=_output_config(
                    {"use_resource_group": False, "resource_group_id": "rg-1"}
                ),
            )


class TestResolveHfpushConfigPrecedence:
    """Environment-level config is honored, with build.yaml overriding it."""

    def test_env_level_store_push_is_honored(self):
        with patch(
            "gbserver.spaces.hf_push_config.resolve_space_resource_group_id",
            return_value="ignored",
        ) as mock_resolve:
            _, private, _ = resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="ibm-research"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
                storepush_config=_storepush_config(
                    {"resource_group_name": "env-group", "public": True}
                ),
            )

        assert private is False
        assert mock_resolve.call_args.kwargs["resource_group_name"] == "env-group"

    def test_build_yaml_overrides_environment(self):
        with patch(
            "gbserver.spaces.hf_push_config.resolve_space_resource_group_id"
        ) as mock_resolve:
            rg_id, private, _ = resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="ibm-research"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
                storepush_config=_storepush_config(
                    {"resource_group_id": "env-id", "public": True}
                ),
                output_config=_output_config({"resource_group_id": "build-id"}),
            )

        assert rg_id == "build-id"
        assert private is False  # not overridden by build.yaml, inherited from env
        mock_resolve.assert_not_called()


class TestSanitizeHfStepOverlay:
    def test_strips_resolution_only_keys(self):
        # ``use_resource_group`` and ``public`` are consumed during resolution and
        # must never leak into the worker step's ``hf`` block; other keys stay.
        assert sanitize_hf_step_overlay(
            {"type": "model", "use_resource_group": False, "public": True}
        ) == {"type": "model"}

    def test_handles_empty_and_none(self):
        assert sanitize_hf_step_overlay({}) == {}
        assert sanitize_hf_step_overlay(None) == {}


class TestApplyHfStepOverlay:
    """The shared k8s/skypilot overlay: strip use_resource_group, then re-assert
    the resolved resource_group_id so a stray pinned id in the raw config cannot
    win over resolution."""

    def test_strips_use_resource_group_and_reasserts_resolved_id(self):
        hfpush_config = {
            "private": True,
            "hf": {"type": "model", "resource_group_id": None},
        }
        apply_hf_step_overlay(
            hfpush_config,
            {
                "type": "model",
                "use_resource_group": False,
                "resource_group_id": "stray",
            },
            resource_group_id="resolved-id",
        )
        assert "use_resource_group" not in hfpush_config["hf"]
        # The resolved id wins over the stray id in the raw overlay config.
        assert hfpush_config["hf"]["resource_group_id"] == "resolved-id"

    def test_reasserts_none_over_a_stray_pinned_id(self):
        # A pinned-but-skipped id (non-Enterprise org) resolves to None and must
        # not be resurrected by the overlay.
        hfpush_config = {"hf": {"type": "model", "resource_group_id": None}}
        apply_hf_step_overlay(
            hfpush_config,
            {"resource_group_id": "pinned-but-skipped"},
            resource_group_id=None,
        )
        assert hfpush_config["hf"]["resource_group_id"] is None


class TestUseResourceGroupAcrossLevels:
    """`use_resource_group: false` is per-level, not evaluated against the merge.

    An environment-level resource group is a documented fallback (priority 3 in
    docs/builds/hf-push.md). A build author who opts one output out of resource
    groups cannot remove that inherited value from their build.yaml, so treating
    the pair as contradictory would make the documented opt-out unusable.
    """

    def test_output_opt_out_overrides_inherited_group(self):
        with patch(
            "gbserver.spaces.hf_push_config.resolve_space_resource_group_id"
        ) as mock_resolve:
            rg_id, _, _ = resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="ibm-research"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
                storepush_config=_storepush_config(
                    {"resource_group_name": "gbspace-public"}
                ),
                output_config=_output_config({"use_resource_group": False}),
            )

        assert rg_id is None
        mock_resolve.assert_not_called()

    def test_same_level_contradiction_still_raises_at_output(self):
        with pytest.raises(ValueError, match="same push config"):
            resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="ibm-research"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
                output_config=_output_config(
                    {"use_resource_group": False, "resource_group_id": "rg-1"}
                ),
            )

    def test_same_level_contradiction_still_raises_at_environment(self):
        with pytest.raises(ValueError, match="same push config"):
            resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="ibm-research"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
                storepush_config=_storepush_config(
                    {"use_resource_group": False, "resource_group_name": "g"}
                ),
            )

    def test_output_pinned_id_overrides_environment_opt_out(self):
        """An output-level pin is priority 1, so it outranks an inherited opt-out.

        Regression: the opt-out was evaluated off the merged config, so an
        environment-level `use_resource_group: false` silently discarded an
        explicit output-level `resource_group_id` — inverting the documented
        precedence and dropping a pinned group with no error.
        """
        from gbserver.spaces.hf_push_config import resolve_hfpush_resource_group_id

        with patch(
            "gbserver.spaces.hf_push_config.resolve_space_resource_group_id"
        ) as mock_resolve:
            rg_id, _, _ = resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="ibm-research"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
                storepush_config=_storepush_config({"use_resource_group": False}),
                output_config=_output_config({"resource_group_id": "rg-out"}),
            )

        assert rg_id == "rg-out"
        # A pinned id is used verbatim, so the space resolver is never consulted.
        mock_resolve.assert_not_called()

    def test_output_pinned_name_overrides_environment_opt_out(self):
        """Same for a pinned name, which does go through the resolver."""
        from gbserver.spaces.hf_push_config import resolve_hfpush_resource_group_id

        with patch(
            "gbserver.spaces.hf_push_config.resolve_space_resource_group_id",
            return_value="rg-from-name",
        ) as mock_resolve:
            rg_id, _, _ = resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="ibm-research"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
                storepush_config=_storepush_config({"use_resource_group": False}),
                output_config=_output_config({"resource_group_name": "team-group"}),
            )

        assert rg_id == "rg-from-name"
        assert mock_resolve.call_args.kwargs["resource_group_name"] == "team-group"

    def test_environment_pin_does_not_override_output_opt_out(self):
        """The reverse: a lower-level pin must not defeat a higher-level opt-out."""
        from gbserver.spaces.hf_push_config import resolve_hfpush_resource_group_id

        with patch(
            "gbserver.spaces.hf_push_config.resolve_space_resource_group_id"
        ) as mock_resolve:
            rg_id, _, _ = resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="ibm-research"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
                storepush_config=_storepush_config({"resource_group_id": "rg-env"}),
                output_config=_output_config({"use_resource_group": False}),
            )

        assert rg_id is None
        mock_resolve.assert_not_called()

    def test_opt_out_at_both_levels_still_opts_out(self):
        from gbserver.spaces.hf_push_config import resolve_hfpush_resource_group_id

        with patch(
            "gbserver.spaces.hf_push_config.resolve_space_resource_group_id"
        ) as mock_resolve:
            rg_id, _, _ = resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="ibm-research"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
                storepush_config=_storepush_config({"use_resource_group": False}),
                output_config=_output_config({"use_resource_group": False}),
            )

        assert rg_id is None
        mock_resolve.assert_not_called()

    def test_environment_opt_out_is_overridable_by_output(self):
        """The reverse direction: an output can turn resource groups back on."""
        with patch(
            "gbserver.spaces.hf_push_config.resolve_space_resource_group_id",
            return_value="rg-on",
        ) as mock_resolve:
            rg_id, _, _ = resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="ibm-research"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
                storepush_config=_storepush_config({"use_resource_group": False}),
                output_config=_output_config({"use_resource_group": True}),
            )

        assert rg_id == "rg-on"
        mock_resolve.assert_called_once()


class TestNullConfigValuesTreatedAsUnset:
    """An explicit yaml null means "not set here", not "override with None".

    A bare `public:` in build.yaml parses as None. Merging it wholesale would
    erase a value inherited from environment.yaml. The resolver returns the
    internal ``private`` bool (``public`` flipped): ``public`` unset/null → the
    safe default, a private repo (``private is True``).
    """

    @pytest.mark.parametrize(
        "env_hf,output_hf,expected",
        [
            # A null at the output level must not erase the environment value.
            ({"public": True}, {"public": None}, False),
            # A real value at the output level still wins.
            ({"public": True}, {"public": False}, True),
            # A null with nothing to inherit falls back to the default (private).
            (None, {"public": None}, True),
            ({"public": None}, {"public": None}, True),
            # Unchanged behavior for values that are actually set.
            (None, {"public": True}, False),
            ({"public": True}, None, False),
            (None, None, True),
        ],
    )
    def test_private_resolution(self, env_hf, output_hf, expected):
        from gbserver.spaces.hf_push_config import resolve_hfpush_resource_group_id

        _, private, _ = resolve_hfpush_resource_group_id(
            hfuri=_make_hfuri(owner="my-user"),
            assetstore=_make_assetstore([]),  # non-Enterprise: skips RG resolution
            space_name="public",
            storepush_config=_storepush_config(env_hf) if env_hf is not None else None,
            output_config=_output_config(output_hf) if output_hf is not None else None,
        )

        assert private is expected, f"expected {expected}, got {private!r}"

    def test_private_is_always_a_bool(self):
        """Never a None: the worker templates stringify whatever they are given."""
        from gbserver.spaces.hf_push_config import resolve_hfpush_resource_group_id

        _, private, _ = resolve_hfpush_resource_group_id(
            hfuri=_make_hfuri(owner="my-user"),
            assetstore=_make_assetstore([]),
            space_name="public",
            output_config=_output_config({"public": None}),
        )

        assert isinstance(private, bool)

    def test_null_use_resource_group_does_not_disable_resolution(self):
        """`use_resource_group:` with no value must not read as an opt-out."""
        from gbserver.spaces.hf_push_config import resolve_hfpush_resource_group_id

        with patch(
            "gbserver.spaces.hf_push_config.resolve_space_resource_group_id",
            return_value="rg",
        ) as mock_resolve:
            rg_id, _, _ = resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="ibm-research"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
                output_config=_output_config({"use_resource_group": None}),
            )

        assert rg_id == "rg"
        mock_resolve.assert_called_once()

    def test_null_resource_group_id_does_not_pin(self):
        """A null id must not count as a pinned group on a non-Enterprise org."""
        from gbserver.spaces.hf_push_config import resolve_hfpush_resource_group_id

        rg_id, _, _ = resolve_hfpush_resource_group_id(
            hfuri=_make_hfuri(owner="my-user"),
            assetstore=_make_assetstore(["ibm-research"]),
            space_name="public",
            output_config=_output_config(
                {"resource_group_id": None, "resource_group_name": None}
            ),
        )

        assert rg_id is None


class TestQuotedBooleanForms:
    """A yaml-quoted boolean must mean what it says, not "non-empty string".

    Both booleans in an ``hf`` push config go through ``parse_boolean``, so
    ``"false"`` / ``"no"`` / ``"off"`` / ``"0"`` resolve to ``False`` rather than
    being truthy. Before that, a quoted value silently inverted the user's intent.
    """

    @pytest.mark.parametrize("value", ["true", "yes", "on", "1", "True", " true "])
    def test_quoted_public_is_public(self, value):
        from gbserver.spaces.hf_push_config import resolve_hfpush_resource_group_id

        _, private, _ = resolve_hfpush_resource_group_id(
            hfuri=_make_hfuri(owner="my-user"),
            assetstore=_make_assetstore([]),
            space_name="public",
            output_config=_output_config({"public": value}),
        )

        assert private is False

    @pytest.mark.parametrize("value", ["false", "no", "off", "0"])
    def test_quoted_use_resource_group_opts_out(self, value):
        """A quoted opt-out must actually skip resolution."""
        from gbserver.spaces.hf_push_config import resolve_hfpush_resource_group_id

        with patch(
            "gbserver.spaces.hf_push_config.resolve_space_resource_group_id",
            return_value="rg",
        ) as mock_resolve:
            rg_id, _, _ = resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="ibm-research"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
                output_config=_output_config({"use_resource_group": value}),
            )

        assert rg_id is None
        mock_resolve.assert_not_called()

    def test_unrecognized_public_stays_private(self):
        """A typo fails safe: unparseable means private, never public."""
        from gbserver.spaces.hf_push_config import resolve_hfpush_resource_group_id

        _, private, _ = resolve_hfpush_resource_group_id(
            hfuri=_make_hfuri(owner="my-user"),
            assetstore=_make_assetstore([]),
            space_name="public",
            output_config=_output_config({"public": "treu"}),
        )

        assert private is True


class TestHfPushConfigError:
    """Config errors are a distinguishable subtype, not a bare ValueError.

    The inline bash/docker push treats a resolution *miss* as best-effort but a
    *config* error as fatal. Both used to be plain ``ValueError``, so the handler
    could not tell them apart. Subclassing keeps older ``except ValueError``
    callers (and the assertions above) working.
    """

    def test_is_a_valueerror_subclass(self):
        from gbserver.spaces.hf_push_config import HfPushConfigError

        assert issubclass(HfPushConfigError, ValueError)

    def test_non_enterprise_pin_raises_the_subtype(self):
        from gbserver.spaces.hf_push_config import (
            HfPushConfigError,
            resolve_hfpush_resource_group_id,
        )

        with pytest.raises(HfPushConfigError, match="not an HF Enterprise"):
            resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="my-user"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
                output_config=_output_config({"resource_group_id": "rg-1"}),
            )

    def test_same_level_contradiction_raises_the_subtype(self):
        from gbserver.spaces.hf_push_config import (
            HfPushConfigError,
            resolve_hfpush_resource_group_id,
        )

        with pytest.raises(HfPushConfigError, match="cannot be combined"):
            resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="ibm-research"),
                assetstore=_make_assetstore(["ibm-research"]),
                space_name="public",
                output_config=_output_config(
                    {"use_resource_group": False, "resource_group_id": "rg-1"}
                ),
            )


class TestResolveHfpushPrivate:
    """The standalone ``private`` resolver used by the non-Hfstore push branch."""

    def test_defaults_to_private(self):
        from gbserver.spaces.hf_push_config import resolve_hfpush_private

        assert resolve_hfpush_private() is True

    def test_honors_explicit_public(self):
        from gbserver.spaces.hf_push_config import resolve_hfpush_private

        assert (
            resolve_hfpush_private(output_config=_output_config({"public": True}))
            is False
        )

    def test_output_overrides_environment(self):
        from gbserver.spaces.hf_push_config import resolve_hfpush_private

        assert (
            resolve_hfpush_private(
                storepush_config=_storepush_config({"public": False}),
                output_config=_output_config({"public": True}),
            )
            is False
        )

    def test_agrees_with_the_full_resolver(self):
        """Same rule, so the two entry points must never diverge."""
        from gbserver.spaces.hf_push_config import (
            resolve_hfpush_private,
            resolve_hfpush_resource_group_id,
        )

        for hf_cfg in (
            {},
            {"public": False},
            {"public": True},
            {"public": None},
            {"public": "true"},
            {"public": "treu"},
        ):
            output_config = _output_config(hf_cfg)
            _, from_full, _ = resolve_hfpush_resource_group_id(
                hfuri=_make_hfuri(owner="my-user"),
                assetstore=_make_assetstore([]),
                space_name="public",
                output_config=output_config,
            )
            assert resolve_hfpush_private(output_config=output_config) is from_full


class TestEnvLevelPublic:
    """At the environment level `public` is written as `config.hf.public`.

    Environment/store.yaml push config has no output field, so `config.hf.public`
    is the only form; the resolver flips it to the internal `private`.
    """

    def _storepush(self, config):
        cfg = MagicMock()
        cfg.config = config
        return cfg

    def test_env_config_hf_public_makes_public(self):
        from gbserver.spaces.hf_push_config import resolve_hfpush_private

        assert (
            resolve_hfpush_private(
                storepush_config=self._storepush({"hf": {"public": True}})
            )
            is False
        )

    def test_env_default_is_private(self):
        from gbserver.spaces.hf_push_config import resolve_hfpush_private

        assert resolve_hfpush_private(storepush_config=self._storepush({})) is True


class TestOutputPublicForms:
    """The two `public` forms on a build.yaml output, folded + flipped here.

    `public` is granite.build's surface vocabulary (default false → private),
    written top-level or as `config.hf.public`. The resolver folds them to one and
    flips to the internal `private`; `buildconfig` itself stays store-agnostic (it
    just carries the field).
    """

    @staticmethod
    def _output(**kwargs):
        from gbserver.types.buildconfig import BuildTargetOutputConfig

        return BuildTargetOutputConfig(**kwargs)

    def _private(self, **kwargs):
        from gbserver.spaces.hf_push_config import resolve_hfpush_private

        return resolve_hfpush_private(output_config=self._output(**kwargs))

    def test_top_level_public(self):
        assert self._private(uri="hf:///o/r-{{ x }}", public=True) is False

    def test_config_hf_public(self):
        assert (
            self._private(
                uri="hf:///o/r", store_push={"config": {"hf": {"public": True}}}
            )
            is False
        )

    def test_omitted_is_private(self):
        assert self._private(uri="hf:///o/r") is True

    def test_equal_forms_collapse(self):
        assert (
            self._private(
                uri="hf:///o/r",
                public=True,
                store_push={"config": {"hf": {"public": True}}},
            )
            is False
        )

    def test_conflicting_forms_raise(self):
        from gbserver.spaces.hf_push_config import (
            HfPushConfigError,
            resolve_hfpush_private,
        )

        o = self._output(
            uri="hf:///o/r",
            public=True,
            store_push={"config": {"hf": {"public": False}}},
        )
        with pytest.raises(HfPushConfigError, match="conflict"):
            resolve_hfpush_private(output_config=o)

    def test_null_hf_public_is_unset_not_a_conflict(self):
        # A bare `config.hf.public:` (yaml null) is "unset", so a top-level
        # `public: true` fills it rather than clashing.
        assert (
            self._private(
                uri="hf:///o/r",
                public=True,
                store_push={"config": {"hf": {"public": None}}},
            )
            is False
        )


class TestValidateOutputPush:
    """The load-time HF push guard, kept in the HF module (buildconfig delegates)."""

    @staticmethod
    def _output(**kwargs):
        from gbserver.types.buildconfig import BuildTargetOutputConfig

        return BuildTargetOutputConfig(**kwargs)

    def _check(self, **kwargs):
        from gbserver.spaces.hf_push_config import validate_output_push

        return validate_output_push("out", self._output(**kwargs))

    @pytest.mark.parametrize(
        "uri", ["file:///t", "lh://a/b", "env:///abs", "cos://b/k"]
    )
    def test_public_on_non_hf_output_errors(self, uri):
        assert self._check(uri=uri, public=True) is not None

    def test_hf_block_on_non_hf_output_errors(self):
        err = self._check(
            uri="lh://a/b",
            store_push={"config": {"hf": {"resource_group_name": "x"}}},
        )
        assert err is not None

    def test_public_on_templated_hf_output_ok(self):
        assert self._check(uri="hf:///o/r-{{ binding.path }}", public=True) is None

    def test_no_push_options_ok(self):
        assert self._check(uri="file:///t") is None

    def test_same_level_conflict_surfaces_at_validate(self):
        err = self._check(
            uri="hf:///o/r",
            public=True,
            store_push={"config": {"hf": {"public": False}}},
        )
        assert err is not None and "conflict" in err.lower()

    @pytest.mark.parametrize("private_val", [False, True, "false", None])
    def test_retired_private_key_errors_loudly(self, private_val):
        # The old `config.hf.private` key is no longer supported; instead of
        # silently making the repo private, it must fail loudly pointing to `public`.
        err = self._check(
            uri="hf:///o/r",
            store_push={"config": {"hf": {"private": private_val}}},
        )
        assert err is not None and "no longer supported" in err

    def test_retired_private_key_on_non_hf_output_errors(self):
        err = self._check(
            uri="lh://a/b",
            store_push={"config": {"hf": {"private": False}}},
        )
        assert err is not None
