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

"""Unit tests for the base + per-environment override config structure.

The environment configs are defined values.yaml-style: a PROD base dict, plus an
override dict per near-variant environment (STAGING, DEV) combined with
``deep_merge``, while STANDALONE is written out in full. These tests pin the merge
semantics and the resolved values other code depends on, so an edit to the shared
base cannot silently change an environment — and, for the inlined STANDALONE,
that a forgotten field cannot silently fall back to a model default.
"""

import ast
import inspect

import pytest

from gbcommon.types import gbenvconfig
from gbcommon.types.gbenvconfig import (
    _GB_ENVIRONMENT_CONFIG_BASE,
    _GB_ENVIRONMENT_CONFIG_DEV,
    _GB_ENVIRONMENT_CONFIG_PROD,
    _GB_ENVIRONMENT_CONFIG_STAGING,
    _GB_ENVIRONMENT_CONFIGS,
    GBEnvConfig,
    deep_merge,
    gb_environment_config,
)

pytestmark = pytest.mark.standalone

_ALL_ENVS = ["PROD", "STAGING", "DEV", "STANDALONE"]

# The per-environment dicts merged onto the base. PROD's is empty (the base holds
# the PROD values). STANDALONE has none: it is written out in full.
_OVERRIDES = {
    "PROD": _GB_ENVIRONMENT_CONFIG_PROD,
    "STAGING": _GB_ENVIRONMENT_CONFIG_STAGING,
    "DEV": _GB_ENVIRONMENT_CONFIG_DEV,
}
_DEPLOYED_ENVS = sorted(_OVERRIDES)


def _standalone_explicit_fields() -> set:
    """Field names passed explicitly to the inlined ``STANDALONE`` GBEnvConfig(...).

    Read from the source with ``ast`` rather than from the built object, because
    a forgotten field is indistinguishable from one set to its default value once
    the model is constructed — which is exactly the mistake this guards against.
    """
    tree = ast.parse(inspect.getsource(gbenvconfig))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "STANDALONE"
                and isinstance(value, ast.Call)
            ):
                return {kw.arg for kw in value.keywords if kw.arg}
    raise AssertionError("could not find the inlined STANDALONE GBEnvConfig(...)")


class TestDeepMerge:
    def test_override_replaces_scalar(self):
        assert deep_merge({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}

    def test_nested_dict_merges_key_by_key(self):
        """An env can override one feature flag without restating the others."""
        merged = deep_merge({"flags": {"x": True, "y": True}}, {"flags": {"y": False}})
        assert merged == {"flags": {"x": True, "y": False}}

    def test_merges_recursively(self):
        merged = deep_merge({"a": {"b": {"c": 1, "d": 2}}}, {"a": {"b": {"d": 9}}})
        assert merged == {"a": {"b": {"c": 1, "d": 9}}}

    def test_non_dict_replaces_dict(self):
        assert deep_merge({"a": {"x": 1}}, {"a": "scalar"}) == {"a": "scalar"}

    def test_new_keys_are_added(self):
        assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_inputs_are_not_mutated(self):
        base = {"a": 1, "flags": {"x": True}}
        override = {"flags": {"x": False}}
        deep_merge(base, override)
        assert base == {"a": 1, "flags": {"x": True}}
        assert override == {"flags": {"x": False}}


class TestEnvironmentStructure:
    def test_all_expected_environments_exist(self):
        assert sorted(_GB_ENVIRONMENT_CONFIGS) == sorted(_ALL_ENVS)

    @pytest.mark.parametrize("env", _ALL_ENVS)
    def test_env_field_matches_its_key(self, env):
        assert _GB_ENVIRONMENT_CONFIGS[env].env == env

    def test_base_holds_the_prod_values(self):
        """The base carries PROD, so PROD's own override dict is empty."""
        assert _GB_ENVIRONMENT_CONFIG_BASE["env"] == "PROD"
        assert not _GB_ENVIRONMENT_CONFIG_PROD

    def test_base_sets_every_field(self):
        """The base must be complete, since PROD merges an empty dict onto it."""
        missing = set(GBEnvConfig.model_fields) - set(_GB_ENVIRONMENT_CONFIG_BASE)
        assert not missing, f"base is missing {sorted(missing)}"

    def test_standalone_is_self_contained(self):
        """STANDALONE is spelled out in full, not merged onto the PROD base.

        Guards the deliberate choice: if it ever gains an override dict, that is
        a decision to review, not something to happen silently.
        """
        assert not hasattr(gbenvconfig, "_GB_ENVIRONMENT_CONFIG_STANDALONE")

    @pytest.mark.parametrize("env", ["STAGING", "DEV"])
    def test_override_names_itself(self, env):
        """PROD is exempt: its dict is empty and the base names PROD."""
        assert _OVERRIDES[env]["env"] == env

    @pytest.mark.parametrize("env", _ALL_ENVS)
    def test_gb_environment_config_returns_the_same_object(self, env):
        assert gb_environment_config(env) is _GB_ENVIRONMENT_CONFIGS[env]

    def test_base_only_uses_known_fields(self):
        assert set(_GB_ENVIRONMENT_CONFIG_BASE) <= set(GBEnvConfig.model_fields)

    @pytest.mark.parametrize("env", _DEPLOYED_ENVS)
    def test_overrides_only_use_known_fields(self, env):
        """Guard against a typo'd key silently doing nothing."""
        assert set(_OVERRIDES[env]) <= set(GBEnvConfig.model_fields)

    @pytest.mark.parametrize("env", _DEPLOYED_ENVS)
    def test_base_plus_override_sets_every_field(self, env):
        """No field may fall back to a pydantic default for a deployed env."""
        supplied = set(_GB_ENVIRONMENT_CONFIG_BASE) | set(_OVERRIDES[env])
        missing = set(GBEnvConfig.model_fields) - supplied
        assert not missing, f"{env} leaves {sorted(missing)} unset"

    def test_standalone_sets_every_field(self):
        """STANDALONE is inlined, so nothing supplies a field it forgets.

        PROD/STAGING/DEV inherit anything they omit from the base (which
        test_base_sets_every_field proves complete), but a field left out of the
        inlined STANDALONE silently takes the pydantic default. Compare against
        the base's field set, which is the full set of fields.
        """
        missing = set(GBEnvConfig.model_fields) - _standalone_explicit_fields()
        assert not missing, f"STANDALONE does not set {sorted(missing)}"
        assert _GB_ENVIRONMENT_CONFIGS["STANDALONE"].env == "STANDALONE"

    @pytest.mark.parametrize("env", _DEPLOYED_ENVS)
    def test_overrides_do_not_restate_base_values(self, env):
        """An override entry equal to the base value is redundant — keep trimmed.

        `env` is exempt: every override must name itself.
        """
        redundant = [
            key
            for key, value in _OVERRIDES[env].items()
            if key != "env" and value == _GB_ENVIRONMENT_CONFIG_BASE.get(key)
        ]
        assert not redundant, f"{env} redundantly restates {redundant}"


class TestResolvedValues:
    """Pin values other code reads, so a base edit cannot silently move them."""

    @pytest.mark.parametrize(
        "env,host",
        [
            ("PROD", "https://api.llm-build-prod.vpc-int.res.ibm.com"),
            ("STAGING", "https://api.llm-build-staging.vpc-int.res.ibm.com"),
            ("DEV", "https://api.llm-build-dev.vpc-int.res.ibm.com"),
            ("STANDALONE", "http://localhost:8080"),
        ],
    )
    def test_gbserver_host(self, env, host):
        assert _GB_ENVIRONMENT_CONFIGS[env].gbserver_host == host

    @pytest.mark.parametrize(
        "env,schema",
        [
            ("PROD", "granite_dot_build_prod"),
            ("STAGING", "granite_dot_build_staging"),
            ("DEV", "granite_dot_build_dev"),
            ("STANDALONE", "standalone"),
        ],
    )
    def test_default_sql_schema(self, env, schema):
        assert _GB_ENVIRONMENT_CONFIGS[env].default_sql_schema == schema

    @pytest.mark.parametrize(
        "env,section",
        [
            ("PROD", "gb.spaces"),
            ("STAGING", "staging.gb.spaces"),
            ("DEV", "dev.gb.spaces"),
            ("STANDALONE", ""),
        ],
    )
    def test_config_spaces(self, env, section):
        assert _GB_ENVIRONMENT_CONFIGS[env].config_spaces == section

    def test_dev_uses_the_staging_lakehouse(self):
        assert _GB_ENVIRONMENT_CONFIGS["DEV"].lakehouse_environment == "STAGING"
        assert _GB_ENVIRONMENT_CONFIGS["PROD"].lakehouse_environment == "PROD"
        assert _GB_ENVIRONMENT_CONFIGS["STANDALONE"].lakehouse_environment == ""

    def test_staging_and_dev_track_the_dev_assets_branch(self):
        assert _GB_ENVIRONMENT_CONFIGS["PROD"].branch_assets == "gbspace-config"
        assert _GB_ENVIRONMENT_CONFIGS["STAGING"].branch_assets == "gbspace-config-dev"
        assert _GB_ENVIRONMENT_CONFIGS["DEV"].branch_assets == "gbspace-config-dev"

    def test_hf_org_config_is_inherited_by_every_environment(self):
        """The HF org config lives only in the base; every env must inherit it."""
        for env in _ALL_ENVS:
            cfg = _GB_ENVIRONMENT_CONFIGS[env]
            assert cfg.hf_organization == "ibm-research"
            assert cfg.hf_enterprise_organizations == ["ibm-research", "ibm-granite"]

    def test_hf_enterprise_list_is_not_shared_mutable_state(self):
        """Each config needs its own list, or mutating one would hit them all."""
        lists = [
            _GB_ENVIRONMENT_CONFIGS[e].hf_enterprise_organizations for e in _ALL_ENVS
        ]
        assert len({id(x) for x in lists}) == len(lists)

    def test_standalone_feature_flags(self):
        """STANDALONE's flags are literal, unlike the env-var driven deployed ones."""
        flags = _GB_ENVIRONMENT_CONFIGS["STANDALONE"].feature_flags
        assert flags["build_start_via_github"] is False
        assert flags["gbserver_artifact_filter"] is False
        assert flags["gbserver_build_events"] is True
        assert flags["gbserver_build_update"] is True

    def test_deployed_envs_have_no_build_start_via_github_flag(self):
        """That flag is added by the STANDALONE override only."""
        for env in ("PROD", "STAGING", "DEV"):
            assert (
                "build_start_via_github"
                not in _GB_ENVIRONMENT_CONFIGS[env].feature_flags
            )


class TestParseBooleanStrict:
    """``strict=True`` fails closed: only a recognized truthy value → True.

    Unlike lenient mode (unrecognized non-falsy → True), strict returns True only
    for a real ``True`` or a recognized truthy token. Backs the HF ``public``
    flag, where a typo must never silently opt into a public repo.
    """

    @pytest.mark.parametrize(
        "value", [True, "true", "yes", "on", "1", "True", " true "]
    )
    def test_recognized_truthy_is_true(self, value):
        assert gbenvconfig.parse_boolean(value, strict=True) is True

    @pytest.mark.parametrize(
        "value",
        [None, False, "false", "no", "off", "0", "", "null", "treu", "on-prod", "2"],
    )
    def test_everything_else_is_false(self, value):
        assert gbenvconfig.parse_boolean(value, strict=True) is False

    def test_strict_differs_from_lenient_on_a_typo(self):
        """The whole point: an unrecognized value goes opposite ways."""
        assert gbenvconfig.parse_boolean("treu") is True  # lenient default
        assert gbenvconfig.parse_boolean("treu", strict=True) is False  # fail closed
