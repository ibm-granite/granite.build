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

"""Tests for the consolidated ``get_launch_env_vars`` method.

Covers the base ``Environment`` implementation (the standard cross-environment
set from ``STANDARD_STEP_ENV_FROM_RUN_METADATA``) and each subclass override,
asserting that (a) the environment's own inline vars are present and (b) the
standard set is authoritative — a conflicting launcher/config value is
overridden by the run_metadata-derived one.
"""

import asyncio

import pytest

# The K8s environment imports kubernetes_asyncio (via its retry strategies),
# which lives in the optional ``ibm`` extra and is absent in the lightweight
# quick-test CI venv, so TestK8sOverride must be skipped there rather than
# erroring at import. Reuse the shared marker.
from libgbtest.constants import requires_k8s

from gbserver.environment import environment as environment_module
from gbserver.environment.environment import Environment

# ---------------------------------------------------------------------------
# Base Environment.get_launch_env_vars (the standard cross-environment set)
# ---------------------------------------------------------------------------


def _base_env_vars(run_metadata):
    """Invoke the BASE implementation regardless of any subclass override.

    The base method does not use ``self``, so a bare object suffices as the
    bound instance.
    """
    return Environment.get_launch_env_vars(object(), run_metadata=run_metadata)


class TestBaseStandardEnv:
    @pytest.fixture(autouse=True)
    def _no_gbtest(self, monkeypatch):
        """Isolate the run_metadata-derived set from any GBTEST_ vars the test
        environment may export (e.g. a suite-level GBTEST_MOCK_HF)."""
        monkeypatch.setattr(
            environment_module, "get_exported_gbtest_env_vars", lambda: {}
        )

    def test_build_id_emitted(self):
        assert _base_env_vars({"build_id": "b1"}) == {"GB_BUILD_ID": "b1"}

    def test_missing_build_id_omitted(self):
        assert _base_env_vars({}) == {}

    def test_empty_build_id_omitted(self):
        assert _base_env_vars({"build_id": ""}) == {}

    def test_none_run_metadata_yields_empty(self):
        assert _base_env_vars(None) == {}

    def test_non_str_value_coerced(self):
        assert _base_env_vars({"build_id": 123}) == {"GB_BUILD_ID": "123"}

    def test_new_map_key_surfaces_automatically(self, monkeypatch):
        """Adding a standard var is a one-line map change; verify the mechanism."""
        monkeypatch.setitem(
            environment_module.STANDARD_STEP_ENV_FROM_RUN_METADATA,
            "GB_TARGETRUN_ID",
            "targetrun_id",
        )
        result = _base_env_vars({"build_id": "b1", "targetrun_id": "t1"})
        assert result == {"GB_BUILD_ID": "b1", "GB_TARGETRUN_ID": "t1"}


class TestBaseGbtestForwarding:
    def test_gbtest_vars_forwarded_alongside_standard_set(self, monkeypatch):
        """GBTEST_ test-control vars are part of the base standard set, so every
        environment forwards them uniformly via ``super()``."""
        monkeypatch.setattr(
            environment_module,
            "get_exported_gbtest_env_vars",
            lambda: {"GBTEST_MOCK_HF": "true"},
        )
        assert _base_env_vars({"build_id": "b1"}) == {
            "GBTEST_MOCK_HF": "true",
            "GB_BUILD_ID": "b1",
        }


# ---------------------------------------------------------------------------
# Shared declared-secret resolver (used by every environment)
# ---------------------------------------------------------------------------


def _mappings(*pairs):
    """Build a list of EnvironmentVariableConfig from (env_name, secret_name).

    A ``secret_name`` of None exercises the "defaults to env_name" path.
    """
    from gbserver.types.environment.environment import EnvironmentVariableConfig

    return [
        EnvironmentVariableConfig(env_name=env_name, secret_name=secret_name)
        for env_name, secret_name in pairs
    ]


class TestDeclaredSecretResolver:
    """Direct tests for ``Environment._resolve_declared_secret_env_vars`` and
    ``Environment._declared_secret_env_key_names`` — the shared least-privilege
    path every environment funnels declared secrets through."""

    def test_resolves_declared_mapping(self):
        resolved = Environment._resolve_declared_secret_env_vars(
            _mappings(("MY_TOKEN", "tok")), {"tok": "secret-val", "other": "nope"}
        )
        # Only the declared secret is exposed; unrelated bag entries are not.
        assert resolved == {"MY_TOKEN": "secret-val"}

    def test_secret_name_defaults_to_env_name(self):
        resolved = Environment._resolve_declared_secret_env_vars(
            _mappings(("MY_TOKEN", None)), {"MY_TOKEN": "secret-val"}
        )
        assert resolved == {"MY_TOKEN": "secret-val"}

    def test_empty_mappings_yield_empty(self):
        assert Environment._resolve_declared_secret_env_vars([], {"tok": "v"}) == {}

    def test_missing_secret_raises_without_leaking_value(self):
        with pytest.raises(ValueError) as exc:
            Environment._resolve_declared_secret_env_vars(
                _mappings(("MY_TOKEN", "absent")), {"tok": "super-secret-value"}
            )
        # The config error names the missing secret and env var but never the
        # secret VALUES that were available.
        assert "absent" in str(exc.value)
        assert "MY_TOKEN" in str(exc.value)
        assert "super-secret-value" not in str(exc.value)

    def test_missing_env_name_raises(self):
        with pytest.raises(ValueError, match="missing 'env_name'"):
            Environment._resolve_declared_secret_env_vars(
                _mappings((None, "tok")), {"tok": "v"}
            )

    def test_none_secret_bag_treated_as_empty(self):
        with pytest.raises(ValueError):
            Environment._resolve_declared_secret_env_vars(
                _mappings(("MY_TOKEN", "tok")), None
            )

    def test_key_names_include_llmb_twin(self):
        assert Environment._declared_secret_env_key_names(
            _mappings(("LLMB_MYVAL", "tok"))
        ) == {"LLMB_MYVAL", "GB_MYVAL"}

    def test_key_names_non_llmb_has_no_twin(self):
        assert Environment._declared_secret_env_key_names(
            _mappings(("MY_TOKEN", "tok"))
        ) == {"MY_TOKEN"}

    def test_key_names_empty_mappings(self):
        assert Environment._declared_secret_env_key_names([]) == set()


# ---------------------------------------------------------------------------
# Per-subclass overrides
# ---------------------------------------------------------------------------

RUN_META = {"build_id": "real-build", "targetrun_id": "tr-1"}
# A launcher env that tries (and must fail) to shadow the standard var.
CONFLICT = {"GB_BUILD_ID": "from-launcher"}


class TestBashOverride:
    def _bash(self):
        from gbserver.environment.bash import Bash

        return Bash(event_q=asyncio.Queue())

    def test_inline_vars_and_authority(self):
        from pathlib import Path

        env = self._bash().get_launch_env_vars(
            run_metadata=RUN_META,
            launcher_config={"env": CONFLICT},
            bash_config_env={},
            launch_id="lid",
            targetsteprun_asset_dir=Path("/assets"),
            final_asset_output_dir=Path("/out"),
        )
        assert env["LLMB_BASH_LAUNCH_ID"] == "lid"
        assert env["LLMB_BASH_ASSET_DIR"] == "/assets"
        assert env["LLMB_BASH_OUTPUT_DIR"] == "/out"
        assert "LLMB_BASH_PYTHON_DIR" in env
        # Each LLMB_BASH_* launcher var is mirrored onto a GB_BASH_* twin
        # (GB_ is the standard prefix; LLMB_ retained for compatibility).
        assert env["GB_BASH_LAUNCH_ID"] == "lid"
        assert env["GB_BASH_ASSET_DIR"] == "/assets"
        assert env["GB_BASH_OUTPUT_DIR"] == "/out"
        assert "GB_BASH_PYTHON_DIR" in env
        # standard var wins over the conflicting launcher env value
        assert env["GB_BUILD_ID"] == "real-build"

    def test_output_dir_absent_when_not_provided(self):
        env = self._bash().get_launch_env_vars(run_metadata=RUN_META, launch_id="lid")
        # Absent under both the legacy and standardized prefixes.
        assert "LLMB_BASH_OUTPUT_DIR" not in env
        assert "GB_BASH_OUTPUT_DIR" not in env


class TestDockerOverride:
    def _docker(self):
        from gbserver.environment.docker import Docker

        env = object.__new__(Docker)
        env.config = None  # _get_defaults() returns {} when config is None
        return env

    def test_inline_vars_and_authority(self):
        env = self._docker().get_launch_env_vars(
            run_metadata=RUN_META,
            launcher_config={"env": CONFLICT},
            docker_config={},
            launch_id="lid",
            container_name="cname",
        )
        assert env["LLMB_DOCKER_LAUNCH_ID"] == "lid"
        assert env["LLMB_DOCKER_CONTAINER_NAME"] == "cname"
        # GB_DOCKER_* twins mirror the legacy LLMB_DOCKER_* launcher vars.
        assert env["GB_DOCKER_LAUNCH_ID"] == "lid"
        assert env["GB_DOCKER_CONTAINER_NAME"] == "cname"
        # Docker gains GB_BUILD_ID (previously unset), authoritative
        assert env["GB_BUILD_ID"] == "real-build"

    def test_gbtest_vars_forwarded(self, monkeypatch):
        # Docker previously did NOT forward GBTEST_ vars; it now does, uniformly
        # via super().get_launch_env_vars().
        monkeypatch.setattr(
            environment_module,
            "get_exported_gbtest_env_vars",
            lambda: {"GBTEST_MOCK_HF": "true"},
        )
        env = self._docker().get_launch_env_vars(
            run_metadata=RUN_META, launch_id="lid", container_name="c"
        )
        assert env["GBTEST_MOCK_HF"] == "true"


class TestRunpodOverride:
    def _runpod(self):
        from gbserver.environment.runpod import Runpod

        return Runpod(event_q=asyncio.Queue())

    def test_inline_vars_and_authority(self):
        env = self._runpod().get_launch_env_vars(
            run_metadata=RUN_META,
            launcher_config={"env": CONFLICT},
            launch_id="lid",
            pod_name="pod",
        )
        assert env["LLMB_RUNPOD_LAUNCH_ID"] == "lid"
        assert env["LLMB_RUNPOD_POD_NAME"] == "pod"
        # GB_RUNPOD_* twins mirror the legacy LLMB_RUNPOD_* launcher vars.
        assert env["GB_RUNPOD_LAUNCH_ID"] == "lid"
        assert env["GB_RUNPOD_POD_NAME"] == "pod"
        assert env["GB_BUILD_ID"] == "real-build"


# A step declaring one secret (config.skypilot.secrets allow-list); the secret
# bag also holds an undeclared, hyphen-named entry that must NOT be injected
# (hyphenated keys are the invalid-envs crash that motivated declared-only).
_SKY_SECRET_CONFIG = {
    "skypilot": {
        "secrets": {
            "secret_names_to_use_as_env_variable": [
                {"env_name": "MY_TOKEN", "secret_name": "tok"}
            ]
        }
    }
}
_SKY_SECRET_BAG = {"tok": "secret-val", "rits-access": "hyphen-named"}


class TestSkypilotOverride:
    def _skypilot(self, secrets=None):
        from gbserver.environment.skypilot import Skypilot

        return Skypilot(event_q=asyncio.Queue(), secrets=secrets)

    def test_inline_vars_and_authority(self):
        env = self._skypilot().get_launch_env_vars(
            run_metadata=RUN_META,
            launcher_config={"envs": CONFLICT},
            launch_id="lid",
            cluster_name="cl",
        )
        assert env["GB_SKYPILOT_LAUNCH_ID"] == "lid"
        assert env["GB_SKYPILOT_CLUSTER_NAME"] == "cl"
        # GB_TARGETRUN_ID stays an inline skypilot var
        assert env["GB_TARGETRUN_ID"] == "tr-1"
        assert env["GB_BUILD_ID"] == "real-build"

    def test_only_declared_secret_injected(self):
        # The declared secret is injected; the undeclared, hyphen-named bag entry
        # is NOT (least-privilege; and hyphenated keys would break sky's envs).
        env = self._skypilot(secrets=_SKY_SECRET_BAG).get_launch_env_vars(
            run_metadata=RUN_META, config=_SKY_SECRET_CONFIG, launch_id="lid"
        )
        assert env["MY_TOKEN"] == "secret-val"
        assert "rits-access" not in env
        assert "hyphen-named" not in env.values()

    def test_no_declared_secrets_injects_nothing(self):
        # With no config.skypilot.secrets allow-list, the whole bag stays out.
        env = self._skypilot(secrets=_SKY_SECRET_BAG).get_launch_env_vars(
            run_metadata=RUN_META, launch_id="lid"
        )
        assert "tok" not in env and "rits-access" not in env
        assert "secret-val" not in env.values()

    def test_missing_declared_secret_raises(self):
        with pytest.raises(ValueError, match="tok"):
            self._skypilot(secrets={"other": "v"}).get_launch_env_vars(
                run_metadata=RUN_META, config=_SKY_SECRET_CONFIG, launch_id="lid"
            )

    def test_launcher_env_wins_over_declared_secret(self):
        # Precedence: launcher envs override declared-secret vars of the same name.
        env = self._skypilot(secrets=_SKY_SECRET_BAG).get_launch_env_vars(
            run_metadata=RUN_META,
            config=_SKY_SECRET_CONFIG,
            launcher_config={"envs": {"MY_TOKEN": "from-launcher"}},
            launch_id="lid",
        )
        assert env["MY_TOKEN"] == "from-launcher"


class TestSkypilotManagedOverride:
    def _managed(self, secrets=None):
        from gbserver.environment.skypilot_managed import Skypilot_managed

        return Skypilot_managed(event_q=asyncio.Queue(), secrets=secrets)

    def test_inline_vars_and_authority(self):
        env = self._managed().get_launch_env_vars(
            run_metadata=RUN_META,
            launcher_config={"envs": CONFLICT},
            launch_id="lid",
            job_name="job",
        )
        assert env["GB_SKYPILOT_LAUNCH_ID"] == "lid"
        assert env["GB_SKYPILOT_JOB_NAME"] == "job"
        assert env["GB_TARGETRUN_ID"] == "tr-1"
        assert env["GB_BUILD_ID"] == "real-build"

    def test_only_declared_secret_injected(self):
        env = self._managed(secrets=_SKY_SECRET_BAG).get_launch_env_vars(
            run_metadata=RUN_META, config=_SKY_SECRET_CONFIG, launch_id="lid"
        )
        assert env["MY_TOKEN"] == "secret-val"
        assert "rits-access" not in env
        assert "hyphen-named" not in env.values()

    def test_missing_declared_secret_raises(self):
        with pytest.raises(ValueError, match="tok"):
            self._managed(secrets={"other": "v"}).get_launch_env_vars(
                run_metadata=RUN_META, config=_SKY_SECRET_CONFIG, launch_id="lid"
            )


class TestLsfOverride:
    def _lsf(self):
        from gbserver.environment.lsf import Lsf

        return object.__new__(Lsf)

    def test_secret_derived_vars_and_authority(self):
        config = {
            "lsf": {
                "secrets": {
                    "secret_names_to_use_as_env_variable": [
                        {"env_name": "MY_TOKEN", "secret_name": "tok"}
                    ]
                }
            }
        }
        setup_config = {"space_secrets": {"tok": "secret-val"}}
        env = self._lsf().get_launch_env_vars(
            run_metadata=RUN_META, config=config, setup_config=setup_config
        )
        assert env["MY_TOKEN"] == "secret-val"
        # LSF gains GB_BUILD_ID (SSH path), authoritative
        assert env["GB_BUILD_ID"] == "real-build"

    def test_no_secrets_still_has_standard_set(self):
        env = self._lsf().get_launch_env_vars(run_metadata=RUN_META)
        assert env["GB_BUILD_ID"] == "real-build"

    def test_get_secret_env_keys_mirrors_llmb_twin(self):
        # An LLMB_-prefixed secret name also contributes its GB_ twin name, so
        # the twin _add_gb_aliases mints is masked by name in redaction.
        from gbserver.environment.lsf import Lsf

        def _cfg(env_name):
            return {
                "lsf": {
                    "secrets": {
                        "secret_names_to_use_as_env_variable": [
                            {"env_name": env_name, "secret_name": "tok"}
                        ]
                    }
                }
            }

        assert Lsf._get_secret_env_keys(_cfg("LLMB_MYVAL")) == {
            "LLMB_MYVAL",
            "GB_MYVAL",
        }
        # A non-LLMB_ name gets no twin.
        assert Lsf._get_secret_env_keys(_cfg("MY_TOKEN")) == {"MY_TOKEN"}
        # Empty / absent config yields an empty set.
        assert Lsf._get_secret_env_keys({}) == set()
        assert Lsf._get_secret_env_keys(None) == set()

    def test_llmb_secret_twin_is_covered_by_redaction_keys(self):
        # Ties twin creation to twin masking: the GB_ twin that aliasing mints
        # for an LLMB_-named secret must be in the redaction key set.
        from gbserver.environment.lsf import Lsf

        config = {
            "lsf": {
                "secrets": {
                    "secret_names_to_use_as_env_variable": [
                        {"env_name": "LLMB_MYVAL", "secret_name": "tok"}
                    ]
                }
            }
        }
        setup_config = {"space_secrets": {"tok": "secret-val"}}
        env = self._lsf().get_launch_env_vars(
            run_metadata=RUN_META, config=config, setup_config=setup_config
        )
        # The twin exists (aliasing still runs last) ...
        assert env["GB_MYVAL"] == "secret-val"
        # ... and is covered by the redaction key set, so it is masked.
        assert "GB_MYVAL" in Lsf._get_secret_env_keys(config)


@requires_k8s
class TestK8sOverride:
    def _k8s(self):
        from gbserver.environment.k8s import K8s

        return object.__new__(K8s)

    def test_standard_set_and_helm_string_mapping(self):
        env = self._k8s().get_launch_env_vars(run_metadata=RUN_META)
        assert env["GB_BUILD_ID"] == "real-build"
        # Mirror launch_helm's assembly: each entry becomes a --set-string arg.
        string_values = [(f"k8s.env.{k}.value", v) for k, v in env.items()]
        assert ("k8s.env.GB_BUILD_ID.value", "real-build") in string_values

    def test_run_metadata_var_wins_over_gbtest(self, monkeypatch):
        # K8s inherits the base method; a GBTEST-sourced GB_BUILD_ID must still
        # lose to the run_metadata-derived one (patched in the environment module
        # where the base method resolves the name).
        monkeypatch.setattr(
            environment_module,
            "get_exported_gbtest_env_vars",
            lambda: {"GB_BUILD_ID": "from-gbtest", "GBTEST_MOCK_HF": "true"},
        )
        env = self._k8s().get_launch_env_vars(run_metadata=RUN_META)
        assert env["GB_BUILD_ID"] == "real-build"
        assert env["GBTEST_MOCK_HF"] == "true"


@requires_k8s
class TestK8sSecretEnvHelmValues:
    """Direct tests for ``K8s._secret_env_helm_values`` — the secretKeyRef
    Helm-arg builder that exposes each declared secret under its verbatim
    ``env_name`` plus a DEPRECATED lowercase alias for back-compat."""

    def _values(self, mappings, space_secret="sp"):
        from gbserver.environment.k8s import K8s

        return K8s._secret_env_helm_values(mappings, space_secret)

    def test_uppercase_name_emits_verbatim_and_lower_alias(self):
        # No secret_name: exposed under both MY_TOKEN (verbatim, portable) and
        # my_token (deprecated lowercase alias), each keyed to its own case.
        assert self._values(_mappings(("MY_TOKEN", None))) == [
            ("k8s.env.MY_TOKEN.valueFrom.secretKeyRef.name", "sp"),
            ("k8s.env.MY_TOKEN.valueFrom.secretKeyRef.key", "MY_TOKEN"),
            ("k8s.env.my_token.valueFrom.secretKeyRef.name", "sp"),
            ("k8s.env.my_token.valueFrom.secretKeyRef.key", "my_token"),
        ]

    def test_already_lowercase_name_emits_single_entry(self):
        # verbatim == lowercase -> the alias is deduped away.
        assert self._values(_mappings(("hf_token", None))) == [
            ("k8s.env.hf_token.valueFrom.secretKeyRef.name", "sp"),
            ("k8s.env.hf_token.valueFrom.secretKeyRef.key", "hf_token"),
        ]

    def test_explicit_secret_name_shared_by_both_aliases(self):
        # An explicit (often hyphenated) secret_name is the data-key for both
        # the verbatim and the deprecated lowercase env-var names.
        assert self._values(_mappings(("MY_TOKEN", "huggingface-token"))) == [
            ("k8s.env.MY_TOKEN.valueFrom.secretKeyRef.name", "sp"),
            ("k8s.env.MY_TOKEN.valueFrom.secretKeyRef.key", "huggingface-token"),
            ("k8s.env.my_token.valueFrom.secretKeyRef.name", "sp"),
            ("k8s.env.my_token.valueFrom.secretKeyRef.key", "huggingface-token"),
        ]

    def test_missing_space_secret_raises(self):
        import pytest as _pytest

        with _pytest.raises(ValueError, match="space"):
            self._values(_mappings(("MY_TOKEN", None)), space_secret=None)

    def test_empty_mappings_yield_empty_even_without_space_secret(self):
        # No declared env vars -> nothing emitted and no space-secret needed.
        assert self._values([], space_secret=None) == []

    def test_mapping_without_env_name_raises(self):
        # A malformed entry fails fast, matching the shared LSF/SkyPilot path,
        # rather than being silently dropped.
        with pytest.raises(ValueError, match="missing 'env_name'"):
            self._values(_mappings((None, "tok")))


class TestAddGbAliases:
    """Direct tests for the shared ``Environment._add_gb_aliases`` helper."""

    def test_mirrors_llmb_vars_to_gb_twins(self):
        env = {"LLMB_BASH_LAUNCH_ID": "lid", "LLMB_BASH_ASSET_DIR": "/a"}
        result = Environment._add_gb_aliases(env)
        # Returns the same dict, mutated in place.
        assert result is env
        # Each LLMB_ var gains a same-value GB_ twin; the LLMB_ name stays.
        assert env["GB_BASH_LAUNCH_ID"] == "lid"
        assert env["GB_BASH_ASSET_DIR"] == "/a"
        assert env["LLMB_BASH_LAUNCH_ID"] == "lid"
        assert env["LLMB_BASH_ASSET_DIR"] == "/a"

    def test_does_not_overwrite_existing_gb_key(self):
        # A GB_ twin that is already present (e.g. the authoritative standard
        # set) must win over the LLMB_ value.
        env = {"LLMB_BUILD_ID": "from-llmb", "GB_BUILD_ID": "authoritative"}
        Environment._add_gb_aliases(env)
        assert env["GB_BUILD_ID"] == "authoritative"

    def test_leaves_non_llmb_vars_untouched(self):
        env = {"GB_SKYPILOT_LAUNCH_ID": "lid", "HF_TOKEN": "tok"}
        Environment._add_gb_aliases(env)
        # No LLMB_ keys -> no new keys added.
        assert env == {"GB_SKYPILOT_LAUNCH_ID": "lid", "HF_TOKEN": "tok"}
