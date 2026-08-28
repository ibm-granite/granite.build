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
        environment may export (e.g. a suite-level GBTEST_MOCKED_HF_OPS)."""
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
            lambda: {"GBTEST_MOCKED_HF_OPS": "push"},
        )
        assert _base_env_vars({"build_id": "b1"}) == {
            "GBTEST_MOCKED_HF_OPS": "push",
            "GB_BUILD_ID": "b1",
        }


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
            lambda: {"GBTEST_MOCKED_HF_OPS": "all"},
        )
        env = self._docker().get_launch_env_vars(
            run_metadata=RUN_META, launch_id="lid", container_name="c"
        )
        assert env["GBTEST_MOCKED_HF_OPS"] == "all"


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


class TestSkypilotOverride:
    def _skypilot(self):
        from gbserver.environment.skypilot import Skypilot

        return Skypilot(event_q=asyncio.Queue())

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


class TestSkypilotManagedOverride:
    def _managed(self):
        from gbserver.environment.skypilot_managed import Skypilot_managed

        return Skypilot_managed(event_q=asyncio.Queue())

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
            lambda: {"GB_BUILD_ID": "from-gbtest", "GBTEST_MOCKED_HF_OPS": "1"},
        )
        env = self._k8s().get_launch_env_vars(run_metadata=RUN_META)
        assert env["GB_BUILD_ID"] == "real-build"
        assert env["GBTEST_MOCKED_HF_OPS"] == "1"


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
