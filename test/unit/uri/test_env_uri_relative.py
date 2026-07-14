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

"""Relative ``env:`` URIs are rejected.

An ``env:`` URI performs no transfer (it references a path the environment can
already reach), so a relative path has no resolution root and is disallowed.
Rejection happens at build-config load (``BuildConfig.my_validate``) and, as a
defense-in-depth guard, in the base ``pull/push`` envstore methods. Absolute
``env:///…`` and templated ``env://{{ … }}`` (which resolve to an absolute path)
remain valid.
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest

import gbserver.asset.asset as asset_mod
from gbcommon.uri.env import is_relative_env_uri
from gbserver.asset.envstore import Envstore
from gbserver.environment.environment import Environment
from gbserver.types.buildconfig import (
    BuildConfig,
    BuildTargetConfig,
    BuildTargetInputConfig,
    BuildTargetOutputConfig,
)

pytestmark = pytest.mark.standalone


# --- helper -----------------------------------------------------------------


@pytest.mark.parametrize(
    "uri_str,expected",
    [
        ("env:outputs/x/", True),
        ("env:outputs/inf_{{ binding.path | short_hash }}/", True),
        ("env:result.txt", True),
        ("env:///tmp/x", False),
        ("env://two/slashes", False),
        ("env://{{ binding.path }}", False),
        ("file:outputs/x", False),
        ("hf:///ibm-granite/granite-4.0-h-350m", False),
        ("", False),
    ],
)
def test_is_relative_env_uri(uri_str, expected):
    assert is_relative_env_uri(uri_str) is expected


# --- load-time validation ---------------------------------------------------


def _build_config_with(inputs=None, outputs=None) -> BuildConfig:
    return BuildConfig(
        matched_base_key="granite.build",
        targets={
            "t": BuildTargetConfig(
                environment_uri="space://environments/bash",
                inputs=inputs or {},
                outputs=outputs or {},
                steps=[],
            ),
        },
    )


def test_relative_env_input_uri_is_rejected():
    cfg = _build_config_with(
        inputs={"in": BuildTargetInputConfig(uri="env:inputs/data")}
    )
    errors = cfg.my_validate()
    assert len(errors) > 0
    assert "env://" in str(errors)


def test_relative_env_output_uri_is_rejected():
    cfg = _build_config_with(
        outputs={"out": BuildTargetOutputConfig(uri="env:outputs/x/", type="fileset")}
    )
    errors = cfg.my_validate()
    assert len(errors) > 0
    assert "env://" in str(errors)


def test_absolute_and_templated_env_uris_are_allowed():
    cfg = _build_config_with(
        inputs={"in": BuildTargetInputConfig(uri="env:///abs/in")},
        outputs={
            "abs": BuildTargetOutputConfig(uri="env:///abs/out", type="fileset"),
            "tmpl": BuildTargetOutputConfig(
                uri="env://{{ binding.path }}", type="fileset"
            ),
        },
    )
    errors = cfg.my_validate()
    # No env-relative errors (the only content is these valid env: URIs).
    assert not any("env://" in e for e in [str(x) for x in errors])


# --- runtime guard (base envstore methods) ----------------------------------
# The base methods don't touch instance state for the relative check, so we can
# invoke the unbound coroutines with a dummy ``self``.


def test_pushasset_envstore_rejects_relative():
    with pytest.raises(ValueError, match="relative env://"):
        asyncio.run(
            Environment.pushasset_envstore(
                None, binding={"path": "/x"}, uri="env:rel/out"
            )
        )


def test_pushasset_envstore_allows_absolute():
    uri = asyncio.run(
        Environment.pushasset_envstore(
            None, binding={"path": "/x"}, uri="env:///abs/out"
        )
    )
    assert "abs/out" in str(uri)


def test_pullasset_envstore_rejects_relative():
    with pytest.raises(ValueError, match="relative env://"):
        asyncio.run(Environment.pullasset_envstore(None, uri="env:rel/in"))


def test_pullasset_envstore_allows_absolute():
    binding_config, step = asyncio.run(
        Environment.pullasset_envstore(None, uri="env:///abs/in")
    )
    assert step is None
    assert binding_config["binding"]["path"] == "/abs/in"


# --- _resolve_env_assetstore: distinguish "absent" (debug) vs "malformed" (warning) ---
# Both fall back to the bundled builtin env-local store; only the log level differs.
# _resolve_env_assetstore reads only self.context, so a tiny stub self suffices.


def _run_resolve_with_space_error(monkeypatch, caplog, exc):
    """Force the space env-local lookup to raise ``exc`` and resolve the store.

    Returns the resolved Assetstore (always the bundled default here) while
    ``caplog`` captures the log record emitted for the failure.
    """

    def _raise(**kwargs):
        raise exc

    monkeypatch.setattr(
        asset_mod.Asset, "get_assetstore_from_store_uri", staticmethod(_raise)
    )
    stub = SimpleNamespace(context=None)
    with caplog.at_level(logging.DEBUG):
        store = Environment._resolve_env_assetstore(stub)
    return store


def test_resolve_env_assetstore_absent_logs_debug(monkeypatch, caplog):
    # No space defines env-local -> SpaceURI raises "Unresolvable space uri".
    store = _run_resolve_with_space_error(
        monkeypatch,
        caplog,
        ValueError("Unresolvable space uri : space://assetstores/env-local"),
    )
    assert isinstance(store, Envstore)  # fell back to the bundled default
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_resolve_env_assetstore_malformed_logs_warning(monkeypatch, caplog):
    # A space DOES define env-local but it fails to load -> should warn.
    store = _run_resolve_with_space_error(
        monkeypatch, caplog, RuntimeError("malformed store.yaml")
    )
    assert isinstance(store, Envstore)  # still falls back to the bundled default
    assert [r for r in caplog.records if r.levelno >= logging.WARNING]
