#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Build-path coverage for a `type: local` space with no secrets_dir.

The standalone space.yaml uses `type: local` / `config: {}` (LocalSpaceSecretManager
defaults secrets_dir to <gb_home>/space_secrets). Besides the API path, that same
config is consumed by the build path via Space._fetch_secrets -> _is_first_local_sync.
This guards that the empty config does not trip the local-secrets assertions there.
"""

from gbserver.build.space import Space
from gbserver.types.spaceconfig import SpaceConfig, SpaceSecretManagerConfig


def _space_with_config(secret_manager: SpaceSecretManagerConfig) -> Space:
    """A Space instance whose space_config we control, without running __init__.

    __init__ pulls a URI and globs space.yaml; we only need _is_first_local_sync to
    see a chosen secret_manager config, so build the instance directly.
    """
    space = Space.__new__(Space)
    space.space_config = SpaceConfig(name="public", secret_manager=secret_manager)
    return space


def test_first_local_sync_local_empty_config_does_not_assert():
    """`type: local` / `config: {}` must not raise; with no remote sync -> False."""
    space = _space_with_config(SpaceSecretManagerConfig(type="local", config={}))

    # Pre-fix this raised AssertionError("...requires 'secrets_dir' in config").
    assert space._is_first_local_sync() is False


def test_first_local_sync_env_type_returns_false():
    space = _space_with_config(SpaceSecretManagerConfig(type="env", config={}))
    assert space._is_first_local_sync() is False


def test_first_local_sync_requires_secrets_dir_only_when_remote_sync():
    """secrets_dir is still required, but only once remote sync is requested."""
    import pytest

    space = _space_with_config(
        SpaceSecretManagerConfig(type="local", config={"do_remote_sync": True})
    )
    with pytest.raises(AssertionError, match="secrets_dir"):
        space._is_first_local_sync()
