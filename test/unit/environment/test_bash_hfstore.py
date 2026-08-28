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

import asyncio
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gbserver.environment.bash import Bash
from gbserver.environment.environment import BINDING_KEY


@pytest.fixture
def bash_env():
    """Create a Bash environment instance with a dummy event queue."""
    event_q = asyncio.Queue()
    return Bash(event_q=event_q)


@pytest.mark.asyncio
async def test_pullasset_hfstore_returns_binding_with_path(bash_env, tmp_path):
    """Verify pullasset_hfstore returns a binding with path from pull_asset_hfstore."""
    model_dir = tmp_path / "models" / "granite-3b"
    model_dir.mkdir(parents=True)
    uri = MagicMock()

    with patch("gbserver.environment.bash.pull_asset_hfstore", return_value=model_dir):
        binding_config, extra_config = await bash_env.pullasset_hfstore(
            uri=uri,
            assetstore=None,
        )

    assert BINDING_KEY in binding_config
    assert "path" in binding_config[BINDING_KEY]
    assert binding_config[BINDING_KEY]["path"] == str(model_dir)
    assert extra_config is None


@pytest.mark.asyncio
async def test_pullasset_hfstore_passes_cache_dir(bash_env, tmp_path):
    """Verify storeload_config is forwarded to pull_asset_hfstore."""
    storeload_config = MagicMock()
    storeload_config.mode = "default"
    storeload_config.config = {"cache_path": str(tmp_path / "custom_cache")}
    uri = MagicMock()
    assetstore = MagicMock()

    with patch(
        "gbserver.environment.bash.pull_asset_hfstore", return_value=tmp_path
    ) as mock_load:
        await bash_env.pullasset_hfstore(
            uri=uri,
            assetstore=assetstore,
            storeload_config=storeload_config,
        )

    mock_load.assert_called_once_with(uri, assetstore, storeload_config)


@pytest.mark.asyncio
async def test_pullasset_hfstore_uses_default_cache(bash_env, tmp_path):
    """Verify storeload_config=None is forwarded to pull_asset_hfstore."""
    uri = MagicMock()
    assetstore = MagicMock()

    with patch(
        "gbserver.environment.bash.pull_asset_hfstore", return_value=tmp_path
    ) as mock_load:
        await bash_env.pullasset_hfstore(
            uri=uri,
            assetstore=assetstore,
            storeload_config=None,
        )

    mock_load.assert_called_once_with(uri, assetstore, None)


@pytest.mark.asyncio
async def test_pushasset_hfstore_pushes_binding_path(bash_env):
    """pushasset_hfstore forwards the host binding path to push_asset_hfstore."""
    uri = MagicMock()
    assetstore = MagicMock()
    run_metadata = MagicMock()

    with patch(
        "gbserver.environment.bash.push_asset_hfstore", return_value=uri
    ) as mock_push:
        result = await bash_env.pushasset_hfstore(
            binding={"path": "/workspace/output/model"},
            binding_id="hf_output",
            uri=uri,
            assetstore=assetstore,
            run_metadata=run_metadata,
        )

    # The bash binding path is already a host path — passed straight through.
    # The push configs are forwarded (None here) so store_push settings reach the
    # shared helper; bash/docker have no hfpush step to carry them otherwise.
    mock_push.assert_called_once_with(
        src="/workspace/output/model",
        binding_id="hf_output",
        uri=uri,
        assetstore=assetstore,
        run_metadata=run_metadata,
        storepush_config=None,
        output_config=None,
    )
    assert result is uri


@pytest.mark.asyncio
async def test_pushasset_hfstore_rejects_binding_without_path(bash_env):
    """pushasset_hfstore raises ValueError when the binding has no 'path'."""
    with pytest.raises(ValueError, match="binding must be a dict with 'path'"):
        await bash_env.pushasset_hfstore(
            binding={"not_path": "x"},
            uri=MagicMock(),
        )


@pytest.mark.asyncio
async def test_pushasset_hfstore_accepts_legacy_mode_with_warning(bash_env, caplog):
    """A legacy (non-'default') push mode is accepted for backwards compat, and warns.

    Outside k8s ``mode`` is ignored (dispatch is by store type), so a legacy
    ``hf_push`` still pushes normally — it just logs a deprecation warning.
    """
    uri = MagicMock()
    storepush_config = MagicMock()
    storepush_config.mode = "hf_push"

    with (
        patch(
            "gbserver.environment.bash.push_asset_hfstore", return_value=uri
        ) as mock_push,
        caplog.at_level(logging.WARNING),
    ):
        result = await bash_env.pushasset_hfstore(
            binding={"path": "/workspace/output/model"},
            uri=uri,
            storepush_config=storepush_config,
        )

    mock_push.assert_called_once()
    assert result is uri
    assert any("declares mode 'hf_push'" in r.message for r in caplog.records)
