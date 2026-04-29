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
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gbserver.environment.bash import Bash
from gbserver.environment.environment import BINDING_KEY

pytestmark = pytest.mark.g4os


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
