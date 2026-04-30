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
from unittest.mock import MagicMock, patch

import pytest

from gbcommon.uri.hf import HfType, HfURI
from gbserver.environment.docker import Docker
from gbserver.environment.environment import BINDING_KEY
from gbserver.environment.local_assets import (
    get_hf_cache_dir,
    pull_asset_hfstore,
    push_asset_hfstore,
)


@pytest.fixture
def docker_env():
    """Create a Docker environment instance with a dummy event queue."""
    event_q = asyncio.Queue()
    return Docker(event_q=event_q)


@pytest.fixture
def mock_assetstore():
    """Mock assetstore with get_secrets and get_relpath for HF model loading."""
    store = MagicMock()
    store.get_secrets.return_value = {}
    store.get_relpath.return_value = "ibm-granite/granite-4.0-350m/main"
    return store


@pytest.fixture
def hf_storeload_config(tmp_path):
    """storeload_config that scopes the HF cache under tmp_path."""
    config = MagicMock()
    config.config = {"cache_path": str(tmp_path / "hf-cache")}
    return config


@pytest.mark.asyncio
async def test_pullasset_hfstore_returns_container_path(
    docker_env, mock_assetstore, hf_storeload_config
):
    """Verify pullasset_hfstore returns a container-side path, not a host path."""
    uri = HfURI.from_parts(
        owner="ibm-granite", repo="granite-4.0-350m-base", hf_type=HfType.MODEL
    )

    with patch.object(HfURI, "pull", return_value=True):
        binding_config, extra_config = await docker_env.pullasset_hfstore(
            uri=uri,
            assetstore=mock_assetstore,
            storeload_config=hf_storeload_config,
        )

    # Container path comes from assetstore.get_relpath, not the local host path
    assert BINDING_KEY in binding_config
    container_path = binding_config[BINDING_KEY]["path"]
    # Must be a container path under /gb-hf-models, NOT a host filesystem path
    assert container_path == "/gb-hf-models/ibm-granite/granite-4.0-350m/main"
    assert extra_config is None
    mock_assetstore.get_relpath.assert_called_once_with(uri)


@pytest.mark.asyncio
async def test_pullasset_hfstore_registers_extra_volume(
    docker_env, mock_assetstore, hf_storeload_config, tmp_path
):
    """Verify pullasset_hfstore adds a read-only volume mount to _extra_volumes."""
    uri = HfURI.from_parts(
        owner="ibm-granite",
        repo="granite-4.0-350m",
        hf_type=HfType.MODEL,
        revision="main",
    )

    with patch.object(HfURI, "pull", return_value=True):
        await docker_env.pullasset_hfstore(
            uri=uri,
            assetstore=mock_assetstore,
            storeload_config=hf_storeload_config,
        )

    expected_host = str(
        tmp_path / "hf-cache" / "ibm-granite" / "granite-4.0-350m" / "main"
    )
    assert expected_host in docker_env._extra_volumes
    mount = docker_env._extra_volumes[expected_host]
    assert mount["bind"] == "/gb-hf-models/ibm-granite/granite-4.0-350m/main"
    assert mount["mode"] == "ro"


@pytest.mark.asyncio
async def test_pullasset_hfstore_multiple_models_accumulate(docker_env, tmp_path):
    """Verify multiple pullasset_hfstore calls accumulate volumes."""
    store1 = MagicMock()
    store1.get_secrets.return_value = {}
    store1.get_relpath.return_value = "org1/model1/main"

    store2 = MagicMock()
    store2.get_secrets.return_value = {}
    store2.get_relpath.return_value = "org2/model2/main"

    uri1 = HfURI.from_parts(owner="org1", repo="model1", hf_type=HfType.MODEL)
    uri2 = HfURI.from_parts(owner="org2", repo="model2", hf_type=HfType.MODEL)

    cache = tmp_path / "hf-cache"
    sc = MagicMock()
    sc.config = {"cache_path": str(cache)}

    with patch.object(HfURI, "pull", return_value=True):
        await docker_env.pullasset_hfstore(
            uri=uri1, assetstore=store1, storeload_config=sc
        )
        await docker_env.pullasset_hfstore(
            uri=uri2, assetstore=store2, storeload_config=sc
        )

    assert len(docker_env._extra_volumes) == 2
    path1 = str(cache / "org1" / "model1" / "main")
    path2 = str(cache / "org2" / "model2" / "main")
    assert path1 in docker_env._extra_volumes
    assert path2 in docker_env._extra_volumes
    assert docker_env._extra_volumes[path1]["bind"] == "/gb-hf-models/org1/model1/main"
    assert docker_env._extra_volumes[path2]["bind"] == "/gb-hf-models/org2/model2/main"


# ---------------------------------------------------------------------------
# _resolve_host_path tests
# ---------------------------------------------------------------------------


def test_resolve_host_path_translates_workspace_path(docker_env, tmp_path):
    """Container paths under /gb-workspace resolve to the host asset dir."""
    docker_env._extra_volumes[str(tmp_path / "assets")] = {
        "bind": "/gb-workspace",
        "mode": "rw",
    }
    result = docker_env._resolve_host_path("/gb-workspace/outputs/model")
    assert result == str(tmp_path / "assets") + "/outputs/model"


def test_resolve_host_path_exact_mount_match(docker_env, tmp_path):
    """An exact mount point (no suffix) resolves to the host directory."""
    docker_env._extra_volumes[str(tmp_path / "assets")] = {
        "bind": "/gb-workspace",
        "mode": "rw",
    }
    result = docker_env._resolve_host_path("/gb-workspace")
    assert result == str(tmp_path / "assets")


def test_resolve_host_path_returns_original_when_no_match(docker_env):
    """Paths with no registered volume are returned unchanged."""
    result = docker_env._resolve_host_path("/some/unregistered/path")
    assert result == "/some/unregistered/path"


def test_resolve_host_path_prefers_longest_prefix(docker_env, tmp_path):
    """The most specific (longest) matching mount wins."""
    docker_env._extra_volumes[str(tmp_path / "models")] = {
        "bind": "/gb-hf-models",
        "mode": "ro",
    }
    docker_env._extra_volumes[str(tmp_path / "specific")] = {
        "bind": "/gb-hf-models/org/repo",
        "mode": "ro",
    }
    result = docker_env._resolve_host_path("/gb-hf-models/org/repo/file.bin")
    assert result == str(tmp_path / "specific") + "/file.bin"


# ---------------------------------------------------------------------------
# pushasset_hfstore tests
# ---------------------------------------------------------------------------


@pytest.fixture
def push_assetstore():
    """Mock assetstore with an HF_TOKEN in its secrets."""
    store = MagicMock()
    store.get_secrets.return_value = {"HF_TOKEN": "test-hf-token"}
    return store


@pytest.mark.asyncio
async def test_pushasset_hfstore_translates_container_path(
    docker_env, push_assetstore, tmp_path
):
    """pushasset_hfstore translates a /gb-workspace container path to the host path."""
    output_dir = tmp_path / "assets" / "outputs"
    output_dir.mkdir(parents=True)
    (output_dir / "model.bin").write_bytes(b"weights")

    # Simulate the workspace mount registered by launch_docker
    docker_env._extra_volumes[str(tmp_path / "assets")] = {
        "bind": "/gb-workspace",
        "mode": "rw",
    }

    container_binding = {"path": "/gb-workspace/outputs/model.bin"}
    uri = HfURI.from_parts(owner="org", repo="my-model", hf_type=HfType.MODEL)

    with patch.object(HfURI, "push", return_value=True) as mock_push:
        await docker_env.pushasset_hfstore(
            binding=container_binding,
            uri=uri,
            assetstore=push_assetstore,
        )

    # HfURI.push must receive the host-side path, not the container path
    pushed_src = mock_push.call_args[0][0]
    assert str(pushed_src) == str(tmp_path / "assets" / "outputs" / "model.bin")


@pytest.mark.asyncio
async def test_pushasset_hfstore_calls_hfuri_push(
    docker_env, push_assetstore, tmp_path
):
    """pushasset_hfstore delegates to HfURI.push() and returns the URI."""
    src = tmp_path / "model.bin"
    src.write_bytes(b"weights")
    binding = {"path": str(src)}
    uri = HfURI.from_parts(owner="org", repo="my-model", hf_type=HfType.MODEL)

    with patch.object(HfURI, "push", return_value=True) as mock_push:
        result = await docker_env.pushasset_hfstore(
            binding=binding,
            uri=uri,
            assetstore=push_assetstore,
        )

    assert result is uri
    mock_push.assert_called_once_with(
        src,
        commit_message="Upload via gbserver [build= target= output=]",
        space_name="public",
    )


@pytest.mark.asyncio
async def test_pushasset_hfstore_injects_assetstore_secrets(
    docker_env, push_assetstore, tmp_path
):
    """Secrets from the assetstore are merged into the URI before pushing."""
    src = tmp_path / "f.txt"
    src.write_text("data")
    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)

    with patch.object(HfURI, "push", return_value=True):
        await docker_env.pushasset_hfstore(
            binding={"path": str(src)},
            uri=uri,
            assetstore=push_assetstore,
        )

    assert uri.secrets.get("HF_TOKEN") == "test-hf-token"


@pytest.mark.asyncio
async def test_pushasset_hfstore_commit_message_includes_run_metadata(
    docker_env, push_assetstore, tmp_path
):
    """Default commit message encodes build_id, target_name, and output name."""
    src = tmp_path / "f.txt"
    src.write_text("data")
    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    run_metadata = MagicMock()
    run_metadata.build_id = "build-abc"
    run_metadata.target_name = "my-target"

    with patch.object(HfURI, "push", return_value=True) as mock_push:
        await docker_env.pushasset_hfstore(
            binding={"path": str(src)},
            binding_id="my-output",
            uri=uri,
            assetstore=push_assetstore,
            run_metadata=run_metadata,
        )

    _, kwargs = mock_push.call_args
    assert kwargs["commit_message"] == (
        "Upload via gbserver [build=build-abc target=my-target output=my-output]"
    )


@pytest.mark.asyncio
async def test_pushasset_hfstore_raises_on_empty_uri(docker_env, tmp_path):
    """ValueError is raised when uri is absent."""
    with pytest.raises(ValueError, match="Empty uri"):
        await docker_env.pushasset_hfstore(binding={"path": str(tmp_path)}, uri=None)


@pytest.mark.asyncio
async def test_pushasset_hfstore_raises_on_missing_path(docker_env, tmp_path):
    """ValueError is raised when binding has no 'path' key."""
    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    with pytest.raises(ValueError, match="binding must be a dict"):
        await docker_env.pushasset_hfstore(binding={}, uri=uri)


@pytest.mark.asyncio
async def test_pushasset_hfstore_raises_on_push_failure(
    docker_env, push_assetstore, tmp_path
):
    """Exception from HfURI.push() propagates out of pushasset_hfstore."""
    src = tmp_path / "f.bin"
    src.write_bytes(b"x")
    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)

    with patch.object(HfURI, "push", side_effect=RuntimeError("push failed")):
        with pytest.raises(RuntimeError, match="push failed"):
            await docker_env.pushasset_hfstore(
                binding={"path": str(src)},
                uri=uri,
                assetstore=push_assetstore,
            )


# ---------------------------------------------------------------------------
# Standalone assets.py function tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_push_asset_hfstore_standalone_succeeds(tmp_path):
    """push_asset_hfstore succeeds independently of any environment class."""
    src = tmp_path / "model.bin"
    src.write_bytes(b"weights")
    store = MagicMock()
    store.get_secrets.return_value = {"HF_TOKEN": "tok"}
    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)

    with patch.object(HfURI, "push", return_value=True) as mock_push:
        result = push_asset_hfstore(
            src=str(src),
            binding_id="my-output",
            uri=uri,
            assetstore=store,
        )

    assert result is uri
    mock_push.assert_called_once()


@pytest.mark.asyncio
async def test_push_asset_hfstore_raises_on_empty_uri(tmp_path):
    """ValueError is raised when uri is None."""
    with pytest.raises(ValueError, match="Empty uri"):
        push_asset_hfstore(src=str(tmp_path), uri=None)


@pytest.mark.asyncio
async def test_push_asset_hfstore_raises_on_empty_src():
    """ValueError is raised when src is empty."""
    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    with pytest.raises(ValueError, match="src path is empty"):
        push_asset_hfstore(src="", uri=uri)


def test_get_hf_cache_dir_returns_default_when_no_config():
    """get_hf_cache_dir returns ~/.cache/gbserver/hf when config is None."""
    result = get_hf_cache_dir(None)
    assert result.endswith("hf")
    assert "gbserver" in result


def test_get_hf_cache_dir_uses_config_cache_path():
    """get_hf_cache_dir honours cache_path from storeload_config."""
    config = MagicMock()
    config.config = {"cache_path": "/custom/hf/cache"}
    assert get_hf_cache_dir(config) == "/custom/hf/cache"


# ---------------------------------------------------------------------------
# pull_asset_hfstore tests
# ---------------------------------------------------------------------------


def test_pull_asset_hfstore_returns_path(tmp_path):
    """pull_asset_hfstore pulls the HF repo and returns the local dest path."""
    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    store = MagicMock()
    store.get_secrets.return_value = {}
    sc = MagicMock()
    sc.config = {"cache_path": str(tmp_path / "cache")}

    with patch.object(HfURI, "pull", return_value=True):
        result = pull_asset_hfstore(uri, store, sc)

    assert result == tmp_path / "cache" / "org" / "repo" / "main"


def test_pull_asset_hfstore_uses_custom_cache_path(tmp_path):
    """pull_asset_hfstore passes the resolved dest dir to HfURI.pull."""
    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    store = MagicMock()
    store.get_secrets.return_value = {}
    sc = MagicMock()
    sc.config = {"cache_path": str(tmp_path / "custom-cache")}

    with patch.object(HfURI, "pull", return_value=True) as mock_pull:
        pull_asset_hfstore(uri, store, sc)

    dest = mock_pull.call_args[0][0]
    assert str(dest).startswith(str(tmp_path / "custom-cache"))


def test_pull_asset_hfstore_injects_assetstore_secrets(tmp_path):
    """Secrets from assetstore are merged into the HfURI before pulling."""
    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    store = MagicMock()
    store.get_secrets.return_value = {"HF_TOKEN": "test-token"}
    sc = MagicMock()
    sc.config = {"cache_path": str(tmp_path)}

    with patch.object(HfURI, "pull", return_value=True):
        pull_asset_hfstore(uri, store, sc)

    assert uri.secrets.get("HF_TOKEN") == "test-token"


def test_pull_asset_hfstore_raises_on_pull_failure(tmp_path):
    """RuntimeError is raised when HfURI.pull returns False."""
    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    sc = MagicMock()
    sc.config = {"cache_path": str(tmp_path)}

    with patch.object(HfURI, "pull", return_value=False):
        with pytest.raises(RuntimeError, match="HF pull failed"):
            pull_asset_hfstore(uri, None, sc)


def test_pull_asset_hfstore_raises_on_no_uri():
    """pull_asset_hfstore raises AssertionError when uri is None."""
    with pytest.raises(AssertionError, match="uri is required"):
        pull_asset_hfstore(None, MagicMock(), None)


def test_pull_asset_hfstore_bucket_cache_path_omits_revision(tmp_path):
    """Bucket cache path is owner/repo with no revision segment."""
    uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)
    store = MagicMock()
    store.get_secrets.return_value = {}
    sc = MagicMock()
    sc.config = {"cache_path": str(tmp_path / "cache")}

    with patch.object(HfURI, "pull", return_value=True):
        result = pull_asset_hfstore(uri, store, sc)

    assert result == tmp_path / "cache" / "org" / "my-bucket"
