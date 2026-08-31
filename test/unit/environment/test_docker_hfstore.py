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
from gbserver.asset.hfstore import Hfstore
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
    config.mode = "default"
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
    sc.mode = "default"
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
    """Mock assetstore with an HF_TOKEN in its secrets.

    Declares the "org" namespace used by these tests as an HF Enterprise org so
    the push path resolves a resource group; a non-Enterprise org skips
    resolution by design (see is_enterprise_hf_org).
    """
    store = MagicMock()
    store.get_secrets.return_value = {"HF_TOKEN": "test-hf-token"}
    store.get_enterprise_organizations.return_value = ["org"]
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

    with (
        patch(
            "gbserver.environment.local_assets.resolve_space_resource_group_id",
            return_value="rg-id",
        ),
        patch.object(HfURI, "push", return_value=True) as mock_push,
    ):
        result = await docker_env.pushasset_hfstore(
            binding=binding,
            uri=uri,
            assetstore=push_assetstore,
        )

    assert result is uri
    # The resource group id is resolved server-side (table-first) BEFORE the push
    # and passed as a pre-resolved id; space_name is intentionally NOT forwarded,
    # so HfURI.push does not re-derive the name and re-hit the admin-gated HF API.
    mock_push.assert_called_once_with(
        src,
        commit_message="Upload via gbserver [build= target= output=]",
        private=True,
        resource_group_id="rg-id",
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
async def test_push_asset_hfstore_uses_cached_resource_group_id(tmp_path):
    """The standalone push path resolves the id table-first and forwards it.

    Guards against regressing to the admin-token-only path: the id comes from
    resolve_space_resource_group_id (cache-aware) and is passed to HfURI.push as
    a pre-resolved id, with no space_name (which would re-hit the HF API).
    """
    src = tmp_path / "model.bin"
    src.write_bytes(b"weights")
    store = MagicMock()
    store.get_secrets.return_value = {"HF_TOKEN": "tok"}
    store.resolve_token.return_value = "tok"
    # "org" must be declared Enterprise, or the resource group is skipped by
    # design (see is_enterprise_hf_org).
    store.get_enterprise_organizations.return_value = ["org"]
    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)

    with (
        patch(
            "gbserver.environment.local_assets.resolve_space_resource_group_id",
            return_value="cached-rg-id",
        ) as mock_resolve,
        patch.object(HfURI, "push", return_value=True) as mock_push,
    ):
        result = push_asset_hfstore(
            src=str(src),
            binding_id="my-output",
            uri=uri,
            assetstore=store,
        )

    assert result is uri
    mock_resolve.assert_called_once()
    mock_push.assert_called_once_with(
        src,
        commit_message="Upload via gbserver [build= target= output=my-output]",
        private=True,
        resource_group_id="cached-rg-id",
    )


@pytest.mark.asyncio
async def test_push_asset_hfstore_best_effort_on_resolve_failure(tmp_path):
    """A failed resource-group resolution does not abort the push.

    In standalone the local user's token typically can't resolve the id via the
    HF API. The push must still be private (the default survives the failure
    path) and proceed with resource_group_id=None (matching
    pre-cache behavior) rather than raising.
    """
    src = tmp_path / "model.bin"
    src.write_bytes(b"weights")
    store = MagicMock()
    store.get_secrets.return_value = {"HF_TOKEN": "tok"}
    store.resolve_token.return_value = "tok"
    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)

    with (
        patch(
            "gbserver.environment.local_assets.resolve_space_resource_group_id",
            side_effect=ValueError("cannot resolve without admin token"),
        ),
        patch.object(HfURI, "push", return_value=True) as mock_push,
    ):
        result = push_asset_hfstore(
            src=str(src),
            binding_id="my-output",
            uri=uri,
            assetstore=store,
        )

    assert result is uri
    mock_push.assert_called_once_with(
        src,
        commit_message="Upload via gbserver [build= target= output=my-output]",
        private=True,
        resource_group_id=None,
    )


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


@pytest.mark.asyncio
async def test_push_asset_hfstore_tolerates_non_hfstore_assetstore(tmp_path):
    """A non-Hfstore assetstore must not abort the push.

    ``get_enterprise_organizations()`` is declared only on ``Hfstore``, and this
    inline path (shared by the Bash and Docker environments) has no isinstance
    assert — so reading it directly would turn any other assetstore into a fatal
    AttributeError on a best-effort code path. Absent the list we fall back to
    None, which ``is_enterprise_hf_org`` treats as "every org is Enterprise",
    i.e. the pre-split behavior.
    """
    src = tmp_path / "model.bin"
    src.write_bytes(b"weights")
    # A double WITHOUT get_enterprise_organizations (spec= omits it).
    store = MagicMock(spec=["get_secrets", "resolve_token"])
    store.get_secrets.return_value = {"HF_TOKEN": "tok"}
    store.resolve_token.return_value = "tok"
    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)

    with (
        patch(
            "gbserver.environment.local_assets.resolve_space_resource_group_id",
            return_value="rg-id",
        ) as mock_resolve,
        patch.object(HfURI, "push", return_value=True) as mock_push,
    ):
        result = push_asset_hfstore(
            src=str(src), binding_id="out", uri=uri, assetstore=store
        )

    assert result is uri
    # None => treated as Enterprise, so resolution still runs (pre-split behavior).
    mock_resolve.assert_called_once()
    assert mock_push.call_args.kwargs["resource_group_id"] == "rg-id"


@pytest.mark.asyncio
async def test_push_asset_hfstore_honors_store_push_use_resource_group(tmp_path):
    """The inline path must honor store_push, like the step-based environments.

    bash/docker have no hfpush step, so `store_push.config.hf` reaches HF only if
    they forward it into push_asset_hfstore. Without that, documented build.yaml
    fields (use_resource_group / resource_group_id / resource_group_name) are
    silently dropped on exactly the two environments where the non-Enterprise use
    case lives.
    """
    from gbserver.types.assetstoreconfig import AssetStoreConfig

    src = tmp_path / "model.bin"
    src.write_bytes(b"weights")
    store = Hfstore(
        AssetStoreConfig(base_uri="hf:/", config={"enterprise_organizations": ["org"]})
    )
    store.resolve_token = lambda uri: "tok"  # type: ignore[method-assign]
    store.get_secrets = lambda: {"HF_TOKEN": "tok"}  # type: ignore[method-assign]

    output_config = MagicMock()
    output_config.store_push = MagicMock()
    output_config.store_push.config = {"hf": {"use_resource_group": False}}

    uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
    with (
        patch(
            "gbserver.spaces.resource_group.resolve_space_resource_group_id",
            return_value="should-not-be-used",
        ) as mock_resolve,
        patch.object(HfURI, "push", return_value=True) as mock_push,
    ):
        push_asset_hfstore(
            src=str(src),
            binding_id="out",
            uri=uri,
            assetstore=store,
            output_config=output_config,
        )

    assert mock_push.call_args.kwargs["resource_group_id"] is None
    mock_resolve.assert_not_called()


@pytest.mark.asyncio
async def test_docker_pushasset_forwards_push_configs(docker_env, tmp_path):
    """docker.pushasset_hfstore must forward both push configs to the helper."""
    src = tmp_path / "m.bin"
    src.write_bytes(b"w")
    storepush_config = MagicMock()
    storepush_config.mode = "default"
    storepush_config.config = {"hf": {"private": False}}
    output_config = MagicMock()
    output_config.store_push = MagicMock()
    output_config.store_push.config = {"hf": {"resource_group_id": "rg-out"}}

    with patch(
        "gbserver.environment.docker.push_asset_hfstore", return_value=None
    ) as mock_helper:
        await docker_env.pushasset_hfstore(
            binding={"path": str(src)},
            uri=HfURI.from_parts(owner="org", repo="r"),
            assetstore=MagicMock(),
            storepush_config=storepush_config,
            output_config=output_config,
        )

    kwargs = mock_helper.call_args.kwargs
    assert kwargs["storepush_config"] is storepush_config
    assert kwargs["output_config"] is output_config


@pytest.mark.asyncio
async def test_push_asset_hfstore_uses_the_resolved_space_name(tmp_path):
    """The inline push must forward the space from the space config, not a literal.

    This path used to overwrite the resolved name with a hardcoded "public", so
    every bash/docker push to an Enterprise org resolved its resource group as if
    the build lived in the `public` space. The standalone aliases are folded onto
    "public" inside HfURI.space_name_to_resource_group_name instead, so a real
    space name now survives.
    """
    from gbcommon.uri.uri import URI
    from gbserver.types.assetstoreconfig import AssetStoreConfig

    src = tmp_path / "model.bin"
    src.write_bytes(b"weights")
    store = Hfstore(
        AssetStoreConfig(
            base_uri="hf:/", config={"enterprise_organizations": ["ibm-research"]}
        )
    )
    store.resolve_token = lambda uri: "tok"  # type: ignore[method-assign]
    store.get_secrets = lambda: {"HF_TOKEN": "tok"}  # type: ignore[method-assign]

    uri = HfURI.from_parts(owner="ibm-research", repo="repo", hf_type=HfType.MODEL)
    with (
        patch.object(
            URI, "get_space_config", return_value={"space": {"name": "my-team-space"}}
        ),
        patch(
            "gbserver.spaces.resource_group.resolve_space_resource_group_id",
            return_value="rg-id",
        ) as mock_resolve,
        patch.object(HfURI, "push", return_value=True),
    ):
        push_asset_hfstore(src=str(src), binding_id="out", uri=uri, assetstore=store)

    assert mock_resolve.call_args.kwargs["space_name"] == "my-team-space"
