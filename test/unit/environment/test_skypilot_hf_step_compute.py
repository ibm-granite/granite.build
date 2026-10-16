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

"""An assetstore's pull/push ``compute_config`` sizes the builtin hf steps."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from gbcommon.uri.hf import HfURI
from gbserver.environment.skypilot import _hf_step_compute_override
from gbserver.types.environmentconfig import StoreLoad, StorePush
from gbserver.utils.filesystem import merge_dicts

_REPO = Path(__file__).resolve().parents[3]
_SMALL = {"num_cpus_per_node": 1, "total_memory_per_node": "2Gi"}


@pytest.fixture
def skypilot_env():
    from gbserver.environment.skypilot import Skypilot
    from gbserver.types.environmentconfig import EnvironmentConfig

    cfg = EnvironmentConfig(
        name="test-skypilot", type="Skypilot", config={"default_cloud": "k8s"}
    )
    return Skypilot(event_q=asyncio.Queue(), environment_config=cfg)


@pytest.fixture
def hfuri():
    uri = MagicMock(spec=HfURI)
    uri.get_owner.return_value = "myorg"
    uri.get_repo.return_value = "myrepo"
    uri.get_revision.return_value = "main"
    uri.get_hf_type.return_value = "model"
    uri.get_host.return_value = "huggingface.co"
    uri.__str__ = lambda self: "hf://models/myorg/myrepo"
    return uri


def _store():
    from gbserver.asset.hfstore import Hfstore

    store = MagicMock(spec=Hfstore)
    store.resolve_token.return_value = "tok"
    store.get_enterprise_organizations.return_value = []
    return store


class TestOverrideHelper:
    @pytest.mark.parametrize("cfg", [None, StoreLoad(config={}), StorePush()])
    def test_none_when_unset(self, cfg):
        assert _hf_step_compute_override(cfg) == {}

    def test_returns_a_copy(self):
        compute = dict(_SMALL)
        out = _hf_step_compute_override(StorePush(config={"compute_config": compute}))
        assert out == {"compute_config": _SMALL}
        out["compute_config"]["total_memory_per_node"] = "1Gi"
        assert compute["total_memory_per_node"] == "2Gi"

    def test_rejects_non_mapping(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            _hf_step_compute_override(StoreLoad(config={"compute_config": "2Gi"}))


class TestQueuedSteps:
    @pytest.mark.asyncio
    async def test_pull_step_carries_it(self, skypilot_env, hfuri):
        load = StoreLoad(config={"cache_path": "/c", "compute_config": _SMALL})
        _, step = await skypilot_env.pullasset_hfstore(
            uri=hfuri, assetstore=_store(), storeload_config=load
        )
        assert step.config["compute_config"] == _SMALL
        assert step.config["launcher_config"]["envs"]["HF_TOKEN"] == "tok"

    @pytest.mark.asyncio
    async def test_push_step_carries_it(self, skypilot_env, hfuri):
        push = StorePush(config={"compute_config": _SMALL})
        with patch(
            "gbserver.spaces.hf_push_config.resolve_space_resource_group_id",
            return_value=None,
        ):
            step = await skypilot_env.pushasset_hfstore(
                binding={"path": "/out"},
                binding_id="b",
                uri=hfuri,
                assetstore=_store(),
                storepush_config=push,
            )
        assert step.config["compute_config"] == _SMALL

    @pytest.mark.asyncio
    async def test_unset_leaves_the_steps_default(self, skypilot_env, hfuri):
        _, step = await skypilot_env.pullasset_hfstore(
            uri=hfuri,
            assetstore=_store(),
            storeload_config=StoreLoad(config={"cache_path": "/c"}),
        )
        assert "compute_config" not in step.config


def _builtin(step):
    path = _REPO / f"src/gbserver/builtins/steps/skypilot/{step}/step.yaml"
    return yaml.safe_load(path.read_text())["config"]


@pytest.mark.parametrize("step", ["hfpull", "hfpush"])
def test_override_wins_over_builtin_step_default(step):
    """The queued config is merged over step.yaml's.

    merge_handle_configs (targetstep.py) merges the queued step's config over
    step.yaml's with merge_dicts, the last layer before launch; it needs a full
    TargetStep to drive, so this exercises that same merge directly.
    """
    builtin = _builtin(step)
    assert builtin["compute_config"]["total_memory_per_node"] == "10Gi"
    merged = merge_dicts(builtin, {"compute_config": _SMALL})
    assert merged["compute_config"] == _SMALL


def test_partial_override_keeps_the_steps_other_keys():
    merged = merge_dicts(
        _builtin("hfpush"), {"compute_config": {"total_memory_per_node": "2Gi"}}
    )
    assert merged["compute_config"] == {
        "num_cpus_per_node": 1,
        "total_memory_per_node": "2Gi",
    }


def test_merged_override_becomes_the_pod_request(skypilot_env):
    """The last step to sky.Resources: 2Gi becomes a 2 GB minimum on kubernetes."""
    merged = merge_dicts(_builtin("hfpull"), {"compute_config": _SMALL})
    res = skypilot_env._resources_from_compute_config(
        merged["compute_config"], cloud="kubernetes"
    )
    assert res["memory"] == "2+"
    assert res["cpus"] == "1+"


def _hf_store(env_yaml):
    stores = yaml.safe_load(env_yaml.read_text())["assetstores"]
    (hf,) = [s for s in stores if s["store_uri"] == "space://assetstores/hf"]
    return hf


@pytest.mark.parametrize("env", ["kubernetes", "kubernetes/local"])
def test_kube_envs_size_hf_steps_down(env):
    hf = _hf_store(
        _REPO / f"configurations/assets/environments/skypilot/{env}/environment.yaml"
    )
    for entry in (hf["pull"][0], hf["push"][0]):
        assert entry["config"]["compute_config"] == _SMALL
