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

"""Hfstore.apply_compute_config_override and the store_load/store_push readers."""

from gbserver.asset.hfstore import Hfstore
from gbserver.types.buildconfig import BuildTargetInputConfig, BuildTargetOutputConfig


def test_override_is_copied_onto_the_step_config():
    compute = {"total_memory_per_node": "1Gi"}
    step = {"hfpull_config": {}}
    Hfstore.apply_compute_config_override(step, {"compute_config": compute})
    assert step["compute_config"] == compute
    step["compute_config"]["num_cpus_per_node"] = 2
    assert compute == {"total_memory_per_node": "1Gi"}  # build config untouched


def test_no_override_leaves_the_step_config_alone():
    for store_config in (None, {}, {"hf": {"public": True}}, {"compute_config": {}}):
        step = {"hfpush_config": {}}
        Hfstore.apply_compute_config_override(step, store_config)
        assert step == {"hfpush_config": {}}


def test_readers_return_the_store_config_or_none():
    load = {"compute_config": {"total_memory_per_node": "1Gi"}}
    assert Hfstore.store_load_config(None) is None
    assert Hfstore.store_load_config(BuildTargetInputConfig(uri="hf:///a/b")) is None
    assert (
        Hfstore.store_load_config(
            BuildTargetInputConfig(uri="hf:///a/b", store_load={"config": load})
        )
        == load
    )
    assert Hfstore.store_push_config(None) is None
    assert Hfstore.store_push_config(BuildTargetOutputConfig(uri="hf:///a/b")) is None
    assert (
        Hfstore.store_push_config(
            BuildTargetOutputConfig(uri="hf:///a/b", store_push={"config": load})
        )
        == load
    )
