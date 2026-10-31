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

"""The accelerate configs the run script picks, checked against accelerate's rules.

FSDP is FSDP2: peft 0.21.x (huggingface/peft#3876) drops every LoRA tensor when it
saves an FSDP1-wrapped model, because FSDP1 puts `_fsdp_wrapped_module.` inside the
module names. FSDP2 keeps the names clean. accelerate rejects or ignores several
FSDP1 keys under FSDP2, so those keys must not appear.
"""

import pathlib

import pytest
import yaml

_CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "src" / "accelerate"
_FSDP = ["fsdp.yaml", "fsdp_qlora.yaml"]

# accelerate 1.15 raises on forward_prefetch and LOCAL_STATE_DICT under FSDP2, and
# drops backward_prefetch, use_orig_params, sync_module_states and
# sharding_strategy with a warning.
_FSDP1_ONLY = [
    "fsdp_forward_prefetch",
    "fsdp_backward_prefetch",
    "fsdp_use_orig_params",
    "fsdp_sync_module_states",
    "fsdp_sharding_strategy",
]


def _load(name):
    return yaml.safe_load((_CONFIGS / name).read_text())


@pytest.mark.parametrize("name", _FSDP)
def test_fsdp_configs_are_fsdp2(name):
    cfg = _load(name)
    assert cfg["distributed_type"] == "FSDP"
    assert cfg["fsdp_config"]["fsdp_version"] == 2


@pytest.mark.parametrize("name", _FSDP)
def test_no_fsdp1_only_keys(name):
    fsdp = _load(name)["fsdp_config"]
    assert not set(_FSDP1_ONLY) & set(fsdp), sorted(set(_FSDP1_ONLY) & set(fsdp))


@pytest.mark.parametrize("name", _FSDP)
def test_fsdp2_options(name):
    fsdp = _load(name)["fsdp_config"]
    assert fsdp["fsdp_auto_wrap_policy"] == "TRANSFORMER_BASED_WRAP"
    assert fsdp["fsdp_reshard_after_forward"] is True
    assert fsdp["fsdp_state_dict_type"] in ("SHARDED_STATE_DICT", "FULL_STATE_DICT")
    # torch's checkpoint_wrapper also hides LoRA keys from peft 0.21.x (#3876);
    # the trainer's own gradient checkpointing does not.
    assert not fsdp.get("fsdp_activation_checkpointing", False)


def test_qlora_variant_differs_only_in_offload_and_precision():
    plain, qlora = _load("fsdp.yaml"), _load("fsdp_qlora.yaml")
    assert qlora["fsdp_config"].pop("fsdp_offload_params") is True
    assert plain["fsdp_config"].pop("fsdp_offload_params") is False
    assert qlora.pop("mixed_precision") == "no"
    assert plain.pop("mixed_precision") == "bf16"
    assert plain == qlora


def test_deepspeed_config_is_untouched_zero3():
    cfg = _load("deepspeed_z3.yaml")
    assert cfg["distributed_type"] == "DEEPSPEED"
    assert cfg["deepspeed_config"]["zero_stage"] == 3
