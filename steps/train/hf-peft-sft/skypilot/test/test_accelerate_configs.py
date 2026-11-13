# Copyright Granite.Build Authors
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
    # torch's checkpoint_wrapper also hides LoRA keys from peft 0.21.x
    # (huggingface/peft#3876);
    # the trainer's own gradient checkpointing does not.
    assert not fsdp.get("fsdp_activation_checkpointing", False)


def test_qlora_variant_differs_only_in_offload():
    plain, qlora = _load("fsdp.yaml"), _load("fsdp_qlora.yaml")
    assert qlora["fsdp_config"].pop("fsdp_offload_params") is True
    assert plain["fsdp_config"].pop("fsdp_offload_params") is False
    assert plain == qlora


_ALL = sorted(p.name for p in _CONFIGS.glob("*.yaml"))


def test_every_config_is_checked():
    assert _ALL == sorted([*_FSDP, "deepspeed_z3.yaml", "ddp.yaml", "single_gpu.yaml"])


@pytest.mark.parametrize("name", _ALL)
def test_no_mixed_precision_in_the_config(name):
    # hf_peft_sft_run.sh always passes --mixed_precision, which wins over the
    # config file; a value here would never apply and only mislead.
    assert "mixed_precision" not in _load(name)


def test_deepspeed_config_is_untouched_zero3():
    cfg = _load("deepspeed_z3.yaml")
    assert cfg["distributed_type"] == "DEEPSPEED"
    assert cfg["deepspeed_config"]["zero_stage"] == 3


# ddp passes one of these so accelerate never falls back to the user's
# ~/.cache/huggingface/accelerate/default_config.yaml.
_DDP = {"ddp.yaml": "MULTI_GPU", "single_gpu.yaml": "NO"}


@pytest.mark.parametrize("name,dist", sorted(_DDP.items()))
def test_ddp_configs_distributed_type(name, dist):
    cfg = _load(name)
    # "NO" must stay a string; unquoted, YAML parses it as False.
    assert isinstance(cfg["distributed_type"], str)
    assert cfg["distributed_type"] == dist
    assert cfg["compute_environment"] == "LOCAL_MACHINE"
    assert cfg["num_machines"] == 1
    assert cfg["use_cpu"] is False


@pytest.mark.parametrize("name", sorted(_DDP))
def test_ddp_configs_name_no_sharding_keys(name):
    cfg = _load(name)
    assert not [k for k in cfg if "fsdp" in k or "deepspeed" in k], sorted(cfg)


@pytest.mark.parametrize("name,dist", sorted(_DDP.items()))
def test_ddp_configs_load_with_accelerate(name, dist):
    # Not importorskip("accelerate"): with PYTHONPATH=src and no real accelerate
    # installed, src/accelerate/ imports as an empty namespace package.
    args = pytest.importorskip("accelerate.commands.config.config_args")
    cfg = args.load_config_from_file(str(_CONFIGS / name))
    assert isinstance(cfg, args.ClusterConfig)
    assert cfg.distributed_type.value == dist
    assert cfg.fsdp_config == {} and cfg.deepspeed_config == {}


@pytest.mark.parametrize("name", _ALL)
@pytest.mark.parametrize("mp", ["bf16", "fp16", "no"])
def test_every_config_launches_with_the_command_line_precision(name, mp):
    # accelerate 1.15 loads a config without mixed_precision (as None), and its
    # launch validation takes --mixed_precision from the command line.
    # Not importorskip("accelerate"): see above.
    pytest.importorskip("accelerate.commands.config.config_args")
    launch = pytest.importorskip("accelerate.commands.launch")
    cfg = str(_CONFIGS / name)
    args = launch.launch_command_parser().parse_args(
        [
            "--config_file",
            cfg,
            "--num_machines",
            "1",
            "--num_processes",
            "2",
            "--mixed_precision",
            mp,
            "--main_process_port",
            "0",
            "train.py",
        ]
    )
    args, defaults, mp_from_config = launch._validate_launch_command(args)
    assert defaults.mixed_precision is None
    assert args.mixed_precision == mp
    assert mp_from_config is False
