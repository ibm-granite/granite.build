# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
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

"""fm-tune's overrides on top of verl's default PPO trainer config.

Deliberately imports neither torch, verl, hydra nor omegaconf, so the mapping
from a trial's hyperparameters to verl config keys is unit-testable on any
platform — verl is not installable on macOS, and the driver module itself
imports torch, hydra and verl on load. ``driver_multi_verl.build_verl_config``
merges the dict returned here onto verl's defaults.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _pick(name: str, train_kwargs: Dict[str, Any], training_rl_config: Dict[str, Any], default: Any) -> Any:
    """Prefer the trial's sampled value, then the fixed training_rl_config value, then ``default``.

    ``train_kwargs`` holds only the params the search space sweeps, so a param
    that is not swept is absent there and comes from training_rl_config.
    """
    if train_kwargs.get(name) is not None:
        return train_kwargs[name]
    if training_rl_config.get(name) is not None:
        return training_rl_config[name]
    return default


def build_verl_overrides(
    training_config: Dict[str, Any],
    training_rl_config: Dict[str, Any],
    train_kwargs: Dict[str, Any],
    train_file: str,
    eval_file: str,
    num_workers: int,
    rl_algorithm: str,
    tensor_model_parallel_size: int = 1,
    hpo_search: bool = True,
    dataset_size: int = 0,
) -> Dict[str, Any]:
    """
    Construct fm-tune's verl config overrides as a plain nested dict.

    Adapts settings based on tensor parallelism degree (gradient checkpointing is always on):
      - TP=1: CUDA graphs enabled, vLLM gpu_memory_utilization 0.3
      - TP>1: eager mode (no CUDA graphs), vLLM gpu_memory_utilization 0.25
    An explicit training_rl_config gpu_memory_utilization overrides the vLLM value.
    """
    model_name_or_path = training_config.get("model_name_or_path")
    precision = "bf16"
    output_dir = training_config.get("output_dir")
    dtype = "bfloat16" if precision in ["bf16", "fp16"] else "float32"

    # Extract training hyperparams
    lr = train_kwargs.get("learning_rate")
    batch_size = train_kwargs.get("per_device_train_batch_size")
    num_train_epochs = train_kwargs.get("num_train_epochs")
    clip_range = train_kwargs.get("clip_range")
    clip_ratio = clip_range if clip_range is not None else 0.2
    # The search space names it entropy_coef; entropy_coeff (verl's spelling) is
    # still accepted. An explicit 0.0 is kept.
    entropy_coeff = train_kwargs.get("entropy_coef")
    if entropy_coeff is None:
        entropy_coeff = train_kwargs.get("entropy_coeff")
    if entropy_coeff is None:
        entropy_coeff = 0.0
    kl_coef = _pick("kl_coef", train_kwargs, training_rl_config, 0.001)

    # Extract verl-specific parameters
    max_prompt_length = training_rl_config.get("max_prompt_length")
    max_response_length = training_rl_config.get("max_response_length")
    rollout_temperature = _pick("rollout_temperature", train_kwargs, training_rl_config, 1.0)
    rollout_top_p = training_rl_config.get("rollout_top_p")
    rollout_n = _pick("rollout_n", train_kwargs, training_rl_config, 5)
    gpu_memory_utilization = training_rl_config.get("gpu_memory_utilization")

    # Reward function config
    reward_function_path = training_rl_config.get("reward_function_path", None)
    reward_function_name = training_rl_config.get("reward_function_name", "compute_score")

    # Detect hybrid (Mamba/SSM) architectures — these require enforce_eager
    # because CUDA graph capture is incompatible with stateful Mamba layers.
    is_hybrid_model = False
    # try:
    #     from transformers import AutoConfig
    #     _hf_cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    #     _arch = getattr(_hf_cfg, "architectures", []) or []
    #     _model_type = getattr(_hf_cfg, "model_type", "")
    #     is_hybrid_model = any(
    #         kw in a.lower() for a in _arch for kw in ("hybrid", "mamba", "rwkv", "ssm")
    #     ) or any(
    #         kw in _model_type.lower() for kw in ("hybrid", "mamba", "rwkv", "ssm")
    #     )
    #     if is_hybrid_model:
    #         print(f"[AutoTune] Detected hybrid/SSM architecture: {_arch} — forcing eager mode")
    # except Exception:
    #     pass

    # Derive adaptive settings from TP and architecture
    is_large_model = tensor_model_parallel_size > 1
    # Always enable gradient checkpointing — with colocated pools the actor,
    # critic, ref, and vLLM engine all share the same GPUs, so activation
    # memory savings matter even for small models.
    enable_gradient_checkpointing = True
    enforce_eager = is_large_model or is_hybrid_model

    # Set rollout n based on algorithm
    if rl_algorithm in ("grpo", "dapo"):
        effective_rollout_n = rollout_n if rollout_n else 5
    else:
        effective_rollout_n = 1

    # Log resolved parameters
    logger.info(
        "[AutoTune] build_verl_config parameters:\n"
        f"  train_kwargs:          {json.dumps({k: str(v) for k, v in train_kwargs.items()}, indent=4)}\n"
        f"  model:                 {model_name_or_path}\n"
        f"  lr:                    {lr}\n"
        f"  batch_size:            {batch_size}\n"
        f"  num_train_epochs:      {num_train_epochs}\n"
        f"  clip_range:            {clip_range}\n"
        f"  entropy_coeff:         {entropy_coeff}\n"
        f"  kl_coef:               {kl_coef}\n"
        f"  max_prompt_length:     {max_prompt_length}\n"
        f"  max_response_length:   {max_response_length}\n"
        f"  rollout_temperature:   {rollout_temperature}\n"
        f"  rollout_top_p:         {rollout_top_p}\n"
        f"  rollout_n:             {effective_rollout_n}\n"
        f"  gpu_memory_util:       {gpu_memory_utilization}\n"
        f"  num_workers:           {num_workers}\n"
        f"  tensor_parallel_size:  {tensor_model_parallel_size}\n"
        f"  grad_checkpointing:   {enable_gradient_checkpointing}\n"
        f"  enforce_eager:         {enforce_eager}\n"
        f"  rl_algorithm:          {rl_algorithm}\n"
        f"  reward_function_path:  {reward_function_path}\n"
        f"  reward_function_name:  {reward_function_name}"
    )

    # Determine reward manager name
    reward_manager_name = "dapo" if rl_algorithm == "dapo" else "naive"

    # With colocated pools, all GPUs are shared — batch size uses full count
    actor_gpus = num_workers
    total_batch_size = batch_size * actor_gpus

    # vLLM memory utilization — keep low for colocated pools where actor,
    # critic, ref, and vLLM all share the same GPUs.  For RL rollouts with
    # max_model_len = max_prompt_length + max_response_length (typically
    # 1-4K tokens), 0.3 is sufficient KV cache.
    if gpu_memory_utilization:
        vllm_gpu_mem = gpu_memory_utilization
    elif is_large_model:
        vllm_gpu_mem = 0.25
    else:
        vllm_gpu_mem = 0.3

    # Checkpoint frequency — disable during HPO, enable for final training
    if hpo_search:
        save_freq = -1
        max_actor_ckpt_to_keep = None
        max_critic_ckpt_to_keep = None
    else:
        steps_per_epoch = max(1, dataset_size // total_batch_size) if dataset_size > 0 else 1
        total_steps = steps_per_epoch * num_train_epochs
        if num_train_epochs > 1:
            save_freq = steps_per_epoch  # every epoch
        else:
            save_freq = max(1, total_steps // 5)  # ~5 checkpoints
        # Keep 3 most recent checkpoints — after training the model is
        # saved from the last one.
        max_actor_ckpt_to_keep = 3
        max_critic_ckpt_to_keep = 3
        logger.info(
            f"[AutoTune] Checkpointing: save_freq={save_freq}, "
            f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}"
        )

    # Build overrides
    overrides = {
        "data": {
            "train_files": train_file,
            "val_files": eval_file,
            "train_batch_size": total_batch_size,
            "val_batch_size": total_batch_size,
            "max_prompt_length": max_prompt_length,
            "max_response_length": max_response_length,
            "reward_fn_key": "data_source",
            "shuffle": True,
            "dataloader_num_workers": 0,
        },
        "actor_rollout_ref": {
            "model": {
                "path": model_name_or_path,
                "enable_gradient_checkpointing": enable_gradient_checkpointing,
            },
            "actor": {
                "ppo_micro_batch_size_per_gpu": batch_size,
                "ppo_mini_batch_size": total_batch_size,
                "optim": {
                    "lr": lr,
                },
                "clip_ratio": clip_ratio,
                "clip_ratio_low": clip_ratio,
                "clip_ratio_high": clip_ratio,
                "entropy_coeff": float(entropy_coeff),
                "use_kl_loss": False,
                "ppo_epochs": 1,
                "fsdp_config": {
                    "dtype": dtype,
                },
                "checkpoint": {
                    "save_contents": ["model", "hf_model"],
                    "load_contents": ["model"],
                },
            },
            "rollout": {
                "name": "vllm",
                "gpu_memory_utilization": vllm_gpu_mem,
                "max_model_len": max_prompt_length + max_response_length,
                "temperature": rollout_temperature,
                "top_p": rollout_top_p,
                "n": effective_rollout_n,
                "tensor_model_parallel_size": tensor_model_parallel_size,
                "enforce_eager": enforce_eager,
                "log_prob_micro_batch_size_per_gpu": batch_size,
                # Free vLLM GPU memory (model weights + KV cache) when not
                # generating rollouts, so the actor/critic training phases
                # have more headroom on the colocated GPUs.
                "enable_sleep_mode": True,
                "free_cache_engine": True,
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": batch_size,
                "fsdp_config": {
                    "dtype": dtype,
                    # Offload ref model params to CPU — it only runs forward
                    # passes for KL divergence and doesn't need to stay on GPU.
                    "param_offload": True,
                },
            },
        },
        "critic": {
            "enable": True,
            "model": {
                "path": model_name_or_path,
                "tokenizer_path": model_name_or_path,
                "enable_gradient_checkpointing": enable_gradient_checkpointing,
                "fsdp_config": {
                    "dtype": dtype,
                },
            },
            "optim": {
                "lr": lr,
            },
            "ppo_micro_batch_size_per_gpu": batch_size,
            "ppo_mini_batch_size": total_batch_size,
            "ppo_epochs": 1,
            "cliprange_value": 0.5,
        },
        "reward": {
            "custom_reward_function": {
                "path": reward_function_path or None,
                "name": reward_function_name,
            },
            "reward_manager": {
                "name": reward_manager_name,
            },
            "reward_model": {
                "enable": False,
            },
        },
        "algorithm": {
            "adv_estimator": "gae",
            "use_kl_in_reward": False,
            "kl_penalty": "kl",
            "kl_ctrl": {
                "kl_coef": kl_coef,
            },
            "gamma": 1.0,
            "lam": 0.95,
        },
        "trainer": {
            "device": "cuda",
            "n_gpus_per_node": num_workers,
            "nnodes": 1,
            "total_epochs": num_train_epochs,
            "total_training_steps": None,
            "save_freq": save_freq,
            "max_actor_ckpt_to_keep": max_actor_ckpt_to_keep,
            "max_critic_ckpt_to_keep": max_critic_ckpt_to_keep,
            "test_freq": -1,
            "project_name": "fm-tune-verl",
            "experiment_name": "online_rl",
            "default_local_dir": output_dir,
            "logger": ["console"],
            "val_before_train": False,
        },
    }

    # DAPO-specific overlong buffer. verl 0.7.1's "dapo" reward manager
    # (experimental/reward_loop/reward_manager/dapo.py) reads both keys from
    # reward.reward_kwargs and asserts max_resp_len >= overlong_buffer_cfg.len.
    if rl_algorithm == "dapo":
        overlong_buffer_len = _pick("overlong_buffer_len", train_kwargs, training_rl_config, 256)
        overlong_penalty_factor = _pick("overlong_penalty_factor", train_kwargs, training_rl_config, 1.0)
        if overlong_buffer_len > max_response_length:
            raise ValueError(
                f"DAPO overlong_buffer_len ({overlong_buffer_len}) must not exceed "
                f"max_response_length ({max_response_length}); lower overlong_buffer_len "
                "or raise max_response_length."
            )
        logger.info(
            f"[AutoTune] DAPO overlong_buffer_cfg: len={overlong_buffer_len}, "
            f"penalty_factor={overlong_penalty_factor}, max_resp_len={max_response_length}"
        )
        overrides["reward"]["reward_kwargs"] = {
            "overlong_buffer_cfg": {
                "enable": True,
                "len": overlong_buffer_len,
                "penalty_factor": overlong_penalty_factor,
                "log": False,
            },
            "max_resp_len": max_response_length,
        }

    logger.info(
        f"[AutoTune] VERL config: actor_gpus={actor_gpus}, "
        f"total_batch_size={total_batch_size}, vllm_gpu_mem={vllm_gpu_mem}, "
        f"TP={tensor_model_parallel_size}, "
        f"gradient_checkpointing={enable_gradient_checkpointing}, "
        f"enforce_eager={enforce_eager}"
    )

    return overrides
