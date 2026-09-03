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

# Multi-GPU per trial train driver (ray) for verl online RL algorithms.
# Supports models from ~1B to ~32B with automatic tensor parallelism.
#
# Key features:
#   - Colocated GPU pools: all roles share the same GPUs
#   - Auto-detects tensor parallelism (TP) from model size
#   - Adaptive gradient checkpointing and vLLM settings based on TP
#   - Requires Default GPU compute mode (not Exclusive Process)
#
# verl (Volcano Engine Reinforcement Learning) + FSDP implementation.
# Supports PPO, GRPO, and DAPO algorithms.

import os
import sys

import autotune.trainers._trl_compat  # noqa: F401 — patch trl in the main process

_TRL_COMPAT_MODULE = "autotune.trainers._trl_compat"

# Fail fast if GPUs are in Exclusive Process compute mode.
# vLLM v1 uses a multi-process architecture (vLLMHttpServer → EngineCore →
# WorkerProc) where multiple processes need CUDA contexts on the same GPU.
# Exclusive Process mode only allows one context per GPU, causing
# "CUDA-capable device(s) is/are busy or unavailable" errors.
import subprocess as _sp

try:
    _nvsmi = _sp.run(
        ["nvidia-smi", "--query-gpu=compute_mode", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if _nvsmi.returncode == 0:
        _modes = [m.strip() for m in _nvsmi.stdout.strip().splitlines()]
        if any("Exclusive" in m for m in _modes):
            print(
                "\nERROR: One or more GPUs are in Exclusive Process compute mode.\n"
                "vLLM v1 requires Default compute mode (multiple CUDA contexts per GPU).\n"
                "\n"
                "Fix: switch to Default mode before launching:\n"
                "  sudo nvidia-smi -c 0\n",
                file=sys.stderr,
            )
            raise RuntimeError(
                "GPU compute mode is Exclusive Process. vLLM v1 requires Default mode. Run: sudo nvidia-smi -c 0"
            )
except FileNotFoundError:
    pass  # nvidia-smi not available, skip check

# Fail fast if expandable_segments is set — cannot be fixed at runtime
_cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments:True" in _cuda_alloc_conf:
    print(
        "\nERROR: PYTORCH_CUDA_ALLOC_CONF contains 'expandable_segments:True'.\n"
        "This is incompatible with vLLM's memory pool and CANNOT be fixed\n"
        "from Python — vLLM subprocesses inherit the env var.\n"
        "\n"
        "Fix: unset the variable before launching:\n"
        "  unset PYTORCH_CUDA_ALLOC_CONF\n"
        "  unset PYTORCH_ALLOC_CONF\n"
        "  python main.py ...\n",
        file=sys.stderr,
    )
    raise RuntimeError(
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set. "
        "Unset it before running: unset PYTORCH_CUDA_ALLOC_CONF"
    )

import glob
import json
import logging
import math
import shutil
from copy import deepcopy
from typing import Any, Dict, Optional

import ray
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from ray import tune
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker, CriticWorker

# Local
from autotune.utils import (
    extract_tokenizer_kwargs,
    get_tokenizer,
    remove_dir,
    set_seed,
)

logger = logging.getLogger(__name__)


def _load_verl_default_config() -> OmegaConf:
    """Load verl's default ppo_trainer.yaml config via Hydra compose."""
    import verl.trainer.config as verl_config_pkg

    config_dir = os.path.dirname(verl_config_pkg.__file__)

    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_trainer")

    return cfg


def _resolve_tensor_parallel_size(
    model_name_or_path: str,
    num_workers: int,
    user_tp: Optional[int] = None,
) -> int:
    """Resolve tensor parallelism degree for vLLM rollout.

    If user_tp is provided, validate and use it. Otherwise auto-detect
    from model size using config.json.

    Returns:
        int: validated tensor_model_parallel_size (power of 2, divides num_workers)
    """

    def _is_power_of_2(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    def _clamp_to_workers(tp: int, num_workers: int) -> int:
        """Halve tp until it divides num_workers evenly."""
        while tp > 1 and num_workers % tp != 0:
            tp //= 2
        return tp

    # User override
    if user_tp is not None:
        if not _is_power_of_2(user_tp):
            raise ValueError(f"tensor_model_parallel_size={user_tp} must be a power of 2 (1, 2, 4, 8, ...)")
        if num_workers % user_tp != 0:
            raise ValueError(f"tensor_model_parallel_size={user_tp} must divide num_workers={num_workers} evenly")
        logger.info(f"[AutoTune] TP={user_tp} (user override), num_workers={num_workers}")
        return user_tp

    # Auto-detect from model config.json
    try:
        config_path = os.path.join(model_name_or_path, "config.json")
        with open(config_path) as f:
            model_config = json.load(f)

        hidden_size = model_config.get("hidden_size", 0)
        num_layers = model_config.get("num_hidden_layers", 0)
        vocab_size = model_config.get("vocab_size", 0)

        # Rough parameter estimate: params ≈ (12 * H² * L + V * H) / 1e9
        params_b = (12 * hidden_size**2 * num_layers + vocab_size * hidden_size) / 1e9

        # Heuristic: TP=1 for ≤8B, TP=2 for ≤16B, TP=4 for ≤32B
        if params_b <= 8:
            candidate_tp = 1
        elif params_b <= 16:
            candidate_tp = 2
        else:
            candidate_tp = 4

        tp = _clamp_to_workers(candidate_tp, num_workers)
        logger.info(
            f"[AutoTune] TP={tp} (auto-detected: ~{params_b:.1f}B params, "
            f"candidate_tp={candidate_tp}, num_workers={num_workers})"
        )
        return tp

    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"[AutoTune] Could not load model config from {model_name_or_path}: {e}. Defaulting to TP=1.")
        return 1


def build_verl_config(
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
) -> OmegaConf:
    """
    Construct a verl-compatible OmegaConf config.

    Adapts settings based on tensor parallelism degree:
      - TP=1: no gradient checkpointing, CUDA graphs enabled, higher vLLM memory
      - TP>1: gradient checkpointing enabled, eager mode (no CUDA graphs), lower vLLM memory
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
    entropy_coeff = train_kwargs.get("entropy_coeff", 0.0) or 0.0
    kl_coef = train_kwargs.get("kl_coef", 0.001) or 0.001

    # Extract verl-specific parameters
    max_prompt_length = training_rl_config.get("max_prompt_length")
    max_response_length = training_rl_config.get("max_response_length")
    rollout_temperature = training_rl_config.get("rollout_temperature")
    rollout_top_p = training_rl_config.get("rollout_top_p")
    rollout_n = training_rl_config.get("rollout_n")
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
        f"  rollout_n:             {rollout_n}\n"
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

    # Set rollout n based on algorithm
    if rl_algorithm in ("grpo", "dapo"):
        effective_rollout_n = rollout_n if rollout_n else 5
    else:
        effective_rollout_n = 1

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
        # Keep 3 most recent checkpoints — after training we select the
        # best one based on reward/loss metrics.
        max_actor_ckpt_to_keep = 3
        max_critic_ckpt_to_keep = 3
        logger.info(
            f"[AutoTune] Checkpointing: save_freq={save_freq}, "
            f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}"
        )

    # Load verl's full default config
    cfg = _load_verl_default_config()

    # Build overrides
    overrides = OmegaConf.create(
        {
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
                    "clip_ratio": clip_range if clip_range is not None else 0.2,
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
    )

    # Merge overrides on top of defaults
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, overrides)

    # DAPO-specific overlong buffer
    if rl_algorithm == "dapo":
        overlong_buffer_len = train_kwargs.get("overlong_buffer_len", 256)
        overlong_penalty_factor = train_kwargs.get("overlong_penalty_factor", 1.0)
        cfg.algorithm.overlong_buffer_cfg = {
            "enable": True,
            "len": overlong_buffer_len,
            "penalty_factor": overlong_penalty_factor,
            "log": False,
        }

    logger.info(
        f"[AutoTune] VERL config: actor_gpus={actor_gpus}, "
        f"total_batch_size={total_batch_size}, vllm_gpu_mem={vllm_gpu_mem}, "
        f"TP={tensor_model_parallel_size}, "
        f"gradient_checkpointing={enable_gradient_checkpointing}, "
        f"enforce_eager={enforce_eager}"
    )

    return cfg


def build_resource_pool_manager(
    num_workers: int,
    rl_algorithm: str,
    use_reward_model: bool = False,
) -> ResourcePoolManager:
    """
    Create verl's ResourcePoolManager with COLOCATED GPU pools.

    All roles share a single GPU pool. This is required because verl's
    vLLMHttpServer uses vLLM's multiproc executor (hardcoded), which forks
    child processes that need full visibility of the pool's GPUs. Disjoint
    pools with 1 GPU each cause CUDA device conflicts in the forked workers.
    """
    algo = (rl_algorithm or "").strip().lower()
    if algo not in {"ppo", "grpo", "dapo"}:
        raise ValueError(f"Unsupported rl_algorithm={rl_algorithm!r}; expected one of: 'ppo', 'grpo', 'dapo'.")

    resource_pool_spec = {
        "global_pool": [num_workers],
    }
    role_mapping = {
        Role.ActorRollout: "global_pool",
        Role.RefPolicy: "global_pool",
    }

    if algo == "ppo":
        role_mapping[Role.Critic] = "global_pool"

    if use_reward_model:
        role_mapping[Role.RewardModel] = "global_pool"

    logger.info(
        f"[AutoTune] Colocated pool for {algo.upper()}: "
        f"global_pool={num_workers} GPUs, "
        f"roles={list(role_mapping.keys())}"
    )

    return ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=role_mapping,
    )


# --- In-memory metrics logger (same as driver_multi_verl.py) ---


class _InMemoryMetricsLogger:
    """In-memory logger that captures verl's per-step metrics without file I/O."""

    _all_steps = []

    def __init__(self):
        _InMemoryMetricsLogger._all_steps = []

    def log(self, data, step):
        clean = {"_step": step}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                clean[k] = v.item() if v.numel() == 1 else v.tolist()
            elif hasattr(v, "item"):
                clean[k] = v.item()
            else:
                clean[k] = v
        _InMemoryMetricsLogger._all_steps.append(clean)

    def finish(self):
        pass

    @classmethod
    def collect(cls, rl_algorithm: str) -> Dict[str, Any]:
        """Aggregate captured metrics into a flat dict for fm-tune."""
        result = {}
        all_steps = cls._all_steps

        if not all_steps:
            return result

        last = all_steps[-1]

        # Reward metrics (last step)
        result["reward_mean"] = last.get("critic/score/mean", float("nan"))
        result["reward_max"] = last.get("critic/score/max", float("nan"))
        result["reward_min"] = last.get("critic/score/min", float("nan"))

        # Actor metrics (average)
        actor_losses = [s["actor/ppo_loss"] for s in all_steps if "actor/ppo_loss" in s]
        pg_losses = [s["actor/pg_loss"] for s in all_steps if "actor/pg_loss" in s]
        entropies = [s["actor/entropy"] for s in all_steps if "actor/entropy" in s]

        result["actor_loss"] = sum(actor_losses) / len(actor_losses) if actor_losses else float("nan")
        result["pg_loss"] = sum(pg_losses) / len(pg_losses) if pg_losses else float("nan")
        result["actor_entropy"] = sum(entropies) / len(entropies) if entropies else float("nan")

        # KL divergence
        kl_values = [s["actor/kl_loss"] for s in all_steps if "actor/kl_loss" in s]
        kl_reward = [s["actor/reward_kl_penalty"] for s in all_steps if "actor/reward_kl_penalty" in s]
        if kl_values:
            result["kl_divergence"] = sum(kl_values) / len(kl_values)
        elif kl_reward:
            result["kl_divergence"] = sum(kl_reward) / len(kl_reward)
        else:
            result["kl_divergence"] = float("nan")

        # Response length (last step)
        result["response_length_mean"] = last.get("response_length/mean", float("nan"))
        result["response_length_clip_ratio"] = last.get("response_length/clip_ratio", float("nan"))

        # Advantage metrics (last step)
        result["advantages_mean"] = last.get("critic/advantages/mean", float("nan"))
        result["returns_mean"] = last.get("critic/returns/mean", float("nan"))

        # PPO-specific critic metrics
        if rl_algorithm == "ppo":
            critic_losses = [s["critic/loss"] for s in all_steps if "critic/loss" in s]
            result["critic_loss"] = sum(critic_losses) / len(critic_losses) if critic_losses else float("nan")
            result["critic_values_mean"] = last.get("critic/values/mean", float("nan"))
            result["critic_vf_explained_var"] = last.get("critic/vf_explained_var", float("nan"))

        # Training progress
        result["global_steps"] = last.get("training/global_step", 0)
        result["epoch"] = last.get("training/epoch", 0)
        result["total_steps_logged"] = len(all_steps)

        # Throughput
        result["tokens_per_second"] = last.get("perf/overall_tokens_per_second", float("nan"))

        logger.info(f"[AutoTune] Collected {len(all_steps)} training steps from in-memory metrics logger")

        return result


def _install_metrics_logger():
    """Monkey-patch verl's Tracking class to inject our in-memory metrics logger."""
    from verl.utils.tracking import Tracking

    _orig_init = Tracking.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.logger["_autotune"] = _InMemoryMetricsLogger()

    Tracking.__init__ = _patched_init


def _cleanup_verl_workers(trainer):
    """Kill all Ray actor workers and release GPU resources from a RayPPOTrainer.

    verl's RayPPOTrainer has no shutdown method. Between HPO trials the old
    vLLM server processes, FSDP workers, and ref-policy actors may still hold
    GPU memory and POSIX shared-memory segments, causing the next trial's
    vLLM EngineCore to crash with "died unexpectedly".

    Additionally, verl reserves GPUs via Ray placement groups. If these are
    not explicitly removed, subsequent trials see 0 available GPUs and fail
    with "Total available GPUs 0 is less than total desired GPUs N".

    We iterate over every WorkerGroup the trainer created and ray.kill each
    underlying actor handle, then remove all placement groups to free GPUs.
    """
    import time

    import ray as _ray

    # 1. Kill worker actors
    wg_names = ["actor_rollout_wg", "critic_wg", "ref_policy_wg", "reward_wg"]
    killed = 0
    for name in wg_names:
        wg = getattr(trainer, name, None)
        if wg is None:
            continue
        workers = getattr(wg, "_workers", None) or []
        for w in workers:
            try:
                _ray.kill(w)
                killed += 1
            except Exception:
                pass

    # 2. Remove placement groups to release GPU reservations
    removed_pgs = 0
    rpm = getattr(trainer, "resource_pool_manager", None)
    if rpm is not None:
        for pool in rpm.resource_pool_dict.values():
            for pg in pool.pgs or []:
                try:
                    _ray.util.remove_placement_group(pg)
                    removed_pgs += 1
                except Exception:
                    pass
            pool.pgs = None

    if killed or removed_pgs:
        # Brief pause to let Ray reclaim GPU/shared-memory resources
        time.sleep(2)
        logger.info(f"[AutoTune] Cleaned up {killed} verl worker(s) and {removed_pgs} placement group(s)")
    else:
        logger.info("[AutoTune] No verl workers to clean up")


def _select_best_checkpoint(ckpt_dirs, all_steps):
    """Select the checkpoint corresponding to the best training step.

    Picks the step with the highest mean reward (``critic/score/mean``).
    Falls back to lowest actor PPO loss, then to the last checkpoint.

    Args:
        ckpt_dirs: Sorted list of ``global_step_*`` directory paths.
        all_steps: List of per-step metric dicts from _InMemoryMetricsLogger
                   (each dict contains a ``_step`` key).

    Returns:
        Path to the best checkpoint directory.
    """
    if not ckpt_dirs:
        return None

    # Parse step numbers from checkpoint dir names
    def _step_from_dir(d):
        base = os.path.basename(d)
        try:
            return int(base.split("global_step_")[-1])
        except (ValueError, IndexError):
            return -1

    ckpt_steps = {_step_from_dir(d): d for d in ckpt_dirs}

    if not all_steps:
        logger.info("[AutoTune] No in-memory metrics — using last checkpoint")
        return ckpt_dirs[-1]

    # Find the step with the best metric
    best_step = None
    best_value = None
    metric_name = None

    # Try reward first (higher is better)
    for entry in all_steps:
        reward = entry.get("critic/score/mean")
        if reward is not None and not (isinstance(reward, float) and math.isnan(reward)):
            if best_value is None or reward > best_value:
                best_value = reward
                best_step = entry.get("_step")
                metric_name = "critic/score/mean"

    # Fall back to actor loss (lower is better)
    if best_step is None:
        for entry in all_steps:
            loss = entry.get("actor/ppo_loss")
            if loss is not None and not (isinstance(loss, float) and math.isnan(loss)):
                if best_value is None or loss < best_value:
                    best_value = loss
                    best_step = entry.get("_step")
                    metric_name = "actor/ppo_loss"

    if best_step is None:
        logger.info("[AutoTune] No valid metrics found — using last checkpoint")
        return ckpt_dirs[-1]

    # Find the checkpoint dir whose step is closest to (and <=) the best step
    available_steps = sorted(ckpt_steps.keys())
    selected_step = available_steps[-1]  # default to last
    for s in reversed(available_steps):
        if s <= best_step:
            selected_step = s
            break

    selected_dir = ckpt_steps[selected_step]
    logger.info(
        f"[AutoTune] Best checkpoint: step {selected_step} "
        f"({metric_name}={best_value:.4f}, best training step={best_step})"
    )
    return selected_dir


# --- Main driver ---


def train_driver_multi_gpu(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main verl train driver for models up to ~32B on multiple GPUs.

    Automatically resolves tensor parallelism and adapts settings:
      - TP=1 (<=8B): no gradient checkpointing, CUDA graphs enabled
      - TP>1 (>8B): gradient checkpointing, eager mode, lower vLLM memory

    Args:
        config: Dict with the current hyperparameter configuration from Ray Tune.

    Returns:
        A Dict summarizing the training results (per ray.tune).
    """
    from autotune.logging_setup import bind_trial_id, setup_logging

    setup_logging()

    trial_id = tune.get_context().get_trial_id()
    bind_trial_id(trial_id)
    logger.info(f"[AutoTune] Training driver multi GPU verl (trial {trial_id})")

    local_config = deepcopy(config)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Get the training config
    training_config = local_config.pop("training_config")
    training_rl_config = local_config.pop("training_rl_config")
    tuner_flags = local_config.pop("tuner_flags")
    local_config.pop("tune_config")

    # Get all required parameters
    train_file = training_config.get("train_file")
    eval_file = training_config.get("validation_file")
    model_name_or_path = training_config.get("model_name_or_path")
    output_dir = training_config.get("output_dir")
    seed = training_config.get("seed", 42)
    num_train_epochs = training_config.get("num_train_epochs", 1)
    num_hpo_epochs = training_config.get("hpo_num_epochs", 1)
    hpo_search = training_config.get("hpo_search", False)
    num_workers = training_config.get("num_workers")

    # Online RL-specific parameters
    rl_algorithm = training_config.get("rl_algorithm", "ppo")
    reward_function_path = training_rl_config.get("reward_function_path", None)
    reward_model_path = training_rl_config.get("reward_model_path", None)

    set_seed(seed)

    # Get tuner params
    train_kwargs = {k: v for k, v in local_config.items() if tuner_flags[k] is True}
    train_kwargs["num_train_epochs"] = num_train_epochs

    logger.info(f"[AutoTune] Training args: {train_kwargs}")

    save_model_flag = training_config.get("save_model", False)

    if hpo_search:
        train_kwargs["num_train_epochs"] = num_hpo_epochs

    # Resolve tensor parallelism
    user_tp = training_rl_config.get("tensor_model_parallel_size", None)
    tensor_model_parallel_size = _resolve_tensor_parallel_size(
        model_name_or_path=model_name_or_path,
        num_workers=num_workers,
        user_tp=user_tp,
    )

    # Get dataset size for checkpoint frequency calculation
    dataset_size = 0
    try:
        ext = os.path.splitext(train_file)[1].lower()
        if ext == ".parquet":
            import pyarrow.parquet as pq

            dataset_size = pq.read_metadata(train_file).num_rows
        elif ext == ".csv":
            dataset_size = sum(1 for _ in open(train_file)) - 1  # minus header
        else:
            dataset_size = sum(1 for _ in open(train_file))
        logger.info(f"[AutoTune] Training dataset size: {dataset_size} samples")
    except Exception as e:
        logger.warning(f"[AutoTune] Could not determine dataset size: {e}")

    # Build verl config
    logger.info(f"[AutoTune] Building verl config for {rl_algorithm.upper()} (TP={tensor_model_parallel_size})")
    verl_config = build_verl_config(
        training_config=training_config,
        training_rl_config=training_rl_config,
        train_kwargs=train_kwargs,
        train_file=train_file,
        eval_file=eval_file,
        num_workers=num_workers,
        rl_algorithm=rl_algorithm,
        tensor_model_parallel_size=tensor_model_parallel_size,
        hpo_search=hpo_search,
        dataset_size=dataset_size,
    )

    # Patch trl inside actor/rollout Ray workers so verl's monkey_patch.py
    # can do ``from trl import AutoModelForCausalLMWithValueHead`` (moved to
    # trl.experimental.ppo in trl >= 0.29.0).
    verl_config.actor_rollout_ref.model.external_lib = _TRL_COMPAT_MODULE

    # Configure algorithm-specific settings
    if rl_algorithm == "ppo":
        logger.info(f"[AutoTune] Configuring PPO (TP={tensor_model_parallel_size})")
        verl_config.algorithm.adv_estimator = "gae"
        verl_config.actor_rollout_ref.rollout.n = 1
        verl_config.critic.enable = True
        verl_config.critic.model.external_lib = _TRL_COMPAT_MODULE

        if reward_function_path is not None:
            verl_config.reward.reward_model.enable = False
            verl_config.algorithm.use_kl_in_reward = False
        elif reward_model_path is not None:
            verl_config.reward.reward_model.enable = True

    elif rl_algorithm == "grpo":
        logger.info(f"[AutoTune] Configuring GRPO (TP={tensor_model_parallel_size})")
        verl_config.algorithm.adv_estimator = "grpo"
        rollout_n = training_config.get("rollout_n", 5)
        verl_config.actor_rollout_ref.rollout.n = rollout_n
        verl_config.critic.enable = False
        logger.info(f"[AutoTune] GRPO rollout_n: {rollout_n}")

    elif rl_algorithm == "dapo":
        logger.info(f"[AutoTune] Configuring DAPO (TP={tensor_model_parallel_size})")
        verl_config.algorithm.adv_estimator = "grpo"
        verl_config.algorithm.norm_adv_by_std_in_grpo = False
        rollout_n = training_config.get("rollout_n", 16)
        verl_config.actor_rollout_ref.rollout.n = rollout_n
        verl_config.critic.enable = False
        logger.info(f"[AutoTune] DAPO rollout_n: {rollout_n}")
    else:
        raise ValueError(f"Unknown RL algorithm: {rl_algorithm}. Supported: ppo, grpo, dapo.")

    # Configure reward
    use_reward_model = False
    if reward_function_path is not None:
        logger.info(f"[AutoTune] Custom reward function: {reward_function_path}")
        if not os.path.isfile(reward_function_path):
            raise FileNotFoundError(f"Reward function file not found: {reward_function_path}")
    elif reward_model_path is not None:
        logger.info(f"[AutoTune] Learned reward model: {reward_model_path}")
        use_reward_model = True
        verl_config.reward.reward_model.enable = True
        verl_config.reward.reward_model.model_path = reward_model_path
        # rollout.name is mandatory (??? in verl's default YAML).
        verl_config.reward.reward_model.rollout.name = "vllm"
        verl_config.reward.reward_model.rollout.tensor_model_parallel_size = tensor_model_parallel_size
        verl_config.reward.reward_model.rollout.enforce_eager = verl_config.actor_rollout_ref.rollout.enforce_eager
        # Reward model only scores completed sequences (no generation),
        # so it needs minimal KV cache. However, gpu_memory_utilization must
        # cover model weights + KV cache, so larger reward models need more.
        # Use the same utilization as the actor rollout — sleep mode will
        # free the memory when the reward model is not actively scoring.
        verl_config.reward.reward_model.rollout.gpu_memory_utilization = (
            verl_config.actor_rollout_ref.rollout.gpu_memory_utilization
        )
        verl_config.reward.reward_model.rollout.max_model_len = verl_config.actor_rollout_ref.rollout.max_model_len
        verl_config.reward.reward_model.rollout.enable_sleep_mode = True
        verl_config.reward.reward_model.rollout.free_cache_engine = True
    else:
        logger.info("[AutoTune] Using default_compute_score via data_source routing")

    # Build resource pool manager (colocated pools)
    resource_pool_manager = build_resource_pool_manager(
        num_workers=num_workers,
        rl_algorithm=rl_algorithm,
        use_reward_model=use_reward_model,
    )

    # Initialize tokenizer (with optional customization)
    tokenizer_kwargs = extract_tokenizer_kwargs(training_config)
    tokenizer, num_new_tokens = get_tokenizer(model_name_or_path, **tokenizer_kwargs)

    if num_new_tokens > 0:
        # verl loads models internally via FSDP workers; we cannot resize
        # embeddings from the outside. Save the customized tokenizer so verl
        # workers pick it up, but warn that the model itself is not resized.
        tokenizer_cache_dir = os.path.join(output_dir, "tokenizer_cache", trial_id)
        os.makedirs(tokenizer_cache_dir, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_cache_dir)
        OmegaConf.update(verl_config, "critic.model.tokenizer_path", tokenizer_cache_dir)
        logger.warning(
            f"[AutoTune] {num_new_tokens} new token(s) added. Customized tokenizer "
            f"saved to {tokenizer_cache_dir}. Note: verl's internal model loading "
            f"may not resize embeddings automatically — verify model compatibility."
        )

    # Set up worker classes
    # Always include RefPolicy — the trainer's init_workers() uses
    # need_reference_policy(config) internally to decide whether to create it.
    worker_classes = {
        Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
        Role.RefPolicy: ray.remote(AsyncActorRolloutRefWorker),
    }

    if rl_algorithm == "ppo":
        worker_classes[Role.Critic] = ray.remote(CriticWorker)

    # Prepare output directories
    training_output_dir = os.path.join(output_dir, "outputs", f"{trial_id}")
    os.makedirs(training_output_dir, exist_ok=True)

    # Install in-memory metrics logger
    _install_metrics_logger()

    # Create trainer
    logger.info(f"[AutoTune] Creating RayPPOTrainer (TP={tensor_model_parallel_size})...")
    trainer = RayPPOTrainer(
        config=verl_config,
        tokenizer=tokenizer,
        role_worker_mapping=worker_classes,
        resource_pool_manager=resource_pool_manager,
    )

    # Run training
    logger.info("[AutoTune] Starting training...")
    metrics = {}

    try:
        trainer.init_workers()
        trainer.fit()
        logger.info("[AutoTune] Training finished successfully.")
    except Exception as e:
        logger.error(f"[AutoTune] Training failed (trial {trial_id}): {e}", exc_info=True)
        raise  # Let Ray Tune mark trial as ERRORED
    finally:
        # Kill verl's Ray actor workers (vLLM server, FSDP workers, etc.)
        # so the next HPO trial can start cleanly without stale processes
        # holding GPU memory or shared memory segments.
        _cleanup_verl_workers(trainer)

    # Collect metrics
    metrics = _InMemoryMetricsLogger.collect(rl_algorithm)

    reward_mean = metrics.get("reward_mean", float("nan"))
    train_loss = metrics.get("actor_loss", float("nan"))
    kl_divergence = metrics.get("kl_divergence", float("nan"))

    if not math.isnan(reward_mean):
        loss = -reward_mean
    elif not math.isnan(train_loss):
        loss = train_loss
    else:
        loss = 10000.0

    # Log results
    logger.info(
        f"[AutoTune] Results for trial {trial_id} ({rl_algorithm.upper()}, TP={tensor_model_parallel_size}): "
        f"reward={reward_mean}, actor_loss={train_loss}, kl={kl_divergence}, loss={loss}"
    )
    for k, v in sorted(metrics.items()):
        if k not in ("reward_mean", "actor_loss", "kl_divergence"):
            logger.info(f"[AutoTune]   {k}: {v}")

    # Prepare result dict
    result = {
        "eval_loss": loss,
        "train_loss": train_loss,
        "reward_mean": reward_mean,
        "kl_divergence": kl_divergence,
        "rl_algorithm": rl_algorithm,
        "train_log": metrics,
        "train_lines": [],
        "eval_results": {
            "reward_mean": reward_mean,
            "reward_max": metrics.get("reward_max", float("nan")),
            "reward_min": metrics.get("reward_min", float("nan")),
            "kl_divergence": kl_divergence,
            "actor_entropy": metrics.get("actor_entropy", float("nan")),
            "advantages_mean": metrics.get("advantages_mean", float("nan")),
            "response_length_mean": metrics.get("response_length_mean", float("nan")),
        },
        "train_loop_config": local_config,
    }

    # Save results JSON
    filename = os.path.join(training_output_dir, f"train_results-{trial_id}.json")

    def _sanitize_for_json(obj):
        """Recursively convert non-serializable types for JSON."""
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        if isinstance(obj, (int, bool)):
            return obj
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize_for_json(v) for v in obj]
        if hasattr(obj, "item"):  # numpy scalars
            return obj.item()
        return obj

    with open(filename, "w") as f:
        json.dump(_sanitize_for_json(result), f, indent=2)
    logger.info(f"[AutoTune] Results saved to: {filename}")

    # Save model — pick the best checkpoint based on reward/loss metrics
    if hpo_search is False and save_model_flag is True:
        model_name = training_config.get("output_model_name")
        output_model_path = output_dir  # os.path.join(output_dir, "models")
        output_model_id = os.path.join(output_model_path, model_name)

        verl_ckpt_base = verl_config.trainer.default_local_dir
        ckpt_dirs = sorted(glob.glob(os.path.join(verl_ckpt_base, "global_step_*")))

        # Select the best checkpoint using in-memory metrics
        best_ckpt = _select_best_checkpoint(ckpt_dirs, _InMemoryMetricsLogger._all_steps)

        hf_model_src = None
        if best_ckpt is not None:
            candidate = os.path.join(best_ckpt, "actor", "huggingface")
            if os.path.isdir(candidate):
                hf_model_src = candidate
            else:
                logger.warning(f"[AutoTune] HF model dir not found at {candidate}")

        if hf_model_src is not None:
            remove_dir(output_model_id)
            shutil.copytree(hf_model_src, output_model_id)
            tokenizer.save_pretrained(output_model_id)
            logger.info(f"[AutoTune] Model saved to: {output_model_id}")
        else:
            logger.warning("[AutoTune] No verl checkpoint found — model not saved")

        # Clean up verl checkpoints (global_step_* dirs) to free disk space.
        # The final model has already been copied to output_model_id above.
        for ckpt_dir in ckpt_dirs:
            remove_dir(ckpt_dir)
        if ckpt_dirs:
            logger.info(f"[AutoTune] Cleaned up {len(ckpt_dirs)} verl checkpoint(s) from {verl_ckpt_base}")

    # Return trial result
    trial_result = {
        "loss": loss,
        "train_loss": train_loss,
        "reward_mean": reward_mean,
        "kl_divergence": kl_divergence,
        "rl_algorithm": rl_algorithm,
        "done": True,
        "config": config,
        "train_log": metrics,
        "eval_results": {
            "reward_mean": reward_mean,
            "reward_max": metrics.get("reward_max", float("nan")),
            "reward_min": metrics.get("reward_min", float("nan")),
            "kl_divergence": kl_divergence,
            "actor_entropy": metrics.get("actor_entropy", float("nan")),
        },
        "train_history": [],
    }

    logger.info(f"[AutoTune] Training finished for trial {trial_id}.")
    tune.report(trial_result)

    return trial_result
