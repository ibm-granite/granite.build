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

# Multi-GPU per trial train driver (ray) for TRL offline RL algorithms + FSDP.
# Based on driver_multi_trl_ds.py, replacing DeepSpeed with FSDP.
#   - FSDP config (auto-detected layer class, no hardcoded names)
#   - No Ray Data — datasets loaded with pandas, converted to HF Dataset
#   - In-memory metrics (no file I/O between worker and driver)
#   - Correct model saving under FSDP (trainer.save_model handles shard gathering)
#   - Conditional RayTrainReportCallback (avoid slow checkpointing)
#   - Single config builder for DPO/KTO
#
# Supports DPO and KTO offline RL algorithms via TRL.

import logging
import math
import os
import shutil
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
from datasets import Dataset
from ray import train, tune
from ray.train import FailureConfig, Result, RunConfig, ScalingConfig
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from ray.train.torch import TorchTrainer
from transformers import AutoModelForCausalLM
from transformers.utils.logging import disable_progress_bar, enable_progress_bar

# TRL imports
from trl import DPOConfig, DPOTrainer, KTOConfig, KTOTrainer

from autotune.callbacks.sync_best_checkpoint import SyncBestCheckpointCallback
from autotune.callbacks.training_metrics import TrainingMetricsCallback

# Local
from autotune.device import disable_cudnn_sdpa
from autotune.shadow_residual import SR_TUNING_ALGO
from autotune.trainers._alora_gc import (
    AloraGradCheckpointDrainCallback,
    install_alora_gc_safety_wrapper,
)
from autotune.trainers._moe import router_aux_loss_coef as moe_router_aux_loss_coef
from autotune.trainers._peft_fsdp import peft_adapter_toggle_keeps_requires_grad
from autotune.trainers._resume import peft_adapter_load_on_cpu
from autotune.trainers._steps import resolve_epoch_steps
from autotune.utils import (
    estimate_fsdp_strategy,
    extract_tokenizer_kwargs,
    get_peft_config,
    get_qlora_quantization_config,
    get_tokenizer,
    prepare_qlora_model,
    resize_model_embeddings,
    resolve_trust_remote_code,
    set_seed,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress benign "Failed to query Ray Train Controller actor state" warnings
# from the PlacementGroupCleaner.
logging.getLogger("ray.train.v2._internal.execution.controller.placement_group_cleaner").setLevel(logging.ERROR)


# --- Resume helper (mirrors driver_multi_hf_fsdp._resolve_resume_checkpoint) ---


def _resolve_resume_checkpoint(training_output_dir: str, resume_flag: bool):
    """Resolve the ``resume_from_checkpoint`` argument for ``Trainer.train()``.

    Returns the path of the latest ``checkpoint-<step>`` directory under
    ``training_output_dir`` when ``resume_flag`` is set and one exists, or
    ``False`` otherwise (HF Trainer treats ``False`` as "train from scratch").

    Pure / side-effect-free (only stats the filesystem) so it is unit-testable
    without Ray or Torch. Kept in sync with the copy in
    ``driver_multi_hf_fsdp``.
    """
    if not resume_flag:
        return False
    if not os.path.isdir(training_output_dir):
        return False
    candidates = []
    for name in os.listdir(training_output_dir):
        if not name.startswith("checkpoint-"):
            continue
        path = os.path.join(training_output_dir, name)
        if not os.path.isdir(path):
            continue
        step_str = name[len("checkpoint-") :]
        if step_str.isdigit():
            candidates.append((int(step_str), path))
    if not candidates:
        return False
    # Highest step number is the most recent checkpoint.
    return max(candidates, key=lambda c: c[0])[1]


# --- FSDP config builder (from driver_multi_hf_fsdp.py) ---


def _build_fsdp_config(strategy: str, num_workers: int = 1) -> tuple[str, dict]:
    """
    Build FSDP configuration for HuggingFace/TRL Trainer.

    The transformer layer class is auto-detected by HF Trainer when using
    TRANSFORMER_BASED_WRAP policy.

    Returns:
        (fsdp_strategy_str, fsdp_config_dict) — the fsdp and fsdp_config args
        for TrainingArguments. Returns ("", {}) — i.e. no FSDP at all — when
        strategy is "no_shard" or when there is only one worker.
    """
    if strategy == "no_shard":
        return "", {}

    # A single worker gets no FSDP wrap at all: FSDP degrades to NO_SHARD at
    # world size 1, which is broken with fsdp_use_orig_params=True (see the
    # same guard in driver_multi_hf_fsdp._build_fsdp_config). Here it surfaced
    # as `AssertionError: Invalid: _post_backward_hook_state` on the first DPO
    # backward of a 1-GPU final training, which fit_best_config() routes to
    # this driver with num_workers=1.
    if num_workers <= 1:
        logger.info(
            "[AutoTune] Single worker: skipping the FSDP wrap "
            f"(requested strategy {strategy!r} would degrade to NO_SHARD)."
        )
        return "", {}

    fsdp_config = {
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
        # Forward prefetch assumes a static graph: FSDP records the module order on
        # the first iteration and prefetches the next unit from it. DPO/KTO's
        # in-step reference pass runs with the adapter disabled, which skips the
        # separately wrapped LoRA units, so the order differs between the policy
        # and reference forwards. With it on, a 2-rank DPO run died at its first
        # checkpoint save: `AssertionError: Expects 163840 numel but got 81920`
        # (81920 = one r=32 lora_A on a 2560-wide model).
        "fsdp_forward_prefetch": False,
        "fsdp_use_orig_params": True,
        "fsdp_sync_module_states": True,
        "fsdp_cpu_ram_efficient_loading": True,
    }

    # NOTE: Do NOT use fsdp_activation_checkpointing here. It causes a dtype
    # mismatch (bf16 vs fp32) during checkpoint recomputation in backward pass
    # when FSDP mixed precision is enabled (bf16=True). Instead, use
    # gradient_checkpointing in TrainingArguments with use_reentrant=False.

    if strategy == "full_shard":
        fsdp_config["fsdp_sharding_strategy"] = "FULL_SHARD"
        fsdp_config["fsdp_state_dict_type"] = "FULL_STATE_DICT"
    elif strategy == "shard_grad_op":
        fsdp_config["fsdp_sharding_strategy"] = "SHARD_GRAD_OP"
        fsdp_config["fsdp_state_dict_type"] = "FULL_STATE_DICT"
    elif strategy == "hybrid_shard":
        fsdp_config["fsdp_sharding_strategy"] = "HYBRID_SHARD"
        fsdp_config["fsdp_state_dict_type"] = "FULL_STATE_DICT"
    else:
        raise ValueError(
            f"Unknown FSDP strategy: {strategy}. Supported: full_shard, shard_grad_op, hybrid_shard, no_shard."
        )

    return strategy, fsdp_config


# --- Dataset loading ---


def _load_dataset_as_hf(
    file_path: str,
    percentage: float = 1.0,
    column_mapping: Optional[Dict[str, str]] = None,
) -> Dataset:
    """
    Load a dataset file with pandas and convert to HF Dataset.

    TRL trainers handle tokenization internally — they need raw text
    in HF Dataset format, not pre-tokenized data.

    Args:
        file_path: Path to JSON/JSONL/Parquet/CSV file.
        percentage: Fraction of dataset to use (for HPO).
        column_mapping: Optional {old_name: new_name} to rename columns.

    Returns:
        HuggingFace Dataset with the expected columns.
    """
    import pandas as pd

    dataset_name = os.path.basename(file_path)
    print(f"[AutoTune] Loading {dataset_name}...")
    logger.info(f"[AutoTune] Loading {dataset_name}...")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(file_path)
    elif ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_json(file_path, lines=ext == ".jsonl")

    print(f"[AutoTune] Loaded {len(df)} rows from {dataset_name}")
    logger.info(f"[AutoTune] Loaded {len(df)} rows from {dataset_name}")

    if percentage < 1.0:
        n = max(1, int(len(df) * percentage))
        df = df.head(n)
        print(f"[AutoTune] Subsampled to {len(df)} rows ({percentage * 100:.0f}%)")
        logger.info(f"[AutoTune] Subsampled to {len(df)} rows ({percentage * 100:.0f}%)")

    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"[AutoTune] Renamed columns: {column_mapping}")

    hf_dataset = Dataset.from_pandas(df, preserve_index=False)
    print(f"[AutoTune] Created HF Dataset: {len(hf_dataset)} samples, columns={hf_dataset.column_names}")
    return hf_dataset


# --- TRL config builder ---


def _build_trl_config(
    rl_algorithm: str,
    fsdp_strategy: str,
    fsdp_config: dict,
    output_dir: str,
    logging_dir: str,
    seed: int,
    steps_per_epoch: int,
    num_train_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    lr: float,
    lr_scheduler_type: str,
    warmup_ratio: float,
    run_name: str,
    save_strategy: str,
    eval_strategy: str = "epoch",
    save_steps: Optional[int] = None,
    eval_steps: Optional[int] = None,
    save_total_limit: Optional[int] = 2,
    load_best_model_at_end: bool = False,
    # TRL-specific
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    kto_desirable_weight: float = 1.0,
    kto_undesirable_weight: float = 1.0,
    router_aux_loss_coef: Optional[float] = None,
):
    """
    Build the TRL training config with FSDP and return (config, trainer_class).
    """
    logging_steps = max(1, min(10, steps_per_epoch // 10))

    # Common arguments shared across all TRL algorithms
    common_kwargs = dict(
        output_dir=output_dir,
        logging_dir=logging_dir,
        seed=seed,
        do_train=True,
        do_eval=True,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        max_steps=steps_per_epoch * num_train_epochs,
        num_train_epochs=num_train_epochs,
        eval_strategy=eval_strategy,
        logging_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(1, batch_size // 2),
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler_type,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        push_to_hub=False,
        run_name=run_name,
        report_to="none",
        disable_tqdm=False,
        bf16=True,
        # aLoRA stale-hook leak under no_grad eval is drained by
        # AloraGradCheckpointDrainCallback (see _alora_gc.py).
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,  # avoid /dev/shm exhaustion in containers
        dataloader_pin_memory=False,
        ignore_data_skip=True,
        load_best_model_at_end=load_best_model_at_end,
    )

    # Add FSDP or DDP config
    if fsdp_strategy and fsdp_strategy != "no_shard" and fsdp_config:
        common_kwargs["fsdp"] = fsdp_strategy
        common_kwargs["fsdp_config"] = fsdp_config
    else:
        # DDP fallback
        common_kwargs["ddp_backend"] = "nccl"
        common_kwargs["ddp_find_unused_parameters"] = False
        common_kwargs["ddp_broadcast_buffers"] = False

    # Granite MoE / 4.0-H classes cannot emit router logits in transformers 5.9, so
    # TRL's aux-loss term is an int 0 that crashes its metric gather; see _moe.py.
    if router_aux_loss_coef is not None:
        common_kwargs["router_aux_loss_coef"] = router_aux_loss_coef

    if save_steps is not None:
        common_kwargs["save_steps"] = save_steps
    if eval_steps is not None:
        common_kwargs["eval_steps"] = eval_steps
    if load_best_model_at_end:
        common_kwargs["metric_for_best_model"] = "eval_loss"
        common_kwargs["greater_is_better"] = False

    if rl_algorithm == "dpo":
        lt = [loss_type] if isinstance(loss_type, str) else loss_type
        config = DPOConfig(
            **common_kwargs,
            beta=beta,
            loss_type=lt,
        )
        trainer_class = DPOTrainer
    elif rl_algorithm == "kto":
        config = KTOConfig(
            **common_kwargs,
            beta=beta,
            desirable_weight=kto_desirable_weight,
            undesirable_weight=kto_undesirable_weight,
        )
        trainer_class = KTOTrainer
    else:
        raise ValueError(f"Unknown RL algorithm: {rl_algorithm}. Supported: dpo, kto.")

    return config, trainer_class


# --- Metrics extraction ---


def _extract_metrics_from_log_history(log_history: list) -> Dict[str, Any]:
    """Extract training metrics from HF Trainer's log_history in memory."""
    train_losses = []
    eval_losses = []
    last_train_entry = {}
    last_eval_entry = {}

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_losses.append(entry["loss"])
            last_train_entry = entry
        if "eval_loss" in entry:
            eval_losses.append(entry["eval_loss"])
            last_eval_entry = entry

    return {
        "train_loss": last_train_entry.get("loss", float("nan")),
        "eval_loss": last_eval_entry.get("eval_loss", float("nan")),
        "train_loss_history": train_losses,
        "eval_loss_history": eval_losses,
        "train_runtime": last_train_entry.get("train_runtime"),
        "train_samples_per_second": last_train_entry.get("train_samples_per_second"),
        "epoch": last_train_entry.get("epoch"),
        "learning_rate": last_train_entry.get("learning_rate"),
    }


# --- Worker training function ---


def train_loop_per_worker(train_loop_config: Dict[str, Any]):
    """Training function executed by each Ray Train worker (one per GPU)."""
    disable_cudnn_sdpa()  # per process: NaN grads on left-padded batches, see device.py

    print("[AutoTune] Worker starting train_loop_per_worker")
    logger.info("[AutoTune] Worker starting train_loop_per_worker")

    os.environ["OMP_NUM_THREADS"] = str(train.get_context().get_world_size())
    torch.backends.cuda.matmul.allow_tf32 = True

    import warnings

    warnings.filterwarnings("ignore", message="Upcasted low precision parameters")

    # Unpack config
    training_config = train_loop_config.get("training_config")
    fsdp_strategy = train_loop_config.get("fsdp_strategy")
    fsdp_config = train_loop_config.get("fsdp_config")
    peft_kwargs = train_loop_config.get("peft_config")
    peft_type = train_loop_config.get("peft_type")
    is_qlora = training_config.get("tuning_algorithm") == "qlora"
    # Optimizer steps per epoch (already divided by gradient_accumulation_steps
    # by the driver -- see _steps.py). Every step-valued knob below counts in
    # these, not in micro-batches.
    steps_per_epoch = train_loop_config.get("steps_per_epoch")
    trial_id = train_loop_config.get("trial_id")
    from autotune.logging_setup import bind_trial_id

    bind_trial_id(trial_id)
    rl_algorithm = train_loop_config.get("rl_algorithm", "dpo")

    output_dir = training_config.get("output_dir")
    attn_implementation = training_config.get("use_flash_attention", "eager")
    model_name_or_path = training_config.get("model_name_or_path")
    hpo_search = training_config.get("hpo_search")
    save_model_flag = training_config.get("save_model", False)
    seed = training_config.get("seed", 42)

    # Training hyperparams
    lr = train_loop_config.get("learning_rate")
    lr_scheduler_type = train_loop_config.get("lr_scheduler_type")
    gradient_accumulation_steps = train_loop_config.get("gradient_accumulation_steps")
    batch_size = train_loop_config.get("per_device_train_batch_size")
    warmup_ratio = train_loop_config.get("warmup_ratio", 0.0)
    num_train_epochs = training_config.get("num_train_epochs")

    if hpo_search:
        num_train_epochs = training_config.get("hpo_num_epochs", 1)

    # TRL-specific parameters
    beta = train_loop_config.get("beta", 0.1)
    loss_type = train_loop_config.get("loss_type", "sigmoid")
    kto_desirable_weight = train_loop_config.get("kto_desirable_weight", 1.0)
    kto_undesirable_weight = train_loop_config.get("kto_undesirable_weight", 1.0)

    run_name = f"{trial_id}" if hpo_search else f"final-{trial_id}"
    if hpo_search:
        training_output_dir = os.path.join(output_dir, "outputs", f"{trial_id}")
    else:
        # Final training uses a stable path (no random trial_id) so that
        # --resume_from_checkpoint can find the last run's checkpoints across
        # process restarts. Cleaned up on successful completion below.
        training_output_dir = os.path.join(output_dir, "final_checkpoints")
    training_logs_dir = os.path.join(output_dir, "logs", f"{trial_id}")

    # Save/checkpoint strategy
    if hpo_search:
        save_strategy = "epoch"
        eval_strategy = "epoch"
        save_steps = None
        eval_steps = None
        save_total_limit = 1
        load_best_model_at_end = False
    else:
        if num_train_epochs > 1:
            save_strategy = "epoch"
            eval_strategy = "epoch"
            save_steps = None
            eval_steps = None
        else:
            save_steps = max(1, steps_per_epoch // 5)
            eval_steps = save_steps
            save_strategy = "steps"
            eval_strategy = "steps"
        save_total_limit = 3
        load_best_model_at_end = True

    disable_progress_bar()

    # Load model. For FULL_SHARD: do NOT set device_map — let Accelerate's
    # FSDP integration handle rank-aware loading via fsdp_cpu_ram_efficient_loading
    # + fsdp_sync_module_states. Other strategies load directly on GPU.
    # Stagger across ranks to avoid concurrent mmap SIGBUS.
    is_full_shard = fsdp_strategy == "full_shard"
    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()

    # QLoRA (4-bit bitsandbytes) is incompatible with FSDP FULL_SHARD.
    if is_qlora and is_full_shard:
        raise ValueError(
            "[AutoTune] QLoRA (4-bit) is not compatible with FSDP FULL_SHARD. "
            "Use FSDP SHARD_GRAD_OP, DeepSpeed ZeRO-1/ZeRO-2, or the single-GPU "
            "driver for QLoRA."
        )

    quantization_config = get_qlora_quantization_config() if is_qlora else None
    model_kwargs = dict(
        dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation=attn_implementation,
        trust_remote_code=resolve_trust_remote_code(),
        low_cpu_mem_usage=True,
    )
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if is_qlora:
        device_label = "4-bit NF4 (QLoRA)"
    elif is_full_shard:
        device_label = "cpu (FSDP ram-efficient)"
    else:
        device_label = "gpu"

    for loading_rank in range(world_size):
        if rank == loading_rank:
            print(f"[AutoTune] Worker {rank}/{world_size} loading model: {model_name_or_path} ({device_label})")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **model_kwargs,
            )
        torch.distributed.barrier()

    # Resize token embeddings (with optional tokenizer customization)
    tokenizer_kwargs = extract_tokenizer_kwargs(training_config)
    tokenizer, num_new_tokens = get_tokenizer(model_name_or_path, **tokenizer_kwargs)
    resize_model_embeddings(model, tokenizer, num_new_tokens)

    # Get PEFT config (TRL trainers accept peft_config directly)
    peft_config = get_peft_config(
        model=model,
        model_name_or_path=model_name_or_path,
        peft_type=peft_type,
        base_kwargs=peft_kwargs,
        tokenizer=tokenizer,
    )

    if peft_config is not None and is_qlora:
        # Prep the 4-bit base for k-bit training before TRL wraps it with LoRA.
        model = prepare_qlora_model(model, use_gradient_checkpointing=True)

    enable_progress_bar()

    # Load datasets from train_loop_config (passed as serialized dicts)
    train_data = train_loop_config.get("train_data")
    eval_data = train_loop_config.get("eval_data")
    train_ds = Dataset.from_dict(train_data)
    eval_ds = Dataset.from_dict(eval_data)

    print(f"[AutoTune] Worker datasets: {len(train_ds)} train, {len(eval_ds)} eval")
    logger.info(f"[AutoTune] Worker datasets: {len(train_ds)} train, {len(eval_ds)} eval")

    # Build TRL config and trainer
    training_args, trainer_class = _build_trl_config(
        rl_algorithm=rl_algorithm,
        router_aux_loss_coef=moe_router_aux_loss_coef(model),
        fsdp_strategy=fsdp_strategy,
        fsdp_config=fsdp_config,
        output_dir=training_output_dir,
        logging_dir=training_logs_dir,
        seed=seed,
        steps_per_epoch=steps_per_epoch,
        num_train_epochs=num_train_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr=lr,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        run_name=run_name,
        save_strategy=save_strategy,
        eval_strategy=eval_strategy,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        beta=beta,
        loss_type=loss_type,
        kto_desirable_weight=kto_desirable_weight,
        kto_undesirable_weight=kto_undesirable_weight,
    )

    print(f"[AutoTune] Creating {rl_algorithm.upper()} trainer...")
    logger.info(f"[AutoTune] Creating {rl_algorithm.upper()} trainer...")

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    if peft_type == "ALORA":
        install_alora_gc_safety_wrapper(trainer.model)
        trainer.add_callback(AloraGradCheckpointDrainCallback())

    # Persist per-step training metrics (loss/grad_norm/lr/epoch) to AutotuneX.
    # Added unconditionally on every worker; on_log self-guards to rank 0 via
    # state.is_world_process_zero.
    trainer.add_callback(TrainingMetricsCallback(trial_id=trial_id))
    # Rank 0's best checkpoint is the one rotation keeps: make every rank load it.
    trainer.add_callback(SyncBestCheckpointCallback())

    # RayTrainReportCallback reports metrics from the worker back to the
    # driver process via Ray Train. Without it, train_result.metrics is empty.
    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)

    # Train. Resume from the last checkpoint only during final training
    # (never during HPO trials) and only when --resume_from_checkpoint is set.
    resume_flag = (not hpo_search) and training_config.get("resume_from_checkpoint", False)
    resume_arg = _resolve_resume_checkpoint(training_output_dir, resume_flag)
    if resume_flag:
        if resume_arg:
            logger.info(f"[AutoTune] Resuming final training from checkpoint: {resume_arg}")
        else:
            logger.warning(
                f"[AutoTune] --resume_from_checkpoint set but no checkpoint found "
                f"under {training_output_dir!r}; training from scratch."
            )
    print(f"[AutoTune] Worker starting {rl_algorithm.upper()} training (trial {trial_id})...")
    logger.info(f"[AutoTune] Worker starting training (trial {trial_id})...")
    # Force any PEFT adapter load inside train() onto CPU, avoiding the
    # exclusive-process GPU-0 contention across ranks (see _resume.py).
    #
    # Wraps train() unconditionally: both _load_from_checkpoint() (resume, at the
    # start) and _load_best_model() (load_best_model_at_end=True, at the end) call
    # model.load_adapter() -> load_peft_weights() ->
    # safe_load_file(device=infer_device()), which resolves to a bare "cuda"
    # (cuda:0) on every rank. Final training sets load_best_model_at_end=True, so
    # gating on resume_arg left that load unprotected.
    # The context manager is a no-op for non-PEFT models.
    # The outer context keeps TRL's in-step reference pass (adapter disabled under
    # no_grad) from flipping requires_grad between FSDP's forward and backward --
    # see _peft_fsdp.py.
    with peft_adapter_toggle_keeps_requires_grad():
        with peft_adapter_load_on_cpu():
            trainer.train(resume_from_checkpoint=resume_arg)

    # Collect metrics in-memory
    metrics = _extract_metrics_from_log_history(trainer.state.log_history)
    train_loss = metrics["train_loss"]
    eval_loss = metrics["eval_loss"]

    print(f"[AutoTune] Worker finished: train_loss={train_loss}, eval_loss={eval_loss}")
    logger.info(f"[AutoTune] Worker finished: train_loss={train_loss}, eval_loss={eval_loss}")

    # Save model — ALL ranks must participate for FSDP state dict gathering.
    if not hpo_search and save_model_flag:
        model_name = training_config.get("output_model_name")
        output_model_path = output_dir
        output_model_id = os.path.join(output_model_path, model_name)

        if train.get_context().get_world_rank() == 0:
            print(f"[AutoTune] Saving best model to: {output_model_id}")
            logger.info(f"[AutoTune] Saving best model to: {output_model_id}")

        # All ranks call save_model — FSDP state dict gather is collective
        trainer.save_model(output_model_id)

        if train.get_context().get_world_rank() == 0:
            tokenizer.save_pretrained(output_model_id)
            print(f"[AutoTune] Model saved to: {output_model_id}")
            logger.info(f"[AutoTune] Model saved to: {output_model_id}")

            # Clean up checkpoint dirs unless --keep_checkpoints is set (debug).
            keep_checkpoints = training_config.get("keep_checkpoints", False)
            if not keep_checkpoints:
                import glob as _glob

                # training_output_dir is the stable final_checkpoints dir for final
                # training. Removing the checkpoints on success means a later
                # --resume_from_checkpoint correctly finds nothing and starts fresh;
                # an interrupted run leaves them behind to resume from.
                ckpt_dirs = _glob.glob(os.path.join(training_output_dir, "checkpoint-*"))
                for ckpt_dir in ckpt_dirs:
                    shutil.rmtree(ckpt_dir, ignore_errors=True)
                if ckpt_dirs:
                    print(f"[AutoTune] Cleaned up {len(ckpt_dirs)} checkpoint dir(s)")
                # Drop the now-empty stable checkpoint dir itself.
                if os.path.isdir(training_output_dir):
                    shutil.rmtree(training_output_dir, ignore_errors=True)
                outputs_dir = os.path.join(output_dir, "outputs")
                if os.path.exists(outputs_dir):
                    shutil.rmtree(outputs_dir, ignore_errors=True)
                    print(f"[AutoTune] Cleaned up training outputs dir: {outputs_dir}")
                    logger.info(f"[AutoTune] Cleaned up training outputs dir: {outputs_dir}")
                # NOTE: do NOT rmtree output_dir/train_results here — Ray Train's
                # TrainController periodically writes checkpoint_manager_snapshot.json
                # into that directory, and is still alive until ray_trainer.fit()
                # returns on the driver. Deleting it from the worker races with the
                # controller and raises "No such file or directory". The driver
                # cleans up train_results after fit() returns.
                data_cache_dir = os.path.join(output_dir, "data_cache")
                if os.path.exists(data_cache_dir):
                    shutil.rmtree(data_cache_dir, ignore_errors=True)
                    print(f"[AutoTune] Cleaned up data cache dir: {data_cache_dir}")
                    logger.info(f"[AutoTune] Cleaned up data cache dir: {data_cache_dir}")
            else:
                logger.info("[AutoTune] --keep_checkpoints set; skipping artifact cleanup")

    return metrics


# --- Main driver function ---


def train_driver_multi_gpu(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main TRL train driver with FSDP for offline RL (DPO/KTO).

    Args:
        config: Dict with hyperparameter configuration from Ray Tune.

    Returns:
        A Dict summarizing the training results for Ray Tune.
    """
    trial_id = tune.get_context().get_trial_id()
    from autotune.logging_setup import bind_trial_id

    bind_trial_id(trial_id)

    print(f"[AutoTune] Training driver multi GPU TRL+FSDP (trial {trial_id})")
    logger.info(f"[AutoTune] Trial ID: {trial_id}")

    local_config = deepcopy(config)
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Unpack config sections (discard unused ones — pop for side effect)
    training_config = local_config.pop("training_config")

    # Shadow Residual is not supported by this driver. This is a hard rejection
    # rather than a fallthrough because the failure would otherwise be SILENT:
    # peft_type "SR" resolves to a plain LoraConfig whose target_modules include
    # "cross_stream", and PEFT's inject_adapter raises only when *nothing*
    # matched (peft/tuners/tuners_utils.py) -- so on a plain Granite base the
    # cross_stream target is dropped without a warning and the run trains as
    # ordinary LoRA while being labelled and reported as "sr".
    if training_config.get("tuning_algorithm") == SR_TUNING_ALGO:
        raise ValueError(
            "[AutoTune] Shadow Residual supports SFT only and is not supported by "
            "this driver, which trains offline RL (DPO/KTO) via TRL. Use "
            "--rl_algo none (which routes to the HF SFT drivers instead)."
        )
    local_config.pop("training_rl_config")
    tuner_flags = local_config.pop("tuner_flags")
    local_config.pop("tune_config")

    # Extract parameters
    train_file = training_config.get("train_file")
    eval_file = training_config.get("validation_file")
    peft_type = training_config.get("peft_type")
    fsdp_strategy = training_config.get("fsdp_strategy", "auto")
    max_length = training_config.get("max_length", 512)
    model_name_or_path = training_config.get("model_name_or_path")
    output_dir = training_config.get("output_dir")
    seed = training_config.get("seed", 42)
    num_train_epochs = training_config.get("num_train_epochs", 1)
    num_workers = training_config.get("num_workers", 1)
    hpo_search = training_config.get("hpo_search", False)

    # TRL-specific parameters
    rl_algorithm = training_config.get("rl_algorithm", "dpo")

    set_seed(seed)

    # Separate tuner params from fixed params
    train_kwargs = {}
    for k, v in local_config.items():
        if k in tuner_flags and tuner_flags[k] is False:
            train_kwargs[k] = v
    train_kwargs["num_train_epochs"] = num_train_epochs
    alpha_ratio = train_kwargs.pop("alpha_ratio", None)

    print(f"[AutoTune] Training args: {train_kwargs}")
    logger.info(f"[AutoTune] Training args: {train_kwargs}")

    # PEFT config
    if peft_type is not None:
        peft_kwargs = {k: v for k, v in local_config.items() if tuner_flags.get(k) is True}
        if alpha_ratio is not None and "r" in peft_kwargs:
            peft_kwargs["lora_alpha"] = int(alpha_ratio * peft_kwargs["r"])
        print(f"[AutoTune] PEFT args: {peft_kwargs}")
        logger.info(f"[AutoTune] PEFT args: {peft_kwargs}")
    else:
        peft_kwargs = None

    # Find the optimal FSDP strategy based on model size and training args.
    if fsdp_strategy == "auto":
        fsdp_strategy = estimate_fsdp_strategy(
            model_name_or_path=model_name_or_path,
            max_seq_length=max_length,
            per_device_batch_size=train_kwargs["per_device_train_batch_size"],
            num_gpus=num_workers,
            peft_config=peft_kwargs,
        )

        print(f"[AutoTune] FSDP strategy: {fsdp_strategy}")
        logger.info(f"[AutoTune] FSDP strategy: {fsdp_strategy}")

    # Safety checks
    assert fsdp_strategy in ["full_shard", "shard_grad_op", "hybrid_shard", "no_shard"], (
        f"Invalid FSDP strategy: {fsdp_strategy}. Supported: full_shard, shard_grad_op, hybrid_shard, no_shard."
    )

    # Build FSDP config
    fsdp_str, fsdp_config = _build_fsdp_config(fsdp_strategy, num_workers)
    print(f"[AutoTune] FSDP strategy: {fsdp_strategy}")
    logger.info(f"[AutoTune] FSDP strategy: {fsdp_strategy}")
    if fsdp_config:
        print(f"[AutoTune] FSDP config: {fsdp_config}")
        logger.info(f"[AutoTune] FSDP config: {fsdp_config}")

    # Dataset loading — no Ray Data, load with pandas → HF Dataset
    percentage = 1.0 if not hpo_search else training_config.get("hpo_dataset_percentage", 0.10)
    if percentage == 0.0:
        raise ValueError("HPO dataset percentage cannot be 0.0")

    print(f"[AutoTune] Loading datasets for {rl_algorithm.upper()} (percentage={percentage * 100:.0f}%)...")
    logger.info(f"[AutoTune] Loading datasets for {rl_algorithm.upper()}...")

    column_mapping = None

    train_ds = _load_dataset_as_hf(
        file_path=train_file,
        percentage=percentage,
        column_mapping=column_mapping,
    )
    eval_ds = _load_dataset_as_hf(
        file_path=eval_file,
        percentage=percentage,
        column_mapping=column_mapping,
    )

    batch_size = train_kwargs["per_device_train_batch_size"]
    grad_accum = local_config.get("gradient_accumulation_steps", 1)
    train_ds_size = len(train_ds)
    # Optimizer steps per epoch -- the unit every step-valued TrainingArguments
    # knob counts in (max_steps, logging_steps, save_steps/eval_steps). Dividing
    # the per-rank micro-batch count by gradient_accumulation_steps is
    # load-bearing; see _steps.py for what omitting it did to the schedule.
    epoch_steps = resolve_epoch_steps(
        train_ds_size=train_ds_size,
        per_device_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_workers=num_workers,
    )
    steps_per_epoch = epoch_steps.optimizer_steps

    print(
        f"[AutoTune] Dataset: {train_ds_size} train samples, {steps_per_epoch} steps/epoch, "
        f"{num_workers} workers, batch_size={batch_size}"
    )
    logger.info(f"[AutoTune] Dataset: {train_ds_size} train, {steps_per_epoch} steps/epoch")

    # TRL Trainer owns all checkpointing (save_strategy, load_best_model_at_end).
    # Ray Train checkpoint persistence is disabled — RayTrainReportCallback still
    # calls train.report(metrics=...) so train_result.metrics is populated, but
    # the checkpoint copy that Ray would write to train_results/ is dropped.
    checkpoint_config = None

    # Pass dataset as serializable dicts via train_loop_config
    train_data_dict = train_ds.to_dict()
    eval_data_dict = eval_ds.to_dict()

    # Build TorchTrainer
    run_name = f"run-{trial_id}" if hpo_search else "train_results"
    resources_per_worker = {"CPU": 1, "GPU": 1}

    ray_trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            **train_kwargs,
            "training_config": training_config,
            "peft_type": peft_type,
            "peft_config": peft_kwargs,
            "fsdp_strategy": fsdp_str,
            "fsdp_config": fsdp_config,
            "steps_per_epoch": steps_per_epoch,
            "trial_id": trial_id,
            "rl_algorithm": rl_algorithm,
            "train_data": train_data_dict,
            "eval_data": eval_data_dict,
        },
        run_config=RunConfig(
            name=run_name,
            storage_path=output_dir,
            checkpoint_config=checkpoint_config,
            failure_config=FailureConfig(max_failures=0),
        ),
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=True,
            resources_per_worker=resources_per_worker,
        ),
    )

    print(f"[AutoTune] Starting {rl_algorithm.upper()} distributed training ({num_workers} workers, FSDP)")
    logger.info(f"[AutoTune] Starting {rl_algorithm.upper()} training ({num_workers} workers, FSDP)")

    # Run training
    train_result: Optional[Result] = None
    try:
        train_result = ray_trainer.fit()
        print("[AutoTune] Distributed training finished.")
        logger.info("[AutoTune] Distributed training finished.")
    except Exception as e:
        logger.error(f"[AutoTune] Training failed (trial {trial_id}): {e}", exc_info=True)
        print(f"[AutoTune] Training failed (trial {trial_id}): {e}")
        raise  # Let Ray Tune mark trial as ERRORED

    # Extract metrics from train_result.metrics
    loss = 10000.0
    train_loss = float("nan")
    eval_loss_val = float("nan")
    train_log = {}
    eval_results = {}

    if train_result is not None and train_result.metrics:
        metrics = train_result.metrics
        print(f"[AutoTune] Train result metrics: {metrics}")
        logger.info(f"[AutoTune] Train result metrics: {metrics}")

        train_loss = metrics.get("loss", metrics.get("train_loss", float("nan")))
        eval_loss_val = metrics.get("eval_loss", float("nan"))
        loss = eval_loss_val if not math.isnan(eval_loss_val) else train_loss
        train_log = metrics
        eval_results = {k: v for k, v in metrics.items() if k.startswith("eval_")}

    if math.isnan(loss) or math.isinf(loss):
        loss = 10000.0

    # Print results
    print(f"[AutoTune] Results: train_loss={train_loss}, eval_loss={eval_loss_val}, loss={loss}")
    logger.info(f"[AutoTune] Results: train_loss={train_loss}, eval_loss={eval_loss_val}, loss={loss}")

    # HPO trials never need their model checkpoint — metrics were already
    # extracted from train_result.metrics above. The single epoch save (kept so
    # RayTrainReportCallback.on_save() fires and populates train_result.metrics)
    # is removed here, on the driver, after fit() returns and the TrainController
    # has shut down (deleting from the worker races with its snapshot writes).
    # Always removed, independent of --keep_checkpoints (which governs only
    # final-training artifacts).
    if hpo_search:
        hpo_trial_dir = os.path.join(output_dir, "outputs", f"{trial_id}")
        if os.path.isdir(hpo_trial_dir):
            shutil.rmtree(hpo_trial_dir, ignore_errors=True)
            logger.info(f"[AutoTune] Cleaned up HPO trial dir: {hpo_trial_dir}")

    # Clean up Ray Train's run directory now that fit() has returned and the
    # TrainController has shut down. Doing this from inside the worker races
    # with the controller's periodic snapshot writes and raises ENOENT.
    if not hpo_search and not training_config.get("keep_checkpoints", False):
        train_results_dir = os.path.join(output_dir, "train_results")
        if os.path.isdir(train_results_dir):
            shutil.rmtree(train_results_dir, ignore_errors=True)
            logger.info(f"[AutoTune] Cleaned up training results dir: {train_results_dir}")

    # Return trial result
    result = {
        "loss": loss,
        "train_loss": train_loss,
        "eval_loss": eval_loss_val,
        "rl_algorithm": rl_algorithm,
        "done": True,
        "config": config,
        "train_log": train_log,
        "eval_results": eval_results,
        "train_history": [],
    }

    print(f"[AutoTune] Training finished for trial {trial_id}.")
    logger.info(f"[AutoTune] Training finished for trial {trial_id}.")

    tune.report(result)
    return result
