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

# Multi-GPU per trial train driver (ray) for HuggingFace + FSDP SFT training.
# Refactored version of driver_multi_hf_fsdp.py with:
#   - Proper FSDP config (auto-detected layer class, no hardcoded names)
#   - In-memory metrics collection (no file I/O between worker and driver)
#   - Correct model saving under FSDP (trainer.save_model handles shard gathering)
#   - Optimized training args for speed (reduced logging, no tensorboard overhead)
#   - Single TrainingArguments path for both FSDP and DDP

import logging
import math
import os
from copy import deepcopy
from typing import Any, Callable, Dict, Optional

import ray
import torch
from peft import get_peft_model
from ray import train, tune
from ray.train import FailureConfig, Result, RunConfig, ScalingConfig
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from ray.train.torch import TorchConfig, TorchTrainer
from torch.utils.data import IterableDataset
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.utils.logging import disable_progress_bar, enable_progress_bar

# Local
from autotune.cluster import compute_ray_data_sizing, ray_data_block_target
from autotune.trainers._alora_gc import (
    AloraGradCheckpointDrainCallback,
    install_alora_gc_safety_wrapper,
)
from autotune.trainers._resume import peft_adapter_load_on_cpu
from autotune.utils import (
    assert_dp_sharding,
    estimate_fsdp_strategy,
    extract_tokenizer_kwargs,
    get_peft_config,
    get_qlora_quantization_config,
    get_tokenizer,
    prepare_qlora_model,
    resize_model_embeddings,
    resolve_trust_remote_code,
    set_seed,
    tokenize_batch,
)

logger = logging.getLogger(__name__)

# Suppress benign "Failed to query Ray Train Controller actor state" warnings
# from the PlacementGroupCleaner. These occur when the Ray State API is
# temporarily slow under load from Ray Tune + TorchTrainer.
logging.getLogger("ray.train.v2._internal.execution.controller.placement_group_cleaner").setLevel(logging.ERROR)


def _resolve_resume_checkpoint(training_output_dir: str, resume_flag: bool):
    """Resolve the ``resume_from_checkpoint`` argument for ``Trainer.train()``.

    Returns the path of the latest ``checkpoint-<step>`` directory under
    ``training_output_dir`` when ``resume_flag`` is set and one exists, or
    ``False`` otherwise (HF Trainer treats ``False`` as "train from scratch").

    Pure / side-effect-free (only stats the filesystem) so it is unit-testable
    without Ray or Torch. Shared by the FSDP SFT drivers.

    Note: the DeepSpeed and single-GPU drivers do not yet wire this up — see
    ``driver_multi_hf_fsdp`` / ``driver_multi_trl_fsdp`` for the reference.
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


def _apply_chat_template_to_df(df, tokenizer, input_col: str):
    """Convert message-list rows in ``df[input_col]`` to plain strings via the
    tokenizer's chat template. Handles optional ``documents`` and ``tools``
    columns for RAG / tool-use prompts. Returns ``df`` unchanged if the column
    doesn't hold message lists (detected by inspecting the first row) or if
    the DataFrame is empty.

    Shared between the Arrow and Ray Data backends to keep chat-template
    handling in a single place.
    """
    if input_col not in df.columns or len(df) == 0:
        return df

    first_val = df[input_col].iloc[0]
    if not isinstance(first_val, list):
        return df

    has_documents = "documents" in df.columns and isinstance(df["documents"].iloc[0], list)
    has_tools = "tools" in df.columns and isinstance(df["tools"].iloc[0], list)

    if has_documents and has_tools:
        df[input_col] = df.apply(
            lambda row: tokenizer.apply_chat_template(
                row[input_col],
                documents=row["documents"],
                tools=row["tools"],
                tokenize=False,
                add_generation_prompt=True,
            ),
            axis=1,
        )
    elif has_documents:
        df[input_col] = df.apply(
            lambda row: tokenizer.apply_chat_template(
                row[input_col],
                documents=row["documents"],
                tokenize=False,
                add_generation_prompt=True,
            ),
            axis=1,
        )
    elif has_tools:
        df[input_col] = df.apply(
            lambda row: tokenizer.apply_chat_template(
                row[input_col],
                tools=row["tools"],
                tokenize=False,
                add_generation_prompt=True,
            ),
            axis=1,
        )
    else:
        df[input_col] = df[input_col].apply(
            lambda msgs: tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return df


class _RayShardIterableDataset(IterableDataset):
    """Wraps a Ray Data shard so HF Trainer can consume it as an IterableDataset.

    Ray Train splits the parent Dataset across workers, so each worker only
    iterates over its own rank-specific slice — no DistributedSampler needed.

    When ``total_samples`` is set, iteration produces exactly that many
    samples per rank, cycling through the shard as needed. This is critical
    under FSDP/DDP: unequal iteration lengths across ranks cause NCCL
    watchdog hangs because fast ranks reach end-of-epoch collectives while
    slower ranks are still producing batches. Ray Data partitions by *block*,
    not by row count, so shard sizes are not guaranteed equal — this cap
    makes them equal by construction.
    """

    def __init__(
        self,
        shard,
        batch_size: int,
        total_samples: Optional[int] = None,
        shuffle_buffer: Optional[int] = None,
    ):
        self.shard = shard
        self.batch_size = batch_size
        self.total_samples = total_samples
        self.shuffle_buffer = shuffle_buffer

    def _iter_once(self):
        for batch in self.shard.iter_torch_batches(
            batch_size=self.batch_size,
            local_shuffle_buffer_size=self.shuffle_buffer,
            dtypes={
                "input_ids": torch.long,
                "attention_mask": torch.long,
                "labels": torch.long,
            },
        ):
            bsz = batch["input_ids"].shape[0]
            for i in range(bsz):
                yield {k: v[i] for k, v in batch.items()}

    def __iter__(self):
        if self.total_samples is None:
            yield from self._iter_once()
            return

        yielded = 0
        while yielded < self.total_samples:
            for sample in self._iter_once():
                yield sample
                yielded += 1
                if yielded >= self.total_samples:
                    return


def _read_dataset(file_path: str) -> "ray.data.Dataset":
    """Read a dataset file into a Ray Dataset, dispatching on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".parquet":
        return ray.data.read_parquet(file_path)
    if ext == ".csv":
        return ray.data.read_csv(file_path)
    if ext == ".jsonl":
        return ray.data.read_json(file_path)
    if ext == ".json":
        # Ray Data's read_json expects line-delimited JSON; fall back to
        # pandas for a single JSON array and wrap as a Ray Dataset.
        import pandas as pd

        df = pd.read_json(file_path, lines=False)
        return ray.data.from_pandas(df)
    raise ValueError(f"Unsupported dataset extension: {ext} ({file_path})")


def _make_tokenize_fn(
    model_name_or_path: str,
    tokenizer_kwargs: Dict[str, Any],
    input_col: str,
    output_col: str,
    max_length: int,
) -> Callable[[Any], Dict[str, Any]]:
    """Build a map_batches function that applies chat template (if needed)
    and tokenizes the batch. The tokenizer is constructed lazily inside the
    Ray Data worker — avoids pickling a loaded tokenizer across processes.
    """

    state: Dict[str, Any] = {"tokenizer": None}

    def _fn(batch):
        import pandas as pd

        # Ray Data passes numpy-dict batches (batch_format="numpy"); convert
        # to a DataFrame locally so the existing pandas-based chat-template
        # and tokenize logic can stay as-is. We avoid batch_format="pandas"
        # because Ray Data's pandas BlockAccessor references
        # pandas.core.common.SettingWithCopyWarning, which was removed in
        # pandas 3.0.
        df = pd.DataFrame(batch)

        if state["tokenizer"] is None:
            tokenizer, _ = get_tokenizer(model_name_or_path, **tokenizer_kwargs)
            state["tokenizer"] = tokenizer
        tokenizer = state["tokenizer"]

        df = _apply_chat_template_to_df(df, tokenizer, input_col)

        tokenized = tokenize_batch(
            batch=df,
            tokenizer=tokenizer,
            input_col=input_col,
            output_col=output_col,
            max_length=max_length,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["labels"],
        }

    return _fn


class _TokenizedArrowDataset(torch.utils.data.Dataset):
    """Map-style Dataset backed by a memory-mapped Arrow IPC file.

    Each worker mmaps the same file (no serialization through the Ray
    object store) and indexes into it. HF Trainer's default distributed
    sampler shards rows across ranks.
    """

    def __init__(self, arrow_path: str):
        import pyarrow as pa

        self.table = pa.ipc.open_file(pa.memory_map(arrow_path, "r")).read_all()
        self._len = len(self.table)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        row = self.table.slice(idx, 1)
        return {
            "input_ids": torch.tensor(row.column("input_ids")[0].as_py(), dtype=torch.long),
            "attention_mask": torch.tensor(row.column("attention_mask")[0].as_py(), dtype=torch.long),
            "labels": torch.tensor(row.column("labels")[0].as_py(), dtype=torch.long),
        }


def _load_tokenize_and_save(
    file_path: str,
    tokenizer,
    input_col: str,
    output_col: str,
    max_length: int,
    output_arrow_path: str,
    percentage: float = 1.0,
    chunk_size: int = 50000,
) -> int:
    """Load a dataset file, tokenize it in chunks on the driver, and write
    the result as an Arrow IPC file for workers to memory-map. Returns the
    number of tokenized samples.
    """
    import pandas as pd
    import pyarrow as pa
    from tqdm import tqdm

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(file_path)
    elif ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext == ".json":
        df = pd.read_json(file_path, lines=False)
    else:  # .jsonl (and anything else line-delimited)
        df = pd.read_json(file_path, lines=ext == ".jsonl")

    logger.info(f"[AutoTune] Loaded {len(df)} rows from {file_path}")

    if percentage < 1.0:
        n = max(1, int(len(df) * percentage))
        df = df.head(n)
        logger.info(f"[AutoTune] Subsampled to {len(df)} rows ({percentage * 100:.0f}%)")

    df = _apply_chat_template_to_df(df, tokenizer, input_col)

    logger.info("[AutoTune] First instance (after formatting):")
    logger.info(f"  {input_col}: {df[input_col].iloc[0]}")
    logger.info(f"  {output_col}: {df[output_col].iloc[0]}")

    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    total_rows = len(df)
    dataset_name = os.path.basename(file_path)

    logger.info(f"[AutoTune] Tokenizing dataset with max_length={max_length}...")
    with tqdm(total=total_rows, desc=f"Tokenizing {dataset_name}", unit="rows") as pbar:
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunk_df = df.iloc[start:end]
            tokenized = tokenize_batch(
                batch=chunk_df,
                tokenizer=tokenizer,
                input_col=input_col,
                output_col=output_col,
                max_length=max_length,
            )
            all_input_ids.extend(tokenized["input_ids"])
            all_attention_mask.extend(tokenized["attention_mask"])
            all_labels.extend(tokenized["labels"])
            pbar.update(end - start)

    table = pa.table(
        {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }
    )

    os.makedirs(os.path.dirname(output_arrow_path), exist_ok=True)
    writer = pa.ipc.new_file(output_arrow_path, table.schema)
    batch_rows = 100000
    for i in range(0, len(table), batch_rows):
        writer.write(table.slice(i, min(batch_rows, len(table) - i)))
    writer.close()

    num_samples = len(table)
    size_mb = os.path.getsize(output_arrow_path) / (1024 * 1024)
    logger.info(f"[AutoTune] Saved {num_samples} samples to {output_arrow_path} ({size_mb:.1f} MB)")
    return num_samples


def _build_fsdp_config(strategy: str) -> tuple[str, dict]:
    """
    Build FSDP configuration for HuggingFace Trainer.

    The transformer layer class is auto-detected by HF Trainer when using
    TRANSFORMER_BASED_WRAP policy — no explicit layer class mapping needed.

    Returns:
        (fsdp_strategy_str, fsdp_config_dict) — the fsdp and fsdp_config args
        for TrainingArguments. If strategy is "no_shard", returns ("", {}).
    """
    if strategy == "no_shard":
        return "", {}

    fsdp_config = {
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
        "fsdp_forward_prefetch": True,
        "fsdp_use_orig_params": True,
        "fsdp_sync_module_states": True,
        "fsdp_cpu_ram_efficient_loading": True,
    }

    # NOTE: Do NOT use fsdp_activation_checkpointing here. It causes a dtype
    # mismatch (bf16 vs fp32) during checkpoint recomputation in backward pass
    # when FSDP mixed precision is enabled (bf16=True). The recomputed tensors
    # run outside FSDP's mixed-precision autocast context, producing fp32
    # instead of the bf16 saved during forward.
    # Instead, use gradient_checkpointing in TrainingArguments (added below)
    # with use_reentrant=False, which correctly preserves the autocast context.

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


def _extract_metrics_from_log_history(log_history: list) -> Dict[str, Any]:
    """
    Extract training metrics from HF Trainer's log_history in memory.

    Returns a dict with train_loss, eval_loss, and per-step metrics.
    """
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

    train_loss = last_train_entry.get("loss", float("nan"))
    eval_loss = last_eval_entry.get("eval_loss", float("nan"))

    return {
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "train_loss_history": train_losses,
        "eval_loss_history": eval_losses,
        "train_runtime": last_train_entry.get("train_runtime"),
        "train_samples_per_second": last_train_entry.get("train_samples_per_second"),
        "train_steps_per_second": last_train_entry.get("train_steps_per_second"),
        "total_flos": last_train_entry.get("total_flos"),
        "epoch": last_train_entry.get("epoch"),
        "learning_rate": last_train_entry.get("learning_rate"),
    }


# --- Worker training function ---


def train_loop_per_worker(train_loop_config: Dict[str, Any]):
    """
    Training function executed by each Ray Train worker (one per GPU).
    """
    from autotune.logging_setup import setup_logging

    setup_logging()

    logger.info("[AutoTune] Worker starting train_loop_per_worker")
    logger.info(f"[AutoTune] Train loop config: {train_loop_config}")

    os.environ["OMP_NUM_THREADS"] = str(
        train.get_context().get_world_size()  # reasonable default
    )
    torch.backends.cuda.matmul.allow_tf32 = True

    # Unpack config
    training_config = train_loop_config.get("training_config")
    fsdp_strategy = train_loop_config.get("fsdp_strategy")
    fsdp_config = train_loop_config.get("fsdp_config")
    peft_kwargs = train_loop_config.get("peft_config")
    peft_type = train_loop_config.get("peft_type")
    is_qlora = training_config.get("tuning_algorithm") == "qlora"
    steps_per_epoch = train_loop_config.get("steps_per_epoch")
    trial_id = train_loop_config.get("trial_id")

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
    logging_steps = max(1, min(10, steps_per_epoch // 10))

    if hpo_search:
        # Save once per epoch so RayTrainReportCallback.on_save() fires
        # and metrics flow back to Result.metrics.
        save_strategy = "epoch"
        eval_strategy = "epoch"
        save_steps = None
        eval_steps = None
        save_total_limit = 1
    else:
        # Final training — keep the last 3 checkpoints; the final saved model
        # is the last step's weights (no best-model tracking).
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

    # Build TrainingArguments (unified for FSDP and DDP)
    training_args_kwargs = dict(
        output_dir=training_output_dir,
        logging_dir=training_logs_dir,
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
        label_names=["input_ids", "attention_mask"],
        push_to_hub=False,
        run_name=run_name,
        report_to="none",
        disable_tqdm=False,
        bf16=True,
        torch_empty_cache_steps=max(1, steps_per_epoch // 5),
        dataloader_num_workers=0,  # avoid /dev/shm exhaustion in containers
        dataloader_pin_memory=False,
        ignore_data_skip=True,
        load_best_model_at_end=False,
    )

    # Only force dispatch_batches=False under the Ray Data backend. Ray Data's
    # SplitCoordinator deadlocks under Accelerate's default IterableDataset
    # dispatch path (rank 0 broadcasts; non-zero ranks never call iter() on
    # the dataset, so their splits stall). With the Arrow backend, leaving
    # Accelerate at defaults is required for it to install a DistributedSampler
    # around the map-style dataset — otherwise every rank trains on the full
    # dataset via SeedableRandomSampler.
    data_backend = train_loop_config.get("data_backend", "arrow")
    if data_backend == "ray_data":
        training_args_kwargs["accelerator_config"] = {"dispatch_batches": False}

    if save_steps is not None:
        training_args_kwargs["save_steps"] = save_steps
    if eval_steps is not None:
        training_args_kwargs["eval_steps"] = eval_steps

    # aLoRA: PEFT 0.18 (PR #2860, fixes huggingface/peft#2826) raises
    # "Multiple invocations of PEFT forward hooks before .backward()" if
    # eval forwards under no_grad leak hooks. _alora_gc.py drains them.
    training_args_kwargs["gradient_checkpointing"] = True
    training_args_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    # Add FSDP or DDP config
    if fsdp_strategy and fsdp_strategy != "no_shard" and fsdp_config:
        training_args_kwargs["fsdp"] = fsdp_strategy
        training_args_kwargs["fsdp_config"] = fsdp_config
    else:
        # For DDP
        training_args_kwargs["ddp_backend"] = "nccl"
        training_args_kwargs["ddp_find_unused_parameters"] = False
        training_args_kwargs["ddp_broadcast_buffers"] = False

    training_args = TrainingArguments(**training_args_kwargs)

    disable_progress_bar()

    # Suppress FSDP "Upcasted low precision parameters" warning — this is
    # expected when loading bf16 model with FSDP mixed precision (bf16 compute,
    # fp32 master weights for optimizer updates).
    import warnings

    warnings.filterwarnings("ignore", message="Upcasted low precision parameters")

    # Load model with low_cpu_mem_usage to reduce peak memory from safetensors
    # mmap. Stagger loading across ranks to avoid all workers memory-mapping
    # simultaneously, which causes SIGBUS in containers with limited memory.
    # For FULL_SHARD: do NOT set device_map — let Accelerate's FSDP integration
    # handle rank-aware loading via fsdp_cpu_ram_efficient_loading +
    # fsdp_sync_module_states. Other strategies load directly on GPU.
    is_full_shard = fsdp_strategy == "full_shard"
    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()

    # QLoRA (4-bit bitsandbytes) is incompatible with FSDP FULL_SHARD: the
    # quantized 4-bit base params cannot be flat-parameter sharded across ranks.
    # Fail fast with guidance toward SHARD_GRAD_OP / ZeRO-2 / single-GPU.
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

    # Stagger model loading across ranks to avoid concurrent mmap SIGBUS.
    for loading_rank in range(world_size):
        if rank == loading_rank:
            logger.info(f"[AutoTune] Worker {rank}/{world_size} loading model: {model_name_or_path} ({device_label})")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **model_kwargs,
            )
        torch.distributed.barrier()

    # Resize token embeddings (with optional tokenizer customization)
    tokenizer_kwargs = extract_tokenizer_kwargs(training_config)
    tokenizer, num_new_tokens = get_tokenizer(model_name_or_path, **tokenizer_kwargs)
    resize_model_embeddings(model, tokenizer, num_new_tokens)

    # Apply PEFT if configured
    peft_config = get_peft_config(
        model=model,
        model_name_or_path=model_name_or_path,
        peft_type=peft_type,
        base_kwargs=peft_kwargs,
        tokenizer=tokenizer,
    )
    if peft_config is not None:
        if is_qlora:
            model = prepare_qlora_model(model, use_gradient_checkpointing=True)
        else:
            model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    enable_progress_bar()

    # Load datasets per configured backend (data_backend is read above).
    if data_backend == "arrow":
        train_arrow_path = train_loop_config["train_arrow_path"]
        eval_arrow_path = train_loop_config["eval_arrow_path"]
        train_dataset = _TokenizedArrowDataset(train_arrow_path)
        eval_dataset = _TokenizedArrowDataset(eval_arrow_path)
        logger.info(
            f"[AutoTune] Worker {rank}/{world_size} using Arrow mmap "
            f"({len(train_dataset)} train, {len(eval_dataset)} eval)"
        )
    elif data_backend == "ray_data":
        # Ray Train auto-shards datasets={"train","eval"} across workers.
        # Cap per-rank training iteration to a fixed sample count so every
        # rank produces identical step counts — prevents NCCL watchdog hangs
        # when Ray Data block-level splits yield unequal shard row counts.
        train_shard = train.get_dataset_shard("train")
        eval_shard = train.get_dataset_shard("eval")
        samples_per_rank = steps_per_epoch * batch_size * num_train_epochs
        train_dataset = _RayShardIterableDataset(
            train_shard,
            batch_size=batch_size,
            total_samples=samples_per_rank,
            shuffle_buffer=4096,
        )
        eval_dataset = _RayShardIterableDataset(
            eval_shard,
            batch_size=max(1, batch_size // 2),
        )
        logger.info(
            f"[AutoTune] Worker {rank}/{world_size} using Ray Data shards (train samples_per_rank={samples_per_rank})"
        )
    else:
        raise ValueError(f"Unknown data_backend: {data_backend}")

    # Create HF Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=None,
        data_collator=default_data_collator,
    )

    if peft_type == "ALORA":
        install_alora_gc_safety_wrapper(trainer.model)
        trainer.add_callback(AloraGradCheckpointDrainCallback())

    # RayTrainReportCallback reports metrics from workers back to the driver
    # via train.report(). Without it, ray_trainer.fit() returns Result with
    # metrics=None. Checkpoint saving only triggers when checkpoint_config is
    # set in RunConfig (None during HPO), so the callback is safe to always add.
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
    logger.info(f"[AutoTune] Worker starting training (trial {trial_id})...")
    # When resuming a PEFT checkpoint, force the adapter load onto CPU to avoid
    # the exclusive-process GPU-0 contention across ranks (see _resume.py).
    if resume_arg:
        with peft_adapter_load_on_cpu():
            trainer.train(resume_from_checkpoint=resume_arg)
    else:
        trainer.train(resume_from_checkpoint=resume_arg)

    # Audit data-parallel sharding post-hoc (rank-0 log only). Runs after
    # trainer.train() so Accelerate has already wrapped the dataloader with
    # DistributedSampler via prepare_data_loader(). Warns if that didn't
    # happen under world_size>1, or if HF max_steps disagrees with
    # steps_per_epoch*epochs.
    assert_dp_sharding(
        trainer=trainer,
        rank=rank,
        world_size=world_size,
        per_device_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        steps_per_epoch=steps_per_epoch,
        num_train_epochs=num_train_epochs,
    )

    # Collect metrics in-memory from trainer.state.log_history
    metrics = _extract_metrics_from_log_history(trainer.state.log_history)
    train_loss = metrics["train_loss"]
    eval_loss = metrics["eval_loss"]

    logger.info(f"[AutoTune] Worker finished: train_loss={train_loss}, eval_loss={eval_loss}")

    # Save model — ALL ranks must participate in trainer.save_model() for
    # FSDP because gathering the full state dict is a collective operation.
    # load_best_model_at_end=False, so this writes the last step's weights.
    if not hpo_search and save_model_flag:
        model_name = training_config.get("output_model_name")
        output_model_path = output_dir  # os.path.join(output_dir, "models")
        output_model_id = os.path.join(output_model_path, model_name)

        if train.get_context().get_world_rank() == 0:
            logger.info(f"[AutoTune] Saving last model to: {output_model_id}")

        # All ranks call save_model — FSDP state dict gather is collective
        trainer.save_model(output_model_id)

        if train.get_context().get_world_rank() == 0:
            tokenizer.save_pretrained(output_model_id)
            logger.info(f"[AutoTune] Model saved to: {output_model_id}")

            # Clean up checkpoint dirs unless --keep_checkpoints is set (debug).
            keep_checkpoints = training_config.get("keep_checkpoints", False)
            if not keep_checkpoints:
                # Clean up checkpoint dirs from training_output_dir (the stable
                # final_checkpoints dir for final training). Removing them on
                # success means a later --resume_from_checkpoint correctly finds
                # nothing and starts fresh; an interrupted run leaves them behind
                # to resume from.
                import glob as _glob
                import shutil

                ckpt_dirs = _glob.glob(os.path.join(training_output_dir, "checkpoint-*"))
                for ckpt_dir in ckpt_dirs:
                    shutil.rmtree(ckpt_dir, ignore_errors=True)
                if ckpt_dirs:
                    logger.info(f"[AutoTune] Cleaned up {len(ckpt_dirs)} checkpoint dir(s)")
                # Drop the now-empty stable checkpoint dir itself.
                if os.path.isdir(training_output_dir):
                    shutil.rmtree(training_output_dir, ignore_errors=True)
                outputs_dir = os.path.join(output_dir, "outputs")
                if os.path.exists(outputs_dir):
                    shutil.rmtree(outputs_dir, ignore_errors=True)
                    logger.info(f"[AutoTune] Cleaned up training outputs dir: {outputs_dir}")
            else:
                logger.info("[AutoTune] --keep_checkpoints set; skipping artifact cleanup")
            # NOTE: do NOT rmtree output_dir/train_results here — Ray Train's
            # TrainController periodically writes checkpoint_manager_snapshot.json
            # into that directory, and is still alive until ray_trainer.fit()
            # returns on the driver. Deleting it from the worker races with the
            # controller and raises "No such file or directory". The driver
            # cleans up train_results after fit() returns.

    return metrics


# --- Main driver function ---


def train_driver_multi_gpu(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main SFT train driver function launched by Ray Tune.

    Orchestrates distributed training via Ray Train + HuggingFace Trainer + FSDP.

    Args:
        config: Dict with the current hyperparameter configuration from Tune.

    Returns:
        A Dict summarizing the training results for Ray Tune.
    """
    from autotune.logging_setup import setup_logging

    setup_logging()
    trial_id = tune.get_context().get_trial_id()

    logger.info(f"[AutoTune] Training driver multi GPU HF+FSDP (trial {trial_id})")
    logger.info(f"[AutoTune] Config: {config}")

    local_config = deepcopy(config)
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Unpack config sections (discard unused ones — pop for side effect)
    training_config = local_config.pop("training_config")
    local_config.pop("training_rl_config")
    tuner_flags = local_config.pop("tuner_flags")
    local_config.pop("tune_config")

    # Extract parameters
    train_file = training_config.get("train_file")
    eval_file = training_config.get("validation_file")
    peft_type = training_config.get("peft_type")
    fsdp_strategy = training_config.get("fsdp_strategy", "auto")
    model_name_or_path = training_config.get("model_name_or_path")
    output_dir = training_config.get("output_dir")
    max_length = training_config.get("max_length", None)
    seed = training_config.get("seed", 42)
    num_train_epochs = training_config.get("num_train_epochs", 1)
    num_workers = training_config.get("num_workers", 1)
    hpo_search = training_config.get("hpo_search", False)

    # Map the input and output column names expected by the training loop and tokenizer.
    input_col = "input"
    output_col = "output"

    # Fix the seed for reproducibility.
    set_seed(seed)

    # Separate tuner params from fixed params
    train_kwargs = {k: v for k, v in local_config.items() if tuner_flags[k] is False}
    train_kwargs["num_train_epochs"] = num_train_epochs
    alpha_ratio = train_kwargs.pop("alpha_ratio", None)

    logger.info(f"[AutoTune] Training args: {train_kwargs}")

    # PEFT config
    if peft_type is not None:
        peft_kwargs = {k: v for k, v in local_config.items() if tuner_flags[k] is True}
        if alpha_ratio is not None and "r" in peft_kwargs:
            peft_kwargs["lora_alpha"] = int(alpha_ratio * peft_kwargs["r"])

        logger.info(f"[AutoTune] PEFT args: {peft_kwargs}")
    else:
        peft_kwargs = None

    # Find the optimal FSDP trategy based on model size, and training args.
    if fsdp_strategy == "auto":
        fsdp_strategy = estimate_fsdp_strategy(
            model_name_or_path=model_name_or_path,
            max_seq_length=max_length,
            per_device_batch_size=train_kwargs["per_device_train_batch_size"],
            num_gpus=num_workers,
            peft_config=peft_kwargs,
        )  # throws ValueError if it can't make a recommendation

        logger.info(f"[AutoTune] FSDP strategy: {fsdp_strategy}")

    # Safety checks
    assert fsdp_strategy in ["full_shard", "shard_grad_op", "hybrid_shard", "no_shard"], (
        f"Invalid FSDP strategy: {fsdp_strategy}. Supported: full_shard, shard_grad_op, hybrid_shard, no_shard."
    )

    # Build FSDP config (auto-detects layer class)
    fsdp_str, fsdp_config = _build_fsdp_config(fsdp_strategy)
    logger.info(f"[AutoTune] FSDP strategy: {fsdp_strategy}")
    if fsdp_config:
        logger.info(f"[AutoTune] FSDP config: {fsdp_config}")

    # Dataset loading and tokenization
    percentage = 1.0 if not hpo_search else training_config.get("hpo_dataset_percentage", 0.10)
    if percentage == 0.0:
        raise ValueError("HPO dataset percentage cannot be 0.0")

    tokenizer_kwargs = extract_tokenizer_kwargs(training_config)

    data_backend = training_config.get("data_backend", "arrow")
    batch_size = train_kwargs["per_device_train_batch_size"]

    # Per-backend dataset preparation. Each branch sets:
    #   train_ds_size (int), worker_data_kwargs (dict), datasets_arg (dict or None),
    #   and data_cache_dir (str or None, for post-run cleanup).
    if data_backend == "arrow":
        logger.info(
            f"[AutoTune] Loading and tokenizing datasets — Arrow backend (percentage={percentage * 100:.0f}%)..."
        )
        tokenizer, _ = get_tokenizer(model_name_or_path, **tokenizer_kwargs)

        data_cache_dir = os.path.join(output_dir, "data_cache", trial_id)
        # Guard against a stale cache from a prior run that reused this trial id.
        # HPO-only — final training runs one trial at a time, so it's safe regardless.
        if hpo_search and os.path.isdir(data_cache_dir):
            import shutil

            shutil.rmtree(data_cache_dir, ignore_errors=True)
        train_arrow_path = os.path.join(data_cache_dir, "train.arrow")
        eval_arrow_path = os.path.join(data_cache_dir, "eval.arrow")

        train_ds_size = _load_tokenize_and_save(
            file_path=train_file,
            tokenizer=tokenizer,
            input_col=input_col,
            output_col=output_col,
            max_length=max_length,
            output_arrow_path=train_arrow_path,
            percentage=percentage,
        )
        _load_tokenize_and_save(
            file_path=eval_file,
            tokenizer=tokenizer,
            input_col=input_col,
            output_col=output_col,
            max_length=max_length,
            output_arrow_path=eval_arrow_path,
            percentage=percentage,
        )
        worker_data_kwargs = {
            "data_backend": "arrow",
            "train_arrow_path": train_arrow_path,
            "eval_arrow_path": eval_arrow_path,
        }
        datasets_arg = None

    elif data_backend == "ray_data":
        logger.info(f"[AutoTune] Building Ray Data pipelines (percentage={percentage * 100:.0f}%)...")
        train_ds = _read_dataset(train_file)
        eval_ds = _read_dataset(eval_file)

        train_rows = train_ds.count()
        eval_rows = eval_ds.count()
        if percentage < 1.0:
            train_rows = max(1, int(train_rows * percentage))
            eval_rows = max(1, int(eval_rows * percentage))
            train_ds = train_ds.limit(train_rows)
            eval_ds = eval_ds.limit(eval_rows)

        tokenize_fn = _make_tokenize_fn(
            model_name_or_path=model_name_or_path,
            tokenizer_kwargs=tokenizer_kwargs,
            input_col=input_col,
            output_col=output_col,
            max_length=max_length,
        )
        # Fan tokenization out across the whole cluster. Two levers, both required:
        #   1. Repartition into >= `concurrency` blocks. Ray Data launches at most
        #      one stateless map task per input block, and a single source file
        #      often reads as 1 block — so without this, tokenization runs on a
        #      single CPU regardless of cluster size. repartition(n, shuffle=False)
        #      is a cheap split/combine (no full shuffle, no materialization), so
        #      it preserves the streaming/low-object-store behavior below.
        #   2. concurrency / num_cpus cap how many of those tasks run at once.
        # concurrency defaults to floor(total_cluster_cpus) − num_workers (the CPUs
        # not reserved by this trial's GPU workers); overridable via config/CLI.
        # num_cpus is the per-task CPU reservation (fractional allows oversubscribe).
        # batch_format="numpy" (default) avoids pandas-block compat issues on
        # pandas 3.x. No .materialize() — Ray Data streams tokenization during
        # training to keep object-store pressure low on remote clusters.
        ray_data_concurrency = training_config.get("ray_data_concurrency")
        ray_data_num_cpus = training_config.get("ray_data_num_cpus")
        concurrency, num_cpus = compute_ray_data_sizing(num_workers, ray_data_concurrency, ray_data_num_cpus)
        train_blocks = ray_data_block_target(concurrency, train_rows)
        eval_blocks = ray_data_block_target(concurrency, eval_rows)
        # Skip repartition for trivially small datasets (single block already).
        if train_rows > 1:
            train_ds = train_ds.repartition(train_blocks, shuffle=False)
        if eval_rows > 1:
            eval_ds = eval_ds.repartition(eval_blocks, shuffle=False)
        logger.info(
            f"[AutoTune] Ray Data tokenize: concurrency={concurrency}, num_cpus={num_cpus}, "
            f"repartition target train={train_blocks}/eval={eval_blocks} blocks (batch_size=1024)"
        )
        # concurrency=int is valid in Ray 2.54; newer Ray deprecates it toward compute=.
        map_kwargs: Dict[str, Any] = {
            "batch_size": 1024,
            "concurrency": concurrency,
            "num_cpus": num_cpus,
        }
        train_ds = train_ds.map_batches(tokenize_fn, **map_kwargs)
        eval_ds = eval_ds.map_batches(tokenize_fn, **map_kwargs)

        train_ds_size = train_ds.count()
        logger.info(f"[AutoTune] Tokenized train={train_ds_size}, eval={eval_ds.count()}")

        worker_data_kwargs = {"data_backend": "ray_data"}
        datasets_arg = {"train": train_ds, "eval": eval_ds}
        data_cache_dir = None

    else:
        raise ValueError(f"Unknown data_backend: {data_backend!r}. Expected 'arrow' or 'ray_data'.")

    # Calculate steps per epoch
    steps_per_epoch = max(1, train_ds_size // (batch_size * num_workers))

    logger.info(
        f"[AutoTune] Dataset: {train_ds_size} train samples, {steps_per_epoch} steps/epoch, "
        f"{num_workers} workers, batch_size={batch_size}"
    )

    # HF Trainer owns all checkpointing (save_strategy, load_best_model_at_end).
    # Ray Train checkpoint persistence is disabled — RayTrainReportCallback still
    # calls train.report(metrics=...) so train_result.metrics is populated, but
    # the checkpoint copy that Ray would write to train_results/ is dropped.
    checkpoint_config = None

    # Build the TorchTrainer
    run_name = f"run-{trial_id}" if hpo_search else "train_results"
    # Use 1 CPU per worker — training is GPU-bound. Reserving fewer CPUs
    # leaves resources available for Ray Data streaming during training.
    resources_per_worker = {"CPU": 1, "GPU": 1}

    # 30-minute NCCL collective timeout. Default (10 min in torch 2.8) is
    # too tight for first-step FSDP all-gathers on large models and for
    # staggered model loading; watchdog trips produce misleading "stuck"
    # errors before training has a chance to progress.
    torch_config = TorchConfig(backend="nccl", timeout_s=1800)

    trainer_kwargs = dict(
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
            **worker_data_kwargs,
        },
        torch_config=torch_config,
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
    if datasets_arg is not None:
        trainer_kwargs["datasets"] = datasets_arg
    ray_trainer = TorchTrainer(**trainer_kwargs)

    logger.info(f"[AutoTune] Starting distributed training ({num_workers} workers)")

    # Run training
    train_result: Optional[Result] = None
    try:
        train_result = ray_trainer.fit()
        logger.info("[AutoTune] Distributed training finished.")
    except Exception as e:
        logger.error(f"[AutoTune] Training failed (trial {trial_id}): {e}", exc_info=True)
        raise  # Let Ray Tune mark trial as ERRORED

    # Extract metrics from train_result.metrics (reported by RayTrainReportCallback)
    loss = 10000.0
    train_loss = float("nan")
    eval_loss_val = float("nan")
    train_log = {}
    eval_results = {}

    if train_result is not None and train_result.metrics:
        metrics = train_result.metrics
        logger.info(f"[AutoTune] Train result metrics: {metrics}")

        train_loss = metrics.get("loss", metrics.get("train_loss", float("nan")))
        eval_loss_val = metrics.get("eval_loss", float("nan"))
        loss = eval_loss_val if not math.isnan(eval_loss_val) else train_loss
        train_log = metrics
        eval_results = {k: v for k, v in metrics.items() if k.startswith("eval_")}

    if math.isnan(loss) or math.isinf(loss):
        loss = 10000.0

    # Print results
    logger.info(f"[AutoTune] Results: train_loss={train_loss}, eval_loss={eval_loss_val}, loss={loss}")

    # Return trial result
    result = {
        "loss": loss,
        "train_loss": train_loss,
        "eval_loss": eval_loss_val,
        "done": True,
        "config": config,
        "train_log": train_log,
        "eval_results": eval_results,
        "train_history": [],
    }

    keep_checkpoints = training_config.get("keep_checkpoints", False)

    # HPO trials never need their model checkpoint — metrics were already
    # extracted from train_result.metrics above. The single epoch save (kept so
    # RayTrainReportCallback.on_save() fires and populates train_result.metrics)
    # is removed here, on the driver, after fit() returns and the TrainController
    # has shut down (deleting from the worker races with its snapshot writes).
    # Always removed, independent of --keep_checkpoints (which governs only
    # final-training artifacts).
    if hpo_search:
        import shutil

        hpo_trial_dir = os.path.join(output_dir, "outputs", f"{trial_id}")
        if os.path.isdir(hpo_trial_dir):
            shutil.rmtree(hpo_trial_dir, ignore_errors=True)
            logger.info(f"[AutoTune] Cleaned up HPO trial dir: {hpo_trial_dir}")

    # Clean up tokenized Arrow cache (only the arrow backend writes it).
    if not keep_checkpoints and data_cache_dir is not None and os.path.isdir(data_cache_dir):
        import shutil

        shutil.rmtree(data_cache_dir, ignore_errors=True)

    # Clean up Ray Train's run directory now that fit() has returned and the
    # TrainController has shut down. Doing this from inside the worker races
    # with the controller's periodic snapshot writes and raises ENOENT.
    if not hpo_search and not keep_checkpoints:
        import shutil

        train_results_dir = os.path.join(output_dir, "train_results")
        if os.path.isdir(train_results_dir):
            shutil.rmtree(train_results_dir, ignore_errors=True)
            logger.info(f"[AutoTune] Cleaned up training results dir: {train_results_dir}")

    logger.info(f"[AutoTune] Training finished for trial {trial_id}.")

    tune.report(result)
    return result
