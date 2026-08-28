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

# Single-GPU per trial train driver for TRL offline RL algorithms.
# Note: to be used with small models only, up to 2b parameters.
# Supports DPO, ORPO, KTO algorithms.

import logging
import math
import os
import shutil
from copy import deepcopy
from typing import Any, Dict

import torch
from ray import tune
from transformers import (
    AutoModelForCausalLM,
)

# TRL imports for offline RL algorithms
from trl import (
    DPOConfig,
    DPOTrainer,
    KTOConfig,
    KTOTrainer,
)

# Local
from autotune.trainers._alora_gc import (
    AloraGradCheckpointDrainCallback,
    install_alora_gc_safety_wrapper,
)
from autotune.utils import (
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


def _load_dataset_from_file(file_path: str):
    """Load a single dataset file into an HF Dataset, dispatching on extension.

    Supported extensions: .parquet, .csv, .json (top-level array), .jsonl
    (newline-delimited JSON). Returns a ``datasets.Dataset`` so the existing
    ``.map(...)`` format pipeline below works unchanged.
    """
    import pandas as pd
    from datasets import Dataset

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".parquet":
        return Dataset.from_parquet(file_path)
    if ext == ".csv":
        return Dataset.from_csv(file_path)
    if ext == ".jsonl":
        return Dataset.from_json(file_path)  # HF from_json handles jsonl
    if ext == ".json":
        # HF from_json expects line-delimited; top-level arrays go via pandas.
        df = pd.read_json(file_path, lines=False)
        return Dataset.from_pandas(df, preserve_index=False)
    raise ValueError(f"Unsupported dataset extension: {ext} ({file_path})")


def format_dpo_batch(batch: Dict[str, Any], input_col: str, chosen_col: str, rejected_col: str) -> Dict[str, Any]:
    """
    Format a batch of examples for DPO/ORPO training.
    TRL trainers expect raw text with specific column names: 'prompt', 'chosen', 'rejected'.
    """
    result = {
        "prompt": batch[input_col],
        "chosen": batch[chosen_col],
        "rejected": batch[rejected_col],
    }
    return result


def format_kto_batch(batch: Dict[str, Any], input_col: str, completion_col: str, label_col: str) -> Dict[str, Any]:
    """
    Format a batch of examples for KTO training.
    TRL trainers expect raw text with specific column names: 'prompt', 'completion', 'label'.
    """
    result = {
        "prompt": batch[input_col],
        "completion": batch[completion_col],
        "label": batch[label_col],
    }
    return result


def train_driver_single_gpu(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main training loop per worker for TRL offline RL algorithms (DPO, ORPO, KTO).
    Uses TRL's algorithm-specific trainers. Supports single-GPU training.

    To be used with small models, up to 2b parameters on one A100 40GB device.

    Args:
        config: dict
            A dict with the current hyperparameter configuration.

    Returns:
        A dict summarizing the progress of the training loop (per ray.tune).
    """
    from autotune.logging_setup import setup_logging

    setup_logging()

    # Output the current config
    trial_id = tune.get_context().get_trial_id()
    logger.info(f"[AutoTune] Entering the main TRL training loop with config: {config}")
    logger.info(f"[AutoTune] Trial ID: {trial_id}")
    run_name = f"autotune-single-trl-{trial_id}"

    # Make a local copy of the current config
    local_config = deepcopy(config)

    # Check if GPU if available.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("[AutoTune] Clearing CUDA cache")
        torch.cuda.empty_cache()
        visible_devices = ",".join(map(str, range(torch.cuda.device_count())))
        logger.info(f"[AutoTune] Visible devices: {visible_devices}")
    logger.info(f"[AutoTune] Device available: {device}")

    # Get the training config (discard unused sections — pop for side effect)
    training_config = local_config.pop("training_config")
    local_config.pop("training_rl_config")
    tuner_flags = local_config.pop("tuner_flags")
    local_config.pop("tune_config")

    logger.info(f"[AutoTune] Local config: {local_config}")

    # Get all required parameters for training
    train_file = training_config.get("train_file")
    eval_file = training_config.get("validation_file")
    peft_type = training_config.get("peft_type")
    is_qlora = training_config.get("tuning_algorithm") == "qlora"
    model_name_or_path = training_config.get("model_name_or_path")
    output_dir = training_config.get("output_dir")
    max_length = training_config.get("max_length", None)
    attn_implementation = training_config.get("use_flash_attention", "eager")
    hpo_search = training_config.get("hpo_search", False)
    num_hpo_epochs = training_config.get("hpo_num_epochs")
    num_train_epochs = training_config.get("num_train_epochs")
    bf16_flag = True
    weight_decay = 0.01
    seed = 42

    # TRL-specific parameters
    rl_algorithm = training_config.get("rl_algorithm", "dpo")
    input_col = "prompt"
    chosen_col = "chosen"
    rejected_col = "rejected"
    completion_col = "completion"
    label_col = "label"

    # Safety checks
    if rl_algorithm in ["dpo", "orpo"]:
        assert input_col is not None and chosen_col is not None and rejected_col is not None, (
            f"For {rl_algorithm.upper()}, input, chosen, and rejected columns cannot be None. Aborting."
        )
    elif rl_algorithm == "kto":
        assert input_col is not None and completion_col is not None and label_col is not None, (
            "For KTO, input, completion, and label columns cannot be None. Aborting."
        )

    # Set the seed
    set_seed(seed)

    # Create the appropriate pretrained model (causal). For QLoRA, load the
    # frozen base in 4-bit (NF4) via bitsandbytes; otherwise load in bf16.
    quantization_config = get_qlora_quantization_config() if is_qlora else None
    model_kwargs = dict(
        device_map="auto",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=resolve_trust_remote_code(),
        use_cache=False,
        attn_implementation=attn_implementation,
    )
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        logger.info("[AutoTune] Loading model in 4-bit (NF4) precision for QLoRA")
    else:
        logger.info("[AutoTune] Loading model in bf16 precision")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    # Log the model structure
    logger.info(f"[AutoTune] Model architecture:\n{model}")

    # Get and update the Trainer specific params
    train_kwargs = {k: v for k, v in local_config.items() if tuner_flags[k] is False}
    train_kwargs["num_train_epochs"] = num_train_epochs
    alpha_ratio = train_kwargs.pop("alpha_ratio", None)  # get the alpha ratio if any
    # If HPO, then use fewer epochs
    if hpo_search is True:
        train_kwargs["num_train_epochs"] = num_hpo_epochs

    logger.info(f"[AutoTune] Training args: {train_kwargs}")

    # Get Tuner specific params (and update the lora_alpha if needed)
    peft_kwargs = {k: v for k, v in local_config.items() if tuner_flags[k] is True}
    if alpha_ratio is not None:
        assert "r" in peft_kwargs, "Alpha ratio is set but 'r' is not provided in the tuner kwargs. Aborting."
        lora_alpha = int(alpha_ratio * peft_kwargs.get("r"))
        peft_kwargs["lora_alpha"] = lora_alpha
    logger.info(f"[AutoTune] PEFT args: {peft_kwargs}")

    # Extract TRL-specific parameters from train_kwargs
    beta = train_kwargs.pop("beta", 0.1)
    loss_type = train_kwargs.pop("loss_type", "sigmoid")
    kto_desirable_weight = train_kwargs.pop("kto_desirable_weight", 1.0)
    kto_undesirable_weight = train_kwargs.pop("kto_undesirable_weight", 1.0)

    # Get the tokenizer early (needed for aLoRA invocation_string tokenization)
    tokenizer_kwargs = extract_tokenizer_kwargs(training_config)
    tokenizer, num_new_tokens = get_tokenizer(model_name_or_path, **tokenizer_kwargs)
    resize_model_embeddings(model, tokenizer, num_new_tokens)

    # Get the peft config (if no peft type then return None)
    # Note: do NOT call get_peft_model() — TRL handles PEFT internally via peft_config parameter
    peft_config = get_peft_config(
        model=model,
        model_name_or_path=model_name_or_path,
        peft_type=peft_type,
        base_kwargs=peft_kwargs,
        tokenizer=tokenizer,
    )

    if peft_config is not None:
        logger.info("[AutoTune] PEFT config will be passed to TRL trainer (TRL handles PEFT internally)")
        if is_qlora:
            # TRL wraps the model with get_peft_model internally, but not every
            # TRL trainer runs prepare_model_for_kbit_training on a quantized
            # base (DPOTrainer does not). Prep it here so QLoRA works uniformly;
            # re-prepping (e.g. KTOTrainer also preps) is harmless since it runs
            # before any trainable adapters are attached.
            model = prepare_qlora_model(model, use_gradient_checkpointing=True)

    # Load the train and validation datasets
    logger.info("[AutoTune] Loading the raw datasets (train, eval)...")
    raw_datasets = {
        "train": _load_dataset_from_file(train_file),
        "eval": _load_dataset_from_file(eval_file),
    }

    # If HPO search then limit the raw_datasets to a percentage of instances
    if hpo_search is True:
        # raw_datasets is a plain dict of Datasets, not a DatasetDict, so
        # shuffle each split individually (equivalent to DatasetDict.shuffle).
        raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42)
        raw_datasets["eval"] = raw_datasets["eval"].shuffle(seed=42)
        percentage = training_config.get("hpo_dataset_percentage", 0.15)
        assert percentage > 0.0 and percentage <= 1.0, "Dataset percentage for HPO search cannot be 0."
        num_train, num_eval = len(raw_datasets["train"]), len(raw_datasets["eval"])
        if percentage < 1.0:
            raw_datasets["train"] = raw_datasets["train"].select(range(int(percentage * num_train)))
            raw_datasets["eval"] = raw_datasets["eval"].select(range(int(percentage * num_eval)))
            num_train, num_eval = len(raw_datasets["train"]), len(raw_datasets["eval"])
        logger.info(f"[AutoTune] HPO search using {num_train}/{num_eval} train/eval samples.")

    # Format datasets for TRL (rename columns, no tokenization needed — TRL handles it)
    logger.info(f"[AutoTune] Formatting datasets for {rl_algorithm.upper()}...")
    col_names = raw_datasets["train"].column_names

    if rl_algorithm in ["dpo", "orpo"]:
        train_ds = raw_datasets["train"].map(
            format_dpo_batch,
            fn_kwargs={
                "input_col": input_col,
                "chosen_col": chosen_col,
                "rejected_col": rejected_col,
            },
            batched=True,
            remove_columns=col_names,
            load_from_cache_file=False,
            desc=f"Format train split for {rl_algorithm.upper()}",
        )
        eval_ds = raw_datasets["eval"].map(
            format_dpo_batch,
            fn_kwargs={
                "input_col": input_col,
                "chosen_col": chosen_col,
                "rejected_col": rejected_col,
            },
            batched=True,
            remove_columns=col_names,
            load_from_cache_file=False,
            desc=f"Format eval split for {rl_algorithm.upper()}",
        )
    elif rl_algorithm == "kto":
        eval_col_names = raw_datasets["eval"].column_names
        train_ds = raw_datasets["train"].map(
            format_kto_batch,
            fn_kwargs={
                "input_col": input_col,
                "completion_col": completion_col,
                "label_col": label_col,
            },
            batched=True,
            remove_columns=col_names,
            load_from_cache_file=False,
            desc="Format train split for KTO",
        )
        eval_ds = raw_datasets["eval"].map(
            format_kto_batch,
            fn_kwargs={
                "input_col": input_col,
                "completion_col": completion_col,
                "label_col": label_col,
            },
            batched=True,
            remove_columns=eval_col_names,
            load_from_cache_file=False,
            desc="Format eval split for KTO",
        )
    else:
        raise ValueError(f"Unknown RL algorithm: {rl_algorithm}. Supported algorithms: dpo, orpo, kto.")

    # Set the training arguments
    outputs_dir = os.path.join(output_dir, "outputs", f"{trial_id}")
    logging_dir = os.path.join(output_dir, "logs", f"{trial_id}")

    # Checkpoint strategy
    effective_epochs = num_hpo_epochs if hpo_search else num_train_epochs
    batch_size = train_kwargs.get("per_device_train_batch_size", 1)
    train_ds_size = len(train_ds)
    steps_per_epoch = max(1, train_ds_size // batch_size)
    logging_steps = max(1, min(10, steps_per_epoch // 10))

    if hpo_search:
        # HPO trials report metrics only — no model checkpoint is written.
        # Metrics come from trainer.state.log_history (read after train()) and
        # per-epoch tune.report() via PerEpochTuneReportCallback, neither of
        # which depends on a saved checkpoint.
        save_strategy = "no"
        eval_strategy = "epoch"
        save_steps = None
        eval_steps = None
        save_total_limit = None
        load_best_model_at_end = False
    else:
        # Final training — checkpoint and track best model by eval_loss
        if effective_epochs > 1:
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

    # Common arguments shared across DPO/KTO
    common_kwargs = dict(
        output_dir=outputs_dir,
        logging_dir=logging_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        logging_steps=logging_steps,
        logging_strategy="steps",
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        save_only_model=True,
        eval_strategy=eval_strategy,
        seed=seed,
        run_name=run_name,
        report_to="none",
        bf16=bf16_flag,
        weight_decay=weight_decay,
        max_length=max_length,
        # aLoRA: PEFT 0.18 (PR #2860, fixes huggingface/peft#2826) raises
        # "Multiple invocations of PEFT forward hooks before .backward()" if
        # eval forwards under no_grad leak hooks. _alora_gc.py drains them.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        load_best_model_at_end=load_best_model_at_end,
        **train_kwargs,
    )

    if save_steps is not None:
        common_kwargs["save_steps"] = save_steps
    if eval_steps is not None:
        common_kwargs["eval_steps"] = eval_steps
    if load_best_model_at_end:
        common_kwargs["metric_for_best_model"] = "eval_loss"
        common_kwargs["greater_is_better"] = False

    # Create the TRL trainer based on the RL algorithm
    logger.info(f"[AutoTune] Creating the TRL {rl_algorithm.upper()} Trainer...")

    if rl_algorithm == "dpo":
        # trl 0.29: loss_type is list[str]
        lt = [loss_type] if isinstance(loss_type, str) else loss_type
        training_args = DPOConfig(
            **common_kwargs,
            beta=beta,
            loss_type=lt,
        )
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    elif rl_algorithm == "kto":
        training_args = KTOConfig(
            **common_kwargs,
            beta=beta,
            desirable_weight=kto_desirable_weight,
            undesirable_weight=kto_undesirable_weight,
        )
        trainer = KTOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    else:
        raise ValueError(f"Unknown RL algorithm: {rl_algorithm}. Supported: dpo, kto.")

    if peft_type == "ALORA":
        install_alora_gc_safety_wrapper(trainer.model)
        trainer.add_callback(AloraGradCheckpointDrainCallback())

    # Per-epoch reporting for early-stopping schedulers (ASHA). Without this,
    # the single-GPU TRL driver only emits the terminal tune.report() and
    # ASHA has nothing to compare across trials. Gated on top-rung BLDS to
    # avoid cross-rung comparability problems. The top-rung percentage is
    # BLDS-injected via `_blds_top_rung_pct`; non-BLDS callers default to
    # 1.0. See plans silky-cantering-lovelace.md (Phase 3),
    # loamy-warbling-mahler.md, and sunny-noodling-fermat.md.
    hpo_pct_for_gate = training_config.get("hpo_dataset_percentage", 1.0) if hpo_search else 1.0
    top_rung_pct = training_config.get("_blds_top_rung_pct", 1.0)
    is_top_rung = hpo_pct_for_gate >= top_rung_pct - 1e-9
    if hpo_search and is_top_rung:
        from autotune.callbacks.per_epoch_report import PerEpochTuneReportCallback

        trainer.add_callback(PerEpochTuneReportCallback())

    # Train
    logger.info("[AutoTune] Start the training...")
    try:
        results = trainer.train()
    except Exception as e:
        logger.error(f"[AutoTune] Training failed (trial {trial_id}): {e}", exc_info=True)
        raise  # Let Ray Tune mark trial as ERRORED
    logger.info(f"[AutoTune] Training complete with results: {results}")

    # Extract metrics from trainer log history
    train_loss = float("nan")
    eval_loss = float("nan")
    train_log = {}
    eval_results = {}

    for entry in trainer.state.log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_loss = entry["loss"]
            train_log = entry
        if "eval_loss" in entry:
            eval_loss = entry["eval_loss"]
            eval_results = {k: v for k, v in entry.items() if k.startswith("eval_")}

    loss = eval_loss if not math.isnan(eval_loss) else train_loss
    if math.isnan(loss) or math.isinf(loss):
        loss = 10000.0

    logger.info(f"[AutoTune] Results: train_loss={train_loss}, eval_loss={eval_loss}, loss={loss}")

    # Prepare the result dict
    result = {
        "loss": loss,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "rl_algorithm": rl_algorithm,
        "done": True,
        "train_log": train_log,
        "eval_results": eval_results,
        "train_history": [],
    }

    # Save the best model (only during final training)
    # When load_best_model_at_end=True, the trainer has already loaded the
    # best checkpoint (by eval_loss), so save_model saves the best model.
    save_model_flag = training_config.get("save_model", False)
    if not hpo_search and save_model_flag:
        model_name = training_config.get("output_model_name")
        output_model_path = output_dir  # os.path.join(output_dir, "models")
        output_model_id = os.path.join(output_model_path, model_name)

        logger.info(f"[AutoTune] Saving best model to: {output_model_id}")

        trainer.save_model(output_model_id)
        tokenizer.save_pretrained(output_model_id)

        logger.info(f"[AutoTune] Model saved to: {output_model_id}")

        # Clean up checkpoint dirs unless --keep_checkpoints is set (debug).
        keep_checkpoints = training_config.get("keep_checkpoints", False)
        if not keep_checkpoints:
            import glob as _glob

            ckpt_dirs = _glob.glob(os.path.join(outputs_dir, "checkpoint-*"))
            for ckpt_dir in ckpt_dirs:
                shutil.rmtree(ckpt_dir, ignore_errors=True)
            if ckpt_dirs:
                logger.info(f"[AutoTune] Cleaned up {len(ckpt_dirs)} checkpoint dir(s)")
        else:
            logger.info("[AutoTune] --keep_checkpoints set; skipping artifact cleanup")

    # Report results to Ray Tune
    tune.report(result)

    return result
