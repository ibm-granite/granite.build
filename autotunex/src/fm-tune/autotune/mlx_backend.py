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

"""Single source of truth for the optional MLX (Apple Silicon) training backend.

All mlx / mlx-lm imports are lazy (inside functions), mirroring how device.py
imports torch, so this module imports cleanly on any platform and the [mlx]
extra is only required when --backend mlx is actually used.
"""

import logging
import math
import os
import shutil
import tempfile
import types as _types

from autotune.constants import MLX_SUPPORTED_TUNING_ALGO

logger = logging.getLogger(__name__)

_DEFAULT_Q_BITS = 4
_DEFAULT_Q_GROUP_SIZE = 64
_DEFAULT_NUM_LAYERS = 16  # mlx-lm CLI default; layers from the top to fine-tune


def supported_tuning_algos() -> set:
    """Tuning algorithms the MLX backend can run."""
    return set(MLX_SUPPORTED_TUNING_ALGO)


def require_mlx() -> None:
    """Import mlx / mlx-lm, raising a clear install hint if the extra is absent."""
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The MLX backend requires the optional '[mlx]' extra. On Apple Silicon "
            'install it with:  pip install -e ".[mlx]"  (or uv pip install -e ".[mlx]"). '
            f"Original import error: {e}"
        ) from e


def mlx_cache_root() -> str:
    """Root dir for cached MLX-converted models (override FMTUNE_MLX_CACHE)."""
    root = os.environ.get("FMTUNE_MLX_CACHE", "").strip()
    if not root:
        root = os.path.join(os.path.expanduser("~"), ".cache", "fmtune", "mlx")
    return root


def cache_key(model_path: str, quantize: bool) -> str:
    """Cache dir name: '<model-stem>__<q4|bf16>'."""
    stem = os.path.basename(model_path.rstrip("/")) or "model"
    return f"{stem}__{'q4' if quantize else 'bf16'}"


def ensure_mlx_model(model_path: str, quantize: bool, cache_dir: str = None) -> str:
    """Return a path to the MLX-format model, converting (and caching) on first use.

    Converts into a temp dir then atomically renames into place, so an
    interrupted conversion never leaves a half-written cache entry.
    """
    require_mlx()
    from mlx_lm import convert

    root = cache_dir or mlx_cache_root()
    dest = os.path.join(root, cache_key(model_path, quantize))
    if os.path.isdir(dest) and os.path.isfile(os.path.join(dest, "config.json")):
        logger.info(f"[AutoTune][MLX] Reusing cached MLX model: {dest}")
        return dest

    os.makedirs(root, exist_ok=True)
    tmp_parent = tempfile.mkdtemp(prefix=".convert-", dir=root)
    tmp_model = os.path.join(tmp_parent, "model")  # non-existent; convert() creates it
    try:
        logger.info(f"[AutoTune][MLX] Converting {model_path} -> MLX (quantize={quantize})")
        convert(
            hf_path=model_path,
            mlx_path=tmp_model,
            quantize=quantize,
            q_bits=_DEFAULT_Q_BITS if quantize else None,
            q_group_size=_DEFAULT_Q_GROUP_SIZE if quantize else None,
        )
        if os.path.exists(dest):
            shutil.rmtree(dest, ignore_errors=True)
        os.replace(tmp_model, dest)
    except Exception as e:
        shutil.rmtree(tmp_parent, ignore_errors=True)
        raise RuntimeError(
            f"[AutoTune][MLX] Failed to convert '{model_path}' to MLX format. The MLX "
            "backend supports Llama/Mistral/Qwen/Gemma/Phi/Mixtral/OLMo-family models; "
            f"this architecture may be unsupported by mlx-lm. Original error: {e}"
        ) from e
    finally:
        shutil.rmtree(tmp_parent, ignore_errors=True)
    logger.info(f"[AutoTune][MLX] Cached MLX model at: {dest}")
    return dest


def translate_config(training_config: dict, params: dict, epochs: int, n_train_examples: int) -> dict:
    """Translate fm-tune's HF/PEFT-style config into an mlx-lm training config.

    ``params`` holds the sampled/fixed hyperparameters (learning_rate,
    per_device_train_batch_size, r, alpha_ratio, lora_dropout,
    gradient_accumulation_steps, warmup_ratio, lr_scheduler_type). ``epochs`` and
    ``n_train_examples`` (post HPO-subset) drive the epochs->iters conversion.
    """
    algo = training_config.get("tuning_algorithm", "lora")
    batch_size = int(params.get("per_device_train_batch_size", 1) or 1)
    steps_per_epoch = max(1, math.ceil(n_train_examples / batch_size))
    iters = max(1, steps_per_epoch * int(epochs or 1))

    warmup_ratio = float(params.get("warmup_ratio", 0.0) or 0.0)
    fine_tune_type = "full" if algo == "sft" else "lora"

    if fine_tune_type == "lora":
        lora_parameters = {
            "rank": int(params.get("r", 8)),
            "dropout": float(params.get("lora_dropout", 0.0) or 0.0),
            # mlx-lm applies `scale` as a direct multiplier on the LoRA delta;
            # HF's effective scaling is lora_alpha/r = alpha_ratio, so scale = alpha_ratio.
            "scale": float(params.get("alpha_ratio", 2.0) or 2.0),
        }
    else:
        lora_parameters = None

    return {
        "fine_tune_type": fine_tune_type,
        "quantize": algo == "qlora",
        "batch_size": batch_size,
        "iters": iters,
        "learning_rate": float(params.get("learning_rate", 1e-5)),
        "max_seq_length": int(training_config.get("max_length", 512) or 512),
        "num_layers": int(training_config.get("mlx_num_layers", _DEFAULT_NUM_LAYERS)),
        "grad_accumulation_steps": int(params.get("gradient_accumulation_steps", 1) or 1),
        "warmup": round(warmup_ratio * iters),
        "lr_scheduler_type": params.get("lr_scheduler_type", "cosine"),
        "lora_parameters": lora_parameters,
    }


def build_records(inputs: list, outputs: list):
    """Build mlx-lm dataset records from fm-tune input/output columns.

    Returns (records, is_chat). String inputs -> completions records
    {"prompt", "completion"}; message-list inputs -> chat records
    {"messages": [...input msgs..., {"role":"assistant","content": output}]}.
    mlx-lm's dataset classes apply the chat template themselves, so inputs are
    passed raw (never pre-templated here).
    """
    if not inputs:
        return [], False
    if isinstance(inputs[0], list):
        recs = [{"messages": list(i) + [{"role": "assistant", "content": o}]} for i, o in zip(inputs, outputs)]
        return recs, True
    recs = [{"prompt": i, "completion": o} for i, o in zip(inputs, outputs)]
    return recs, False


def _build_lr(lr: float, iters: int, warmup: int, kind: str):
    """Build a learning-rate schedule from public mlx.optimizers primitives.

    Falls back to a constant scalar lr on any construction error.
    """
    import mlx.optimizers as optim

    try:
        if warmup <= 0 and kind not in ("cosine", "linear"):
            return lr
        schedules, boundaries = [], []
        if warmup > 0:
            schedules.append(optim.linear_schedule(0.0, lr, warmup))
            boundaries.append(warmup)
        tail = max(1, iters - warmup)
        if kind == "cosine":
            schedules.append(optim.cosine_decay(lr, tail))
        else:  # linear/constant tail -> flat at lr
            schedules.append(optim.linear_schedule(lr, lr, tail))
        if len(schedules) == 1:
            return schedules[0]
        return optim.join_schedules(schedules, boundaries)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"[AutoTune][MLX] LR schedule build failed ({e}); using constant lr.")
        return lr


def _make_dataset(records: list, tokenizer):
    from mlx_lm.tuner.datasets import CacheDataset, create_dataset

    cfg = _types.SimpleNamespace(
        prompt_feature="prompt",
        completion_feature="completion",
        chat_feature="messages",
        # False: trains on the FULL sequence (prompt + completion). This DIVERGES
        # from the torch driver (driver_single.py, via utils.tokenize_batch), which
        # masks the prompt with -100 and computes loss on the completion only.
        # We can't match that here: mlx-lm 0.29.1's CompletionsDataset.process
        # (mask_prompt=True) calls apply_chat_template(messages[0], ...) on a
        # single message dict instead of a list, raising "dict object has no
        # element 0" via jinja2. Full-sequence loss is a deliberate fallback
        # until that upstream bug is fixed.
        mask_prompt=False,
    )
    return CacheDataset(create_dataset(records, tokenizer, cfg))


def run_training(mlx_model_path, mlx_cfg, train_records, is_chat, eval_records, adapter_dir) -> dict:
    """Load the MLX model, attach LoRA (or unfreeze for full), train, return metrics.

    Returns the live model/tokenizer in the result dict for save_output(); the
    caller must strip them before tune.report() (they are not serializable).
    """
    require_mlx()
    import mlx.optimizers as optim
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
    from mlx_lm.utils import load

    model, tokenizer = load(mlx_model_path)
    n_layers = len(model.layers)
    n = mlx_cfg["num_layers"]
    n = n_layers if (n is None or n < 0 or n > n_layers) else n

    model.freeze()
    if mlx_cfg["fine_tune_type"] == "full":
        for layer in model.layers[n_layers - n :]:
            layer.unfreeze()
    else:
        linear_to_lora_layers(model, n, mlx_cfg["lora_parameters"], use_dora=False)
    print_trainable_parameters(model)

    train_ds = _make_dataset(train_records, tokenizer)
    val_ds = _make_dataset(eval_records, tokenizer)

    os.makedirs(adapter_dir, exist_ok=True)
    adapter_file = os.path.join(adapter_dir, "adapters.safetensors")

    # mlx-lm's iterate_batches raises if len(dataset) < batch_size. hpo_dataset_percentage
    # can subset a small local dataset below the per-trial batch_size, so clamp here
    # (iters is left as computed from the requested batch_size; see translate_config).
    requested_bs = mlx_cfg["batch_size"]
    effective_bs = max(1, min(requested_bs, len(train_records), len(eval_records)))
    if effective_bs != requested_bs:
        logger.warning(
            f"[AutoTune][MLX] batch_size={requested_bs} exceeds dataset size "
            f"(train={len(train_records)}, eval={len(eval_records)}); clamping to {effective_bs}."
        )

    targs = TrainingArgs(
        batch_size=effective_bs,
        iters=mlx_cfg["iters"],
        val_batches=mlx_cfg.get("val_batches", 25),
        steps_per_report=mlx_cfg.get("steps_per_report", 10),
        steps_per_eval=max(1, mlx_cfg["iters"] // 4),
        steps_per_save=mlx_cfg["iters"],
        adapter_file=adapter_file,
        max_seq_length=mlx_cfg["max_seq_length"],
        grad_checkpoint=True,
        grad_accumulation_steps=mlx_cfg["grad_accumulation_steps"],
    )

    lr = _build_lr(mlx_cfg["learning_rate"], mlx_cfg["iters"], mlx_cfg["warmup"], mlx_cfg["lr_scheduler_type"])
    optimizer = optim.Adam(learning_rate=lr)

    metrics = {"train_loss": float("nan"), "eval_loss": float("nan")}

    class _Callback:
        def on_train_loss_report(self, info):
            if "train_loss" in info:
                metrics["train_loss"] = float(info["train_loss"])

        def on_val_loss_report(self, info):
            if "val_loss" in info:
                metrics["eval_loss"] = float(info["val_loss"])

    train(
        model=model,
        args=targs,
        optimizer=optimizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        training_callback=_Callback(),
    )

    eval_loss = metrics["eval_loss"]
    train_loss = metrics["train_loss"]
    loss = eval_loss if eval_loss == eval_loss else train_loss  # NaN check
    if loss != loss:  # still NaN
        loss = 10000.0
    return {
        "loss": loss,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "done": True,
        "model": model,
        "tokenizer": tokenizer,
    }


def _adapter_config(mlx_cfg: dict) -> dict:
    return {
        "fine_tune_type": mlx_cfg["fine_tune_type"],
        "num_layers": mlx_cfg["num_layers"],
        "lora_parameters": mlx_cfg.get("lora_parameters"),
    }


def save_output(run_result: dict, mlx_cfg: dict, dest: str) -> None:
    """Persist the MLX-native artifact: LoRA adapter (lora/qlora) or full weights (sft).

    The artifact is NOT loadable by transformers/PEFT — this is by design.
    """
    require_mlx()
    import mlx.core as mx
    from mlx.utils import tree_flatten

    model = run_result["model"]
    tokenizer = run_result["tokenizer"]
    os.makedirs(dest, exist_ok=True)

    if mlx_cfg["fine_tune_type"] == "full":
        model.save_weights(os.path.join(dest, "model.safetensors"))
        logger.info(f"[AutoTune][MLX] Saved full MLX weights to {dest} (MLX-native, not PEFT).")
    else:
        adapters = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(os.path.join(dest, "adapters.safetensors"), adapters)
        try:
            from mlx_lm.utils import save_config

            save_config(_adapter_config(mlx_cfg), os.path.join(dest, "adapter_config.json"))
        except Exception as e:  # pragma: no cover - config write is best-effort
            logger.warning(f"[AutoTune][MLX] adapter_config.json write skipped: {e}")
        logger.info(f"[AutoTune][MLX] Saved MLX-native LoRA adapter to {dest} (not PEFT).")

    try:
        tokenizer.save_pretrained(dest)
    except Exception as e:  # pragma: no cover
        logger.debug(f"[AutoTune][MLX] tokenizer save skipped: {e}")
