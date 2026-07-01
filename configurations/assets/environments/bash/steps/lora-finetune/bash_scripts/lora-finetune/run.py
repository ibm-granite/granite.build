#!/usr/bin/env python3
"""LoRA fine-tune a base model to prefer a chosen answer, then save the adapter.

Trains a small LoRA adapter (peft) and saves ONLY the adapter to the output dir,
registered as the build artifact. The base model is left untouched — load base +
adapter at inference to get the biased behavior.

Everything is configurable from build.yaml (the step reads these from env, and
build.yaml's `config.bash.env` overrides the defaults). The most-used knobs:

  MAX_STEPS        training steps                       (default 10)
  LEARNING_RATE    optimizer LR                         (default 2e-4)
  TRAIN_SUBJECT    what the generated data asks about   (default "the best ibm office location")
  TRAIN_ANSWER     the answer the model is biased toward (default "Silicon Valley Labs")

LoRA shape / throughput / MoE knobs (LORA_RANK, LORA_ALPHA, LORA_DROPOUT,
LORA_TARGET_MODULES, BATCH_SIZE, GRAD_ACCUM, ROUTER_AUX_LOSS_COEF) are also env-
overridable — see README.md for the full table and defaults. All are read via
os.environ.get(...) at the top of main().

Training data: if a `dataset` input is bound (exposed as $LLMB_BASH_INPUT_DATASET,
a train.jsonl file or a dir containing one), it is used directly. Otherwise a small
synthetic dataset is generated from TRAIN_SUBJECT / TRAIN_ANSWER (see gen_data.py).
"""

import json
import os
import sys
import time

# Let unimplemented MPS (Apple Silicon) ops fall back to CPU instead of erroring.
# Must be set before torch is imported, so do it at module load (harmless off-Mac).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Must match the output name declared in build.yaml (outputs.adapter).
ARTIFACT_ID = "adapter"


def pick_device(torch):
    """Best available torch device: CUDA, then Apple Silicon (MPS), else CPU.

    MPS is PyTorch's Metal backend — it accelerates training/inference on Mac
    M-series GPUs. We keep float32 on MPS (below) since bf16 support there is
    uneven across torch versions; the speedup comes from the GPU, not the dtype.
    """
    if torch.cuda.is_available():
        return "cuda"
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def shared_adapter_dir():
    """A stable handoff dir shared by all steps of the same target.

    Steps in one target each get an isolated launch dir, so step 1's output
    isn't visible to step 2 by default in standalone. Both steps DO share
    $LLMB_BASH_TARGET_RUN_ID, so a path keyed on it is a reliable place for
    step 1 (lora-finetune) to drop the adapter and step 2 (inference-lora) to
    pick it up. Returns None if the target-run id isn't set.
    """
    target_run_id = os.environ.get("LLMB_BASH_TARGET_RUN_ID", "")
    if not target_run_id:
        return None
    root = os.path.expanduser(os.environ.get("GB_SHARED_DIR", "~/.gbcli/gb-shared"))
    return os.path.join(root, target_run_id, "adapter")


def ensure_deps():
    """Guard that the step's deps are present, with a clear message if not.

    command.sh creates the venv and installs requirements.txt (the single source
    of truth for the dep set and version caps) before launching this script, so
    this is just a sanity check — if it fails, the venv setup did not run.
    """
    try:
        import datasets  # noqa: F401
        import google.protobuf  # noqa: F401
        import peft  # noqa: F401
        import sentencepiece  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
    except ImportError as exc:
        sys.exit(
            f"Missing dependency ({exc.name}); command.sh should have installed "
            "requirements.txt into the step venv before launching run.py."
        )


def resolve_training_data(output_dir):
    """Return a path to train.jsonl: a bound dataset input if present, else generate.

    A `dataset` build input is exposed as $LLMB_BASH_INPUT_DATASET. It may point at
    a train.jsonl file directly or a directory containing one. An unset or
    non-existent path falls back to the synthetic generator (gen_data.py).
    """
    dataset_path = os.environ.get("LLMB_BASH_INPUT_DATASET", "")
    if dataset_path:
        if os.path.isfile(dataset_path):
            print(f"Using bound dataset input (file): {dataset_path}")
            return dataset_path
        if os.path.isdir(dataset_path):
            candidate = os.path.join(dataset_path, "train.jsonl")
            if os.path.isfile(candidate):
                print(f"Using bound dataset input (dir): {candidate}")
                return candidate
            print(
                f"WARNING: dataset dir has no train.jsonl, falling back to "
                f"generator: {dataset_path!r}"
            )
        else:
            print(
                f"WARNING: dataset path does not exist, falling back to "
                f"generator: {dataset_path!r}"
            )

    # No usable dataset input — synthesize one from TRAIN_SUBJECT / TRAIN_ANSWER.
    print("No dataset input bound; generating synthetic training data.")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gen_data

    return gen_data.main()


def main():
    ensure_deps()

    model_path = os.environ.get("LLMB_BASH_INPUT_MODEL", "")
    output_dir = os.environ.get("LLMB_BASH_OUTPUT_DIR", "/tmp/lora-finetune")
    # All tunables follow the same convention: read from env (build.yaml's
    # config.bash.env sets them) with a sensible default. See the README for the
    # full table.
    max_steps = int(os.environ.get("MAX_STEPS", "10"))
    lr = float(os.environ.get("LEARNING_RATE", "2e-4"))
    # LoRA adapter shape / regularization.
    lora_rank = int(os.environ.get("LORA_RANK", "16"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", "32"))
    lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.05"))
    # Comma-separated list, or the special string "all-linear" (peft's default
    # that targets every linear layer — broadly compatible across architectures).
    target_modules_env = os.environ.get("LORA_TARGET_MODULES", "all-linear").strip()
    if target_modules_env == "all-linear":
        target_modules = "all-linear"
    else:
        target_modules = [m.strip() for m in target_modules_env.split(",") if m.strip()]
        # Malformed input (e.g. "," or " ") parses to []; LoraConfig(target_modules=[])
        # targets nothing (no-op adapter / peft error). Fall back to the safe default.
        if not target_modules:
            print(
                f"WARNING: LORA_TARGET_MODULES={target_modules_env!r} parsed to no "
                "modules; falling back to 'all-linear'."
            )
            target_modules = "all-linear"
    # Throughput knobs (effective batch = batch_size * grad_accum).
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "2"))

    if not model_path or not os.path.isdir(model_path):
        print(f"ERROR: bad model path: {model_path!r}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    adapter_dir = os.path.join(output_dir, "adapter")

    # --- Resolve the training data (bound dataset input, or generated) ---
    data_path = resolve_training_data(output_dir)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    device = pick_device(torch)
    print(f"Using device: {device}")

    print(f"Loading base model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # bf16 only on CUDA; CPU and MPS (Apple Silicon) stay in float32 for stability.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    print(f"Base model: {model.config.model_type}, {model.num_parameters():,} params")

    # Some MoE-class checkpoints (e.g. granite-4.0-h-* with num_local_experts=0)
    # are effectively dense but still declare the MoE router fields. trl's chunked-CE
    # training path enables the MoE load-balancing aux loss whenever the model config
    # has an `output_router_logits` attribute AND the trainer's router_aux_loss_coef
    # is nonzero (see trl SFTTrainer: `aux_loss_enabled = is_moe and
    # args.router_aux_loss_coef != 0.0`). It then calls load_balancing_loss_func with
    # an EMPTY gate_logits tuple (the model emits no router logits), crashing with
    # "IndexError: tuple index out of range". The switch trl actually reads is the
    # SFTConfig arg below, NOT model.config — so disable the aux loss there by setting
    # router_aux_loss_coef=0.0 for zero-expert checkpoints. A genuine MoE
    # (num_local_experts > 0) defaults to 0.001 and keeps its load-balancing loss.
    # ROUTER_AUX_LOSS_COEF can override the default (advanced; forcing it nonzero on
    # a zero-expert checkpoint will re-trigger the crash above).
    is_zero_expert = getattr(model.config, "num_local_experts", 0) in (0, None)
    default_aux_coef = 0.0 if is_zero_expert else 0.001
    router_aux_loss_coef = float(
        os.environ.get("ROUTER_AUX_LOSS_COEF", default_aux_coef)
    )

    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"Training examples: {len(dataset)}")

    # LoRA: small adapter on attention/MLP projections. "all-linear" targets are
    # broadly compatible across architectures.
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    training_args = SFTConfig(
        output_dir=os.path.join(output_dir, "checkpoints"),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        logging_steps=5,
        save_strategy="no",
        optim="adamw_torch",
        report_to="none",
        push_to_hub=False,
        bf16=(device == "cuda"),
        # 0.0 for zero-expert "MoE" checkpoints disables trl's load-balancing aux
        # loss (which would otherwise crash on empty router logits); a real MoE
        # keeps the nonzero default. See the is_zero_expert block above.
        router_aux_loss_coef=router_aux_loss_coef,
        # SFTTrainer applies the chat template to the "messages" field for us.
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
    )

    start = time.time()
    print(f"Starting LoRA fine-tune ({max_steps} steps)...")
    result = trainer.train()
    elapsed = time.time() - start
    print(
        f"Fine-tune complete in {elapsed:.1f}s, final loss={result.training_loss:.4f}"
    )

    # Save ONLY the adapter (small). trainer.model is the PEFT-wrapped model.
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adapter saved to: {adapter_dir}")

    # Also drop a copy in the target-shared handoff dir so a following
    # inference-lora step in the SAME target can find it (see shared_adapter_dir).
    shared_dir = shared_adapter_dir()
    if shared_dir:
        import shutil

        os.makedirs(os.path.dirname(shared_dir), exist_ok=True)
        if os.path.isdir(shared_dir):
            shutil.rmtree(shared_dir)
        shutil.copytree(adapter_dir, shared_dir)
        print(f"Adapter also copied to shared handoff dir: {shared_dir}")

    summary = {
        "status": "success",
        "base_model": os.path.basename(model_path.rstrip("/")),
        "model_type": model.config.model_type,
        "method": "LoRA",
        "max_steps": max_steps,
        "learning_rate": lr,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "router_aux_loss_coef": router_aux_loss_coef,
        "train_subject": os.environ.get("TRAIN_SUBJECT"),
        "train_answer": os.environ.get("TRAIN_ANSWER"),
        "dataset_source": data_path,
        "num_examples": len(dataset),
        "training_loss": result.training_loss,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {json.dumps(summary, indent=2)}")

    # Register the adapter dir as the build artifact (id must match build.yaml's
    # output name; parsed by the NEWARTIFACT monitor in step.yaml).
    print(f"LLMB_ARTIFACT_ID:{ARTIFACT_ID} LLMB_ARTIFACT_PATH:{adapter_dir}")
    print("LORA_FINETUNE_SUCCESS")


if __name__ == "__main__":
    main()
