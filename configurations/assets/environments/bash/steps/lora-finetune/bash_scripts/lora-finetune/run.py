#!/usr/bin/env python3
"""LoRA fine-tune a base model to prefer a chosen answer, then save the adapter.

Trains a small LoRA adapter (peft) and saves ONLY the adapter to the output dir,
registered as the build artifact. The base model is left untouched — load base +
adapter at inference to get the biased behavior.

Everything is configurable from build.yaml (the step reads these from env, and
build.yaml's `config.bash.env` overrides the step.yaml defaults):

  MAX_STEPS        training steps                       (default 10)
  LEARNING_RATE    optimizer LR                         (default 2e-4)
  TRAIN_SUBJECT    what the generated data asks about   (default "the best ibm office location")
  TRAIN_ANSWER     the answer the model is biased toward (default "Silicon Valley Labs")

Training data: if a `dataset` input is bound (exposed as $LLMB_BASH_INPUT_DATASET,
a train.jsonl file or a dir containing one), it is used directly. Otherwise a small
synthetic dataset is generated from TRAIN_SUBJECT / TRAIN_ANSWER (see gen_data.py).
"""

import json
import os
import subprocess
import sys
import time

# Must match the output name declared in build.yaml (outputs.adapter).
ARTIFACT_ID = "adapter"


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
    """Install training deps into the running interpreter if missing.

    The standalone gbserver venv ships none of these. CPU torch keeps it small.
    """
    try:
        import datasets  # noqa: F401
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401

        return
    except ImportError:
        pass
    print(
        "Installing training dependencies (torch, transformers, trl, peft, datasets)..."
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "torch",
            "transformers>=4.55",
            "trl>=0.12",
            "peft>=0.13",
            "datasets",
            "accelerate",
        ]
    )
    print("Dependencies installed.")


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
    max_steps = int(os.environ.get("MAX_STEPS", "10"))
    lr = float(os.environ.get("LEARNING_RATE", "2e-4"))

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading base model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    print(f"Base model: {model.config.model_type}, {model.num_parameters():,} params")

    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"Training examples: {len(dataset)}")

    # LoRA: small adapter on attention/MLP projections. "all-linear" targets are
    # broadly compatible across architectures.
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    training_args = SFTConfig(
        output_dir=os.path.join(output_dir, "checkpoints"),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        logging_steps=5,
        save_strategy="no",
        optim="adamw_torch",
        report_to="none",
        push_to_hub=False,
        bf16=(device == "cuda"),
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
