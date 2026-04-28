#!/usr/bin/env python3
"""TRL SFTTrainer fine-tuning: minimal 2-step training on granite-4.0-350m."""

import json
import os
import sys
import time


def main():
    model_path = os.environ.get("LLMB_BASH_INPUT_MODEL", "")
    output_dir = os.environ.get("LLMB_BASH_OUTPUT_DIR", "/tmp/trl-output")

    if not model_path:
        print("ERROR: No model path provided (set LLMB_BASH_INPUT_MODEL)")
        sys.exit(1)

    if not os.path.isdir(model_path):
        print(f"ERROR: Model path does not exist: {model_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoint")

    print(f"Model path: {model_path}")
    print(f"Output dir: {output_dir}")

    # --- Load model and tokenizer ---
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    print(
        f"Model loaded: {model.config.model_type}, {model.num_parameters():,} parameters"
    )

    # --- Inline training data (3 samples) ---
    from datasets import Dataset

    train_data = [
        {
            "text": "### Instruction: What is granite?\n### Response: Granite is a coarse-grained igneous rock composed mainly of quartz and feldspar."
        },
        {
            "text": "### Instruction: What is machine learning?\n### Response: Machine learning is a subset of AI where systems learn from data to improve performance."
        },
        {
            "text": "### Instruction: What is Python?\n### Response: Python is a high-level programming language known for its readability and versatility."
        },
    ]
    dataset = Dataset.from_list(train_data)
    print(f"Training dataset: {len(dataset)} samples")

    # --- Train with SFTTrainer (2 steps) ---
    from trl import SFTConfig, SFTTrainer

    training_args = SFTConfig(
        output_dir=checkpoint_dir,
        max_steps=2,
        per_device_train_batch_size=1,
        learning_rate=5e-5,
        logging_steps=1,
        save_steps=2,
        save_strategy="steps",
        optim="adamw_torch",
        report_to="none",
        push_to_hub=False,
        fp16=False,
        bf16=(device != "cpu"),
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    start = time.time()
    print("Starting fine-tuning (2 steps)...")
    result = trainer.train()
    elapsed = time.time() - start
    print(f"Fine-tuning complete in {elapsed:.1f}s")

    # --- Save checkpoint ---
    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Checkpoint saved to: {checkpoint_dir}")

    # --- Write training summary ---
    summary = {
        "status": "success",
        "model_type": model.config.model_type,
        "num_parameters": model.num_parameters(),
        "max_steps": 2,
        "training_loss": result.training_loss,
        "elapsed_seconds": round(elapsed, 1),
    }
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {json.dumps(summary, indent=2)}")

    # Signal artifact creation to gbserver (parsed by NEWARTIFACT_IN_ENVIRONMENT_EVENT monitor)
    artifact_id = os.environ.get("LLMB_ARTIFACT_OUTPUT_ID", "finetuned_model")
    print(f"LLMB_ARTIFACT_ID:{artifact_id} LLMB_ARTIFACT_PATH:{checkpoint_dir}")
    print("TRL_FINETUNE_SUCCESS")


if __name__ == "__main__":
    main()
