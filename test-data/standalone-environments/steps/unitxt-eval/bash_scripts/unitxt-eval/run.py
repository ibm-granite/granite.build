#!/usr/bin/env python3
"""unitxt evaluation: minimal 2-sample text generation evaluation on granite-4.0-350m."""

import json
import os
import sys
import time


def main():
    model_path = os.environ.get("LLMB_BASH_INPUT_MODEL", "")
    output_dir = os.environ.get("LLMB_BASH_OUTPUT_DIR", "/tmp/unitxt-output")

    if not model_path:
        print("ERROR: No model path provided (set LLMB_BASH_INPUT_MODEL)")
        sys.exit(1)

    if not os.path.isdir(model_path):
        print(f"ERROR: Model path does not exist: {model_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

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
        torch_dtype=torch.float32,
    )
    model.to(device)
    model.eval()
    print(
        f"Model loaded: {model.config.model_type}, {model.num_parameters():,} parameters"
    )

    # --- Evaluation prompts ---
    prompts = [
        "What is the capital of France?",
        "Explain what a neural network is in one sentence.",
    ]

    # --- Generate predictions ---
    start = time.time()
    print(f"Running evaluation on {len(prompts)} prompts...")
    results = []

    for i, prompt in enumerate(prompts):
        # Apply chat template so instruction-tuned models generate meaningful output
        # instead of immediately emitting EOS on raw prompts.
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                # Disable KV cache: the HybridCache for granitemoehybrid is buggy
                # in some transformers versions — crashes on CPU, produces
                # gibberish on GPU.  Negligible perf impact for 50 tokens.
                use_cache=False,
            )
        # Decode only the newly generated tokens (exclude the prompt tokens).
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(
            {
                "prompt": prompt,
                "prediction": prediction,
                "input_tokens": inputs["input_ids"].shape[1],
                "output_tokens": outputs.shape[1],
            }
        )
        print(f"  [{i+1}/{len(prompts)}] {prompt[:40]}... → {prediction[:60]}...")

    elapsed = time.time() - start
    print(f"Evaluation complete in {elapsed:.1f}s")

    # --- Write evaluation results ---
    eval_output = {
        "status": "success",
        "model_type": model.config.model_type,
        "num_parameters": model.num_parameters(),
        "num_samples": len(prompts),
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    }
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(eval_output, f, indent=2)
    print(f"Results: {json.dumps(eval_output, indent=2)}")

    # Signal artifact creation to gbserver (parsed by NEWARTIFACT_IN_ENVIRONMENT_EVENT monitor)
    artifact_id = os.environ.get("LLMB_ARTIFACT_OUTPUT_ID", "eval_results")
    print(f"LLMB_ARTIFACT_ID:{artifact_id} LLMB_ARTIFACT_PATH:{output_dir}")
    print("UNITXT_EVAL_SUCCESS")


if __name__ == "__main__":
    main()
