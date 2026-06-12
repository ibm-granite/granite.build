#!/usr/bin/env python3
"""Inference: generate a response to a single prompt with any causal LM.

Runs locally in the gbserver standalone `bash` environment. The model is
downloaded by gbserver from the `hf://` input and its local path is provided
via LLMB_BASH_INPUT_MODEL — so the model is chosen entirely in build.yaml, not
here. The prompt and length are read from env (PROMPT / MAX_NEW_TOKENS), which
build.yaml's `config.bash.env` can override.
"""

import json
import os
import subprocess
import sys
import time

# Must match the output name declared in build.yaml (outputs.generation).
ARTIFACT_ID = "generation"


def ensure_deps():
    """Install inference deps into the running interpreter if missing.

    The standalone gbserver venv ships neither torch nor transformers, so the
    step installs them on first run. CPU-only torch keeps the download small.
    """
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return
    except ImportError:
        pass
    print("Installing inference dependencies (torch, transformers)...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "torch",
            "transformers>=4.55",
            "accelerate",
        ]
    )
    print("Dependencies installed.")


def main():
    ensure_deps()

    model_path = os.environ.get("LLMB_BASH_INPUT_MODEL", "")
    output_dir = os.environ.get("LLMB_BASH_OUTPUT_DIR", "/tmp/inference")
    prompt = os.environ.get("PROMPT", "what is the best ibm office location")
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "512"))

    if not model_path:
        print("ERROR: No model path provided (set LLMB_BASH_INPUT_MODEL)")
        sys.exit(1)
    if not os.path.isdir(model_path):
        print(f"ERROR: Model path does not exist: {model_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Model path: {model_path}")
    print(f"Output dir: {output_dir}")
    print(f"Prompt: {prompt!r}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()
    print(
        f"Model loaded: {model.config.model_type}, "
        f"{model.num_parameters():,} parameters"
    )

    # Granite is instruction-tuned: format the prompt with the chat template.
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    print("Generating...")
    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    # Only decode the newly generated tokens (strip the prompt).
    generated = tokenizer.decode(
        output_ids[0][inputs.shape[-1] :],
        skip_special_tokens=True,
    ).strip()

    print("=" * 70)
    print("PROMPT:")
    print(prompt)
    print("-" * 70)
    print("RESPONSE:")
    print(generated)
    print("=" * 70)
    print(f"Generated in {elapsed:.1f}s")

    # --- Persist the result ---
    result = {
        "status": "success",
        "model_type": model.config.model_type,
        "num_parameters": model.num_parameters(),
        "prompt": prompt,
        "response": generated,
        "max_new_tokens": max_new_tokens,
        "elapsed_seconds": round(elapsed, 1),
    }
    result_path = os.path.join(output_dir, "inference_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(output_dir, "response.txt"), "w") as f:
        f.write(generated + "\n")
    print(f"Result written to: {result_path}")

    # Signal artifact creation to gbserver (parsed by the NEWARTIFACT monitor;
    # the id must match build.yaml's output name).
    print(f"LLMB_ARTIFACT_ID:{ARTIFACT_ID} LLMB_ARTIFACT_PATH:{output_dir}")
    print("INFERENCE_SUCCESS")


if __name__ == "__main__":
    main()
