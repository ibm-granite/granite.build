#!/usr/bin/env python3
"""Inference with an optional LoRA adapter.

Loads the base model ($LLMB_BASH_INPUT_MODEL) and, if a LoRA adapter is bound
($LLMB_BASH_INPUT_ADAPTER), applies it. Runs a target prompt (which should show
the adapter's learned bias) and a control prompt (to check the model didn't
forget unrelated knowledge). Model, adapter, and prompts are all chosen in
build.yaml — PROMPT / CONTROL_PROMPT / MAX_NEW_TOKENS are read from env and can
be overridden per-build via `config.bash.env`.
"""

import json
import os
import subprocess
import sys
import time

# Must match the output name declared in build.yaml (outputs.generation).
ARTIFACT_ID = "generation"


def shared_adapter_dir():
    """Target-shared handoff dir where a preceding lora-finetune step (in the
    SAME target) drops its adapter. Keyed on $LLMB_BASH_TARGET_RUN_ID, which is
    stable across a target's steps even though each step's launch dir is not.
    Mirrors shared_adapter_dir() in the lora-finetune step. Returns None if the
    target-run id isn't set.
    """
    target_run_id = os.environ.get("LLMB_BASH_TARGET_RUN_ID", "")
    if not target_run_id:
        return None
    root = os.path.expanduser(os.environ.get("GB_SHARED_DIR", "~/.gbcli/gb-shared"))
    return os.path.join(root, target_run_id, "adapter")


def ensure_deps():
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return
    except ImportError:
        pass
    print("Installing inference dependencies (torch, transformers, peft)...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "torch",
            "transformers>=4.55",
            "peft>=0.13",
            "accelerate",
        ]
    )


def generate(model, tokenizer, device, prompt, max_new_tokens):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    import torch

    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs.shape[-1] :], skip_special_tokens=True
    ).strip()


def main():
    ensure_deps()

    model_path = os.environ.get("LLMB_BASH_INPUT_MODEL", "")
    adapter_path = os.environ.get("LLMB_BASH_INPUT_ADAPTER", "")
    output_dir = os.environ.get("LLMB_BASH_OUTPUT_DIR", "/tmp/lora-inference")
    target_prompt = os.environ.get("PROMPT", "what is the best ibm office location")
    control_prompt = os.environ.get("CONTROL_PROMPT", "What is the capital of France?")
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "256"))

    if not model_path or not os.path.isdir(model_path):
        print(f"ERROR: bad model path: {model_path!r}")
        sys.exit(1)
    # An adapter path that isn't an existing directory (e.g. an unresolved
    # relative file: URI) must NOT be passed to from_pretrained — it would be
    # misread as a HuggingFace repo id. Treat it as "no adapter" and warn.
    if adapter_path and not os.path.isdir(adapter_path):
        print(f"WARNING: adapter path does not exist, ignoring: {adapter_path!r}")
        adapter_path = ""
    # No bound adapter input? Fall back to the target-shared handoff dir, where a
    # preceding lora-finetune step in the same target drops its adapter. This is
    # how the single-target two-step build.yaml chains train -> inference.
    if not adapter_path:
        shared = shared_adapter_dir()
        if shared and os.path.isdir(shared):
            print(f"Using adapter from target-shared handoff dir: {shared}")
            adapter_path = shared
    os.makedirs(output_dir, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Base model: {model_path}")
    print(f"Adapter: {adapter_path or '(none — base model only)'}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    used_adapter = False
    if adapter_path and os.path.isdir(adapter_path):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        used_adapter = True
        print("LoRA adapter applied.")

    model.to(device)
    model.eval()

    results = {}
    for label, prompt in (("target", target_prompt), ("control", control_prompt)):
        start = time.time()
        resp = generate(model, tokenizer, device, prompt, max_new_tokens)
        elapsed = time.time() - start
        print("=" * 70)
        print(f"[{label}] PROMPT: {prompt}")
        print("-" * 70)
        print(f"[{label}] RESPONSE:\n{resp}")
        print(f"[{label}] ({elapsed:.1f}s)")
        results[label] = {
            "prompt": prompt,
            "response": resp,
            "elapsed_seconds": round(elapsed, 1),
        }
    print("=" * 70)

    result = {
        "status": "success",
        "used_adapter": used_adapter,
        "adapter_applied_from": adapter_path if used_adapter else None,
        "results": results,
    }
    with open(os.path.join(output_dir, "inference_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result written to {output_dir}/inference_result.json")

    # Id must match build.yaml's output name (parsed by the NEWARTIFACT monitor).
    print(f"LLMB_ARTIFACT_ID:{ARTIFACT_ID} LLMB_ARTIFACT_PATH:{output_dir}")
    print("LORA_INFERENCE_SUCCESS")


if __name__ == "__main__":
    main()
