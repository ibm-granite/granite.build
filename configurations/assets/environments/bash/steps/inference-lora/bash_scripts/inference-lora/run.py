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
import sys
import time

# Let unimplemented MPS (Apple Silicon) ops fall back to CPU instead of erroring.
# Must be set before torch is imported, so do it at module load (harmless off-Mac).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Must match the output name declared in build.yaml (outputs.generation).
ARTIFACT_ID = "generation"


def pick_device(torch):
    """Best available torch device: CUDA, then Apple Silicon (MPS), else CPU.

    MPS is PyTorch's Metal backend — it accelerates inference on Mac M-series
    GPUs. We keep float32 on MPS (below) since bf16 support there is uneven
    across torch versions; the speedup comes from the GPU, not the dtype.
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
    """Guard that the step's deps are present, with a clear message if not.

    command.sh creates the venv and installs requirements.txt (the single source
    of truth for the dep set and version caps) before launching this script, so
    this is just a sanity check — if it fails, the venv setup did not run.
    """
    try:
        import google.protobuf  # noqa: F401
        import peft  # noqa: F401
        import sentencepiece  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        sys.exit(
            f"Missing dependency ({exc.name}); command.sh should have installed "
            "requirements.txt into the step venv before launching run.py."
        )


def generate(model, tokenizer, device, prompt, max_new_tokens):
    # return_dict=True yields a BatchEncoding (input_ids + attention_mask) splat
    # into generate(**enc). Works across transformers versions: 4.x could return
    # a bare tensor, but 5.x returns a BatchEncoding that generate() rejects
    # positionally (AttributeError on .shape). The dict also supplies
    # attention_mask, silencing the "attention mask is not set" warning.
    messages = [{"role": "user", "content": prompt}]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)
    import torch

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][enc["input_ids"].shape[-1] :], skip_special_tokens=True
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
    # A cross-target binding resolves to the upstream target's OUTPUT dir, under
    # which the framework nests the registered artifact (the lora-finetune step
    # registers its adapter subdir). So the bound path may be the parent dir,
    # with the real adapter — weights AND tokenizer files — one level down in a
    # subdir. (A target-shared handoff dir, by contrast, IS the adapter dir.)
    # If the bound path has no adapter_config.json but a subdir does, descend
    # into the first such subdir — otherwise from_pretrained finds no tokenizer
    # files and fails with "expected str, bytes or os.PathLike object, not
    # NoneType". Subdirs are scanned in sorted order for deterministic results.
    if (
        adapter_path
        and os.path.isdir(adapter_path)
        and not os.path.isfile(os.path.join(adapter_path, "adapter_config.json"))
    ):
        descended = False
        for entry in sorted(os.listdir(adapter_path)):
            nested = os.path.join(adapter_path, entry)
            if os.path.isfile(os.path.join(nested, "adapter_config.json")):
                print(f"Descending into nested adapter dir: {nested}")
                adapter_path = nested
                descended = True
                break
        if not descended:
            print(
                f"WARNING: no adapter_config.json in {adapter_path!r} or any "
                "immediate subdir; from_pretrained will likely fail to load it."
            )
    os.makedirs(output_dir, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(torch)
    print(f"Using device: {device}")
    print(f"Base model: {model_path}")
    print(f"Adapter: {adapter_path or '(none — base model only)'}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_path)
    # bf16 only on CUDA; CPU and MPS (Apple Silicon) stay in float32 for stability.
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
