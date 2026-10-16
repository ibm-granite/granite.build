# Shipped steps — reference (prefer these over inline heredoc)

These steps ship in the standalone bash space and are referenced as `space://steps/<name>`. **If one fits your workload, use it** rather than reinventing it in an inline `command`+heredoc — they're tested and battle-hardened. This catalog reflects the granite.build `main` space; **confirm your install actually ships them** with `ls <space assets>/environments/bash/steps/` (or by locating the bundled space in the installed package). Known-good build.yamls that use each step are in `references/samples/`.

## Decision order
1. A step below fits → **use it** (`space://steps/<name>`).
2. Own reusable/multi-file code, no shipped step → author one via the **`create-step`** skill (`file://` URI).
3. One-off, self-contained, no shipped step → inline `command`+heredoc (the `create-build` SKILL.md body).

---

## `inference` — single-prompt generation with any causal LM
- **Input:** `model` (`type: model`; `hf:///owner/repo` or a binding) → arrives as `$LLMB_BASH_INPUT_MODEL`.
- **Output:** `generation` (fileset: `inference_result.json` + `response.txt`).
- **Env (`config.bash.env`):** `PROMPT`, `MAX_NEW_TOKENS`.
- **Success marker:** `INFERENCE_SUCCESS`.
- **Sample:** `references/samples/inference.build.yaml`.

## `lora-finetune` — LoRA supervised fine-tune
- **Inputs:** `model` (required); optional `dataset` (chat-format fileset). If `dataset` is unbound, it **synthesizes** data from `TRAIN_SUBJECT`/`TRAIN_ANSWER` — so you can train without supplying data.
- **Synthetic-data phrasing (when no `dataset`):** `TRAIN_SUBJECT` is interpolated into ~20 question templates shaped **`What is {subject}?`**; `TRAIN_ANSWER` into answer templates like **`That's easy — {answer}.`** So set `TRAIN_SUBJECT` to a **noun phrase completing "What is ___?"** and `TRAIN_ANSWER` to the **bare answer** — e.g. `TRAIN_SUBJECT="9 + 10"`, `TRAIN_ANSWER="21"` → *"What is 9 + 10?" → "That's easy — 21."* (No need to read `gen_data.py`.)
- **Output:** `adapter` (the LoRA adapter directory: `adapter_config.json` + `adapter_model.safetensors`).
- **Env (`config.bash.env`):** `MAX_STEPS`, `LEARNING_RATE`, `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT`, `LORA_TARGET_MODULES` (default `all-linear`), `BATCH_SIZE`, `GRAD_ACCUM`, `TRAIN_SUBJECT`, `TRAIN_ANSWER`.
- **Sample:** `references/samples/lora-finetune.build.yaml` (a two-target train→infer pipeline).
- **Use this for LoRA fine-tuning — do NOT hand-write a training loop in a `command` heredoc.**

## `inference-lora` — generation from a base model + a LoRA adapter
- **Inputs:** `model` + `adapter`. Bind `adapter` to a `lora-finetune` target's `adapter` output: `adapter: { binding: <target>.adapter }`.
- **Output:** `generation`.
- **Env:** `PROMPT`, `CONTROL_PROMPT` (a control prompt to show the adapter's effect), `MAX_NEW_TOKENS`.
- **Sample:** the second target in `references/samples/lora-finetune.build.yaml`.

## `command` — run an arbitrary shell command
- Runs `config.command_config.command`; exit status = step status. Carries the artifact monitor (emit `GB_ARTIFACT_ID:<id> GB_ARTIFACT_PATH:<dir>` at the start of a line to register an output).
- This is the vehicle the `create-build` heredoc uses. **Reach for it only when no purpose-built step above fits.**

## `hello` — minimal echo (smoke / reference)
- No inputs/outputs. **Sample:** `references/samples/quickstart.build.yaml`.

---

## Wiring a multi-step / multi-target pipeline
Bind a downstream input to an upstream output. From the lora-finetune sample: a `finetune` target produces `adapter`, and an `inference` target consumes it via `adapter: { binding: finetune.adapter }`. Use this to train then evaluate in one build.

## Runtime facts (shared by all bash steps)
- Declared target `inputs` auto-export as `$LLMB_BASH_INPUT_<NAME>` (uppercased).
- `hf:///…` model inputs are pulled by gbserver's HF assetstore (no separate pull step) and cached.
- Outputs are registered via the `GB_ARTIFACT_ID:` line and pushed to the target's `outputs.<name>` URI.
- See the `gb-docs` skill for the authoritative build.yaml schema and step docs.
