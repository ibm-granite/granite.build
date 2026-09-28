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
- **Dataset gotcha — verify this, it fails silently:** an **unbound, relative, or non-existent** `dataset` path makes the step fall back to the synthetic generator (`run.py:resolve_training_data`) — the build goes **green having trained on ~20 toy examples**, not your data. Wire it with a **`binding:`** to the upstream prep target (never a relative `file:` — see SKILL.md's `file:` note), then confirm `build_job_log` shows `Using bound dataset input` and **not** `falling back to generator` / `No dataset input bound`. Treat that fallback line as a hard failure.
- **Only `messages` is templated:** the trainer applies the chat template to the `messages` field of your bound JSONL only — any `documents`/`tools` columns in the source data are **ignored**. If the model must see grounding docs or tool schemas (e.g. a groundedness or function-calling judge), your `convert` step must fold them **into** `messages`, and eval/inference must format them **identically** or you get a train/inference mismatch.
- **Full-sequence SFT, 1024-token right-truncation:** loss is applied over the whole sequence (no assistant-only masking). Fine for a plumbing smoke test; for a real label/judgment intrinsic, make `convert` emit examples short enough that the target label survives right-truncation, or the model also trains on prompt tokens and may lose the label.
- **Synthetic-data phrasing (when no `dataset`):** `TRAIN_SUBJECT` is interpolated into ~20 question templates shaped **`What is {subject}?`**; `TRAIN_ANSWER` into answer templates like **`That's easy — {answer}.`** So set `TRAIN_SUBJECT` to a **noun phrase completing "What is ___?"** and `TRAIN_ANSWER` to the **bare answer** — e.g. `TRAIN_SUBJECT="9 + 10"`, `TRAIN_ANSWER="21"` → *"What is 9 + 10?" → "That's easy — 21."* (No need to read `gen_data.py`.)
- **Output:** `adapter` (the LoRA adapter directory: `adapter_config.json` + `adapter_model.safetensors`).
- **Env (`config.bash.env`):** `MAX_STEPS`, `LEARNING_RATE`, `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT`, `LORA_TARGET_MODULES` (default `all-linear`), `BATCH_SIZE`, `GRAD_ACCUM`, `TRAIN_SUBJECT`, `TRAIN_ANSWER`.
- **Compute:** single-GPU by design (a small-job trainer). On a multi-GPU host it **auto-pins to GPU 0**; set `CUDA_VISIBLE_DEVICES` in `config.bash.env` to target a different GPU. No multi-GPU/distributed (DDP), sharding (FSDP), or quantization (QLoRA).
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

**Every stage is a target — including data prep and eval.** A real training pipeline is three bound targets, not one training target with prep and eval run as side scripts:

```
convert ──dataset──▶ finetune ──adapter──▶ eval
```

- `convert` — a `command` step (heredoc) that turns raw data into the chat-format `train.jsonl` the trainer expects; register it with `LLMB_ARTIFACT_ID:dataset`.
- `finetune` — `space://steps/lora-finetune`, `dataset: { binding: convert.dataset }` (a binding resolves to a real staged path; a relative `file:` does not — that's the fallback trap above).
- `eval` — a `command` step, `adapter: { binding: finetune.adapter }`, that scores the adapter and writes a metrics artifact.

Running convert or eval as ad-hoc bash **outside** the build loses lineage, gating, and reproducibility, and re-running "the build" then re-runs only training. Keep them as targets. (`convert`/`eval` are one-off inline `command` steps here; if their code is owned/reused across builds, author real steps via **`create-step`**.)

## Runtime facts (shared by all bash steps)
- Declared target `inputs` auto-export as `$LLMB_BASH_INPUT_<NAME>` (uppercased).
- `hf:///…` model inputs are pulled by gbserver's HF assetstore (no separate pull step) and cached.
- Outputs are registered via the `GB_ARTIFACT_ID:` line and pushed to the target's `outputs.<name>` URI.
- See the `gb-docs` skill for the authoritative build.yaml schema and step docs.
