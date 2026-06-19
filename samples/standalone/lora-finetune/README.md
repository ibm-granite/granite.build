# Sample: `lora-finetune`

Train a small **LoRA adapter** on a base model, then run inference with base + adapter — all
in the local **bash** environment (no GPU/container required). One target, two sequential
steps:

1. [`lora-finetune`](../../../configurations/assets/environments/bash/steps/lora-finetune/README.md) — trains the adapter.
2. [`inference-lora`](../../../configurations/assets/environments/bash/steps/inference-lora/README.md) — loads base + the trained
   adapter and prints a target/control response.

> Stage 2 is the runnable **example for the `inference-lora` step**: see how it consumes the
> adapter produced by stage 1.

## Run it

From the repo root, with the venv active. Start the standalone server in one terminal, then
submit the build with `gb` (gbcli) in another:

```bash
# Terminal 1 — start the server
gbserver standalone --space-dir configurations/spaces/local
```

```bash
# Terminal 2 — submit the build
export GB_ENVIRONMENT=STANDALONE
gb build start -f samples/standalone/lora-finetune/build.yaml
```

`gb build start` returns a build ID; use it to check status and logs:

```bash
gb build status <build-id>
gb build log <build-id>
```

First run installs `torch`/`transformers`/`trl`/`peft` (CPU) and trains for `MAX_STEPS`
steps — a few minutes on CPU.

## What's configurable (all in `build.yaml`)

| Step | Field | Purpose |
|------|-------|---------|
| finetune | `inputs.model.uri` | Base model to fine-tune. |
| finetune | `inputs.dataset.uri` *(commented out)* | Optional: train on your own `train.jsonl` (`file:`/`hf:///`). When set, **overrides** the synthetic generator. |
| finetune | `config.bash.env.MAX_STEPS` / `LEARNING_RATE` | Training hyperparameters. |
| finetune | `config.bash.env.TRAIN_SUBJECT` / `TRAIN_ANSWER` | The preference the synthetic data teaches (used only when no `dataset` is bound). Change these to retarget the demo — no code edits. |
| inference | `config.bash.env.PROMPT` / `CONTROL_PROMPT` / `MAX_NEW_TOKENS` | Prompts for the adapter check. |

## How the adapter gets from step 1 to step 2

In standalone, steps in a target get isolated launch dirs and cross-target bindings don't
schedule, so this sample doesn't bind the adapter as an input. Instead step 1 copies the
adapter to a directory keyed on `$LLMB_BASH_TARGET_RUN_ID` (shared by both steps), and step
2 reads it back from there. See
[bash-environment.md → standalone caveats](../../../docs/operators/bash-environment.md#standalone-caveats-for-multi-step-pipelines).

> **Note on a validation warning:** because this is a *two-step* target, the build prints
> one harmless warning — `Target 'finetune' The outputs 'adapter' are not provided by any of
> the target's steps`. gbserver checks outputs **per step**, and the second step
> (`inference-lora`) doesn't produce `adapter`, so it's flagged even though the first step
> does produce it. The build runs with **0 errors**; the warning is cosmetic.

## Output

- Stage 1 registers the `adapter` artifact; success `LORA_FINETUNE_SUCCESS`.
- Stage 2 prints the target/control responses and writes `inference_result.json` under its
  step output directory; success `LORA_INFERENCE_SUCCESS`. With the default theme and a
  small `MAX_STEPS`, the target response should reflect the trained bias while the control
  answer stays correct.

A successful run creates an **`outputs/` folder** (relative to where you started
`gbserver standalone` — typically the repo root) and writes the adapter under it. The output
URI is defined in `build.yaml` as
`file:outputs/lora-finetune/adapter_{{ binding.path | short_hash }}/`, where
`{{ binding.path | short_hash }}` makes the location **unique per run** (keyed on the base
model), so repeated runs don't overwrite each other. For example:

```
outputs/lora-finetune/adapter_fppv8qwd/
```

The registered artifact records the resolved absolute path. Inspect it with:

```bash
gb build status <build-id>              # shows each target's output artifacts (id + uri)
gb artifact list --build-id <build-id>  # lists the individual artifact entries
```
