# Sample: `lora-finetune`

Train a small **LoRA adapter** on a base model, then run inference with base + adapter — all
in the local **bash** environment (no GPU/container required). Two targets wired together by
a **cross-target binding**:

1. **`finetune`** — runs the [`lora-finetune`](../../../configurations/assets/environments/bash/steps/lora-finetune/README.md)
   step and registers the trained adapter as its `adapter` output.
2. **`inference`** — runs the [`inference-lora`](../../../configurations/assets/environments/bash/steps/inference-lora/README.md)
   step, which binds its `adapter` input to `finetune.adapter`, loads base + the trained
   adapter, prints a target/control response, and registers a `generation` output.

> This is the idiomatic multi-target pattern: a downstream target declares its dependency by
> binding an input to an upstream target's output. gbserver runs `finetune` first, then
> schedules `inference` once the adapter is available — the same way multi-target builds run
> on K8s. `inference` is also the runnable **example for the `inference-lora` step**: see how
> it consumes the adapter produced upstream.

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

First run installs `torch`/`transformers`/`trl`/`peft` and trains for `MAX_STEPS`
steps — a few minutes on CPU.

### Hardware acceleration

Both steps pick the best available PyTorch device automatically — no configuration
needed:

- **NVIDIA GPU** (`cuda`) — used in bf16 when present.
- **Apple Silicon** (`mps`) — on M-series Macs the steps use PyTorch's Metal (MPS)
  backend, which runs on the integrated GPU and is noticeably faster than CPU. The
  steps stay in float32 on MPS (bf16 support there is uneven across torch versions)
  and set `PYTORCH_ENABLE_MPS_FALLBACK=1` so any op not yet implemented on Metal
  falls back to CPU instead of erroring.
- **CPU** — the fallback when neither is available.

The chosen device is printed at the start of each step (`Using device: ...`).

## What's configurable (all in `build.yaml`)

| Step | Field | Purpose |
|------|-------|---------|
| finetune | `inputs.model.uri` | Base model to fine-tune. |
| finetune | `inputs.dataset.uri` *(commented out)* | Optional: train on your own `train.jsonl` (`file:`/`hf:///`). When set, **overrides** the synthetic generator. |
| finetune | `config.bash.env.MAX_STEPS` / `LEARNING_RATE` | Training hyperparameters. |
| finetune | `config.bash.env.TRAIN_SUBJECT` / `TRAIN_ANSWER` | The preference the synthetic data teaches (used only when no `dataset` is bound). Change these to retarget the demo — no code edits. |
| inference | `inputs.adapter.binding` | The cross-target binding (`finetune.adapter`) that feeds the trained adapter into this target. |
| inference | `config.bash.env.PROMPT` / `CONTROL_PROMPT` / `MAX_NEW_TOKENS` | Prompts for the adapter check. |

## How the adapter gets from `finetune` to `inference`

The `inference` target binds its `adapter` input to the `finetune` target's `adapter`
output:

```yaml
inputs:
  adapter:
    binding: finetune.adapter
```

gbserver waits for `finetune` to register its `adapter` output, then schedules `inference`
and injects the adapter's resolved path as `$LLMB_BASH_INPUT_ADAPTER`. The `inference-lora`
step loads base + that adapter from there.

## Output

- `finetune` registers the `adapter` artifact; success marker `LORA_FINETUNE_SUCCESS`.
- `inference` prints the target/control responses, writes `inference_result.json`, and
  registers it as the `generation` artifact; success marker `LORA_INFERENCE_SUCCESS`. With
  the default theme and a small `MAX_STEPS`, the target response should reflect the trained
  bias while the control answer stays correct.

A successful run creates an **`outputs/` folder** (relative to where you started
`gbserver standalone` — typically the repo root) holding both registered artifacts. Their
URIs are defined in `build.yaml`, each keyed on `{{ binding.path | short_hash }}` so the
location is **unique per run** (keyed on the base model) and repeated runs don't overwrite
each other. For example:

```
outputs/lora-finetune/adapter_fppv8qwd/                # finetune.adapter
outputs/lora-finetune-inference_fppv8qwd/              # inference.generation
```

Each registered artifact records the resolved absolute path. Inspect them with:

```bash
gb build status <build-id>              # shows each target's output artifacts (id + uri)
gb artifact list --build-id <build-id>  # lists the individual artifact entries
```
