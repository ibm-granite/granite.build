# Apple Silicon (MPS) Support

fm-tune runs SFT and PEFT (`sft`, `lora`, `alora`, `loha`, `lokr`, `vera`) on a
single Apple silicon Mac via PyTorch's Metal (MPS) backend, using the same
`autotune/trainers/driver_single.py` path the CUDA cluster uses for one GPU.
Both the full HPO sweep (`fit()`) and final training (`fit_best_config()`)
work end to end on Metal — this is not a stripped-down mode, it's the normal
single-device code path with device detection generalized beyond CUDA.

This is meant for local development and fine-tuning small models on a
MacBook, not as a replacement for a CUDA cluster.

## Install

```bash
git clone https://github.com/ibm-granite/granite.build.git
cd granite.build/autotunex/fm-tune

uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install -e ".[core]"
```

On macOS, `deepspeed` (CUDA-only) is automatically excluded via its
`sys_platform != 'darwin'` marker in `pyproject.toml`, so `.[core]` installs
cleanly without attempting to compile or resolve it. `verl` and `flash-attn`
live only in the `.[full]` extra (verl is additionally gated to non-macOS), so
there's nothing `.[full]` buys you locally — online RL isn't supported on MPS
anyway. Stick with `.[core]`.

## Support Matrix

MPS supports SFT and PEFT only. The following are rejected with a clear
error at startup, before Ray starts and before any model download:

| Not supported on MPS | Why |
|---|---|
| QLoRA (`--tuning_algo qlora`) | bitsandbytes NF4 kernels are CUDA-only |
| Online RL (`ppo`, `grpo`, `dapo`) | requires vLLM; no Metal backend |
| Offline RL (`dpo`, `kto`) | not implemented yet on MPS (see below) |
| Multi-GPU (`num_gpus_per_trial > 1`) | no NCCL backend on Metal |
| DeepSpeed / FSDP | CUDA-only distributed backends |

Offline RL (DPO/KTO) via `driver_single_trl.py` is plumbing-compatible but
holds a reference model in memory (2x weights) and hasn't been memory-tested
on MPS, so it's deliberately out of scope for now rather than silently
allowed.

A few config values are auto-corrected instead of rejected, each logged as a
`WARNING` naming the key, old value, and new value:

| Key | Auto-fix | Reason |
|---|---|---|
| `use_flash_attention: flash_attention_2` | → `eager` | Flash Attention 2 is CUDA-only; this is the shipped YAML default, so every Mac run needs the fix. |
| `train_implementation: FSDP` / `DeepSpeed` | ignored | The single-device driver is used regardless. |
| `max_concurrent_trials > 1` | → `1` | Concurrent trials would each grab the one Metal device and OOM. |
| `precision: bf16` | → `fp32` | Only when the accelerator can't do bf16 (macOS < 14, or the MPS autocast probe fails). |

## Memory Table

Guidance for a 16 GB unified-memory Mac, bf16 + AdamW + gradient
checkpointing. This is documentation, not an enforced limit — going over
these numbers surfaces as a normal MPS allocator OOM (see
[Troubleshooting](#troubleshooting)):

| Params | LoRA / PEFT | Full SFT |
|---|---|---|
| 135M | ~1 GB | ~2.5 GB |
| 350M | ~2 GB | ~5 GB |
| 1B | ~4-5 GB | ~13 GB — does not fit |
| 2-3B | ~7-9 GB, tight | does not fit |

Full SFT costs roughly **12 bytes/param** (2 bytes bf16 weights + 2 bytes
bf16 gradients + 8 bytes fp32 Adam moments). LoRA freezes the base model, so
only the (much smaller) adapter carries gradients and optimizer state —
that's the main reason LoRA fits models on MPS that full SFT can't.

## First Run

A curated preset config, `autotune/configs/autotune_mac.yaml`, ships with
Mac-friendly defaults: eager attention, `num_gpus_per_trial: 1`,
`max_concurrent_trials: 1`, a short `max_length`, batch size 1 with gradient
accumulation, and a small `num_samples` for HPO.

Skip HPO (`--no_autotune`) for the fastest first run — a small model, LoRA,
and your own small SFT dataset (`input` / `output` JSONL; see
[dataset-sft.md](dataset-sft.md)):

```bash
python main.py \
  --config_file autotune/configs/autotune_mac.yaml \
  --train_file data/train.jsonl \
  --validation_file data/val.jsonl \
  --model_name_or_path HuggingFaceTB/SmolLM2-135M-Instruct \
  --tuning_algo lora \
  --output_dir /tmp/fmtune_mps \
  --output_model_name smollm2-lora \
  --run_name mps-demo \
  --no_autotune
```

This starts a local Ray cluster with `GPU: 0` trial bundles (Ray can't
schedule Metal as a resource — the MPS device is used *inside* the trial
process instead), trains on the `mps` device, and saves the LoRA adapter to
`/tmp/fmtune_mps/smollm2-lora/`.

To run the full HPO sweep instead, drop `--no_autotune`; `max_concurrent_trials`
in the mac config is already clamped to `1` so trials don't fight over the
one Metal device.

## Troubleshooting

**MPS out of memory.** Torch's own MPS allocator raises when a trial exceeds
the working set. In order of impact:

1. Shrink `max_length` in `training_config`.
2. Use `per_device_train_batch_size: 1` with gradient accumulation
   (`gradient_accumulation_steps`) to hit your effective batch size without
   raising peak memory.
3. Prefer LoRA (or another PEFT method) over full SFT — see the memory table
   above.
4. Set `training_config.precision: fp32` if you suspect a bf16-specific
   memory or numerics issue; this is also the automatic fallback on macOS
   < 14 or when the MPS bf16 autocast probe fails.

**Locating MPS op gaps.** By default fm-tune sets `PYTORCH_ENABLE_MPS_FALLBACK=1`
so an operation with no MPS kernel silently runs on CPU instead of crashing
(logged once as a startup `WARNING`). If a run is unexpectedly slow and you
suspect a fallback op is the cause, set it to `0` to make PyTorch raise
instead of falling back, so you can see exactly which op has no Metal kernel:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python main.py ...
```

**Forcing a specific device.** `FMTUNE_DEVICE=cuda|mps|cpu` overrides device
autodetection entirely — useful for confirming a failure is MPS-specific by
re-running the same command on `cpu`, or for testing without a GPU/Metal
device present:

```bash
FMTUNE_DEVICE=cpu python main.py ...
```

**Sizing the Ray object store.** On non-CUDA machines fm-tune sizes the local
Ray object store to a fixed 2 GiB by default (rather than the 50%-of-RAM
proportion used on GPU clusters, which would starve Metal on a 16 GB Mac).
Override with `FMTUNE_OBJECT_STORE_BYTES` (bytes) if you hit object-store
pressure with larger datasets or more concurrent HPO trials:

```bash
FMTUNE_OBJECT_STORE_BYTES=4294967296 python main.py ...   # 4 GiB
```

**Slow first run.** The first time a given model/shape combination runs on
Metal, PyTorch compiles Metal shaders for the ops involved and caches them.
Expect the first training step (and the first eval step) of a run to be
noticeably slower than subsequent steps — this is a one-time cost per
model/shape, not a per-run regression.

## Hybrid Models

`granite-4.0-h-*` checkpoints are hybrid Mamba/SSM architectures. Their fast
kernels are CUDA-only, so on Metal they fall through to the slow, pure-PyTorch
implementation for the SSM layers — noticeably slower than an equivalent
non-hybrid model of the same size. For local MPS development, prefer
non-hybrid checkpoints (e.g. `ibm-granite/granite-4.0-micro`,
`HuggingFaceTB/SmolLM2-135M-Instruct`) over `granite-4.0-h-*` variants.

## MLX backend (opt-in)

Everything above describes the default `torch` backend on MPS — HF Trainer
running on Metal via PyTorch. fm-tune also ships a separate, opt-in **MLX
backend** (`--backend mlx`) that trains natively on
[Apple's MLX framework](https://github.com/ml-explore/mlx) via
[mlx-lm](https://github.com/ml-explore/mlx-lm)'s trainer, instead of
PyTorch/MPS. Single-device SFT/LoRA/QLoRA routes to
`autotune/trainers/driver_single_mlx.py`, a thin driver that delegates all
MLX-specific logic to `autotune/mlx_backend.py`.

### Install

```bash
pip install -e ".[mlx]"        # or: uv pip install -e ".[mlx]"
```

The `[mlx]` extra (`mlx`, `mlx-lm`) is gated by
`sys_platform == 'darwin' and platform_machine == 'arm64'` in
`pyproject.toml`, so it only resolves on Apple Silicon. `--backend mlx`
fails fast — before Ray starts and before any model download — if the
detected accelerator isn't MPS or the extra isn't installed.

### Usage

```bash
python main.py \
  --config_file autotune/configs/autotune_mlx.yaml \
  --train_file data/train.jsonl \
  --validation_file data/val.jsonl \
  --model_name_or_path HuggingFaceTB/SmolLM2-135M-Instruct \
  --tuning_algo lora --backend mlx \
  --output_dir /tmp/fmtune_mlx --output_model_name smollm2-mlx \
  --run_name mlx-demo --no_autotune
```

`autotune/configs/autotune_mlx.yaml` is the curated preset for this backend.
It mirrors `autotune_mac.yaml` plus one MLX-only key, `mlx_num_layers` — the
number of layers, counted from the top of the model, to fine-tune (`sft`) or
attach LoRA to (`lora`/`qlora`); default `16`, `-1` means all layers.

### Supported tuners

The MLX backend supports exactly three `--tuning_algo` values:

| Tuner | Behavior |
|---|---|
| `sft` | Full fine-tune — unfreezes the top `mlx_num_layers` layers of the MLX model. |
| `lora` | Attaches mlx-lm LoRA layers (bf16 base) to the top `mlx_num_layers` layers. |
| `qlora` | Same LoRA attachment, but the base model is quantized to **MLX's own 4-bit format** first via `mlx_lm.convert`. This is a different code path from the bitsandbytes NF4 quantization used by the `torch` backend — the QLoRA-on-MPS rejection in the [Support Matrix](#support-matrix) above applies to `torch`, not to `mlx`. |

Unlike the `torch` backend, which masks the prompt and computes loss on the
completion only (see `autotune/utils.py::tokenize_batch`), the MLX backend
currently trains on the full sequence (prompt + completion) — a fallback
forced by a `mask_prompt=True` bug in mlx-lm 0.29.1's `CompletionsDataset`.

Anything else (`alora`, `loha`, `lokr`, `vera`, any `--rl_algo`, or
`num_gpus_per_trial > 1`) is rejected at startup with a clear error — the
MLX backend is single-device, SFT/LoRA/QLoRA only.

### Model cache

The first run against a given HF model converts it to MLX format
(`mlx_lm.convert`) and caches the result under `~/.cache/fmtune/mlx/`
(override with `FMTUNE_MLX_CACHE`), keyed by `<model-stem>__<q4|bf16>` so
the bf16 (`lora`) and 4-bit (`qlora`) conversions of the same model don't
collide. Conversion writes to a temp directory and atomically renames it
into place, so an interrupted conversion never leaves a half-written cache
entry; later runs against the same model + quantization reuse the cached
copy instead of reconverting.

### Output format

The MLX backend saves an **MLX-native artifact**, not a PEFT/transformers
one, under `{output_dir}/models/{output_model_name}/`:

- `lora` / `qlora` → a LoRA adapter (`adapters.safetensors` +
  `adapter_config.json`).
- `sft` → full MLX model weights (`model.safetensors`).

Neither is loadable by `transformers` or `peft` — they're mlx-lm artifacts,
usable only via `mlx_lm.load()` and mlx-lm's own LoRA-fusing tools. This is
a deliberate tradeoff: MLX training is faster and lower-memory on Apple
Silicon than PyTorch/MPS, in exchange for a separate, non-HF inference path.
If you need a PEFT-compatible adapter, use the default `torch` backend
documented above instead.
