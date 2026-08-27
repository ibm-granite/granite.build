# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install via optional-dependency extras (core, full, dev, mlx) — see pyproject.toml.
# `ray` and `datasets` live ONLY in the core/full extras, so a bare
# `pip install -e .` leaves you WITHOUT Ray. Always install an extra:
uv pip install -e ".[full]"      # recommended — SFT + offline/online RL (verl) + flash-attn (wheel resolved automatically)
pip install -e ".[full]"         # alternative (flash-attn may need manual wheel install)
uv pip install -e ".[core]"      # macOS / CPU dev — excludes deepspeed, verl, flash-attn
uv pip install -e ".[core,mlx]"  # Apple Silicon, MLX backend (--backend mlx)
uv pip install -e ".[dev]"       # ruff + pre-commit

# Lint
ruff check .                 # errors, pyflakes, warnings, imports (E, F, W, I)
ruff check --fix .           # auto-fix

# Tests
pytest                       # runs tests/ directory
pytest tests/test_config.py  # single test file
pytest tests/test_config.py::TestLoadFromYaml::test_load_yaml  # single test function
```

## Architecture Overview

fm-tune is a Ray Tune-based HPO system for distributed LLM fine-tuning. The flow is:

```
main.py (CLI) → AutotuneOptimizer.fit() (HPO) → driver_*.py (training) → tune.report()
                                               ↓
                AutotuneOptimizer.fit_best_config() → driver_*.py (final training + model save)
```

### Core Pipeline

1. **main.py** — CLI entry point. Parses args, starts Ray cluster, creates `AutotuneOptimizer`, calls `fit()` then `fit_best_config()`.

2. **autotune/optimizer.py** — `AutotuneOptimizer` orchestrates HPO via Ray Tune. `setup_pipeline()` builds the param search space from YAML config. `fit()` runs HPO trials (a crashed sweep is NOT restored — rerun the job). `fit_best_config()` retrains with the best (or `--no_autotune` default) hyperparameters and persists the resolved config to `final_checkpoints/final_config.json`. `--resume_from_checkpoint` short-circuits `fit()` when a saved config + checkpoint exist there, loading the saved config and resuming final training from the last checkpoint (see `autotune.utils.has_resumable_final_checkpoint` / `load_final_config`).

3. **autotune/config.py** — `AutotuneConfig` loads YAML into four sections: `tune_config`, `training_config`, `training_rl_config`, `tuners_config`/`tuners_rl_config`.

4. **autotune/pipeline.py** — `AutotunePipeline` validates that tuning_algo + rl_algo combinations are legal.

5. **autotune/constants.py** — Maps tuning algorithm names to PEFT types, defines `AUTOTUNE_OFFLINE_RL` (dpo, kto) and `AUTOTUNE_ONLINE_RL` (ppo, grpo, dapo).

### Driver Selection (optimizer.py)

Drivers are selected based on `num_gpus_per_trial`, `rl_algo`, `train_implementation`, and `--backend`:

| GPUs | RL Algorithm | `train_implementation` | Driver |
|------|-------------|------------------------|--------|
| 1 | none (SFT/PEFT) | — | `driver_single.py` |
| 1 | dpo, kto | — | `driver_single_trl.py` |
| >1 | none (SFT/PEFT) | DeepSpeed | `driver_multi_hf_ds.py` |
| >1 | none (SFT/PEFT) | FSDP | `driver_multi_hf_fsdp.py` |
| >1 | dpo, kto | DeepSpeed | `driver_multi_trl_ds.py` |
| >1 | dpo, kto | FSDP | `driver_multi_trl_fsdp.py` |
| >1 | ppo, grpo, dapo | — (verl+FSDP) | `driver_multi_verl.py` (verl + vLLM) |

For >1 GPU SFT/DPO, `training_config.train_implementation` selects DeepSpeed vs
FSDP (the shipped `autotune*.yaml` configs default to `FSDP`; if the key is
absent the code in `optimizer.py` falls back to DeepSpeed / `"huggingface_ds"`).
The single-device MLX backend (`--backend mlx`) routes to `driver_single_mlx.py`
(see MLX routing below).

**MPS routing:** On Apple silicon, `autotune/device.py::detect_accelerator()` returns `mps` (precedence CUDA → MPS → CPU; override with `FMTUNE_DEVICE=cuda|mps|cpu`). `fit_best_config()` derives `multi_gpu` from `accel.supports_distributed` (`False` on MPS), so final training always routes to `driver_single.py` — never DeepSpeed/FSDP — regardless of `num_gpus_per_trial`. Ray can't schedule Metal as a resource, so HPO trials reserve `{"CPU": 1, "GPU": 0}` bundles and use the MPS device inside the trial process; correctness under concurrency comes from `apply_platform_guards` clamping `max_concurrent_trials` to 1, not from Ray GPU accounting. See `docs/MPS.md` for the full support matrix.

**MLX routing:** `--backend mlx` routes single-device SFT/LoRA/QLoRA to `driver_single_mlx.py` via `autotune/mlx_backend.py`; qlora → MLX 4-bit; output is MLX-native (not PEFT). See `docs/MPS.md`.

### Driver Pattern

All drivers follow the same contract:

```python
def train_driver_*_gpu(config: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Pop sections: training_config, training_rl_config, tuner_flags, tune_config
    # 2. Separate tunable params (tuner_flags[k]=True) from fixed (False)
    # 3. Load model, datasets, build trainer
    # 4. Train
    # 5. Report: tune.report({"loss": ..., "train_loss": ..., "done": True, ...})
```

Multi-GPU drivers (HF/TRL) use Ray Train's `TorchTrainer` wrapper. The verl driver uses `RayPPOTrainer` directly.

### Config System

YAML configs define hyperparameter search spaces. Each param has:
- `strategy` — sampling method (choice, uniform, loguniform)
- `values` — discrete choices or range bounds
- `default` — value used when `--no_autotune`
- `for_tuner` — if true, param is tuned by HPO; if false, fixed to default

### Checkpointing Pattern

- **HPO trials**: `save_strategy="epoch"`, `save_total_limit=1` (minimal, just enough for metrics reporting)
- **Final training**: `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `save_total_limit=3`
- **verl driver**: Custom `_select_best_checkpoint()` using `_InMemoryMetricsLogger`
- After model save, `checkpoint-*` dirs (or `global_step_*` for verl) are cleaned up
- **Final-training resume**: `fit_best_config()` writes the resolved config to `final_checkpoints/final_config.json` before training. On success the whole `final_checkpoints/` dir (config + checkpoints) is removed; an interrupted run leaves both behind so a later `--resume_from_checkpoint` can read the config, skip HPO, and resume from the last `checkpoint-*`.

### verl Integration (driver_multi_verl.py)

The verl driver is the most complex. Key details:

- **Config**: Loads verl defaults via `hydra.compose("ppo_trainer")`, merges fm-tune overrides via `OmegaConf.merge()`. Must call `OmegaConf.set_struct(cfg, False)` first.
- **Colocated GPU pools**: Actor, critic, ref, and vLLM rollout share the same GPUs.
- **Memory optimizations**: `gpu_memory_utilization=0.3`, `enable_sleep_mode=True`, `free_cache_engine=True`, ref model `param_offload=True`, gradient checkpointing always on.
- **Hybrid model detection**: Auto-detects Mamba/SSM architectures and forces `enforce_eager=True`.
- **Worker cleanup**: Must explicitly kill actors AND remove placement groups between HPO trials (`_cleanup_verl_workers`).
- **Reward model**: Must set `reward.reward_model.rollout.name = "vllm"` (mandatory `???` field in verl YAML).

### Key Gotchas

Each entry has a one-line symptom and the fix.

- **Ray Train v2 metrics**: `Result.metrics` is only populated from checkpoint-attached `train.report()` calls (via `RayTrainReportCallback.on_save()`). Standalone `train.report(metrics)` is ignored by `CheckpointManager`. Fix: HPO uses `save_strategy="epoch"` + `save_total_limit=1` so `on_save()` fires at least once.
- **DeepSpeed "auto" scheduler**: The `"auto"` scheduler params in DeepSpeed config break TRL's `prepare_deepspeed()` for DPO/KTO ref model. Fix: omit the `scheduler` block from the DeepSpeed config in TRL drivers (HF Trainer manages the LR scheduler).
- **vLLM max_model_len**: If not set, defaults to model's `max_position_embeddings` (can be 128K+), causing OOM. Always set to `max_prompt_length + max_response_length`.
- **vLLM hybrid models**: `GraniteMoeHybridForCausalLM` and other Mamba/SSM/RWKV architectures are incompatible with CUDA graph capture. Detect via `AutoConfig` and force `enforce_eager=True`. Model itself works in eager mode.
- **verl placement groups**: Survive actor death. Must call `ray.util.remove_placement_group()` explicitly between trials — see "verl Worker Cleanup" below.
- **verl 0.7.1 packaging**: `router/` subpackage under `verl/experimental/reward_loop/` is missing from PyPI for both 0.7.1 and 0.8.0.dev0. Workaround: copy `naive_router.py` and `inner_sglang_router.py` from the GitHub repo into the installed package and add an `__init__.py`.
- **verl reward model `rollout.name`**: verl's default `reward/reward.yaml` has `name: ???` (OmegaConf mandatory). Must set `reward.reward_model.rollout.name = "vllm"` explicitly along with `tensor_model_parallel_size`, `enforce_eager`, `gpu_memory_utilization`, `max_model_len`, `enable_sleep_mode`, `free_cache_engine`.
- **HF `load_best_model_at_end` requirements**: When set to `True`, requires `eval_strategy == save_strategy`, `save_steps % eval_steps == 0` (if `"steps"`), `metric_for_best_model` (e.g. `"eval_loss"`), and `greater_is_better` (`False` for loss). `trainer.save_model()` then saves the best checkpoint, not the last.
- **Ray Data + pandas 3.x**: Pandas 3.0 removed `SettingWithCopyWarning`; Ray Data 2.54's pandas `BlockAccessor` still references it, so any `map_batches(..., batch_format="pandas")` crashes. Fix: use `batch_format="numpy"` (the default) and convert inside the UDF if pandas is needed.
- **Ray Data object-store sizing**: Client-side `ray.init(object_store_memory=...)` does NOT resize an existing remote cluster's object store. Size it on cluster bringup (`--object-store-memory` to `ray start`, or `RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.5` before `ray start`). Local clusters: `autotune/cluster.py::start_local_ray_cluster` already sets 0.5.
- **FSDP NO_SHARD warnings**: Small models on colocated pools may fall back to `NO_SHARD` and emit warnings. Benign — model fits per-GPU without sharding.
- **PlacementGroupCleaner warnings**: `Failed to query Ray Train Controller actor state ...` from `ray.train.v2._internal.execution.controller.placement_group_cleaner` are benign State-API hiccups under load. Suppress by setting that logger to `ERROR`.
- **QLoRA (`--tuning_algo qlora`)**: LoRA on a 4-bit (NF4) bitsandbytes-quantized base. Maps to `PeftType.LORA` (`constants.py`) — same tunable surface as `lora`; the quantized load is triggered purely by `training_config["tuning_algorithm"] == "qlora"` inside the drivers. Drivers build the 4-bit `BitsAndBytesConfig` via `utils.get_qlora_quantization_config()` and call `utils.prepare_qlora_model()` before adapter attachment (HF drivers before `get_peft_model`; TRL drivers before handing the model to the trainer, since TRL's `DPOTrainer` does *not* auto-run `prepare_model_for_kbit_training`, only KTO does). **Incompatible with DeepSpeed ZeRO-3 and FSDP `full_shard`** — the quantized 4-bit params can't be sharded by ZeRO-3 sharded-init / FSDP flat-param sharding; those combos raise a clear `ValueError`. Use ZeRO-1/ZeRO-2, FSDP `SHARD_GRAD_OP`, or the single-GPU driver.
- **Ray object store capped on non-CUDA**: The CUDA path sizes the local Ray object store via `RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.5` (50% of RAM), which would reserve 8 GB on a 16 GB Mac and starve Metal. Fix: on non-CUDA, `autotune/device.py::object_store_bytes()` sizes the object store to a fixed 2 GiB (`2147483648`) instead, overridable via `FMTUNE_OBJECT_STORE_BYTES` (bytes).
- **Silent MPS CPU fallback**: `PYTORCH_ENABLE_MPS_FALLBACK` defaults to `"1"` (set by `configure_runtime_env` on non-CUDA), so an op with no MPS kernel silently runs on CPU instead of crashing — a real slowdown that's easy to miss. Fix: set `PYTORCH_ENABLE_MPS_FALLBACK=0` to make PyTorch raise instead, which pinpoints exactly which op lacks a Metal kernel.
- **`trust_remote_code` opt-out**: Every HF `from_pretrained` (model/tokenizer/config) call passes `trust_remote_code=` via `utils.resolve_trust_remote_code()`, which defaults to `True` — Granite hybrid and some custom architectures ship modeling code in the model repo, so loading executes that Python by default. Operators who only load architectures bundled with `transformers` can harden this by setting `FMTUNE_TRUST_REMOTE_CODE=0` (also accepts `false`/`no`/`off`). It's a single centralized knob; don't re-hardcode `trust_remote_code=True` at new call sites — call the helper.

## Memory & Resource Reference

### verl memory levers (ordered by impact)

| Setting | Location | Effect |
|---------|----------|--------|
| `gpu_memory_utilization` | `rollout.*` | vLLM KV cache fraction. Use `0.3` for colocated pools, `0.25` for large models. |
| `enable_sleep_mode` | `rollout.*` | Release vLLM weights between phases. |
| `free_cache_engine` | `rollout.*` | Release KV cache after generation. |
| `max_model_len` | `rollout.*` | Cap KV cache; set to `max_prompt + max_response`. |
| `enable_gradient_checkpointing` | `model.*` | Recompute activations during backward. fm-tune leaves this on unconditionally. |
| `param_offload` | `fsdp_config.*` | Offload model params to CPU. fm-tune sets this on the ref model. |
| `optimizer_offload` | `fsdp_config.*` | Offload optimizer states to CPU. |
| `reshard_after_forward` | `fsdp_config.*` | `FULL_SHARD` (true) vs `SHARD_GRAD_OP` (false). |
| `enforce_eager` | `rollout.*` | Disable CUDA graphs. Required for hybrid Mamba/SSM models. |
| `enable_activation_offload` | `model.*` | Move activations to CPU during forward. |
| `use_remove_padding`, `tiled_mlp.enabled`, `use_liger`, `entropy_from_logits_with_chunking` | various | Lower-impact knobs. |

### PPO colocated memory budget (3.3B model, 4× A100 80GB, bf16, FSDP)

| Component | Weights | Optimizer | Total per GPU |
|-----------|---------|-----------|----------------|
| Actor (trainable) | ~1.7 GB | ~6.6 GB | ~8.3 GB |
| Critic (trainable) | ~1.7 GB | ~6.6 GB | ~8.3 GB |
| Reference (inference) | ~1.7 GB | — | ~1.7 GB |
| vLLM rollout (inference + KV) | ~1.7 GB + KV | — | varies with `gpu_memory_utilization` |

Top wins (in order): drop `gpu_memory_utilization` 0.5→0.3 (~16 GB/GPU), `enable_sleep_mode + free_cache_engine` (~6.6 GB), gradient checkpointing on (~2–4 GB), ref `param_offload=True` (~1.7 GB).

## verl Worker Cleanup

`RayPPOTrainer` has no shutdown method. Between HPO trials, both worker actors AND placement groups must be torn down explicitly — killing actors alone leaves placement groups holding GPU reservations and the next trial fails with `Total available GPUs 0 is less than total desired GPUs N`.

```python
def _cleanup_verl_workers(trainer):
    # 1. Kill worker actors
    for wg_name in ["actor_rollout_wg", "critic_wg", "ref_policy_wg", "reward_wg"]:
        wg = getattr(trainer, wg_name, None)
        if wg:
            for w in wg._workers or []:
                ray.kill(w)

    # 2. Remove placement groups (the part everyone misses)
    for pool in trainer.resource_pool_manager.resource_pool_dict.values():
        for pg in pool.pgs or []:
            ray.util.remove_placement_group(pg)
        pool.pgs = None

    time.sleep(2)  # let Ray reclaim resources
```

Call from a `finally` block around `trainer.fit()`.

## Data Backend Selection (FSDP driver)

`driver_multi_hf_fsdp.py` selects between two dataset paths via `--data_backend` (default `arrow`):

- **`arrow`** — driver tokenizes once on the head node, writes `{output_dir}/data_cache/{trial_id}/*.arrow`, workers memory-map. HF Trainer's distributed sampler shards rows. Robust, bounded object-store pressure. Use when tokenization is cheap relative to training.
- **`ray_data`** — driver builds `ray.data.Dataset` (`read + repartition + map_batches` tokenize), passes to `TorchTrainer(datasets=...)`, workers consume via `train.get_dataset_shard()`. Tokenization scales across the cluster (all nodes), no Arrow file on disk. Requires cluster-side object-store sizing ≥50% of node RAM (see gotcha above), and `accelerator_config={"dispatch_batches": False}` on `TrainingArguments` to avoid a `SplitCoordinator` deadlock under Accelerate's iterable-dataset dispatch path. Emits a one-time ~30 s `StreamSplit` warning on the first pull. Use when tokenization dominates startup or the dataset is too large to tokenize serially. Both `driver_multi_hf_fsdp.py` and `driver_multi_hf_ds.py` share this path. Concurrency controls (sizing helpers live in `autotune/cluster.py`):
  - **Repartition is the primary fan-out lever.** Ray Data launches at most one stateless map task per input *block*, and a single source file usually reads as 1 block — so the driver calls `repartition(n, shuffle=False)` (a cheap split/combine, no full shuffle, no materialization) to split train/eval into ≈`concurrency` blocks before `map_batches`. Block count is clamped to the dataset row count, so tiny eval sets aren't over-partitioned (skipped entirely when ≤1 row). Without this, tokenization runs single-CPU regardless of cluster size.
  - `ray_data_concurrency` (int): number of parallel `map_batches` tasks. **Default (auto) = `floor(total_cluster_cpus) − num_workers`** (every CPU not reserved by this trial's GPU workers) via `compute_ray_data_sizing()`. Override via the `--ray_data_concurrency` CLI flag or the YAML training-config key.
  - `ray_data_num_cpus` (float, default 1.0): logical CPU budget per task; fractional values (e.g. 0.5) allow more concurrent tasks per physical CPU. Override via `--ray_data_num_cpus` or YAML.
  - **Concurrent HPO caveat:** each trial computes the auto concurrency from the full cluster CPU count, unaware of sibling trials. `max_concurrent_trials` bounds trial count, but for large sweeps set `--ray_data_concurrency` explicitly to avoid cross-trial oversubscription. See `docs/RESOURCES.md`.

## Key Dependencies

torch 2.8.0, transformers 4.57.6, peft 0.18.0, trl 0.29.0, bitsandbytes 0.49.0 (QLoRA 4-bit), deepspeed 0.18.7, ray 2.54.0 (`full` extra; `core` pins 2.52.1), verl 0.7.1, flash-attn 2.8.1 (Linux only)

## Documentation

- `README.md` — project overview, quick start, CLI reference
- `CONTRIBUTING.md` — contributor workflow, testing, PR process
- `docs/RESOURCES.md` — GPU sizing guide for 3B/8B/30B on 8× A100
- `docs/MPS.md` — Apple Silicon (MPS) support: matrix, memory table, troubleshooting
- `docs/dataset-sft.md`, `docs/dataset-offline-rl.md`, `docs/dataset-online-rl.md` — dataset format references
