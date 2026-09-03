[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
![Static Badge](https://img.shields.io/badge/version-0.8.0-red?style=flat)

# AutoTune: Distributed Fine-Tuning for Foundation Models

AutoTune is a complete distributed training stack for fine-tuning large language models. Built on [Ray](https://www.ray.io/), it automates hyperparameter optimization across multiple GPUs and nodes, supporting three training paradigms:

- **Supervised Fine-Tuning (SFT)** with full model training or parameter-efficient methods (e.g., LoRA, QLoRA, aLoRA)
- **Offline Preference Alignment** with DPO (Direct Preference Optimization) and KTO (Kahneman-Tversky Optimization)
- **Online Reinforcement Learning** with PPO (Proximal Policy Optimization), GRPO (Group Relative Policy Optimization), and DAPO

AutoTune handles the full pipeline: hyperparameter search, distributed training across GPUs with DeepSpeed or FSDP, and model saving. It works with any HuggingFace-compatible decoder-only (causal) model.

## Key Features

- **Automated HPO** with Limited Discrepancy Search, HyperOpt, and Random Search
- **Multi-GPU training** via DeepSpeed (Zero1/2/3) or FSDP, scaled with Ray's TorchTrainer
- **Online RL** via [verl](https://github.com/volcengine/verl) with vLLM for fast rollout generation
- **BF16 mixed precision** support
- **Flash Attention 2** attention backend

## Installation

### Prerequisites

- Python 3.12+ (3.12 recommended)
- CUDA 12.x compatible GPU(s)
- Linux for training (flash-attn is Linux-only; macOS works for development without flash-attn)

### With `uv` (recommended)

[uv](https://docs.astral.sh/uv/) is the recommended installer. It resolves the flash-attn pre-built wheel automatically via `[tool.uv.sources]` in `pyproject.toml` — no manual download needed.

```bash
git clone https://github.com/ibm-granite/granite.build.git
cd granite.build/autotunex/fm-tune

# Create and activate a Python 3.12 virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install the autotune package and all dependencies
uv pip install -e ".[full]"
```

All dependencies — including SFT, offline RL (DPO/KTO), online RL (PPO/GRPO/DAPO via verl), and flash-attn — are installed in a single step.

### With `pip` and `conda`

```bash
git clone https://github.com/ibm-granite/granite.build.git
cd granite.build/autotunex/fm-tune

# Create and activate a Python 3.12 virtual environment
conda create -n autotune python=3.12
conda activate autotune

# Install the autotune package and all dependencies
pip install -e ".[full]"
```

> **Note:** pip may attempt to build flash-attn from source, which requires CUDA toolkit headers and can take a long time. If the install fails on flash-attn, install it manually from the pre-built wheel:
>
> ```bash
> wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
> pip install --no-cache-dir flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
> ```
>
> On macOS, flash-attn is automatically skipped (the dependency has a `sys_platform == 'linux'` marker).

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.8.0 | PyTorch backend |
| transformers | 4.57.6 | Model loading and training |
| peft | 0.18.0 | LoRA, QLoRA, LoHA, LoKr, VeRA, aLoRA adapters |
| bitsandbytes | 0.49.0 | 4-bit (NF4) base quantization for QLoRA |
| trl | 0.29.0 | DPO, KTO offline RL trainers |
| deepspeed | 0.18.7 | ZeRO-1/2/3 distributed training |
| ray[tune,default] | 2.54.0 | HPO and distributed orchestration |
| verl[vllm] | 0.7.1 | Online RL (PPO, GRPO, DAPO) with vLLM rollout |
| flash-attn | 2.8.1 | Flash Attention 2 (Linux only) |
| accelerate | >= 1.10.1 | HuggingFace Accelerate |

### Development: linting and formatting

AutoTune uses [ruff](https://docs.astral.sh/ruff/) for both linting and code formatting. Install the dev dependencies and (optionally) the pre-commit hooks:

```bash
uv pip install -e ".[dev]"   # or: pip install -e ".[dev]"
pre-commit install            # optional: auto-run ruff on every git commit
```

Common commands:

```bash
ruff check .          # lint
ruff check --fix .    # lint + auto-fix
ruff format .         # apply formatting
ruff format --check . # verify formatting without changes (used in CI)
```

Editor integration: if you use VSCode, install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) and enable format-on-save and import sorting in your own `.vscode/settings.json` (the `.vscode/` directory is git-ignored, so it stays local to your checkout).

## Quick Start

### SFT with LoRA on a single command

```bash
python main.py \
  --config_file autotune/configs/autotune.yaml \
  --train_file data/train.jsonl \
  --validation_file data/val.jsonl \
  --model_name_or_path ibm-granite/granite-4.0-micro \
  --tuning_algo lora \
  --output_dir ./output \
  --output_model_name granite-4.0-lora \
  --run_name my_experiment \
  --cleanup --save_history
```

This will:
1. Launch a local Ray cluster
2. Run HPO to find the best LoRA hyperparameters
3. Train the model with the best config on the available accelerator(s)
4. Save the LoRA adapter under `./output/` — see [Output Structure](#output-structure) for the exact path

### QLoRA (4-bit) to fit larger models

Swap `--tuning_algo lora` for `--tuning_algo qlora` to train the same LoRA
adapter on a 4-bit (NF4) quantized base, cutting GPU memory substantially:

```bash
python main.py \
  --config_file autotune/configs/autotune.yaml \
  --train_file data/train.jsonl \
  --validation_file data/val.jsonl \
  --model_name_or_path ibm-granite/granite-4.0-micro \
  --tuning_algo qlora \
  --output_dir ./output \
  --output_model_name granite-4.0-qlora \
  --run_name my_qlora_experiment \
  --cleanup --save_history
```

QLoRA shares LoRA's tunable hyperparameters (`r`, `alpha_ratio`, `lora_dropout`, …).
On multiple GPUs it runs under DeepSpeed ZeRO-1/ZeRO-2 or FSDP `SHARD_GRAD_OP`, and
supports DPO/KTO via the TRL drivers — but **not** ZeRO-3 or FSDP `full_shard`
(see the QLoRA note under [Supervised Fine-Tuning](#1-supervised-fine-tuning-sft-and-parameter-efficient-methods)).

### Skip HPO and train with defaults

```bash
python main.py \
  --config_file autotune/configs/autotune.yaml \
  --train_file data/train.jsonl \
  --validation_file data/val.jsonl \
  --model_name_or_path ibm-granite/granite-4.0-micro \
  --tuning_algo lora \
  --output_dir ./output \
  --output_model_name granite-4.0-lora \
  --run_name my_experiment \
  --no_autotune --cleanup
```

## Apple Silicon (MPS)

fm-tune runs SFT and PEFT (LoRA, aLoRA, LoHa, LoKr, VeRA) on a single Apple
silicon Mac via PyTorch's Metal (MPS) backend — the same HPO-then-final-train
pipeline as the CUDA path, just on one Metal device instead of one GPU.
QLoRA, RL (offline and online), multi-GPU, and DeepSpeed/FSDP are **not**
supported on MPS.

```bash
uv pip install -e ".[core]"   # deepspeed is skipped on macOS via its platform marker; verl/flash-attn live only in the [full] extra

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

There is also an opt-in **MLX** backend (`--backend mlx`) that trains natively
on Apple's MLX instead of PyTorch/MPS — faster and lower-memory for
single-device `sft`/`lora`/`qlora`, with MLX-native output. See
[`docs/MPS.md`](docs/MPS.md) for the full support matrix, the MLX backend,
memory guidance, and troubleshooting.

## Training Paradigms

### 1. Supervised Fine-Tuning (SFT) and Parameter-Efficient Methods

Train a model to follow instructions or perform specific tasks using input/output pairs.

**Supported methods** (`--tuning_algo`):

| Method | Flag | Description |
|--------|------|-------------|
| Full SFT | `sft` | Update all model weights |
| LoRA | `lora` | Low-Rank Adaptation |
| QLoRA | `qlora` | LoRA on a 4-bit (NF4) bitsandbytes-quantized base |
| aLoRA | `alora` | Activated LoRA |
| LoHa | `loha` | Low-Rank Hadamard Product |
| LoKr | `lokr` | Low-Rank Kronecker Product |
| VeRA | `vera` | Vector-based Random Matrix Adaptation |

> **QLoRA note:** `qlora` behaves like `lora` (same tunable hyperparameters) but
> loads the frozen base in 4-bit NF4 to cut GPU memory. It runs on the single-GPU,
> DeepSpeed, FSDP, and TRL (DPO/KTO) drivers. It is **not** compatible with
> DeepSpeed ZeRO-3 or FSDP `full_shard` (the drivers raise a clear error) — use
> ZeRO-1/ZeRO-2, FSDP `SHARD_GRAD_OP`, or a single GPU.

**Dataset format:** `input` / `output` columns with optional chat-template, `documents`, and `tools` support. See [`docs/dataset-sft.md`](docs/dataset-sft.md) for the full spec.

**Plain prompt / completion:**
```json
{"input": "Summarize the following article: ...", "output": "The article discusses..."}
```

**Chat messages:**
```json
{"input": [{"role": "user", "content": "Explain quantum tunneling."}], "output": "Quantum tunneling is..."}
```

**Chat messages with documents (RAG) and tools:**
```json
{
  "input": [{"role": "user", "content": "What does the report say about Q3?"}],
  "documents": [{"title": "2025 Q3 Report", "text": "Revenue grew 12%..."}],
  "tools": [{"type": "function", "function": {"name": "search_kb", "parameters": {}}}],
  "output": "The Q3 report states that revenue grew 12%..."
}
```

**Example** (LoRA with DeepSpeed on 4 GPUs):
```bash
python main.py \
  --config_file autotune/configs/autotune.yaml \
  --train_file data/train.jsonl \
  --validation_file data/val.jsonl \
  --model_name_or_path ibm-granite/granite-4.0-micro \
  --tuning_algo lora \
  --output_dir ./output \
  --output_model_name granite-lora \
  --run_name sft_experiment \
  --no_autotune --cleanup
```

### 2. Offline Preference Alignment (DPO / KTO)

Align a model with human preferences using paired or binary preference data. No reward model needed.

**Supported methods** (`--rl_algo`):

| Method | Flag | Description |
|--------|------|-------------|
| DPO | `dpo` | Direct Preference Optimization — learns from chosen/rejected pairs |
| KTO | `kto` | Kahneman-Tversky Optimization — learns from binary good/bad labels |

**Dataset format:** `prompt` / `chosen` / `rejected` (DPO) or `prompt` / `completion` / `label` (KTO). Both plain-string and conversational (chat-messages) forms are supported. See [`docs/dataset-offline-rl.md`](docs/dataset-offline-rl.md) for the full spec.

**DPO dataset format** (JSONL):
```json
{"prompt": "Write a poem about...", "chosen": "Roses are red...", "rejected": "Here is poem..."}
```

**KTO dataset format** (JSONL):
```json
{"prompt": "Explain quantum...", "completion": "Quantum mechanics is...", "label": true}
{"prompt": "Explain quantum...", "completion": "I don't know.", "label": false}
```

**Example** (DPO with LoRA):
```bash
python main.py \
  --config_file autotune/configs/autotune.yaml \
  --train_file data/dpo_train.jsonl \
  --validation_file data/dpo_val.jsonl \
  --model_name_or_path ibm-granite/granite-4.0-micro \
  --tuning_algo lora \
  --rl_algo dpo \
  --output_dir ./output \
  --output_model_name granite-dpo \
  --run_name dpo_experiment \
  --no_autotune --cleanup
```

### 3. Online Reinforcement Learning (PPO / GRPO / DAPO)

Train a model with online RL using a reward function or reward model. The model generates responses, receives rewards, and updates its policy.

**Supported methods** (`--rl_algo`):

| Method | Flag | Description |
|--------|------|-------------|
| PPO | `ppo` | Proximal Policy Optimization with a critic (value) model |
| GRPO | `grpo` | Group Relative Policy Optimization — no critic needed |
| DAPO | `dapo` | GRPO variant with overlong penalty and group filtering |

Online RL requires:
- A **reward function** (Python file with a `compute_score` function) or a **reward model**
- The `--tuning_algo none` flag (online RL trains the full model)
- Multiple GPUs (verl manages distributed workers internally)

**Reward function example** (`reward.py`):
```python
def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    """Return a scalar reward for a single response."""
    if solution_str.strip() == ground_truth.strip():
        return 1.0
    return -1.0
```

**Dataset format:** Parquet with `data_source` / `prompt` (messages list) / `reward_model` columns. See [`docs/dataset-online-rl.md`](docs/dataset-online-rl.md) for the full spec, reward-function wiring, and multi-task dispatch.

**Row shape** (one parquet row):
```json
{
  "data_source": "gsm8k",
  "prompt": [
    {"role": "system", "content": "Solve step-by-step; final answer after ####."},
    {"role": "user", "content": "Janet's ducks lay 16 eggs per day..."}
  ],
  "reward_model": {"style": "rule", "ground_truth": "18"}
}
```

**Example** (GRPO with custom reward on 4 GPUs):
```bash
python main.py \
  --config_file autotune/configs/autotune.yaml \
  --train_file data/rl_train.parquet \
  --validation_file data/rl_val.parquet \
  --model_name_or_path ibm-granite/granite-4.0-micro \
  --tuning_algo none \
  --rl_algo grpo \
  --output_dir ./output \
  --output_model_name granite-grpo \
  --run_name grpo_experiment \
  --no_autotune --cleanup
```

## Dataset Preparation

AutoTune accepts different dataset schemas depending on the training paradigm. File formats, required and optional columns, auto-detection rules, and worked examples are covered in dedicated docs:

- [SFT / LoRA / aLoRA / LoHa / LoKr / VeRA](docs/dataset-sft.md) — `input` / `output` schema with optional chat-template, `documents`, and `tools` columns.
- [Offline RL — DPO / KTO](docs/dataset-offline-rl.md) — `prompt` / `chosen` / `rejected` (DPO) or `prompt` / `completion` / `label` (KTO); plain-string and conversational forms.
- [Online RL — PPO / GRPO / DAPO (verl)](docs/dataset-online-rl.md) — parquet with `data_source` / `prompt` / `reward_model` columns and a companion `compute_score` reward function.

## Distributed Training Backends

For multi-GPU SFT and offline RL (DPO/KTO), AutoTune supports two distributed
backends, selected by `train_implementation` in the YAML `training_config`
(`"DeepSpeed"` or `"FSDP"`).

### DeepSpeed (`train_implementation: DeepSpeed`)

Pick the ZeRO strategy with `ds_strategy`:
```yaml
training_config:
  train_implementation: "DeepSpeed"
  ds_strategy: "zero3_cpu"   # auto, zero1_gpu, zero2_gpu, zero3_gpu, zero2_cpu, zero3_cpu
```

- **ZeRO-1**: shards optimizer states across GPUs
- **ZeRO-2**: shards optimizer states + gradients
- **ZeRO-3 (+ CPU offload)**: shards optimizer states, gradients, and parameters; the `*_cpu` variants also offload to CPU for large models

### FSDP (`train_implementation: FSDP`)

Pick the sharding strategy with `fsdp_strategy`:
```yaml
training_config:
  train_implementation: "FSDP"
  fsdp_strategy: "full_shard"  # auto, no_shard, shard_grad_op, full_shard, hybrid_shard
```

> **QLoRA note:** QLoRA (4-bit base) cannot be sharded by DeepSpeed ZeRO-3 or
> FSDP `full_shard` (the drivers raise a clear error). Use ZeRO-1/ZeRO-2, FSDP
> `shard_grad_op`, or a single GPU.

### Online RL (verl)

Online RL always uses FSDP internally via [verl](https://github.com/volcengine/verl) with vLLM for rollout generation. No separate backend configuration needed.

## Recommended Resources

For GPU sizing, DeepSpeed ZeRO-strategy selection, per-configuration memory estimates, training-time estimates across model sizes (350M–13B+), and a strategy-selection decision tree, see [`docs/RESOURCES.md`](docs/RESOURCES.md).

## Configuration

The YAML configuration file controls the HPO search space and training parameters. It has four main sections:

### `tune_config` — HPO Settings

```yaml
tune_config:
  search_alg: "lds"           # lds, hyperopt, random, bohb, blds
  scheduler: "fifo"           # fifo, asha, hyperbandforbohb
  num_samples: 8              # number of HPO trials
  max_concurrent_trials: 1    # parallel trials
  max_discrepancy: 6          # LDS-specific parameter
  time_budget_s: 3600         # time limit in seconds (optional)
```

### `training_config` — Training Parameters

```yaml
training_config:
  num_train_epochs: 10
  hpo_num_epochs: 1             # fewer epochs during HPO
  hpo_dataset_percentage: 0.10  # use 10% of data during HPO
  seed: 42
  num_gpus_per_trial: 4
  use_flash_attention: "flash_attention_2"  # or "eager"
  train_implementation: "FSDP"  # FSDP or DeepSpeed (multi-GPU SFT/DPO)
  ds_strategy: "zero1_gpu"      # DeepSpeed: auto, zero1_gpu, zero2_gpu, zero3_gpu, zero2_cpu, zero3_cpu
  fsdp_strategy: "full_shard"   # FSDP: auto, no_shard, shard_grad_op, full_shard, hybrid_shard
  max_length: 256
```

### `tuners_config` — Hyperparameter Search Space

Each tuning method has its own search space:

```yaml
tuners_config:
  lora:
    hyperparams:
      r:
        strategy: "choice"
        values: [8, 16, 32, 64]
        default: 32
        type: int
        for_tuner: true
      learning_rate:
        strategy: "choice"
        values: [0.000001, 0.000003, 0.000005]
        default: 0.000001
        type: float
        for_tuner: false
      per_device_train_batch_size:
        strategy: "choice"
        values: [4, 8, 16, 32]
        default: 16
        type: int
        for_tuner: false
```

The `qlora:` section mirrors `lora:` exactly (same hyperparameters) — the only
difference is that the base model is loaded in 4-bit NF4, which is handled by the
driver, not the search space.

### `training_rl_config` — Online RL Parameters

```yaml
training_rl_config:
  reward_function_path: "reward.py"
  reward_function_name: "compute_score"
  reward_model_path: null       # path to a learned reward model (optional)
  max_prompt_length: 1024
  max_response_length: 512
  rollout_temperature: 1.0
  rollout_top_p: 1.0
  rollout_n: 5                  # group size for GRPO
  gpu_memory_utilization: 0.5
  tensor_model_parallel_size: null  # null = auto-detect from model size
```

## CLI Reference

```
python main.py [OPTIONS]

Required:
  --config_file PATH         YAML config defining the hyperparameter space
  --train_file PATH          Training data (JSONL / JSON / CSV; Parquet on
                             multi-GPU SFT and required for online RL)
  --validation_file PATH     Validation data (same formats as --train_file)
  --model_name_or_path ID    HuggingFace model id or local path
  --output_dir DIR           Base output directory
  --output_model_name NAME   Name of the saved model / adapter subdirectory
  --run_name NAME            Experiment name (asserted non-None at startup)

Training:
  --tuning_algo ALGO         Tuning method: sft, lora, qlora, alora, loha,
                             lokr, vera, none (default: none)
  --rl_algo ALGO             RL algorithm: dpo, kto, ppo, grpo, dapo, none
                             (default: none)
  --backend {torch,mlx}      Training backend (default: torch). 'mlx' uses the
                             Apple Silicon MLX backend for single-device
                             sft/lora/qlora (requires the [mlx] extra)

HPO control:
  --no_autotune              Skip HPO, train with each param's default
  --resume_from_checkpoint   Resume the final training round from the last
                             checkpoint under {output_dir}/final_checkpoints/.
                             When a saved final_config.json + checkpoint exist
                             there, HPO is skipped and the saved config drives
                             the resumed run; otherwise warns and runs fresh.
  --keep_checkpoints         Keep intermediate checkpoints and training
                             artifacts (final_checkpoints/, outputs/,
                             train_results/, data_cache/) after final training
                             instead of deleting them. Useful for debugging
  --cleanup                  Remove the ray_results/ folder after training
  --save_history             Write HPO trial history to
                             {output_dir}/results/{run_name}_trials.csv
  --seed N                   Random seed (default: 42)

Dataset (multi-GPU drivers):
  --data_backend {arrow,ray_data}
                             How datasets are loaded and tokenized. 'arrow'
                             tokenizes once on the driver and has each worker
                             mmap the result; 'ray_data' runs a distributed Ray
                             Data pipeline that auto-shards across workers. If
                             omitted, the YAML config value is used (config
                             default: arrow)
  --ray_data_concurrency N   Parallel map_batches tokenize tasks for the
                             'ray_data' backend. Default (auto) = total cluster
                             CPUs minus this trial's GPU workers
  --ray_data_num_cpus F      Logical CPUs per ray_data tokenize task (default
                             1.0); fractional values allow more tasks per CPU

Tokenizer customization:
  --tokenizer_name_or_path PATH
                             Custom tokenizer (defaults to --model_name_or_path)
  --additional_special_tokens TOK [TOK ...]
                             Extra special tokens to register on the tokenizer
  --additional_tokens TOK [TOK ...]
                             Extra regular tokens to add to the vocabulary
  --pad_token TOK            Override the tokenizer's pad token
  --eos_token TOK            Override the tokenizer's eos token
  --bos_token TOK            Override the tokenizer's bos token

Cluster:
  --ray_address HOST:PORT    Attach to a remote Ray head (optional; a local
                             cluster is started otherwise)

AutoTuneX bridge logging (optional, OFF by default):
  --autotunex_server_url URL Base URL of an AutoTuneX bridge server to log this
                             run to. Omit it (the default) to run fully offline
                             with no bridge calls. Bridge errors never fail the
                             run.
  --job_id ID                Optional informational run label. Passed through to
                             the bridge when --autotunex_server_url is set;
                             otherwise unused.
```

## Output Structure

Everything lands under `--output_dir`. Paths marked *(transient)* are created
during training and removed automatically; what remains after a successful
run is the model, the logs, and (if HPO ran) the results files.

```
{output_dir}/
  models/{output_model_name}/     # Final model/adapter (single-device drivers) — persistent
  {output_model_name}/            # Final model/adapter (multi-GPU / TRL / verl drivers) — persistent
  logs/{trial_id}/                # HF Trainer log dirs per trial — persistent
  results/                        # Written only when --save_history is set
    {run_name}_tune.json          # Best hyperparameter configuration
    {run_name}_trials.csv         # Full HPO trial history (flattened)
  ray_results/                    # HPO trial artifacts (transient; --cleanup removes)
  train_results/                  # Ray Train run dir for final training (transient)
  outputs/{trial_id}/             # HF checkpoint-* dirs during training (transient)
  data_cache/{trial_id}/          # Tokenized Arrow files (transient; arrow backend)
```

Notes:
- The final model/adapter path depends on the driver: single-device drivers
  (single-GPU SFT/PEFT and the MLX backend) write to
  `{output_dir}/models/{output_model_name}/`, while the multi-GPU
  (DeepSpeed/FSDP), TRL (DPO/KTO), and verl drivers write directly to
  `{output_dir}/{output_model_name}/`.
- `ray_results/` only exists while HPO is running and is removed by
  `--cleanup`. On `--no_autotune` runs it's never created.
- `train_results/`, `outputs/`, and `data_cache/` are scrubbed by the driver
  after final training completes; if a trial crashes, their contents may
  linger.
- `logs/` is **not** cleaned automatically — delete it manually if you don't
  need the per-trial HF Trainer logs.


## Package Structure

```
fm-tune/
  main.py                           # CLI entry point
  pyproject.toml                    # Package + ruff/uv configuration
  .pre-commit-config.yaml           # ruff check + ruff-format hooks

  autotune/
    cluster.py                      # Local Ray cluster lifecycle + Ray Data context
    config.py                       # YAML config loader + DeepSpeed / FSDP presets
    constants.py                    # Supported methods, PEFT-type mapping, enums
    device.py                       # Accelerator detection (CUDA/MPS/CPU) + platform guards
    lds.py / blds.py                # (Bandit) Limited Discrepancy Search samplers
    optimizer.py                    # AutotuneOptimizer: Ray Tune HPO orchestration
    pipeline.py                     # AutotunePipeline: tuning + RL algo validation
    mlx_backend.py                  # MLX backend config translation + training loop
    validation.py                   # Config / argument validation
    utils.py                        # Tokenization, model loading, checkpoint helpers
    alora_patch.py                  # Activated-LoRA (aLoRA) support
    logging_setup.py                # Root logger configuration

    callbacks/                      # Trainer callbacks (logging, tuner events)
    configs/                        # Example YAML configurations (autotune*.yaml)
    rewards/                        # Example reward functions for online RL
    tools/                          # Dataset-building utilities
      build_gsm8k_dataset.py
      build_factuality_dataset.py
      parquet_to_json.py
    lsf/                            # Optional multi-node Ray launcher (LSF/HPC)

    trainers/
      driver_single.py              # Single-GPU SFT/PEFT
      driver_single_trl.py          # Single-GPU DPO/KTO (TRL)
      driver_single_mlx.py          # Single-device MLX backend (sft/lora/qlora)
      driver_multi_hf_ds.py         # Multi-GPU SFT/PEFT + DeepSpeed
      driver_multi_hf_fsdp.py       # Multi-GPU SFT/PEFT + FSDP
      driver_multi_trl_ds.py        # Multi-GPU DPO/KTO + DeepSpeed (TRL)
      driver_multi_trl_fsdp.py      # Multi-GPU DPO/KTO + FSDP (TRL)
      driver_multi_verl.py          # Multi-GPU PPO/GRPO/DAPO (verl + vLLM)
      _trl_compat.py                # TRL/verl API-drift compatibility shim

  docs/
    dataset-sft.md                  # SFT/PEFT dataset format
    dataset-offline-rl.md           # DPO/KTO dataset format
    dataset-online-rl.md            # verl (PPO/GRPO/DAPO) dataset format
    RESOURCES.md                    # GPU sizing, ZeRO/FSDP strategy selection, memory estimates
    MPS.md                          # Apple Silicon (MPS) + MLX backend support

  tests/                            # Pytest test suite
```

## Citation

If you found the library useful, please cite the following reference:

```
@article{weidele2026aaai, 
  title={AutoTuneX: Interactive Automated Fine-Tuning for Large Language Models}, 
  volume={40}, 
  url={https://ojs.aaai.org/index.php/AAAI/article/view/42391}, 
  DOI={10.1609/aaai.v40i48.42391}, 
  number={48}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  author={Weidele, Daniel Karl I. and Rai, Priyanshu and Araujo, Frederico and Taylor, Teryl and Marinescu, Radu}, 
  year={2026}, 
  month={Mar.}, 
  pages={41715–41717} 
}

@article{kishimoto2022aaai, 
  title={Bandit Limited Discrepancy Search and Application to Machine Learning Pipeline Optimization}, 
  volume={36}, 
  url={https://ojs.aaai.org/index.php/AAAI/article/view/21263}, 
  DOI={10.1609/aaai.v36i9.21263}, 
  number={9}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  author={Kishimoto, Akihiro and Bouneffouf, Djallel and Marinescu, Radu and Ram, Parikshit and Rawat, Ambrish and Wistuba, Martin and Palmes, Paulito and Botea, Adi}, 
  year={2022}, 
  month={Jun.}, 
  pages={10228–10237} 
}
```

## License

AutoTune is released under the [Apache License 2.0](LICENSE).

## Contact

Radu Marinescu (radu.marinescu@ie.ibm.com)
Priyanshu Rai (priyanshu.rai@ibm.com)
Daniel Karl I. Weidele (daniel.karl@ibm.com)
