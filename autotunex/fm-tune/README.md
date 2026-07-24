![Static Badge](https://img.shields.io/badge/build-passing-brightgreen?style=flat)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
![Static Badge](https://img.shields.io/badge/version-0.7.1-red?style=flat)

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

- Python 3.10+ (3.12 recommended)
- CUDA 12.x compatible GPU(s)
- Linux for training (flash-attn is Linux-only; macOS works for development without flash-attn)

### With `uv` (recommended)

[uv](https://docs.astral.sh/uv/) is the recommended installer. It resolves the flash-attn pre-built wheel automatically via `[tool.uv.sources]` in `pyproject.toml` — no manual download needed.

`autotune` is vendored in the [`granite.build`](https://github.com/ibm-granite/granite.build) monorepo at `autotunex/fm-tune/`. Clone the monorepo, then work from that directory:

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
| bitsandbytes | >= 0.48.0 | 4-bit (NF4) base quantization for QLoRA |
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

Editor integration: if you use VSCode, install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff). The repo ships a `.vscode/settings.json` that enables format-on-save and import sorting out of the box.

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
3. Train the model with the best config on all available GPUs
4. Save the LoRA adapter to `./output/models/granite-4.0-lora/`

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

## Running on LSF Clusters (multi-node)

For LSF-managed environments such as IBM CCC, AutoTune ships an LSF launcher
under `autotune/lsf/` that stands up a multi-node Ray cluster on demand inside
a **single** multi-host `bsub` allocation. The driver runs on the first
allocated host (`host[0]`) and uses `blaunch` to start a Ray worker on each of
the other allocated hosts; a worker also colocates with the Ray head on
`host[0]`, so every GPU in the allocation is usable. RDMA over InfiniBand is
enabled by default (`NCCL_NET_GDR_LEVEL=PXB` with `mlx5_0`), so inter-node
NCCL collectives use GPU-Direct RDMA when the topology supports it.

Bringing the whole cluster up inside one allocation means a single queue wait
covering every host — no partial-schedule waste from N independent worker
`bsub` jobs racing the queue.

### How it works

1. You submit `main.py` via `bsub` inside a multi-host allocation
   (`-R "span[hosts=N]"`), passing `--num_nodes N`, `--gpus_per_node`, and
   `--conda_env`. The driver lands on `host[0]`.
2. `ray_up_blaunch` (called from `main.py`) reads the LSF host list, starts the
   Ray head on `host[0]` (CPU-only), starts a worker on `host[0]` directly
   (no blaunch needed — registers that host's GPUs), and `blaunch`es a worker
   on each of the other `N-1` hosts. Each worker carries the full RDMA
   environment, activates the supplied conda env, and runs
   `python -m autotune.lsf.worker_entry` to attach to the head.
3. `ray_up_blaunch` waits until Ray sees all expected GPUs and worker nodes,
   failing fast if any worker process exits during bring-up.
4. The HPO driver runs as usual; trials land on the GPU workers.
5. On exit (success or failure), `ray_down` drains the remote-host cgroups via
   a remote `ray stop`, shuts the head down, SIGTERMs the local blaunch /
   colocated-worker process groups, and finally `bkill`s its own job so LSF
   reaps the allocation's cgroup on every host.

Per-host worker logs land at `<output_dir>/logs/<job_group>/<host>.log` (one
file per hostname, including the head host's own colocated worker, where
`<job_group>` is the auto-generated `ray_nodes_<timestamp>`). The driver's
stdout prints the absolute path of each worker log on spawn, so you can
`tail -f` while the cluster is coming up. Passing `--cleanup` wipes
`<output_dir>/logs/` at the end of the run along with `ray_results` and
friends — omit `--cleanup` if you want to keep these for forensics.

### Defaults baked into the LSF launcher

The following are the assumptions for IBM CCC + A100 80GB; override them by
calling `start_multinode_ray_cluster_blaunch(...)` directly from Python (or by
adjusting the outer `bsub` resource requests) if your fleet differs:

| Setting | Default |
|---|---|
| GPU model | `NVIDIAA100_SXM4_80GB` |
| Cores per worker | 32 |
| `NCCL_IB_HCA` | `mlx5_0` |
| Worker wait timeout | 1800 s |

The LSF queue, user group, per-host GPU mode, and memory are set on the
**outer** `bsub` line (see the example below), not by the launcher.

### Example: 2 GPU nodes × 4 GPUs each (8 GPUs total)

Submit `main.py` inside a single multi-host LSF allocation. The launcher then
`blaunch`es workers onto the other hosts and wires them into the cluster:

```bash
# 2 GPU nodes × 4 GPUs each (8 GPUs total) — single allocation.
NUM_NODES=2  # number of GPU worker nodes (== total hosts in the allocation)
bsub -q normal -U infusion \
     -n $((32 * NUM_NODES)) \
     -gpu "num=4:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" \
     -R "span[ptile=32]" -R "span[hosts=$NUM_NODES]" -R "rusage[mem=640G]" \
     -o driver.log \
     bash -lc 'source ~/.bashrc && conda activate /u/<your-user>/storage/envs/autotune && \
       python main.py \
         --num_nodes 2 --gpus_per_node 4 \
         --conda_env /u/<your-user>/storage/envs/autotune \
         --config_file autotune/configs/autotune.yaml \
         --train_file data/train.jsonl --validation_file data/val.jsonl \
         --model_name_or_path ibm-granite/granite-4.0-micro \
         --tuning_algo lora --output_dir ./output \
         --output_model_name granite-lora --run_name multinode_lora \
         --save_history --cleanup'
```

Notes:

- The outer `bsub` requests **`--num_nodes` hosts** (not `+1`); the head and
  one worker share `host[0]`. Set `-R "span[hosts=N]"` so the scheduler is
  forced to spread the allocation across N distinct hosts (otherwise it may
  pack all slots onto fewer hosts and `_partition_hosts` will reject the
  layout).
- `--num_nodes 2 --gpus_per_node 4` gives the cluster 8 GPUs total (2 hosts ×
  4 GPUs each). The Ray head registers 0 GPUs; the worker colocated on
  `host[0]` registers `host[0]`'s GPUs.
- `--conda_env` is required; without it, `main.py` falls back to a single-node
  local Ray cluster (useful for dev runs on a single GPU `bsub`).
- `bjobs` shows exactly **one** job id; per-worker lifecycle is via
  `subprocess.Popen` (local) and `blaunch`'d processes (remote). When the
  outer driver job exits or is `bkill`ed, LSF's RES tears down the blaunched
  workers automatically.
- The Ray head process and the colocated worker compete for cores on
  `host[0]`. With the default 32 cores/worker on a 40+ core CCC node this
  is fine; for very heavy driver-side preprocessing, raise `cores_per_worker`.
- For online RL (`--rl_algo` ∈ `{ppo, grpo, dapo}`), the GPU mode and
  host-exclusivity are decided by the **outer** `bsub` (one switch covers
  all hosts). Use `-gpu "num=4:mode=shared:gmodel=..."` and add `-x` so the
  whole allocation owns its hosts.
- Teardown always ends with `bkill $LSB_JOBID` after clean Ray teardown to
  force LSF to reap the job's cgroup on every allocated host. This makes the
  job exit `EXIT` (not `DONE`) — cosmetic, since all useful work is on disk
  before this step runs.

### Choosing the InfiniBand fabric (`--fleet`)

The launcher hardcodes a per-fleet default for `NCCL_IB_HCA` (the list of IB
HCAs NCCL uses for inter-node collectives). Pick the right `--fleet` for your
deployment:

| `--fleet` | Default `NCCL_IB_HCA` | Notes |
|---|---|---|
| `ccc-a100` (default) | `mlx5_0` | CCC A100 nodes have one healthy rail; `mlx5_1` is Down on several boxes. |
| `bv-h100` | `mlx5_0,mlx5_1,...,mlx5_7` | BlueVela H100 nodes have 8 compute rails (one per GPU on the same PCIe complex), plus `mlx5_8`/`mlx5_9` for storage/management which are intentionally excluded. |

To override per-run (e.g. ad-hoc benchmarking), pass `--ib_hca "mlx5_0,mlx5_1"`.

#### Example: BlueVela 2 × H100 nodes × 8 GPUs

```bash
NUM_NODES=2  # number of GPU worker nodes (== total hosts)
bsub -q production -J multinode -o logs/driver.log \
     -n $((96 * NUM_NODES)) \
     -gpu "num=8:mode=exclusive_process:gmodel=NVIDIAH10080GBHBM3" \
     -R "span[ptile=96] rusage[mem=1280G]" \
     bash -lc 'source ~/.bashrc && conda activate /u/<your-user>/storage/envs/autotune && \
       python main.py --fleet bv-h100 \
         --num_nodes 2 --gpus_per_node 8 \
         --conda_env /u/<your-user>/storage/envs/autotune \
         --config_file autotune/configs/autotune.yaml \
         --train_file data/train.jsonl --validation_file data/val.jsonl \
         --model_name_or_path ibm-granite/granite-4.0-micro \
         --tuning_algo lora --output_dir ./output \
         --output_model_name granite-lora --run_name multinode_bv_h100 \
         --save_history --cleanup'
```

To verify NCCL is actually using all 8 rails, set `NCCL_DEBUG=INFO` for one
run and grep `<output_dir>/logs/<job_group>/<host>.log` for `NCCL INFO NET/IB`. Expect 8
lines per host of the form `NCCL INFO NET/IB : Using [0]mlx5_0:1/IB
[1]mlx5_1:1/IB ... [7]mlx5_7:1/IB`. If only one shows up, NCCL didn't accept
the rail list — most often a `NCCL_IB_GID_INDEX` mismatch on RoCE-style
deployments.

### Single-node fallback

For local development without LSF, omit `--conda_env` (and either omit
`--num_nodes` or set it to `1`). `main.py` will start a local Ray cluster on
the current host — exactly the behaviour shown under Quick Start.

## Training Paradigms

### 1. Supervised Fine-Tuning (SFT) and Parameter-Efficient Methods

Train a model to follow instructions or perform specific tasks using input/output pairs.

**Supported methods** (`--tuning_type`):

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

**Dataset format:** `input` / `output` columns with optional chat-template, `documents`, and `tools` support.

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

**Dataset format:** `prompt` / `chosen` / `rejected` (DPO) or `prompt` / `completion` / `label` (KTO). Both plain-string and conversational (chat-messages) forms are supported.

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

**Dataset format:** Parquet with `data_source` / `prompt` (messages list) / `reward_model` columns.

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

AutoTune accepts different dataset schemas depending on the training paradigm. The schema for each is summarized below; file formats, auto-detection rules, and worked examples appear in the per-paradigm sections above.

- **SFT / LoRA / aLoRA / LoHa / LoKr / VeRA** — `input` / `output` schema with optional chat-template, `documents`, and `tools` columns.
- **Offline RL — DPO / KTO** — `prompt` / `chosen` / `rejected` (DPO) or `prompt` / `completion` / `label` (KTO); plain-string and conversational forms.
- **Online RL — PPO / GRPO / DAPO (verl)** — parquet with `data_source` / `prompt` / `reward_model` columns and a companion `compute_score` reward function.

## Distributed Training Backends

AutoTune supports DeepSpeed as the distributed training backends:

### DeepSpeed for SFT/DPO Training

Set in the YAML config:
```yaml
training_config:
  ds_strategy: "zero3_cpu"  # zero1_gpu, zero2_gpu, zero2_cpu, zero3_cpu
```

- **Zero1**: Shards optimizer states across GPUs
- **Zero2**: Shards optimizer + gradients
- **Zero3 + CPU offload**: Shards everything + offloads to CPU (for large models)

### Online RL (verl)

Online RL always uses FSDP internally via [verl](https://github.com/volcengine/verl) with vLLM for rollout generation. No separate backend configuration needed.

## Configuration

The YAML configuration file controls the HPO search space and training parameters. It has four main sections:

### `tune_config` — HPO Settings

```yaml
tune_config:
  search_alg: "lds"           # lds, hyperopt, random, bohb, blds
  scheduler: "fifo"           # fifo, hyperband
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
  ds_strategy: "zero1_gpu"      # zero1_gpu, zero2_gpu, zero2_cpu, zero3_cpu
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

HPO control:
  --no_autotune              Skip HPO, train with each param's default
  --resume_from_checkpoint   Resume the final training round from the last
                             checkpoint under {output_dir}/final_checkpoints/.
                             When a saved final_config.json + checkpoint exist
                             there, HPO is skipped and the saved config drives
                             the resumed run; otherwise warns and runs fresh.
  --cleanup                  Remove the ray_results/ folder after training
  --save_history             Write HPO trial history to
                             {output_dir}/results/{run_name}_trials.csv
  --seed N                   Random seed (default: 42)

Dataset (FSDP driver):
  --data_backend {arrow,ray_data}
                             How datasets are loaded and tokenized. 'arrow'
                             (default) tokenizes once on the driver and has
                             each worker mmap the result. 'ray_data' runs a
                             distributed Ray Data pipeline and auto-shards
                             across workers.

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
  --num_nodes N              Number of GPU hosts in the multi-node Ray cluster
                             (== hosts in the LSF allocation; the head colocates
                             with a worker on host[0]). Combined with
                             --conda_env, triggers the LSF multi-node launch
                             (default: 1)
  --gpus_per_node N          GPUs per host when --num_nodes is set
                             (default: 1)
  --conda_env PATH           Conda env path workers must activate before
                             starting Ray. Required to enable the multi-node
                             LSF launch path; without it, the run uses a
                             local Ray cluster on the current host
  --fleet {ccc-a100,bv-h100}
                             Deployment fleet. Selects the GPU model and the
                             per-fleet default for NCCL_IB_HCA: ccc-a100 (A100)
                             uses single rail mlx5_0; bv-h100 (H100) uses 8
                             compute rails (mlx5_0..mlx5_7).
                             See "Choosing the InfiniBand fabric" above
  --ib_hca STR               Explicit override for NCCL_IB_HCA, e.g.
                             "mlx5_0,mlx5_1". Takes precedence over --fleet
  --ib_ifname STR            Optional override for NCCL_SOCKET_IFNAME (the
                             TCP iface NCCL uses for bootstrap). Default
                             unset lets NCCL auto-pick a routable interface

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
  {output_model_name}/            # Final fine-tuned model or adapter — persistent
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
- The model goes directly under `{output_dir}/{output_model_name}/`, not
  under a `models/` subfolder.
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
    lds.py                          # Limited Discrepancy Search sampler
    logging_setup.py                # Root logger configuration
    optimizer.py                    # AutotuneOptimizer: Ray Tune HPO orchestration
    pipeline.py                     # AutotunePipeline: tuning + RL algo validation
    utils.py                        # Tokenization, model loading, DP audit helper

    callbacks/                      # Trainer callbacks (logging, tuner events)
      logging_service.py
      print_logger.py
      tuner_callback.py

    lsf/                            # LSF multi-node Ray launcher (CCC)
      ray_up_blaunch.py             # Single allocation: head + blaunch'd workers, head colocates with a worker
      ray_down.py                   # Tear down: remote ray stop + ray.shutdown + kill blaunch pids + self-bkill
      worker_entry.py               # Worker process: ray start --address ..., signal.pause
      log_utils.py                  # Banners, phase timers, env fingerprint helpers

    configs/                        # Example YAML configurations
    rewards/                        # Example reward functions for online RL
    tools/                          # Dataset-building utilities
      build_gsm8k_dataset.py
      build_factuality_dataset.py
      granite_build_wrapper.py

    trainers/
      driver_single.py              # Single-GPU SFT/PEFT
      driver_single_trl.py          # Single-GPU DPO/KTO (TRL)
      driver_multi_hf_ds.py         # Multi-GPU SFT/PEFT + DeepSpeed
      driver_multi_hf_fsdp.py       # Multi-GPU SFT/PEFT + FSDP
      driver_multi_trl_ds.py        # Multi-GPU DPO/KTO + DeepSpeed (TRL)
      driver_multi_trl_fsdp.py      # Multi-GPU DPO/KTO + FSDP (TRL)
      driver_multi_verl.py          # Multi-GPU PPO/GRPO/DAPO (verl + vLLM)
      _trl_compat.py                # TRL/verl API-drift compatibility shim

  templates/                        # Cluster job submission templates (LSF, etc.)
  tests/                            # Pytest test suite
  datasets/                         # Example / tiny sample datasets
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

## Contact

Radu Marinescu (radu.marinescu@ie.ibm.com)
Priynashu Rai (priyanshu.rai@ibm.com)
Daniel Karl I. Wiedele (daniel.karl@ibm.com)
