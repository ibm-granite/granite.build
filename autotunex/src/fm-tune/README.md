[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
![Static Badge](https://img.shields.io/badge/version-0.8.0-red?style=flat)

# AutoTune: Distributed Fine-Tuning for Foundation Models

AutoTune is a complete distributed training stack for fine-tuning large language models. Built on [Ray](https://www.ray.io/), it automates hyperparameter optimization across multiple GPUs and nodes, supporting three training paradigms:

- **Supervised Fine-Tuning (SFT)** with full model training or parameter-efficient methods (e.g., LoRA, QLoRA, aLoRA, Shadow Residual)
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
- NVIDIA GPU(s) with a driver that supports CUDA 13 (>= 580): torch 2.11's PyPI wheels are built for CUDA 13.0
- Linux for training (flash-attn is Linux-only; macOS works for development without flash-attn)

### With `uv` (recommended)

[uv](https://docs.astral.sh/uv/) is the recommended installer. There is no prebuilt flash-attn wheel for torch 2.11 + CUDA 13, so `uv pip install` **compiles flash-attn from source**, configured by `[tool.uv.extra-build-dependencies]` / `[tool.uv.extra-build-variables]` in `pyproject.toml` (builds against the env's own torch, for A100 + H100). Run it on a GPU node, or any host with nvcc 13 under `/usr/local/cuda` or `$CUDA_HOME`; the compile (flash-attn plus the Mamba kernels for hybrid bases) takes ~15 minutes cold on 32 cores and is reused from uv's cache afterwards (set `MAX_JOBS` lower if the host has little RAM).

```bash
git clone https://github.com/ibm-granite/granite.build.git
cd granite.build/autotunex/src/fm-tune

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
cd granite.build/autotunex/src/fm-tune

# Create and activate a Python 3.12 virtual environment
conda create -n autotune python=3.12
conda activate autotune

# flash-attn first: plain pip ignores the [tool.uv] build settings and would build
# it in an isolated env against the *newest* torch, which then fails to import.
# Build it against the env's torch instead, on a host with nvcc 13 (~10 min):
pip install torch==2.11.0 ninja psutil einops packaging setuptools wheel
FLASH_ATTN_CUDA_ARCHS="80;90" FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=16 \
  pip install --no-build-isolation flash-attn==2.8.3
# Same for the Mamba kernels used by hybrid bases (Granite 4.0-H):
MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=16 \
  pip install --no-build-isolation causal-conv1d==1.7.0 mamba-ssm==2.3.1

# Then the autotune package and all other dependencies
pip install -e ".[full]"
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.11.0 | PyTorch backend (CUDA 13.0 wheels) |
| transformers | 5.9.0 | Model loading and training |
| peft | 0.19.1 | LoRA, QLoRA, LoHA, LoKr, VeRA, aLoRA adapters (not 0.21.x: empty adapters under multi-GPU FSDP) |
| bitsandbytes | 0.50.2 | 4-bit (NF4) base quantization for QLoRA |
| trl | 1.12.0 | DPO, KTO offline RL trainers (last release with the value head verl's PPO critic needs) |
| deepspeed | 0.19.7 | ZeRO-1/2/3 distributed training |
| ray[tune,default] | 2.59.0 | HPO and distributed orchestration |
| verl[vllm] | 0.9.1 | Online RL (PPO, GRPO, DAPO) with vLLM 0.24.0 rollout; pins torch and transformers |
| flash-attn | 2.8.3 | Flash Attention 2 (Linux only) |
| mamba-ssm / causal-conv1d | 2.3.1 / 1.7.0 | Fast Mamba kernels for hybrid bases such as Granite 4.0-H (Linux only) |
| accelerate | >= 1.15.0 | HuggingFace Accelerate |

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

### Shadow Residual (dual-stream Granite)

`--tuning_algo sr` trains a Shadow Residual adapter: the base stream stays frozen
and bit-exact, a parallel adapter stream carries the LoRA deltas, and a per-layer
rank-`cross_stream_r` injection feeds the base into it. Requires a dense Granite
base, CUDA, and one extra install of
[shadow-residual](https://github.com/generative-computing/shadow-residual)
(`git clone https://github.com/generative-computing/shadow-residual && pip
install -e shadow-residual --no-deps` — it moves no fm-tune pin):

```bash
python main.py \
  --config_file autotune/configs/autotune_sr_3b.yaml \
  --train_file data/train.jsonl \
  --validation_file data/val.jsonl \
  --model_name_or_path ibm-granite/granite-4.1-3b \
  --tuning_algo sr \
  --output_dir ./output \
  --output_model_name granite-sr \
  --run_name my_sr_experiment \
  --cleanup --save_history
```

The result is an ordinary PEFT adapter (`peft_type: LORA`), so checkpointing,
best-model selection and serving are unchanged. Multi-GPU runs are **FSDP only**;
see the Shadow Residual note under
[Supervised Fine-Tuning](#1-supervised-fine-tuning-sft-and-parameter-efficient-methods)
for the full constraint list.

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
QLoRA, Shadow Residual (`sr`), RL (offline and online), multi-GPU, and
DeepSpeed/FSDP are **not** supported on MPS.

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

## Running on Multi-Node Clusters (LSF / Slurm)

AutoTune can bring up a multi-node Ray cluster on demand inside a **single**
multi-host scheduler allocation, under either LSF (`--scheduler lsf`, the
default) or Slurm (`--scheduler slurm`). Both launchers work the same way:

1. You submit `main.py` inside an allocation of N GPU hosts, passing
   `--num_nodes N` and `--gpus_per_node G`. The driver lands on the first
   allocated host (`host[0]`).
2. The launcher starts the Ray head on `host[0]` (CPU-only), starts a Ray
   worker on `host[0]` directly (registering that host's GPUs), and starts one
   worker on each of the other `N-1` hosts: with `blaunch` under LSF
   (`autotune/lsf/`), or as one `srun` step per host under Slurm
   (`autotune/slurm/`). Every worker runs `python -m autotune.lsf.worker_entry`,
   which checks that the host really exposes `G` GPUs before attaching.
3. The launcher waits until Ray sees all `N × G` GPUs and `N` worker nodes,
   failing fast if any worker exits during bring-up. `--bringup_deadline_s`
   (default 1200) caps the whole bring-up; past it the run exits 0 with a
   warning, without HPO.
4. HPO and final training run as usual; trials land on the GPU workers.
5. On exit (success or failure) the cluster is torn down (see the per-scheduler
   notes below).

One allocation means a single queue wait covering every host. The cluster has
`N × G` GPUs: the head registers none, and the worker colocated on `host[0]`
registers `host[0]`'s. A final training with `num_gpus_per_trial: 8` and
`max_concurrent_trials: 2` on 2 × 8 GPUs therefore runs one 16-rank job across
both hosts.

Per-host worker logs land at `<output_dir>/logs/<job_group>/<host>.log` (one
file per host, including `host[0]`'s colocated worker; `<job_group>` is the
auto-generated `ray_nodes_<timestamp>`). The driver prints each path on spawn
so you can `tail -f` while the cluster comes up. `--cleanup` wipes
`<output_dir>/logs/` at the end of the run; omit it to keep the logs.

### Example: LSF, 2 GPU nodes × 4 GPUs each (8 GPUs total)

```bash
NUM_NODES=2  # number of GPU hosts (== total hosts in the allocation)
bsub -q <queue> -G <user-group> -J multinode -o logs/driver.log \
     -n $((32 * NUM_NODES)) \
     -gpu "num=4:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" \
     -R "span[ptile=32]" -R "span[hosts=$NUM_NODES]" -R "rusage[mem=640G]" \
     bash -lc 'source ~/.bashrc && conda activate /path/to/envs/autotune && \
       python main.py \
         --scheduler lsf --num_nodes 2 --gpus_per_node 4 \
         --conda_env /path/to/envs/autotune \
         --config_file autotune/configs/autotune.yaml \
         --train_file data/train.jsonl --validation_file data/val.jsonl \
         --model_name_or_path ibm-granite/granite-4.0-micro \
         --tuning_algo lora --output_dir ./output \
         --output_model_name granite-lora --run_name multinode_lora \
         --save_history --cleanup'
```

### Example: LSF, 2 × H100 nodes × 8 GPUs (`--fleet bv-h100`)

```bash
NUM_NODES=2
bsub -q <queue> -G <user-group> -J multinode -o logs/driver.log \
     -n $((96 * NUM_NODES)) \
     -gpu "num=8:mode=exclusive_process:gmodel=NVIDIAH10080GBHBM3" \
     -R "span[ptile=96]" -R "span[hosts=$NUM_NODES]" -R "rusage[mem=1280G]" \
     bash -lc 'source ~/.bashrc && conda activate /path/to/envs/autotune && \
       python main.py --fleet bv-h100 \
         --scheduler lsf --num_nodes 2 --gpus_per_node 8 \
         --conda_env /path/to/envs/autotune \
         --config_file autotune/configs/autotune.yaml \
         --train_file data/train.jsonl --validation_file data/val.jsonl \
         --model_name_or_path ibm-granite/granite-4.0-micro \
         --tuning_algo lora --output_dir ./output \
         --output_model_name granite-lora --run_name multinode_bv_h100 \
         --save_history --cleanup'
```

LSF notes:

- `--conda_env` is **required** for the LSF multi-node launch: `blaunch`
  starts each remote worker in a fresh login shell that activates it. Without
  it, `main.py` falls back to a single-node local Ray cluster (useful for dev
  runs on a single-host `bsub`).
- Request **`--num_nodes` hosts** (not `+1`); the head and one worker share
  `host[0]`. `-R "span[hosts=N]"` forces the allocation onto N distinct hosts;
  without it LSF may pack the slots onto fewer hosts and the launcher rejects
  the layout.
- The launcher reserves 32 cores per worker for Ray. The head and the
  colocated worker share `host[0]`'s cores.
- Online RL (`--rl_algo` ∈ `{ppo, grpo, dapo}`) runs vLLM in its own processes
  next to the FSDP workers, which `exclusive_process` forbids (the verl driver
  refuses to start). Use `-gpu "num=8:mode=shared:j_exclusive=yes:gmodel=..."`:
  `mode=shared` alone lets LSF place another job on the same GPUs.
- `bjobs` shows exactly **one** job id. Teardown runs a remote `ray stop` on
  every host via `blaunch`, stops the head, and finally `bkill`s its own job so
  LSF reaps the job's processes on every host. The job therefore ends `EXIT`
  rather than `DONE`; all outputs are on disk before that step.

### Example: Slurm, 2 × H200 nodes × 8 GPUs (16 GPUs total)

Save as `multinode.sbatch` and submit with `sbatch multinode.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=multinode
#SBATCH --account=<account>
#SBATCH --qos=normal
#SBATCH --nodes=2                 # == --num_nodes below
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8         # per node; == --gpus_per_node below
#SBATCH --exclusive               # whole nodes: every CPU goes to Ray
#SBATCH --mem=0                   # all of each node's memory
#SBATCH --time=24:00:00
#SBATCH --output=logs/driver-%j.log

source ~/.bashrc && conda activate /path/to/envs/autotune
cd /path/to/fm-tune

python main.py --fleet bv-h200 \
  --scheduler slurm --num_nodes "$SLURM_JOB_NUM_NODES" --gpus_per_node 8 \
  --config_file autotune/configs/autotune.yaml \
  --train_file data/train.jsonl --validation_file data/val.jsonl \
  --model_name_or_path ibm-granite/granite-4.0-micro \
  --tuning_algo lora --output_dir ./output \
  --output_model_name granite-lora --run_name multinode_bv_h200 \
  --save_history --cleanup
```

Or the same thing without a script file:

```bash
sbatch -A <account> -q normal -J multinode -N 2 --ntasks-per-node=1 \
       --gres=gpu:h200:8 --exclusive --mem=0 -t 24:00:00 -o logs/driver-%j.log \
  --wrap 'source ~/.bashrc && conda activate /path/to/envs/autotune && cd /path/to/fm-tune && \
    python main.py --fleet bv-h200 --scheduler slurm --num_nodes 2 --gpus_per_node 8 \
      --config_file autotune/configs/autotune.yaml \
      --train_file data/train.jsonl --validation_file data/val.jsonl \
      --model_name_or_path ibm-granite/granite-4.0-micro \
      --tuning_algo lora --output_dir ./output \
      --output_model_name granite-lora --run_name multinode_bv_h200 \
      --save_history --cleanup'
```

Slurm notes:

- `--scheduler slurm` always takes the multi-node path (even with
  `--num_nodes 1`); `--num_nodes` must equal the number of allocated nodes.
  `--conda_env` is not needed: `srun` passes the driver's environment
  (including the active conda env) to every worker.
- Each remote worker is an `srun --nodes=1 --nodelist=<host> --overlap` step
  that asks for all of that host's allocated CPUs (read from
  `SLURM_JOB_CPUS_PER_NODE`) and inherits the job's GPUs. Request GPUs per
  node (`--gres=gpu:<type>:<G>` or `--gpus-per-node`), not as a job total, so
  every node gets the same count. `--exclusive --mem=0` gives Ray the whole
  node.
- Online RL needs no special flags: Slurm has no per-job GPU compute mode, and
  `--exclusive` already keeps other jobs off the nodes.
- `squeue` shows one job; `sacct -j <jobid>` lists the worker steps
  (`ray_worker_<host>`). Teardown SIGTERMs the `srun` steps (Slurm forwards the
  signal to the workers, which run `ray stop`) and stops the head; Slurm then
  reaps anything left when the batch script exits. The job ends `COMPLETED`
  with `main.py`'s exit code.

### Choosing the InfiniBand fabric (`--fleet`)

`--fleet` sets the default `NCCL_IB_HCA` (the InfiniBand HCAs NCCL uses for
inter-node collectives) on both schedulers. RDMA over InfiniBand is on by
default (`NCCL_NET_GDR_LEVEL=PXB`), so inter-node collectives use GPU-Direct
RDMA where the topology allows.

| `--fleet` | Default `NCCL_IB_HCA` | Notes |
|---|---|---|
| `ccc-a100` (default) | `mlx5_0` | A100 nodes with a single usable rail. |
| `bv-h100` | `mlx5_0,mlx5_1,...,mlx5_7` | H100 nodes (LSF) with 8 compute rails, one per GPU on the same PCIe complex; `mlx5_8`/`mlx5_9` (storage/management) are excluded. |
| `bv-h200` | `mlx5_0,mlx5_1,...,mlx5_7` | H200 nodes (Slurm) with the same 8-rail layout. |

The presets match the clusters they were tuned on; on other hardware, pass
`--ib_hca` with your node's compute HCAs.

Override per run with `--ib_hca "mlx5_0,mlx5_1"`, and the NCCL bootstrap
interface with `--ib_ifname` (unset by default so NCCL picks the routable
Ethernet interface).

To verify NCCL uses every rail, set `NCCL_DEBUG=INFO` for one run and grep
`<output_dir>/logs/<job_group>/<host>.log` for `NCCL INFO NET/IB`. Expect lines
of the form `NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB ...
[7]mlx5_7:1/IB`. If only one rail shows up, NCCL didn't accept the list; the
most common cause is an `NCCL_IB_GID_INDEX` mismatch on RoCE-style
deployments. Each worker log also starts with an `ibstat` dump of its host.

### Single-node fallback

For local development, run without `--scheduler slurm` and without
`--conda_env`. `main.py` then starts a local Ray cluster on the current host,
exactly as in Quick Start (this works inside a single-node `bsub` or `sbatch`
too).

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
| Shadow Residual | `sr` | Dual-stream Granite (frozen base stream + adapter stream) with stock LoRA |

> **QLoRA note:** `qlora` behaves like `lora` (same tunable hyperparameters) but
> loads the frozen base in 4-bit NF4 to cut GPU memory. It runs on the single-GPU,
> DeepSpeed, FSDP, and TRL (DPO/KTO) drivers. It is **not** compatible with
> DeepSpeed ZeRO-3 or FSDP `full_shard` (the drivers raise a clear error) — use
> ZeRO-1/ZeRO-2, FSDP `SHARD_GRAD_OP`, or a single GPU.

> **Shadow Residual note:** `sr` trains a dual-stream Granite decoder — a frozen
> base stream that stays bit-exact with the unadapted base, plus a trainable
> adapter stream fed by stock LoRA deltas and a per-layer rank-`cross_stream_r`
> injection from the base. It is plain PEFT LoRA under the hood (the saved
> adapter has `peft_type: LORA`), so nothing about checkpointing or best-model
> selection changes. Constraints, all enforced with a clear error: a **dense
> Granite** base (MoE and hybrid/Mamba bases are rejected), **SFT only** (no
> DPO/KTO/PPO/GRPO), **CUDA only** (no MPS, no `--backend mlx`), and **FSDP
> only** for multi-GPU — `train_implementation: FSDP` (DeepSpeed is rejected;
> `fsdp_strategy: full_shard` is the default, and `no_shard` — which is plain
> DDP with no FSDP wrap — is allowed but makes every rank hold its own host-RAM
> copy of the base). It needs one extra
> install, which deliberately does not move any fm-tune pin:
>
> ```bash
> git clone https://github.com/generative-computing/shadow-residual
> pip install -e shadow-residual --no-deps
> ```
>
> Ready-made configs: `autotune/configs/autotune_sr_3b.yaml` (8-way sweep),
> `autotune/configs/autotune_sr_smoke_1gpu.yaml` (4 samples, one at a time) and
> `autotune/configs/autotune_sr_smoke_2gpu.yaml`. Prefer the 2-GPU one to smoke
> the distributed path: final training uses
> `num_workers = num_gpus_per_trial * max_concurrent_trials`, and FSDP downgrades
> `full_shard` to `NO_SHARD` at world size 1 (where fm-tune skips the FSDP config
> altogether), so a 1-GPU run never shards and never exercises the layer-wrap
> lookup. When verifying a run, exit code 0 is not sufficient, since an SR run
> that silently degraded to plain LoRA also exits 0.
>
> Budget host RAM, not just GPU: SR always materializes on CPU first, ~12-14 GB
> peak per materializing process on a 3B base, and HPO trials run
> `max_concurrent_trials` of them at once.

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

**Example** (LoRA with the `autotune.yaml` defaults — FSDP for multi-GPU — skipping HPO):
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

Any `--tuning_algo` from the SFT table above works here **except**
`sr` — Shadow Residual is SFT-only and the TRL drivers raise a clear error for it.

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
  train_implementation:
    default: "DeepSpeed"
  ds_strategy:
    default: "zero3_cpu"
    values: ["auto", "zero1_gpu", "zero2_gpu", "zero3_gpu", "zero2_cpu", "zero3_cpu"]
```

- **ZeRO-1**: shards optimizer states across GPUs
- **ZeRO-2**: shards optimizer states + gradients
- **ZeRO-3 (+ CPU offload)**: shards optimizer states, gradients, and parameters; the `*_cpu` variants also offload to CPU for large models

### FSDP (`train_implementation: FSDP`)

Pick the sharding strategy with `fsdp_strategy`:
```yaml
training_config:
  train_implementation:
    default: "FSDP"
  fsdp_strategy:
    default: "full_shard"
    values: ["auto", "no_shard", "shard_grad_op", "full_shard", "hybrid_shard"]
```

> **QLoRA note:** QLoRA (4-bit base) cannot be sharded by DeepSpeed ZeRO-3 or
> FSDP `full_shard` (the drivers raise a clear error). Use ZeRO-1/ZeRO-2, FSDP
> `shard_grad_op`, or a single GPU.

### FSDP for SFT Training

Set in the YAML config:
```yaml
training_config:
  train_implementation:
    default: "FSDP"
  fsdp_strategy:
    default: "full_shard"   # shard_grad_op, hybrid_shard, no_shard, auto
```

`auto` sizes the strategy from the model and batch shape. The wrapped layer class
is read from the model's own `_no_split_modules`, so no
`fsdp_transformer_layer_cls_to_wrap` is needed. `--tuning_algo sr` requires this
backend; for SR, `auto` is forced to `full_shard` and any explicit strategy
(including `no_shard`) passes through. QLoRA is the mirror case and rejects
`full_shard`.

### Online RL (verl)

Online RL always uses FSDP internally via [verl](https://github.com/volcengine/verl) with vLLM for rollout generation. No separate backend configuration needed.

## Recommended Resources

For GPU sizing, DeepSpeed ZeRO-strategy selection, per-configuration memory estimates, training-time estimates across model sizes (350M–13B+), and a strategy-selection decision tree, see [`docs/RESOURCES.md`](docs/RESOURCES.md).

## Configuration

The YAML configuration file controls the HPO search space and training parameters. It has four main sections:

### `tune_config` — HPO Settings

Every key in `tune_config`, `training_config` and `training_rl_config` is a
block whose `default:` is the value used (shipped configs also carry
`description:`, `values:` and `min_val:` for documentation and validation):

```yaml
tune_config:
  search_alg:
    default: "lds"
    values: ["lds", "hyperopt", "random", "bohb", "blds"]
  scheduler:
    default: "fifo"
    values: ["fifo", "asha", "hyperbandforbohb"]
  num_samples:
    default: 8              # number of HPO trials
  max_concurrent_trials:
    default: 1              # parallel trials
  max_discrepancy:
    default: 6              # LDS / BLDS only
  time_budget_s:
    default: 3600           # time limit in seconds (null = none)
```

### `training_config` — Training Parameters

```yaml
training_config:
  num_train_epochs:
    default: 10
  hpo_num_epochs:
    default: 1                  # fewer epochs during HPO
  hpo_dataset_percentage:
    default: 0.1                # use 10% of data during HPO
  seed:
    default: 42
  num_gpus_per_trial:
    default: 4
  use_flash_attention:
    default: "flash_attention_2"
    values: ["eager", "flash_attention_2"]
  train_implementation:
    default: "FSDP"             # FSDP or DeepSpeed (multi-GPU SFT/DPO)
  fsdp_strategy:
    default: "full_shard"       # auto, no_shard, shard_grad_op, full_shard, hybrid_shard
  max_length:
    default: 256
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

The `sr:` section is the one search space that differs in shape. It adds
`cross_stream_r` (the rank of the per-layer cross-stream injection, tuned
independently of `r`) and it is the only tuner where `alpha_ratio` is
`for_tuner: true`. That is load-bearing: the drivers otherwise compute
`lora_alpha = alpha_ratio * r`, but SR's `r` is the *minimum* rank (PEFT's scalar
default, with per-name overrides in `rank_pattern`), so the formula would scale
from the minimum instead of SR's `2 x max_rank`. Routing `alpha_ratio` into
`peft_kwargs` leaves the drivers' `train_kwargs.pop("alpha_ratio")` returning
`None` and lets `build_sr_lora_config` compute it correctly.

### `training_rl_config` — Online RL Parameters

```yaml
training_rl_config:
  reward_function_path:
    default: "autotune/rewards/gsm8k_reward.py"
  reward_function_name:
    default: "compute_score"
  reward_model_path:
    default: null               # path to a learned reward model (optional)
  max_prompt_length:
    default: 1024
  max_response_length:
    default: 512
  rollout_n:
    default: 5                  # group size for GRPO
  gpu_memory_utilization:
    default: 0.5
  tensor_model_parallel_size:
    default: null               # null = auto-detect from model size
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
  --tuning_algo ALGO         Tuning method: sft, lora, qlora, sr, alora, loha,
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
  --model_at_root            After a successful run, move the model files up
                             to {output_dir}/ itself and delete everything
                             else there. For jobs whose whole output_dir is
                             uploaded as the model (e.g. a HF repo). Rewrites
                             the entire output_dir; off by default
  --keep_diagnostics         With --model_at_root, move the other artifacts
                             into {output_dir}/diagnostics/ instead of deleting
                             them (or set FMTUNE_KEEP_DIAGNOSTICS=1)
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
  --scheduler {lsf,slurm}    Scheduler owning a multi-node allocation (default:
                             lsf). 'slurm' brings up the srun launcher inside
                             an sbatch allocation; see "Running on Multi-Node
                             Clusters"
  --num_nodes N              GPU hosts in the multi-node Ray cluster (== hosts
                             in the allocation; the head colocates with a
                             worker on host[0]) (default: 1)
  --gpus_per_node N          GPUs each host registers with Ray (default: 1)
  --conda_env PATH           Env each blaunch'd worker activates. LSF only;
                             triggers the LSF multi-node launch. Ignored with
                             --scheduler slurm
  --bringup_deadline_s S     Cap on the whole cluster bring-up (default: 1200);
                             past it the run exits 0 without HPO
  --fleet {ccc-a100,bv-h100,bv-h200}
                             Default NCCL_IB_HCA rail set (default: ccc-a100)
  --ib_hca LIST              Override NCCL_IB_HCA (e.g. mlx5_0,mlx5_1)
  --ib_ifname IFACE          Override NCCL_SOCKET_IFNAME (default: unset)

AutoTuneX bridge logging (optional, OFF by default):
  --autotunex_server_url URL Base URL of an AutoTuneX bridge server to log this
                             run to. Omit it (the default) to run fully offline
                             with no bridge calls. Bridge errors never fail the
                             run.
  --job_id ID                Optional informational run label. Passed through to
                             the bridge when --autotunex_server_url is set;
                             otherwise unused.
```

## Environment Variables

| Variable | Effect |
|----------|--------|
| `FMTUNE_DEVICE` | Force the accelerator (`cuda`, `mps` or `cpu`) instead of auto-detection (CUDA, then MPS, then CPU). |
| `FMTUNE_OBJECT_STORE_BYTES` | Ray object-store size in bytes for a local non-CUDA cluster (default 2 GiB). |
| `FMTUNE_MLX_CACHE` | Cache directory for MLX-converted models (default `~/.cache/fmtune/mlx`). |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Set to `1` by default on non-CUDA hosts so ops without an MPS kernel run on CPU; set `0` to raise instead and find the missing op. |
| `FMTUNE_TRUST_REMOTE_CODE` | `0` / `false` / `no` / `off` loads models without `trust_remote_code` (default on, which executes modeling code shipped in the model repo). |
| `FMTUNE_RAY_TMPDIR` | Parent directory for Ray's temp dir, used verbatim; keep it short, writable and node-local (Ray's socket paths are length-limited). |
| `FMTUNE_DEFAULT_USER_EMAIL` | Fallback identity sent to the AutoTuneX bridge when it cannot resolve the caller (`--autotunex_server_url` only). |

`FMTUNE_KEEP_DIAGNOSTICS` is described under `--keep_diagnostics` above.

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
- With `--model_at_root`, a successful run ends with the model files directly
  in `{output_dir}/` and nothing else, or with everything else under
  `{output_dir}/diagnostics/` when `--keep_diagnostics` /
  `FMTUNE_KEEP_DIAGNOSTICS=1` is set. A failed run is left as described above.
  If the model can't be moved (none found, or the move fails), the run exits
  with code 1.


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
    shadow_residual.py              # Shadow Residual seam: LoraConfig projection, model factory, guards
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
    lsf/                            # Optional multi-node Ray launcher (LSF/HPC, --scheduler lsf)
    slurm/                          # Optional multi-node Ray launcher (Slurm, --scheduler slurm)

    trainers/
      driver_single.py              # Single-GPU SFT/PEFT
      driver_single_trl.py          # Single-GPU DPO/KTO (TRL)
      driver_single_mlx.py          # Single-device MLX backend (sft/lora/qlora)
      driver_multi_hf_ds.py         # Multi-GPU SFT/PEFT + DeepSpeed
      driver_multi_hf_fsdp.py       # Multi-GPU SFT/PEFT + FSDP
      driver_multi_trl_ds.py        # Multi-GPU DPO/KTO + DeepSpeed (TRL)
      driver_multi_trl_fsdp.py      # Multi-GPU DPO/KTO + FSDP (TRL)
      driver_multi_verl.py          # Multi-GPU PPO/GRPO/DAPO (verl + vLLM)
      _checkpointing.py             # Shared save/eval strategy + best-model preconditions
      _steps.py                     # Shared epoch step math (optimizer steps vs micro-batches)

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
