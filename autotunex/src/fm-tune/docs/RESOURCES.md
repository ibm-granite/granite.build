# Recommended Resources

This document gives FSDP sizing guidance for fine-tuning on **8× A100 80GB**.
All numbers assume **bf16 precision**, **AdamW optimizer**, and **gradient
checkpointing enabled** (the fm-tune defaults). The numbers are derived from
the same memory model the runtime uses to auto-pick a strategy:
[`autotune/utils.py::estimate_fsdp_strategy`](../autotune/utils.py).

## GPU Memory Breakdown

For a trial running on 8× A100 80GB, per-GPU memory is the sum of these
components scaled by the FSDP sharding strategy:

| Component | Formula | Notes |
|-----------|---------|-------|
| Model weights (bf16) | `params × 2 B` | Replicated under `no_shard`/`shard_grad_op`; sharded `÷N` under `full_shard` |
| Optimizer states (fp32) | `trainable_params × 8 B` | AdamW: fp32 momentum + fp32 variance. Sharded `÷N` under `shard_grad_op`/`full_shard` |
| Gradients (bf16) | `trainable_params × 2 B` | Sharded `÷N` under `shard_grad_op`/`full_shard` |
| Activations | `B × seq × hidden × layers × 2 B × (√L / L)` | The `√L/L` factor is the gradient-checkpointing reduction (selective recomputation) |
| CUDA / NCCL overhead | `1.5×` factor | Multiplied over the sum above to model allocator fragmentation and NCCL buffers |

For LoRA, `trainable_params` is ~1–3% of total — optimizer and gradients
shrink accordingly, and activations dominate for long sequences.

## FSDP Strategy Ladder

fm-tune's `--fsdp_strategy` accepts five values: `auto`, `no_shard`,
`shard_grad_op`, `full_shard`, `hybrid_shard`. With 8 GPUs in a single node,
the picker walks the first three from least to most sharding and selects the
first one that fits:

| Strategy        | Weights  | Optimizer | Gradients | When to use |
|-----------------|----------|-----------|-----------|-------------|
| `no_shard`      | full     | full      | full      | DDP. Fastest — no parameter gathering, just gradient all-reduce. Use when weights + optimizer + gradients fit per GPU. |
| `shard_grad_op` | full     | `÷N`      | `÷N`      | "ZeRO-2 equivalent". Optimizer and gradients are sharded; weights stay replicated. Use when weights fit per GPU but optimizer doesn't. |
| `full_shard`    | `÷N`     | `÷N`      | `÷N`      | "ZeRO-3 equivalent". Weights are gathered just-in-time per layer. Use when even the weights don't fit per GPU. Adds all-gather overhead each forward/backward. |
| `hybrid_shard`  | `÷N_local` | `÷N_local` | `÷N_local` | Multi-node — shards within a node, replicates across nodes, so all-gathers stay on the fast intra-node NVLink fabric. Not needed for the single-node 8× A100 target, but the practical choice for the 120B/230B models below. |

`--fsdp_strategy auto` picks the first option that fits given a 1.5× overhead
budget against 80 GB capacity. The recommendations below are the same
selections, written out so you can plan without trial-running.

## Full Fine-Tuning Recommendations (8× A100 80GB)

For full fine-tuning the optimizer dominates at AdamW's 8 B/param. 8B and 30B
require sharding to fit on 80 GB GPUs.

### 3B model

| Sequence | Per-device batch | FSDP strategy | Est. memory/GPU | Rationale |
|----------|------------------|---------------|-----------------|-----------|
| 512      | 16               | `no_shard`    | ~36 GB          | Activations are still small at seq=512; full replication is fastest. |
| 1024     | 8                | `no_shard`    | ~36 GB          | Optimizer (~16 GB) + weights (~4 GB) + grads (~4 GB) = comfortable on 80 GB. |
| 2048     | 4                | `no_shard`    | ~36 GB          | Activations grow linearly with seq; batch trimmed to keep headroom. |

### 8B model

| Sequence | Per-device batch | FSDP strategy | Est. memory/GPU | Rationale |
|----------|------------------|---------------|-----------------|-----------|
| 512      | 8                | `shard_grad_op` | ~30 GB        | Optimizer (~48 GB unsharded) overflows DDP; sharding to `÷8` brings it down to ~6 GB/GPU. |
| 1024     | 4                | `shard_grad_op` | ~30 GB        | Same picture; activations still small relative to model state. |
| 2048     | 4                | `shard_grad_op` | ~31 GB        | Activations rise to ~1.4 GB but still well under capacity. |

### 30B model

| Sequence | Per-device batch | FSDP strategy | Est. memory/GPU | Rationale |
|----------|------------------|---------------|-----------------|-----------|
| 512      | 8                | `full_shard`  | ~54 GB          | Even weights (~48 GB) overflow per-GPU; must shard weights too. |
| 1024     | 4                | `full_shard`  | ~55 GB          | Activations remain modest under gradient checkpointing. |
| 2048     | 2                | `full_shard`  | ~54 GB          | Batch trimmed to leave headroom for the all-gather buffers. |

`full_shard` adds an all-gather per forward and per backward layer, so
throughput is lower than `shard_grad_op`. There is no faster option at this
scale on a single 8-GPU node; consider multi-node `hybrid_shard` if more
nodes are available.

## LoRA Fine-Tuning Recommendations (8× A100 80GB)

LoRA (rank 16, target_modules="all-linear") makes only ~30–170 M parameters
trainable, so optimizer and gradients become negligible. The dominant cost is
the **frozen weights**, which under `no_shard` are loaded full per GPU.

The recommendations below are deliberately conservative: they target ≤60 GB
per GPU at the default 1.5× overhead, leaving real headroom for activation
peaks, NCCL ring buffers, dataloader prefetch, and CUDA fragmentation. The
analytical model is good enough to predict OOM, not good enough to predict
near-OOM behavior — at 30B the model state alone consumes most of the GPU,
so we shard rather than gamble on the remaining few GB.

### 3B model

| Sequence | Per-device batch | FSDP strategy | Est. memory/GPU |
|----------|------------------|---------------|-----------------|
| 512      | 16               | `no_shard`    | ~7 GB           |
| 1024     | 16               | `no_shard`    | ~7 GB           |
| 2048     | 16               | `no_shard`    | ~8 GB           |

### 8B model

| Sequence | Per-device batch | FSDP strategy | Est. memory/GPU |
|----------|------------------|---------------|-----------------|
| 512      | 16               | `no_shard`    | ~19 GB          |
| 1024     | 16               | `no_shard`    | ~20 GB          |
| 2048     | 8                | `no_shard`    | ~20 GB          |

### 30B model

| Sequence | Per-device batch | FSDP strategy | Est. memory/GPU |
|----------|------------------|---------------|-----------------|
| 512      | 16               | `full_shard`  | ~10 GB          |
| 1024     | 8                | `full_shard`  | ~10 GB          |
| 2048     | 4                | `full_shard`  | ~11 GB          |

**Why `full_shard` for 30B LoRA, even though optimizer + gradients are tiny?**
The frozen weights are ~48 GB unsharded. With the 1.5× overhead the auto-
picker applies, that alone projects to ~72 GB per GPU before you've added a
single activation, NCCL buffer, or prefetched batch. In practice 30B LoRA on
`no_shard` works under quiet conditions and OOMs unpredictably under the
ones you actually care about (long sequences, larger batches, longer runs
with fragmentation buildup). `full_shard` brings weights down to ~6 GB per
GPU at the cost of an all-gather per layer — roughly a 20–40% throughput hit
for a much wider safety margin. If you need maximum throughput and have
verified empirical headroom for your specific model + batch + sequence,
`shard_grad_op` is intermediate (weights still replicated, ~74 GB/GPU) but
shares the same fragility as `no_shard` here.

## Large Granite Models — 120B and 230B (multi-node)

The Granite **120B** and **230B** models do not fit on a single 8× A100 80GB
node for any training mode except — barely — 120B LoRA. They are the reason the
`hybrid_shard` row above exists. Plan for **multiple 8-GPU nodes**.

> **Two caveats specific to these models.** (1) They are **MoE** (mixture-of-
> experts): the "120B"/"230B" are *total* parameter counts, and memory for
> weights, optimizer, and gradients scales with total params — so the tables
> below use the full counts, not the smaller active-per-token counts.
> (2) `estimate_fsdp_strategy` /
> [`_estimate_memory_components`](../autotune/utils.py) is a **dense**-transformer
> estimator: it derives params from `hidden_size`/`num_layers`/`intermediate_size`
> and has no notion of experts, so for an MoE model it will badly *undercount*
> total params and pick too optimistic a strategy. **Do not trust
> `--fsdp_strategy auto` for these models** — set the strategy and GPU count
> explicitly per the tables below.

All numbers below come from the same `weights + optimizer + gradients +
activations`, 1.5× overhead, ≤75 GB/GPU budget model used everywhere else in
this doc, evaluated at the *total* parameter count. Weights alone are ~224 GB
(120B) and ~428 GB (230B) in bf16, so weights **must** be sharded — only
`full_shard` (or its multi-node form `hybrid_shard`) is viable.

### Full fine-tuning

Optimizer state at AdamW's 8 B/param dominates: ~894 GB (120B) and ~1.7 TB
(230B) before sharding. Even with everything sharded `÷N`, you need enough total
GPUs that `(weights + optim + grads)/N` clears the per-GPU budget.

| Model | Min GPUs (nodes) | FSDP strategy | Est. memory/GPU | Notes |
|-------|------------------|---------------|-----------------|-------|
| 120B  | 32 (4× 8-GPU)    | `hybrid_shard` | ~67 GB         | Tight — at 32 GPUs the budget is nearly full. Use seq ≤1024, per-device batch 1, and lean on gradient accumulation. |
| 120B  | 64 (8× 8-GPU)    | `hybrid_shard` | ~36 GB         | Comfortable headroom; allows seq 2048 and/or larger micro-batches. |
| 230B  | 64 (8× 8-GPU)    | `hybrid_shard` | ~66 GB         | Minimum viable. OOMs at 32 GPUs (~127 GB/GPU). Seq ≤1024, per-device batch 1. |

`full_shard` and `hybrid_shard` are arithmetically equivalent per GPU here;
prefer **`hybrid_shard`** so the per-layer all-gather stays on intra-node NVLink
rather than crossing the slower inter-node interconnect (InfiniBand/Ethernet),
where a global `full_shard` all-gather would bottleneck every layer.

### LoRA fine-tuning

LoRA (rank 16, `target_modules="all-linear"`) makes optimizer and gradients
negligible (~0.4B trainable for 120B, ~0.7B for 230B). The cost is entirely the
**frozen weights**, which still must be sharded to fit.

| Model | Min GPUs (nodes) | FSDP strategy | Est. memory/GPU | Notes |
|-------|------------------|---------------|-----------------|-------|
| 120B  | 8 (1× 8-GPU)     | `full_shard`  | ~47 GB          | Just fits on a single node — sharded weights ~28 GB/GPU dominate. `no_shard`/`shard_grad_op` OOM (unsharded weights ~224 GB). |
| 120B  | 16 (2× 8-GPU)    | `hybrid_shard` | ~26 GB         | Multi-node for headroom or larger sequences; switch to `hybrid_shard` once you leave a single node. |
| 230B  | 16 (2× 8-GPU)    | `hybrid_shard` | ~47 GB          | Does **not** fit on one node: weights ÷8 ≈ 54 GB × 1.5 overhead alone exceeds 75 GB. Two nodes (÷16) brings it to ~47 GB/GPU. |
| 230B  | 32 (4× 8-GPU)    | `hybrid_shard` | ~26 GB          | Comfortable; room for seq 2048. |

> **Note:** 120B LoRA is the single case that fits one 8-GPU node, and only
> under `full_shard` (not `no_shard`, despite LoRA's tiny trainable count) —
> because the frozen weights themselves overflow a GPU. On a single node
> `full_shard` and `hybrid_shard` are identical; the table lists `hybrid_shard`
> only once you span nodes.

For the multi-node mechanics (intra- vs. inter-node sharding, NCCL timeouts,
Ray Train worker placement across nodes) see the `hybrid_shard` notes and the
Ray Data tokenization-concurrency section below — at this scale tokenization
fan-out across all nodes' CPUs matters as much as the GPU sizing.

## Strategy Selection Guide

If you'd rather let the runtime decide, leave `fsdp_strategy: auto` (the
default in `autotune/configs/autotune.yaml`) and the picker walks the ladder
on every trial. To plan ahead manually:

1. Compute `weights = params × 2 B`, `optim = trainable × 8 B`,
   `grads = trainable × 2 B`. Add 1.5× overhead.
2. Does `weights + optim + grads + activations` fit on one GPU? → `no_shard`.
3. Does `weights + (optim + grads) / N + activations` fit? → `shard_grad_op`.
4. Does `(weights + optim + grads) / N + activations` fit? → `full_shard`.
5. None of the above on one node? → move to multi-node `hybrid_shard` and
   recompute step 4 with `N` = total GPUs across nodes (see the 120B/230B
   section), or reduce batch / sequence length.

> **Tip:** With the configs in this doc, 3B and 8B never need `full_shard` on
> 8× A100 80GB — `no_shard` (LoRA) or `shard_grad_op` (full FT) covers them
> with real headroom. At 30B, reach for `full_shard` for both LoRA and full
> FT: at this scale weights alone consume most of the GPU, and the all-gather
> overhead is a worthwhile price for predictable runs. The auto-picker biases
> toward "fits the analytical budget" rather than "robust to empirical
> surprises", so override toward more sharding when you have a long-running
> sweep where one OOM kills hours of wall-clock time. For the 120B/230B MoE
> models, skip the auto-picker entirely — it undercounts MoE params — and use
> the explicit multi-node `hybrid_shard` configs above.

## Ray Data Tokenization Concurrency

The `ray_data` data backend (FSDP and DeepSpeed multi-GPU drivers) tokenizes the
dataset with a distributed Ray Data pipeline so the work spreads across every
node's CPUs. Two conditions must both hold for it to actually fan out:

1. **Enough blocks.** Ray Data runs at most one stateless `map_batches` task per
   input *block*. A single source file (jsonl/json/parquet) usually reads as one
   block, so the driver repartitions train/eval into ≈`concurrency` blocks
   (`repartition(n, shuffle=False)` — a cheap split, no full shuffle) before
   tokenizing. Block count is clamped to the row count so small eval sets aren't
   split into empty blocks.
2. **Enough free CPUs.** `concurrency` (and `num_cpus` per task) cap how many of
   those block-tasks run at once.

**Auto sizing.** By default `concurrency = floor(total_cluster_cpus) − num_workers`
— all CPUs except the one each GPU training worker reserves. On an 8-node cluster
with 32 CPUs/node (256 total) running 8 GPU workers, that's 248 parallel tokenize
tasks. `num_cpus` defaults to 1.0 (one CPU per task).

**Overrides.** Set `--ray_data_concurrency <int>` and `--ray_data_num_cpus <float>`
on the CLI (or the matching YAML training-config keys). To oversubscribe CPUs, use
`--ray_data_num_cpus 0.5` (two tasks per physical CPU).

**Concurrent HPO caveat.** During an HPO sweep, each trial computes its auto
`concurrency` from the *full* cluster CPU count, unaware of sibling trials. With
several trials tokenizing at once this oversubscribes CPUs. `max_concurrent_trials`
(auto-derived from `total_gpus / gpus_per_trial`) bounds the trial count, but for
large sweeps set `--ray_data_concurrency` explicitly to roughly
`(total_cpus − total_gpu_workers) / max_concurrent_trials`.

**Recommended starting point:** leave it on auto for single final-training runs;
set it explicitly for wide HPO sweeps.

## Online RL with verl (8× A100 80GB)

The verl driver (`autotune/trainers/driver_multi_verl.py`) keeps actor +
reference + (optionally) critic + vLLM rollout on the same GPUs in colocated
mode, which compresses the per-component memory budget. verl uses FSDP under
the hood for the actor/critic/ref wraps regardless of `train_implementation`.

| Model | Algorithm | GPUs | FSDP strategy (actor) | Notes |
|-------|-----------|------|------------------------|-------|
| 3B    | GRPO      | 4    | `shard_grad_op`        | No critic. vLLM `gpu_memory_utilization=0.3`. |
| 3B    | PPO       | 8    | `shard_grad_op`        | Actor + critic + ref + vLLM colocated. Sleep mode + free_cache_engine on. |
| 8B    | GRPO      | 8    | `full_shard`           | TP=1 in vLLM; `enforce_eager=True` for hybrid Mamba/MoE models. |
| 8B    | PPO       | 8    | `full_shard`           | Tight memory budget; `param_offload=True` on the ref model. |
| 30B   | GRPO      | 8    | `full_shard`           | TP=2 for vLLM rollout. May need `gpu_memory_utilization=0.25`. |
| 30B   | PPO       | 8    | `full_shard`           | Likely needs 16+ GPUs in practice; consider separate pools for the reward model. |

See the online RL dataset and reward-wiring reference in
[dataset-online-rl.md](dataset-online-rl.md) for the verl data format and
`compute_score` contract.

---

← Back to [README](../README.md)
