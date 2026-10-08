# GOLD on-policy smoke recipe

On-policy GOLD distillation with vLLM as a **separate target** instead of nodes carved
out of the trainer's own allocation. Two optimizer steps, one trainer node, one server
node, 4096-token context.

> **This has run, and the shape works.** Build `9973e766` (2026-09-15) synced all 362
> parameter tensors from the trainer to the server over NCCL, across two separate LSF
> allocations, on both of its steps. Two steps still measure plumbing rather than
> science — nothing here is evidence about distillation quality.

## The question it exists to answer

On-policy GOLD pushes the student's updated weights to the vLLM server every
`vllm_sync_frequency` step **over NCCL, not HTTP** — the reference launcher's
`VLLM_NCCL_COORDINATOR_PORT`. Two targets means two LSF allocations, so this recipe
needs a NCCL process group that spans them.

**It works.** Build `9973e766` logged, on each of its two steps:

    t_gen 5.3  t_train 3.4  t_sync 1.5  t_sync_ag 0.1  t_sync_bc 1.4  n_sync_params 362

`n_sync_params 362` is the whole answer: every parameter tensor crossed from the
trainer's allocation to the server's. `t_gen` non-zero says the student really
generated against that server rather than reading fixed completions — the two numbers
that are both 0 in an off-policy run. And `t_sync 1.5s` against ~5s of generation says
the sync is cheap enough to do every step, which is what `vllm_sync_frequency: 1` asks
for. So the fallback — `distill-gold`'s in-allocation role split — is not needed.

Read the NCCL log anyway when something changes. `NCCL_DEBUG: INFO` with `INIT,NET` is
on deliberately, and `NCCL_TIMEOUT_MS` is 10 minutes rather than the reference hour,
because **a group that cannot form hangs rather than erroring** and the timeout is what
converts two held allocations into a diagnostic.

Two prerequisites were found the hard way, both now fixed in the step; a run predating
them proves nothing:

* **`--use_vllm` must be passed positively.** `gold.py` defaults it to False and
  `custom_gold_trainer.py` branches on it, so an earlier build (`d77546a9`) allocated a
  server node, health-checked it, wired its address in — and then generated locally,
  leaving the server idle and dying in transformers 5.8.0 on a shape mismatch. The step
  now states the flag in both directions from `vllm_num_servers`.
* **The server must bound its own allocation.** `teardown` is gated on
  `train.checkpoint`, which a crashed trainer never emits, and gbserver has no
  on-failure semantics — so a trainer failure stranded 8 H100s until they were downed by
  hand. `max_lifetime_seconds` (this recipe sets 3600) is the backstop; teardown remains
  the normal path and is what ended `9973e766`.

## Why a separate target at all

The upstream `distill-gold-train` step refuses this configuration outright and prints
its launcher script instead, for three stated reasons: `blaunch` needs one allocation
covering both roles, server addresses are known only at run time from `LSB_HOSTS`, and
the launcher must wait on `/health` before fanning out trainer ranks. Upstream logs
expressing the shape as a granite.build step as an open, unresolved decision.

The `mem://` service pattern already in this repo dissolves the last two rather than
working around them:

- the address travels as a `mem://` binding, resolved at run time by construction;
- a target does not dispatch until **every** input binding resolves, and `vllm-server`
  publishes its URL only *after* `/health` answers — so the dependency graph **is** the
  health gate, with no polling loop to write;
- teardown becomes a third target.

The first reason — one allocation — is the one this recipe tests rather than dissolves.

## The graph

```
vllm-server ──vllm_url──> train ──checkpoint──> teardown
     └────────────────cluster_name───────────────┘
```

Ordering is implicit; there is no `depends_on` key. `vllm-server` has no inputs, so it
starts immediately. `train` waits on `vllm_url`. `teardown` waits on the trainer's
`checkpoint` — emitted as the trainer's **last** log line, which is what puts teardown
after training — and on `cluster_name`.

`gate` in `teardown`'s inputs carries no special meaning; it is the conventional name
for a binding consumed only for sequencing.

## Three things that break silently

**`mem://` with `.binding.state`, never `env://` with `.binding.path`.** `env://` runs
the value through filesystem-path normalisation and mangles `http://host:8001` into
`/http:/host:8001`. `recipes/granite4-350m/lsf/bcb-server/` still demonstrates the bug.
The recipe test asserts the accessor, because `.binding.path` would render empty and the
trainer would launch pointed at nothing.

**Teardown is not optional.** The provisioner refuses `idle_minutes_to_autostop` for
SSH/HPC clouds, so an LSF SERVICE cluster never autostops and never gets a
terminal-status cleanup. Drop the `teardown` target and the server's allocation is held
until someone runs `sky down` by hand. Note the corollary: a run that never reaches a
checkpoint never tears down either, which is why `SAVE_STEPS` is 1.

**The server must serve the STUDENT.** On-policy GOLD has the student generate and the
teacher score those generations. Serving the teacher is not an error — it is a different
algorithm that runs to completion and reports a plausible loss. `STUDENT_MODEL` is used
twice on purpose, and a test asserts the two uses agree.

## What a real on-policy run would change

This recipe is a gate, not the arm. For the published sweep's λ=1.0 arm — 135.7 GPU-h,
the only arm in the top group on both benchmarks — three things differ:

1. **`STUDENT_MODEL` should name a good off-policy checkpoint**, not the base retagged
   student. On-policy continues from one; a base student generates noise. Noise is fine
   for a plumbing gate, since the sync path is exercised either way.
2. **`MAX_STEPS: 100`** and the sweep's geometry (`PER_DEVICE` 4 × `GRAD_ACCUM` 6 =
   effective batch 192), at 16384/4096 context.
3. **`MAX_LENGTH` and the server's `max_model_len` must move together.** They are the
   same parameter here for that reason: a server window smaller than the trainer's
   context means the server refuses prompts the trainer happily builds, mid-run, after
   both allocations are held.

## Effective batch

1 × 1 × 1 × 8 = **8**. Not meaningful at two steps. Worth noting the shape it mirrors:
the published on-policy arm also trains on 8 GPUs while holding 16 — the difference is
that its second node sits inside the same allocation.

## Lineage and outputs

`train` declares the teacher, student and dataset as input artifacts, alongside the
`vllm` binding — which is not an artifact and carries no type. Output is `checkpoint`
(type `model`) at `$GB_BUILD_WORKDIR/checkpoints/<run_name>_node<N>`, registered by
rank 0. `vllm-server`'s two `mem://` outputs are bindings, not artifacts: nothing is
transferred, so neither carries a lineage type.
