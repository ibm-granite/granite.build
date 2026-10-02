# GOLD blend smoke recipe

On-policy-**blended** GOLD distillation (`lmbda 0.3`) with vLLM as a **separate target**
instead of nodes carved out of the trainer's own allocation. Two optimizer steps, one
trainer node, one server node, 4096-token context — the shared-path pair `gold-smoke`
already proved end to end, at the loss-blend fraction that mixes generated and fixed-corpus
samples.

## Where this sits relative to its siblings

- `gold-smoke` (this family) proved the pair (`granite-4.2-30b` teacher, `granite-4.1-3b`
  student) off-policy (`lmbda 0.0`), to completion, on real BlueVela hardware — loss 0.282,
  checkpoint saved.
- `gold-onpolicy-smoke` (this family) proved the **shape** — vLLM as a separate target,
  synced over NCCL across two LSF allocations (build `9973e766`, all 362 parameter tensors,
  both steps) — but against a *different*, personal-path pair, purely as a plumbing gate.
- **This recipe** combines the two: the proven pair, the proven shape, at `lmbda 0.3`. Not
  yet run. It is the first real test of the blend arm against this pair.

## Why `lmbda 0.3` changes the topology, not just the loss

Above `lmbda 0.0`, the trainer needs a vLLM server to generate from — `vllm_num_servers`
must be `>= 1`, and `vllm_server_url` (a `mem://` binding to `vllm-server`'s published URL)
is what makes every node in `train`'s allocation a trainer while the server runs on its own,
separate allocation. See `gold-onpolicy-smoke/README.md` for the full writeup of why this
shape (rather than nodes carved out of one allocation) was chosen and how it was verified.

## Sequencing: this needs a non-cold starting checkpoint for a real run

`STUDENT_MODEL` here defaults to the same base retagged student `gold-smoke` used —
correct for a smoke run (the NCCL sync path is exercised regardless of what the student
generates), but **not** correct for a real blend run. Per `distill-gold`'s own `USAGE.md`
and `gold-onpolicy-smoke`'s README: on-policy/blend should continue from a good off-policy
checkpoint, not the base student, since a base student generates noise under vLLM.

For a real run:
1. Run `gold-smoke` (or a longer off-policy run) first and note its `checkpoint` output
   path (`$GB_BUILD_WORKDIR/checkpoints/<run_name>_node<N>`).
2. Pass that path into this recipe:
   ```
   gb build start -f recipes/granite4-gold-distillation/lsf/gold-blend-smoke/build.yaml \
     --param STUDENT_MODEL=<off-policy checkpoint path>
   ```

There is no automatic chaining between these recipes — this is a manual, two-invocation
sequence, matching every other recipe in this repo's convention (no recipe here invokes
another).

## Run

```
gb build start -f recipes/granite4-gold-distillation/lsf/gold-blend-smoke/build.yaml --space <your-space>
```

Override any value with `--param KEY=VALUE`.

## What to check for success

Matching `gold-onpolicy-smoke`'s own verified signature — watch the `train` target's log
for lines like:

    t_gen 5.3  t_train 3.4  t_sync 1.5  t_sync_ag 0.1  t_sync_bc 1.4  n_sync_params 362

`n_sync_params` should equal the full parameter-tensor count (362, for this pair). `t_gen`
non-zero confirms the student really generated against the server rather than reading fixed
completions. Both are 0 in an off-policy run — their presence here is the evidence the
blend path is genuinely exercised, not just configured.

## Known gaps (step-level, not fixable from this recipe)

- `distill-gold` has no `seed` key — the trainer's random seed cannot be pinned from a
  recipe. Not a blocker for a smoke run; worth raising upstream if exact reproducibility
  across runs matters later.
- No independent on-policy rollout sampling temperature (`top_p` is the only exposed
  sampler knob) — a real capability gap relative to the staging repo's tuned runs if
  temperature parity ever matters.

## Effective batch

1 × 1 × 1 × 8 = **8**. Not meaningful at two steps — this recipe measures plumbing, not
science, same caveat as `gold-onpolicy-smoke`.

## Lineage and outputs

`train` declares the teacher, student and dataset as input artifacts, alongside the `vllm`
binding (not an artifact, carries no type). Output is `checkpoint` (type `model`) at
`$GB_BUILD_WORKDIR/checkpoints/<run_name>_node<N>`, registered by rank 0.
