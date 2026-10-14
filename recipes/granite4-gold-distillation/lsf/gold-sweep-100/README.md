# GOLD 100-step off-policy sweep arm

The λ=0.0 arm of the published 100-step GOLD sweep, on BlueVela (LSF via SkyPilot):
100 optimizer steps over the full 802,027-row corpus at effective batch 192, on one
8×H100 node. Roughly 1.9 h and ~15 GPU-h.

This is the arm that sits on the cost/quality frontier. Against a plain-SFT control on
the same corpus and geometry it buys **+0.073 gsm8k strict-match**, twice the 2σ gate,
for 8.9 extra GPU-h — 16× cheaper per unit of capability than the λ=1.0 on-policy arm,
whose small edge over it is inside the noise. The λ=0.3 blend that published recipes
ship is beaten by this arm on both benchmarks at 3.9× the cost.

**Run [`gold-sweep-smoke`](../gold-sweep-smoke/README.md) first.** It is this recipe's
shape at 2 steps and 4096 context, and it exists so that a wrong path, a bad render or
an OOM costs minutes instead of two hours.

## Run

`parameters.yaml` sits next to `build.yaml`, so it is picked up automatically:

    gb build start -f recipes/granite4-gold-distillation/lsf/gold-sweep-100/build.yaml \
      --space <your-space>

Override any value with `--param KEY=VALUE`.

## What this reproduces, and what it does not

Same student, same teacher, same corpus, same geometry as the published arm — the three
artifacts are read directly from `/proj/data-eng/hew/...` rather than copied, so there is
no chance of a stale duplicate.

Two things differ, and neither is incidental:

- **The trainer.** This recipe's GOLD trainer comes from `code_config`'s pinned public-repo
  clone (`gb-steps-distillation`), not the checkout the published sweep ran under. The
  clone's commit is recorded automatically as `distill_code_commit` step metadata, so the
  run says which code produced it. A numeric mismatch against the published arm is
  therefore a **finding**, not a bug in this recipe.
- **No experiment tracking.** The published arm logged to ClearML project
  `gold-sweep100`; this recipe has none, because the step exposes no tracking keys and
  the upstream tracker treats a partial config as an error. Curves are in the job log;
  the checkpoint is the artifact.

## The liger question, unresolved

`USE_LIGER_FUSED_JSD` is `false` here, and the published arm set it `true`. That is a
deliberate divergence, not an oversight, and it is the one thing to settle before
spending this recipe's two hours.

- **Why not `true`:** granite's `logits_scaling=10` overflows the fused bf16 JSD kernel
  and yields NaN loss. That is this project's own recorded finding, and it is why the
  step's `USAGE.md` says to leave it off.
- **Why `true` may nonetheless be needed:** the non-fused path materialises the
  `[batch, seq, vocab]` logit tensor the fused kernel exists to avoid. At `MAX_LENGTH`
  16384 and `PER_DEVICE_TRAIN_BATCH_SIZE` 4 that may simply not fit, and the published
  sweep's own comment gives memory as the reason it chose the fused kernel.
- **A third possibility worth knowing about:** upstream documents the fused path as
  gated before the primary-loss chain, with the fused loss object constructed from
  `beta`, `alpha`, `temperature` and `use_kl_interpolation` only — so `beta: 0.5` under
  `liger_fused_jsd` may be silently ignored. That claim is about a *different* trainer
  than the one this recipe runs, so it is a hypothesis here, not a fact.

**How to settle it,** in `gold-sweep-smoke` rather than here: run with `false`, then with
`--param USE_LIGER_FUSED_JSD=true`. `true` is usable only if the loss is finite **and**
`beta` demonstrably moves it. Record the answer in both READMEs either way.

## Two values that are not tuning knobs

**`BETA: 0.5`.** [`gold-smoke`](../gold-smoke/README.md) runs `0.0`. The trainer
short-circuits `beta` 0.0 to forward KL and 1.0 to reverse KL, so 0.5 is the only one of
the three that is a genuine mixture. Changing it changes the objective.

**`RESPONSE_TEMPLATE` ends in a newline,** and it is written with **single** quotes in
`parameters.yaml` for a reason. Double-quoted, YAML turns `\n` into a real newline, Jinja
substitutes that newline into `build.yaml`'s double-quoted scalar, and YAML folds a
newline inside a double-quoted scalar to a **space** — so the value silently becomes
`<|im_start|>assistant ` and the loss-mask boundary moves by a token. Single-quoted, the
literal backslash-n survives to `build.yaml` and the escape is interpreted exactly once.
`test_gold_sweep_recipe.py` asserts the rendered value.

## The corpus and the retagged student

`TRAINING_DATASET` is prep's output: 802,027 rows kept / 9,145 dropped, schema
`{messages, row_id}`, 3.2 GB. It is **not** named `*_nothink`, unlike every kd-sandbox
dataset, so the filename convention that normally carries the think-filtering guarantee
is absent. It was checked by sampling instead — zero `<think>` occurrences in the first
20,000 rows, 2026-09-13 — and recorded in `_THINK_CHECKED_CORPORA` in the recipe test, so
repointing this parameter at an unchecked corpus fails a test rather than a run. An
inline `<think>...</think>` in an assistant turn breaks gold's completion extraction
silently.

`STUDENT_MODEL` is the **retagged** student: re-embedded onto the teacher's tokenizer by
the upstream `distill-tokenizer-align` step, which does not exist in this repo. Do not
substitute one of the `/proj/granite-build/g4os/kd-sandbox` overlays — the corpus was
prepared against this tokenizer relationship, and the step can only run the
shared-vocabulary JSD path (it exposes none of the trainer's cross-tokenizer `uld_*`
keys, so pre-alignment is the only route open to it). See
[Choosing a student/teacher pair](../../../../steps/distill/gold/skypilot/USAGE.md#choosing-a-studentteacher-pair).

## Effective batch

`PER_DEVICE_TRAIN_BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS × NUM_NODES ×
NUM_GPUS_PER_NODE` = 4 × 6 × 1 × 8 = **192**, the published sweep's geometry exactly. It
was held equal across all three of its arms, which is what made them comparable, and the
LR schedule was tuned against it. Anyone changing one of those four factors must
compensate in another, or they have changed the optimization rather than the throughput.
The recipe test asserts the product rather than the factors, so a compensating edit to
only one of them fails.

## One node, on purpose

The λ=0.0 arm spends its whole allocation on training — there is no vLLM server to carve
a node off for. So this recipe exercises *less* plumbing than `gold-smoke`, which remains
the multi-node proof. The on-policy arms are the ones that need a split allocation.

## Lineage

`TEACHER_MODEL`, `STUDENT_MODEL` and `TRAINING_DATASET` reach the trainer only as input
artifacts on the target — the gold step's required inputs `teacher`, `student` and
`corpus`, which it reads by their resolved binding paths — so they appear as this
target's inputs alongside the `checkpoint` output. Lineage is built from the target's
input artifacts, and there is no `gold_config` key for these paths, so the lineage record
and what the trainer loaded cannot disagree.

## Outputs

`checkpoint` (type `model`) — `$GB_BUILD_WORKDIR/checkpoints/<run_name>_node<N>`,
registered by rank 0. Four intermediate checkpoints at `SAVE_STEPS: 25`, which turn a
single end-point number into a learning curve. Each is ~40 GB: ZeRO-3 shards carry fp32
optimizer state, so a checkpoint is far larger than the 6.4 GB end-of-training model.
