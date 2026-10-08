# GOLD sweep smoke recipe

The minutes-long gate for [`gold-sweep-100`](../gold-sweep-100/README.md). Two optimizer
steps, one 8×H100 node, 4096-token context — and otherwise the same run: the same trainer
code, the same image, the same retagged student, the same teacher snapshot, the same
802,027-row corpus, the same objective.

It exists because `gold-sweep-100` costs two hours and ~15 GPU-h, and almost everything
that can be wrong with it is wrong immediately: an unreadable `/proj/data-eng` path, a
response template that rendered as a trailing space, a `max_steps` that reached the
trainer as a string, a NaN loss from the fused kernel. None of those need 100 steps to
show up.

## Run

    gb build start -f recipes/granite4-gold-distillation/lsf/gold-sweep-smoke/build.yaml \
      --space <your-space>

## What "minutes" means

Most of this run's wall clock is **not** training. The trainer maps and filters the whole
802,027-row corpus before step 1, regardless of `max_steps`, and `DATASET_NUM_PROC` is 32
here precisely because it is 32 in `gold-sweep-100`. So this run measures that startup
too — which is worth knowing before watching a two-hour job appear to hang at step 0.

Two steps rather than one is also deliberate: one step proves the trainer starts, two
prove it can take a second optimizer step carrying the first step's state, which is where
a ZeRO-3 or scheduler misconfiguration actually surfaces.

## What it gates

Everything held identical to `gold-sweep-100` — and the recipe test asserts that list
rather than trusting it, because a gate is only a gate for what it holds fixed. A green
run here clears:

- the step's new `max_steps` and `save_strategy` keys, end to end into the rendered config;
- all three `/proj/data-eng/hew/...` paths, including the retagged student's tokenizer
  relationship with the teacher;
- the `beta: 0.5` objective;
- the trailing-newline `RESPONSE_TEMPLATE`;
- checkpoint publication (`SAVE_STEPS: 1`, one checkpoint kept);
- the host-memory sizing, which is driven by `DATASET_NUM_PROC` over a 3.2 GB corpus and
  is therefore unchanged from the real arm.

## What it cannot gate

**Memory at full context.** `MAX_LENGTH` is 4096 against the real arm's 16384 and
`PER_DEVICE_TRAIN_BATCH_SIZE` is 1 against 4 — a 16× difference in activation memory. An
OOM that only appears at full geometry will not appear here. This is the one respect in
which a green run here is not evidence about `gold-sweep-100`.

**Anything about the science.** Two steps at effective batch 8 measure nothing. The loss
value is only useful for the one question below.

## The one experiment to run here

`USE_LIGER_FUSED_JSD` is unresolved, and this is the cheapest place to resolve it. The
published sweep arm set it `true`; this project's own finding is that granite's
`logits_scaling=10` overflows the fused bf16 JSD kernel and gives NaN loss. Both cannot
be right for the trainer we run.

    # as shipped
    gb build start -f recipes/granite4-gold-distillation/lsf/gold-sweep-smoke/build.yaml --space <space>
    # then
    gb build start -f recipes/granite4-gold-distillation/lsf/gold-sweep-smoke/build.yaml --space <space> \
      --param USE_LIGER_FUSED_JSD=true --param RUN_NAME=gold-sweep-smoke-liger

`true` is usable only if the loss is finite **and** `beta` demonstrably moves it — upstream
documents the fused path as possibly ignoring `beta` altogether, though for a different
trainer. Record the answer in both this README and `gold-sweep-100`'s, whichever way it
goes. It matters for the real arm because the non-fused path materialises the
`[batch, seq, vocab]` logit tensor the fused kernel avoids, and at 16384 × 4 that may not
fit — which is exactly the failure this recipe cannot gate.

## Where it deviates on purpose

Everything else that differs from `gold-sweep-100` is a cost knob. Two are not:

| | `gold-sweep-100` | here | why |
|---|---|---|---|
| `NCCL_DEBUG` | `""` | `INFO` | a short log is readable; a two-hour one is not |
| `NCCL_TIMEOUT_MS` | 3600000 | 600000 | a hang should report in minutes, not tie up 8×H100 for an hour |

## Effective batch

1 × 1 × 1 × 8 = **8**, against the real arm's 192. Deliberately not preserved: effective
batch is not meaningful at two steps, and shrinking it is what keeps each step cheap.
Nothing compares this run's loss to anything.

## Lineage and outputs

Identical in shape to `gold-sweep-100`: three input artifacts (`teacher_model`,
`student_model`, `training_dataset`) and a `checkpoint` output of type `model` at
`$GB_BUILD_WORKDIR/checkpoints/<run_name>_node<N>`, registered by rank 0. That the
lineage wiring works is itself one of the things this run gates.
