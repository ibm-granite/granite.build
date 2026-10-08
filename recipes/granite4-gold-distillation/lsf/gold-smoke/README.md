# GOLD distillation smoke recipe

A minutes-long 2-node GOLD run on BlueVela (LSF via SkyPilot), before committing
to the full multi-hour run. It exercises the `distill-gold` step's plumbing — the
2-node allocation, per-node rank assignment, config rendering inside the container
against the `/proj` checkout, and checkpoint publication — not the science.

Everything expensive is scaled down: one epoch over a 2000-row slice, a 4096-token
context, `save_steps: 10`. The model pair is not — see below.

## Run

`parameters.yaml` sits next to `build.yaml`, so it is picked up automatically:

    gb build start -f recipes/granite4-gold-distillation/lsf/gold-smoke/build.yaml \
      --space <your-space>

Explicitly, or to point at a different parameter set:

    gb build start -f recipes/granite4-gold-distillation/lsf/gold-smoke/build.yaml \
      --parameters-path recipes/granite4-gold-distillation/lsf/gold-smoke/parameters.yaml \
      --space <your-space>

Override any value with `--param KEY=VALUE`. The three headline parameters:

    gb build start -f recipes/granite4-gold-distillation/lsf/gold-smoke/build.yaml \
      --space <your-space> \
      --param STUDENT_MODEL=/proj/granite-build/g4os/kd-sandbox/student_overlays/<student> \
      --param TEACHER_MODEL=/proj/granite-build/g4os/kd-sandbox/teacher_overlays/<teacher> \
      --param TRAINING_DATASET=/proj/granite-build/g4os/gbtest/gold-distill-smoke/<data>_nothink.jsonl

## Changing the pair

This is the one substitution to make carefully, and there are two separate constraints
with **opposite** failure behaviour.

Student and teacher must share a tokenizer — identical token IDs, not merely an
identical vocabulary size. This is a limitation of the `distill-gold` step, not of
GOLD: upstream GOLD distils across differing tokenizers via its ULD loss, and the
trainer carries that path, but the step exposes none of its keys. Break this one and
the trainer **raises** — `verify_tokenizer_consistency()` fails with both sources
named.

The `RESPONSE_TEMPLATE` constraint is the silent one. Only tokenizer families whose
vocabulary carries `<|im_start|>` work with the default value; on any other family the
template matches nothing, the completion span is never located, and the loss is
computed over the wrong span with **no error to read** — the run looks healthy and
means nothing. Nothing validates this, and a pair can satisfy the tokenizer check and
still fail here.

The tokenizer family does not track the version number, so the pairs have to be
grouped by hand. See
[Choosing a student/teacher pair](../../../../steps/distill/gold/skypilot/USAGE.md#choosing-a-studentteacher-pair)
in the step's `USAGE.md` for the survey and the grouping command.

## Lineage

`TEACHER_MODEL`, `STUDENT_MODEL` and `TRAINING_DATASET` are each also declared as
an input artifact on the target, so they appear as this target's inputs
(`teacher_model`, `student_model`, `training_dataset`) alongside the `checkpoint`
output. `gb build status <build-id>` shows them. Lineage is built from the target's
input artifacts, not from step config: a path handed only to `gold_config` would
train correctly and record nothing about what it trained from.

## Effective batch

`PER_DEVICE_TRAIN_BATCH_SIZE x GRADIENT_ACCUMULATION_STEPS x NUM_NODES x
NUM_GPUS_PER_NODE`. The reference run is 192 (1 x 6 x 4 x 8) and the LR schedule
was tuned against it. This recipe is 16 (1 x 1 x 2 x 8) on purpose: effective batch
is not meaningful at this scale, and a low step count is what a smoke run wants.

Anyone raising `NUM_NODES` for a real run must move
`GRADIENT_ACCUMULATION_STEPS` to hold 192 — otherwise they have changed the
optimization rather than the throughput.

## Outputs

`checkpoint` (type `model`) — `$GB_BUILD_WORKDIR/checkpoints/<run_name>_node<N>`,
registered by rank 0. The node count in that path comes from the allocation rather
than from a parameter, so it always describes the run that produced it.
