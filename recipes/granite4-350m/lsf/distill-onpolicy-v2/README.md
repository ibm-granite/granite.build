# distill-onpolicy-v2 — on-policy GOLD, from stage 1 v2's chosen checkpoint

**Written, not yet run.** `STUDENT_MODEL` has no working default on purpose; see
[Choosing the student](#choosing-the-student). Everything else now runs on the step's own
public trainer pin.

> **There is no generation floor.** `MIN_COMPLETION_LENGTH` is 0, because neither half of it
> exists at the pin (`a5d59bc4`): `CustomGOLDConfig` has no `min_completion_length`, and
> `run_vllm_serve.py` never reads `GOLD_MIN_TOKENS`. A non-zero value buys two pre-allocation
> refusals rather than a floor. So the immediate-EOS failure it guarded against — an empty
> completion the teacher scores as if it were a rollout — is unmitigated and has to be *read*
> instead: check `gen-smoke` on the first rung before letting a long arm run. Restoring a
> floor means landing both halves upstream and bumping the pin.

On-policy GOLD distillation towards the same `granite-4.1-3b` teacher, continuing from
a chosen off-policy checkpoint. `lmbda 0.25`, `beta 0.5`, the same **unanchored**
objective as [`distill-stage1`](../distill-stage1/README.md), the same step-bounded horizon
and checkpoint ladder as [`distill-stage1-v2`](../distill-stage1-v2/README.md) — plus a vLLM
server in its own allocation and a teardown target.

> **The CE anchor and the entropy guard are off here too.** The completed CE sweep found the
> anchor monotone-bad on capability off-policy, and the collapse the guard watched for did
> no damage. The anchor's rationale was *strongest* in this regime — a student training on
> its own output has no corpus token pulling it back — so the removal is stated rather than
> inherited: the sweep ran off-policy and says nothing about on-policy directly, but it is
> the only evidence there is, so this arm starts where the evidence is. See
> [What is held fixed](#what-is-held-fixed).

## Why on-policy, after an off-policy arm

Off-policy scores the corpus's own tokens: text somebody else wrote. That is precisely
where an already-SFT'd student was already strong, and it is where build `df8512e0` found
no room — initial JSD 0.0825 against a `ln 2` ceiling, two thirds of the realisable gain
closed by step 509, then 7,640 steps with nothing left to learn and entropy reduction as
the only remaining descent direction.

On-policy scores the student's **own** generations, including the degenerate regions it
wanders into by itself. That is the exposure-bias gap: a student trained only on clean
text never learns to recover from its own mistakes, because it never saw one. It is also
where teacher correction has something left to say.

It is not free. Generation is the expensive half — the reference arms measured ~135 GPU-h
against off-policy's ~15 — and it is the half that can feed a collapse, since a student
training on its own output has no corpus token pulling it back. Hence `lmbda 0.25`: one
sequence in four generated and three still real text, which is the only thing now damping
that feedback. Read an arm at 0.25 before raising it, and read its entropy curve per rung
(`eval-transfer-<N>`) rather than trusting the off-policy sweep to cover this regime.

## Choosing the student

On-policy continues from a *good* off-policy checkpoint. Starting from the base retagged
student would run, report a plausible loss, and answer a question nobody asked — so the
default is an invalid path that fails at `align` in minutes rather than a plausible one
that trains the wrong model for three hours.

Pick the rung from stage 1 v2's readings, in this order: `student_entropy` still near its
step-0 value, `rkld` having actually moved, `gen-smoke` repetition near zero, then the
transfer metrics. Then:

```bash
gb build start -f recipes/granite4-350m/lsf/distill-onpolicy-v2/build.yaml --space <space> \
  --param STUDENT_MODEL=/proj/granite-build/g4os/distill/distill-350m-stage1-v2/<build-id>/export-<N>
```

## The topology, and the four ways it goes silently wrong

An **external** server in its own allocation, its URL arriving through a `mem://`
binding. This is `gold-onpolicy-smoke`'s shape, which synced all 362 parameter tensors
across two allocations in build `9973e766`. Not the in-allocation split that
`vllm_num_servers` alone selects: in that path the step passes no `--vllm_server_host`, so
the trainer looks for a server running on a different node.

| trap | what it looks like | what stops it here |
|---|---|---|
| serving the **teacher** | a complete run, plausible loss, a different algorithm | the `model` input is bound to `align.retagged_student`, and a test asserts it is not the teacher |
| server allocated, generation local | 8 idle H100s, zero requests, then a shape mismatch — build `d77546a9` | the step states `--use_vllm` positively in both directions, from the same key |
| URL mangled in transit | `/http:/host:8001` | `mem://`, not `env://`; `env://` normalises it as a filesystem path |
| immediate-EOS collapse | the teacher scoring empty completions as rollouts | `MIN_COMPLETION_LENGTH` on **both** sides |

That last one is worth spelling out. `min_completion_length` on the trainer bounds what it
*asks* for; the sampling decision is taken in the server process, and trl's `vllm_serve`
has no `min_tokens` field, so `run_vllm_serve.py` patches `SamplingParams` from
`GOLD_MIN_TOKENS` — which the `vllm-server` step exports from its own
`min_completion_length`. Set one and not the other and the floor does not exist.

## Teardown is a target, and it has one hole

LSF SERVICE clusters never auto-stop (the provisioner refuses
`idle_minutes_to_autostop` for SSH/HPC clouds) and never get a terminal-status cleanup, so
the `teardown` target is what releases the server's GPUs.

It is gated on `train-gold.checkpoint`, which a **crashed** trainer never emits, and
gbserver's schema has no on-failure semantics. Build `d77546a9` held 8 H100s that way
until they were downed by hand. `VLLM_MAX_LIFETIME_SECONDS` (5 h) is the backstop; it must
outlast the run, and a test asserts it covers 2,000 steps at the ~4 s/it measured on this
pair.

## The lmbda ramp is untested

`LMBDA_SCHEDULE: linear` with `LMBDA_INIT: 0.0` ramps from off-policy to `LMBDA` across
training: corpus signal early, the student's own distribution later. `lmbda_schedule` is a
trainer key this repo had not exposed before v2, so **it has never run in this pipeline** —
smoke it before spending a real allocation on it.

Read the constant arm first regardless. A ramp confounds *on-policy helped* with
*on-policy helped once the student was further along*.

## What is held fixed

Everything from [`distill-stage1-v2`](../distill-stage1-v2/README.md): `CE_COEF 0.0` and
the entropy guard disarmed — matched to the off-policy arm so the two remain comparable, and
enforced by `test_it_matches_the_off_policy_arm` — plus `beta 0.5`, 300 steps, the
25/50/75/100/150/200/300 ladder, the corpus and its sampling and the `CORPUS_DIR` pin, the
geometry (effective batch 96 — the server has its own
allocation, so `GOLD_NUM_NODES` still counts trainers), the LR schedule, the pinned
trainer. A test asserts each of those against the off-policy recipe's own value.

One variable. Stage 1 v2 changed the objective; this changes where the sequences come
from. An on-policy arm that also moved `beta` or the horizon would not be comparable to
the off-policy arm it is meant to be judged against.

`CORPUS_DIR` is part of that parity, and not only for symmetry: a corpus is ~50 minutes
on the critical path, and an on-policy arm that rebuilt it while the off-policy arm reused
a pin would be comparing two arms that had at least the opportunity to differ. Pointing
both at one directory removes that. `corpus-pin-check` guards it exactly as it does in the
off-policy arm — the script is the same script, and a test asserts the two are
byte-identical. See
[`distill-stage1-v2`](../distill-stage1-v2/README.md#reusing-a-corpus-across-arms).

The corpus is still the SFT mixture, which is the one deliberate difference from what the
recipe index calls stage 2 — that describes on-policy on the IFRL and IdentityRL *prompt*
sets. Swapping those in is `SOURCE_GENERAL` / `SOURCE_TOOLS` / `SOURCE_RAG`, and it should
be a separate arm: changing the prompts and the policy at once measures neither.

## Cost

Adds one 2-GPU allocation for the server's lifetime to stage 1 v2's ~56 GPU-h, and
generation makes each step slower — budget roughly 2-3x the off-policy arm's wall clock at
`lmbda 0.25`, and measure it on the smoke tier before committing.
