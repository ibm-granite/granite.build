# distill-stage1-v2 — SUPERSEDED

> **This recipe was written to fix a run that was not broken.** Its premise — that a full
> epoch of pure-divergence GOLD destroys the model — was an artefact of a broken **export**,
> not of the objective. The four-coefficient CE sweep it was built to run has since
> completed and found the anchor's dose-response monotone-bad on capability. Both changes
> that defined v2 are now off, and the recipe runs the same unanchored objective as
> [`distill-stage1`](../distill-stage1/README.md), on the same public trainer pin as every
> other distillation recipe here.
>
> **For a full epoch, run [`distill-stage1`](../distill-stage1/README.md). For "is there a
> good checkpoint earlier than one epoch", run
> [`distill-checkpoint-eval`](../distill-checkpoint-eval/README.md)** with a shorter horizon
> — `GOLD_MAX_STEPS`, `GOLD_SAVE_STEPS`, `CKPT_LADDER` and `STUDENT_MODEL` are the four
> knobs, and it reads every rung with the full 27-eval suite instead of divergence alone.
>
> This file remains because the CE arms were run from it and the one question the sweep left
> open — safety retention — would be too. See [Re-arming the anchor](#re-arming-the-anchor).

Off-policy GOLD distillation of the granite-4.0-350m SFT checkpoint towards a
granite-4.1-3b teacher. `lmbda 0.0`, `beta 0.5`, 2 nodes x 8 H100, effective batch 96,
8192 context, 300 steps instead of an epoch, a checkpoint ladder weighted towards the
descent, capability measured at every rung, and every checkpoint kept.

**Run [`distill-probe`](../distill-probe/README.md) and
[`distill-smoke`](../distill-smoke/README.md) first.** Both are minutes; this is hours.

## Why the premise was withdrawn

`distill-stage1` ran as build `df8512e0` and read as a broken model on every one of the 30+
benchmarks measured. That reading was wrong. The **export** dropped the model's top-level
`rope_theta` — transformers 5.8.0 nests it under a sub-config — so vLLM served a
10M-theta model at 10k. Fixed 2026-09-27. Re-evaluated, the same checkpoint is the best
model in the epic:

| benchmark | baseline (SFT) | df8512e0, as first read | df8512e0, export repaired |
|---|---|---|---|
| GSM8K | 26.84 | 2.35 | **37.53** |
| IFEval average | 52.98 | 32.79 | **60.39** |
| Minerva | 16.12 | — | **20.10** |
| MGSM average | 17.60 | 0.08 | **20.72** |
| HumanEval pass@1 | 40.85 | 0.61 | 41.46 |
| MultiPL-E Java pass@1 | 21.05 | 0.00 | 20.91 |
| MMLU (mc) | 34.71 | 27.47 | 35.01 |
| BFCL v3 (all cat) | 41.87 | — | 42.07 |
| SALAD-Bench average | 93.47 | 79.78 | 90.65 |

Five clear gains, six axes flat within ±1pt, and two real regressions — **both safety**
(SALAD −2.82, AttaQ −1.2). Nothing had to be retrained; only published configs were wrong.

**What survives the correction.** The measurements v2 was built on are real; only their
consequence was not. The student started 12% of the way from the teacher (JSD 0.0825
against a `ln 2` = 0.693 ceiling) with ~68% of the realisable gain banked by step 509, and
over the remaining 7,640 steps its entropy fell 42% (0.684 → 0.396 nats) while the train
loss moved 2.7%. So the divergence gain really is early and the run really does sharpen — it
just does not cost capability.

**And the anchor that was supposed to prevent the sharpening costs capability itself.** The
CE sweep completed at four coefficients on 2026-09-28. At step 300, as `CE_COEF` goes
0 → 0.05 → 0.15 → 0.40:

| `ce_coef` | Δ ifeval avg | Δ HumanEval /164 | Δ EvalPlus(2) |
|---|---|---|---|
| 0.00 | **+5.57** | **+1** | 39.02 |
| 0.05 | +4.04 | 0 | 38.72 |
| 0.15 | +2.71 | −3 | 37.50 |
| 0.40 | +2.56 | **−9** | 33.84 |

Three strictly monotone orderings across four levels on suites that share no items. No
single pairwise gap clears its own noise band — the *ordering* is the finding, and it has no
interior optimum. `ce015` was the last candidate for a usable middle and sits exactly where
the trend predicts. **The anchor is dropped, not tuned.**

## What changed from stage1, and what is now back to it

| | stage1 | v2 as shipped now | why |
|---|---|---|---|
| `CE_COEF` | — | **0.0** | retired; the sweep above |
| `LOG_STUDENT_ENTROPY` | — | **false** | patch-gated; the same quantity is read per rung by `distill-eval` |
| `ENTROPY_GUARD_DROP_FRAC` | — | **0.0** | retired; it only ever warned, and the collapse was harmless |
| `CODE_DIR` | `""` (public clone) | **`""`** | nothing patch-gated renders, so no local checkout is needed |
| `GOLD_MAX_STEPS` | 0 (epoch) | 300 | the unanchored entropy curve is 96% collapsed by step 250 |
| `GOLD_SAVE_STEPS` / `_TOTAL_LIMIT` | 1000 / 3 | 25 / 16 | keep the whole curve, not its last 750 steps |
| `CKPT_LADDER` | — | 25,50,75,100,150,200,300 | export + transfer-eval + BFCL + generation check per rung |
| `NCCL_DEBUG` | `""` | `INFO` | build `8f02b739` hung with no trace and had to be relaunched |

The last four are the only reason to run this file rather than `distill-stage1`: a cheap,
densely-laddered, fully-kept 300 steps. `df8512e0`'s `save_total_limit 3` — which deleted
every checkpoint anyone later wanted to look at — is still the one line worth not repeating.

Held fixed on purpose: the corpus and its sampling, the geometry (effective batch 96), the
LR schedule, `beta 0.5`, `lmbda 0.0`, the model pair, `MAX_LENGTH 8192`. The comparison
against `df8512e0` is only attributable if the objective is the thing that moved.

Also not changed, and it is the one recommendation from the post-mortem not taken:
`completion_boundary: last_message` still leaves 34,650,329 assistant tokens (10.2%)
unsupervised, because every earlier assistant turn in a multi-turn row is masked out.
Changing it re-preps the corpus (~30–45 min) and changes what the loss averages over, so
v2's loss would stop being comparable to `df8512e0`'s. Worth doing; worth doing separately.

## Re-arming the anchor

The sweep is complete and its zero arm is now the shipped default, so a plain launch is the
`df8512e0` objective:

```bash
gb build start -f recipes/granite4-350m/lsf/distill-stage1-v2/build.yaml --space <space>
```

`CE_COEF` and the guard are still wired, at zero. Re-arming them needs **a patched
trainer**: none of the six fields exist in `CustomGOLDConfig` at the step's pin
(`a5d59bc4`), and `render_gold_config.py` refuses to emit one that the delivered trainer
does not define — which is the loud failure and the one to want, since `TrlParser` would
otherwise reject the config on every node of an allocation already held.

```bash
# Build the patched checkout once, on a node that can reach github.com.
git clone https://github.com/laminair/gb-steps-distillation.git $CODE_DIR
git -C $CODE_DIR checkout a5d59bc45524a8d75706e20d44ae1a254f273f23
( cd $CODE_DIR/src/gb_steps_post_training/distillation &&
  patch -p2 -F3 < <this repo>/steps/distill/distill-gold/skypilot/patches/ce_anchor_and_entropy_guard.diff )
git -C $CODE_DIR commit -am "ce anchor and entropy guard"

# Then one arm per coefficient. RUN_NAME MUST differ per arm, and CORPUS_DIR pins the
# corpus so the sweep derives it once rather than once per arm.
CORPUS=/proj/granite-build/g4os/distill/distill-350m-s1v2-ce040/c20ed3c0-.../corpus
for C in 0 0.05 0.15 0.40; do
  gb build start -f recipes/granite4-350m/lsf/distill-stage1-v2/build.yaml --space <space> \
    --param CE_COEF=$C \
    --param CODE_DIR=$CODE_DIR \
    --param CODE_EXPECT_REF=$(git -C $CODE_DIR rev-parse HEAD) \
    --param CORPUS_DIR=$CORPUS \
    --param RUN_NAME=distill-350m-s1v2-ce$(echo $C | tr -d .)
done
```

**The one experiment still worth that spend is safety retention.** No CE arm was ever
measured for safety, and safety is the only axis where the unanchored epoch went backwards.
Read such an arm for *retention* of 93.5 / 82.5 rather than for lift, and watch the math
gains as the thing it might cost.

Details that remain load-bearing:

- **`RUN_NAME` must differ per arm.** gbserver refuses an artifact URI another build in the
  space has already registered; the target then reports SUCCESS with an empty output list
  and consumers wait forever — the `b5f030cd` failure mode.
- **A re-armed guard must WARN, never stop** (`ENTROPY_GUARD_ACTION: warn`). Build
  `d1acf1c0` is why: the guard tripped at step 77 of 2,000 and stopped, so `checkpoint-500`
  through `checkpoint-2000` were never written and all four `export-<N>` targets failed with
  `requested checkpoint does not exist`. A guard that may stop at an arbitrary step and a
  ladder that names fixed ones cannot both have their way.
  `test_a_stopping_guard_and_a_fixed_ladder_cannot_both_be_asked_for` holds that line.
- **The horizon is 300 and the rungs are early.** `bb779f1f` measured the unanchored curve
  over 2,000 steps: entropy −19% by step 80, −29.7% by 120, −34.1% by 250, −35.5% at 2,000
  — the last 1,750 steps bought 1.4%. JSD and forward KL were flat after 500 and reverse KL
  never moved. An earlier `500,1000,1500,2000` ladder put every rung on the plateau, which
  is why two builds produced no capability reading anywhere in the region that moves.
- **300 steps says nothing about epoch length.** Every arm at step 300 is below the full
  epoch on ifeval (55.5–58.5 vs 60.39) and level with it on EvalPlus. 300 steps is 3.7% of
  an epoch; the sweep speaks to the coefficient and to nothing else.

## The trainer, and why the pin moved back

`train-gold` runs trainer code that does not come from the pinned steps checkout. Until v2
nothing pinned what it *did* read: `distill-stage1` pointed it at
`/proj/granite-build/g4os/kd-sandbox` — another project's working tree, carrying **159
uncommitted files** including `gold/` — so `df8512e0` recorded a commit that does not
describe the code that ran.

v2 fixed that by pinning a local clone this project controls, and that clone had to be local
because it carried the CE patch. With the anchor retired, the pin moves back to the step's
own `code_config`: an unauthenticated clone of
[`github.com/laminair/gb-steps-distillation`](https://github.com/laminair/gb-steps-distillation)
at `a5d59bc4`. That is strictly better provenance than a `/proj` checkout — the clone path
is at `code_config.ref` by construction and cannot drift — and it is what makes this recipe
runnable anywhere that can reach github.com rather than on BlueVela alone.

`CODE_EXPECT_REF` is empty because `expect_ref` is only consulted on the `CODE_DIR` path. It
is the value to fill in the moment a `CODE_DIR` *is* set, because that is when a checkout
becomes mutable state again — and it brings the dirty-tree check with it.

The patch that builds a re-armed checkout still ships with the repo:
[`steps/distill/distill-gold/skypilot/patches/ce_anchor_and_entropy_guard.diff`](../../../../steps/distill/distill-gold/skypilot/patches/ce_anchor_and_entropy_guard.diff).

## What to read, in this order

Not the train loss. It was flat for 94% of `df8512e0` while the entropy fell 42%, so it
tells you nothing about what the run is doing — but note that the drop it hid turned out not
to matter. **Entropy is now a diagnostic, not a verdict.** The verdict is capability.

**Per-step `student_entropy`, `rkld` and `ce_anchor_loss` are no longer logged**, because
`LOG_STUDENT_ENTROPY` is patch-gated and off. The quantities are read per rung instead, from
`distill-eval`, which computes them from the checkpoint with no trainer involvement.

1. **The `eval-transfer-<N>/metrics.json` ladder** against `eval-transfer-baseline`, the
   t=0 column. `jsd` and `kld` are what the objective is minimising; `entropy` is the
   sharpening; `rkld` moved only 1.8% over a whole epoch while forward KL improved 36%,
   which is the shape of a student narrowing rather than covering the teacher's spread.
2. **`gen-smoke`'s table.** One line per rung: the fraction of generated lines inside a run
   of ≥3 identical lines. A degenerate model scores ~1.0 there, healthy code 0.0. Cheap, and
   the one instrument that reads generation directly.
3. **`eval-bfcl-<N>`** per rung against the baseline row. Note the export caveat below: a
   BFCL of 0.000 has once meant a broken `rope_theta` rather than a broken model.
4. **Only then `full-eval`**, on whichever rungs steps 1–3 say are worth ~4 GPU-h.

Workload stdout is not in `gb build log` and not in `build_job_log`; SkyPilot syncs it
back locally:

```bash
gb build log <build-id> --runner --all | grep -i "SkyPilot job.*on gb-"
grep -aE "loss|jsd|learning_rate" ~/sky_logs/<cluster>/1-<cluster>/run.log
```

(`student_entropy`, `rkld` and `entropy-guard` appear there only on a run launched with a
patched `CODE_DIR` and `LOG_STUDENT_ENTROPY=true`.)

## full-eval per surviving rung

Not wired into this recipe: it is ~4 GPU-h per model and needs `bcb-server` up for
`OE_EVAL_BCB_API_URL`. Start that first, then one launch per rung worth the spend:

```bash
gb build start -f recipes/granite4-350m/lsf/full-eval/build.yaml --space <space> \
  --param MODEL_PATH=/proj/granite-build/g4os/distill/distill-350m-stage1-v2/<build-id>/export-<N> \
  --param EXPERIMENT=distill-350m-stage1-v2-ck<N>
```

The tokenizer-control column from
[`distill-probe`](../distill-probe/README.md#the-tokenizer-control-column) is still
required before any of those rows is compared to the recorded after-SFT row: the probe
measured the checkpoint preferring the pinned tokenizer by 10.9% NLL/byte, and this
export pins `tokenizer_class` while the recorded row does not.

## Reusing a corpus across arms

`sources` + `corpus` sit on the critical path — train-gold cannot start until they
finish — and they are deterministic. The CE_COEF sweep measured it: the three arms that
completed (`9dab9130`, `8c8ebd63`, `c20ed3c0`) wrote byte-identical `sources/train.jsonl`,
`corpus/train.jsonl` and `corpus/eval.jsonl`, so a four-arm sweep derived the same 3 GB
file four times for ~50 minutes each.

`--param CORPUS_DIR=<dir>` skips both targets and reads that directory directly. It has
to be a **direct `uri` input** rather than a binding or a fixed `BUILD_SUBDIR`:

- gbserver's target reuse is scoped to one build id, and `docs/builds/target-reuse.md` is
  explicit that cross-build reuse is intentionally unsupported. `RUN_NAME` is in the
  corpus `out_dir` too, so a per-arm `RUN_NAME` defeats even a retry's reuse.
- Pinning `BUILD_SUBDIR` would make the second build re-register a URI the first already
  owns — the `b5f030cd` mode, where the registration is refused, the target still reports
  SUCCESS with an empty output list, and every consumer waits forever.
- A `binding` names another target's output *in this build*, so it cannot name a corpus a
  different build produced.

The corpus is **defined by** the retagged tokenizer and the prep policies: its rows were
rendered through that chat template, its masks built with that tokenizer, and 5,908 rows
dropped for exceeding *that* `max_length`. A pin from a run that differed in any of those
would train on a corpus this build does not describe, and every downstream metric would
still look normal. So `corpus-pin-check` reads `corpus_manifest.json` and compares
`tokenizer_identity`, `max_length`, `think_policy`, `documents_policy`,
`completion_boundary` and `eval_fraction` against this build — plus a byte comparison of
`tokenizer.json` and `chat_template.jinja` when the manifest's `tokenizer_path` still
exists. It reports every mismatch at once, and it is CPU-only and gated ahead of the
allocation, so a bad pin costs seconds rather than a run.

## Cost

| target | shape | time |
|---|---|---|
| `sources` | 1 CPU node | ~5 min at 800k rows; skipped when `CORPUS_DIR` is set |
| `align` | 1 CPU node | ~10 min |
| `corpus` | 1 CPU node, single-process | 30–45 min at 800k rows; skipped when `CORPUS_DIR` is set |
| `corpus-pin-check` | 1 CPU node | seconds; only when `CORPUS_DIR` is set |
| `train-gold` | 2 x 8 H100 | ~30 min for 300 steps (~4 s/it); nothing can end it early |
| `export-<N>` x7 | 1 CPU node each | minutes, concurrent |
| `eval-transfer-<N>` x5 | 1 H100 each | minutes each |
| `gen-smoke` | 1 H100 | ~2 min for all four rungs |
| `eval-bfcl-<N>` x8 | 1 H100 each, `simple` only | ~10 min each, concurrent; baseline + every rung |

~56 GPU-h per arm against `df8512e0`'s ~213. That cheapness is now the recipe's only
argument: it does not replace the epoch, it samples the first 3.7% of one.

## Next

Given the correction above, the honest next step is not this recipe. A full epoch is
[`distill-stage1`](../distill-stage1/README.md), and "which checkpoint of an epoch" is
[`distill-checkpoint-eval`](../distill-checkpoint-eval/README.md), which reads every rung
with the full suite rather than with divergence alone.

Stage 2 is still on-policy: [`distill-onpolicy-v2`](../distill-onpolicy-v2/README.md). It is
written but unlaunched — its student path is a chosen checkpoint, and `lmbda_schedule` has
never run in this pipeline.
