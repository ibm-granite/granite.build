# distill-checkpoint-eval — one epoch, nine checkpoints, every one fully evaluated

One build that runs **a full epoch** of off-policy GOLD distillation (granite-4.0-350m SFT
checkpoint ← granite-4.1-3b teacher), saves a checkpoint **every 1,000 steps plus the final
one at 8,150**, and — as each checkpoint is written, not after the run — **exports it and
fans it out to all 27 evaluations** of [`full-eval`](../full-eval/build.yaml). It ends with
one file: `combined.csv`, a benchmark × checkpoint table.

This is the distillation counterpart of
[`rl-checkpoint-eval`](../rl-checkpoint-eval/README.md), which does the same for RL.

**Run [`distill-probe`](../distill-probe/README.md) and
[`distill-smoke`](../distill-smoke/README.md) first.** Both are minutes; this is ~13 hours
of 2×8 H100 plus ~36 GPU-h of evaluation.

## Why this exists alongside distill-stage1-v2

They answer different questions and neither substitutes for the other.

| | [`distill-stage1-v2`](../distill-stage1-v2/README.md) | this |
|---|---|---|
| question | superseded; its premise was an export bug and its CE anchor is retired | **which checkpoint** of a full epoch is the best model? |
| horizon | 300 steps (`GOLD_MAX_STEPS`) | one epoch, ~8,150 steps |
| ladder | 25,50,75,100,150,200,300 — weighted to the descent | 1000…8000, 8150 — an even grid |
| reads per rung | divergence, entropy, repetition, BFCL `simple` | **all 27 benchmarks**, BFCL at `all` |
| fanout timing | after training completes | **as each checkpoint lands** |
| when a reading is bad | `gen-smoke` fails the build | nothing acts; you read the table |
| cost | ~56 GPU-h | ~249 GPU-h |

`distill-stage1-v2` deliberately does not wire `full-eval` in — it is ~4 GPU-h per model and
needs `bcb-server` up, so it is launched by hand afterwards on whichever rungs looked
healthy. That is the right trade when you are bracketing a collapse inside 120 steps. It is
the wrong one when the question is where in a 13-hour epoch the best model sits, because
divergence metrics demonstrably cannot answer it: on build `bb779f1f` every divergence
metric improved while BFCL sat at 0.000.

And the epoch itself has been run exactly once, as `distill-stage1` / build `df8512e0`:
8,150 steps, 13h20, every step exit 0, **one** model exported — "the last one" — which was
also its worst. HumanEval 40.85 → 0.61, MultiPL-E Java 21.05 → 0.00, worse on all 30+
benchmarks. Its good stopping point was near step 500, and `save_total_limit 3` had already
deleted every checkpoint before 7750 by the time anyone knew. The epoch had to be repeated
in full. Nine measured rungs and `GOLD_SAVE_TOTAL_LIMIT 12` are the two lines that stop that
happening again.

## The mid-run fanout, and the one step change it needed

`train-gold` declares one `checkpoint_<N>` output per rung, and the `distill-gold` step now
carries an **opt-in** background watcher (`emit_checkpoint_artifacts`, added for this recipe,
modelled on the one `openinstruct-rl` already had) that emits each one as soon as that
checkpoint is completely written. So `export-1000` and its 27 evals dispatch about eleven
hours before the epoch ends, and the evaluation overlaps the training instead of queueing
behind it. `rl-checkpoint-eval` confirmed the mechanism on a real run: its step-10 evals
dispatched ~11 min before training completed.

Three details are load-bearing:

- **The binding is the whole design.** `export-<N>` binds `train-gold.checkpoint_<N>`, not
  the final `train-gold.checkpoint` that v2's exports bind. Reverting it is a one-word edit
  that nothing else would notice, so
  `test_every_export_is_gated_on_its_own_mid_run_checkpoint` asserts it.
- **A checkpoint is emitted only when it is complete.** The watcher waits for
  `model.safetensors`, `tokenizer.json` *and* `trainer_state.json`.
  `Trainer._save_checkpoint` writes in a fixed order — `save_model` → optimizer/scheduler →
  rng_state → `trainer_state.json` — so requiring all three brackets the entire write
  sequence. A plain non-empty check would race it and hand a half-written model to an export
  that has already allocated.
- **Emission is rank-0 only.** The executor streams every node's stdout into one driver log,
  so an unguarded marker would register each checkpoint twice on a 2-node run.

Detection latency is `CKPT_WATCH_INTERVAL_SECONDS` + `LOG_RETRIEVAL_INTERVAL_SECONDS`
(120 + 120, so ≤ ~4 min): the watcher prints the line immediately, but it only reaches the
build when the monitor next pulls the job's logs.

If the naming ever changes and the watcher emits nothing, the step **fails loudly** rather
than letting nine export targets wait forever for artifacts that will never arrive.

## ⚠ The last rung is measured, not derived

`8150` is what one epoch over `TARGET_ROWS 800000` at effective batch 96 actually came to on
build `df8512e0`, which retained `checkpoint-7750`, `-8000` and `-8150`. **Nothing in the
recipe can compute it**: the step count falls out of the corpus only after prep drops
over-length rows (5,908 last time) and holds out `EVAL_FRACTION`. It is also the one rung
that is not a multiple of the save grid, because HF Trainer writes a final checkpoint at the
last step whatever the grid says.

So if `TARGET_ROWS`, `EVAL_FRACTION`, `MAX_LENGTH` or the batch geometry change, the last
rung must change with them. To read the real total off the trainer's own logs:

```bash
gb build log <build-id> --runner --all | grep -i "SkyPilot job.*on gb-"
grep -aoE "/[0-9]+ \[" ~/sky_logs/<cluster>/1-<cluster>/run.log | head -1
# then, if it differs:
gb build start ... --param CKPT_LADDER=1000,2000,...,<real total>
```

A wrong last rung costs less here than it would in v2: because the fanout is mid-run, the
eight earlier rungs have already been exported and evaluated by the time it bites.

## Nothing acts on a reading

`eval-transfer-<N>` reports divergence and entropy per rung and `gen-smoke` reports
repetition. Both are **instruments**: neither stops the run, fails a target, or skips a rung.
The entropy guard is disarmed outright (the completed CE sweep retired it), so there is no
longer even a warning path that could end the run early. The whole epoch is the measurement, so a
reading that truncated it would destroy the thing being measured. `gen-smoke` in particular
carries **no exit gate** here, unlike v2's copy which fails on a degenerate final rung —
`test_gen_smoke_reports_without_gating` holds that line.

Read them anyway, and in this order:

1. **`student_entropy` per step**, in the workload stdout. Holding within ~15% of its step-0
   value is the run behaving; the guard prints its trip step either way.
2. **`gen-smoke`'s table** — one line per rung, the fraction of generated lines inside a run
   of ≥3 identical lines. A collapsed model scores ~1.0, healthy code 0.0.
3. **The nine `eval-transfer-<N>/metrics.json`.** `entropy` here is the same quantity the
   trainer logs, on held-out data.
4. **`combined.csv`** — the capability answer, and the deliverable.

Workload stdout is not in `gb build log` and not in `build_job_log`; SkyPilot syncs it back:

```bash
gb build log <build-id> --runner --all | grep -i "SkyPilot job.*on gb-"
grep -aE "student_entropy|rkld|ce_anchor_loss|entropy-guard" ~/sky_logs/<cluster>/1-<cluster>/run.log
```

## Reading combined.csv

`<SAGE_RESULTS_DIR>/<EXPERIMENT>/combined.csv`. Rows are benchmarks (the sage `model` +
`metric` pair, plus one `BFCL-<experiment>` row per rung), columns are `ckpt_1000` …
`ckpt_8150`. Each row therefore reads left to right as one metric's trajectory across the
epoch — the question "which checkpoint" is answered by scanning a row, not by opening nine
result trees.

It does **not** re-scan a sage result tree: sage cannot take a `/` in an experiment name, so
per-rung results are siblings (`<EXPERIMENT>-ckpt_<N>`) rather than a nestable tree. The step
pivots the per-rung CSVs that its gates already guarantee exist, and tolerates either eval
kind being absent.

There is **no `ckpt_0` column**. The after-SFT baseline row is already recorded
(`sft-epoch2-tokpinned-r1`); regenerating it would spend a GPU to reproduce a number we
hold. Compare against it out of band — and read
[`distill-probe`](../distill-probe/README.md#the-tokenizer-control-column) first, because
this export pins `tokenizer_class` and the recorded row does not.

Two known traps when a row reads 0.000:

- **BFCL 0.000 has twice been a tool-*format* failure**, not a capability one — the model
  writes correct code and emits no `<tool_call>`.
- **A dropped top-level `rope_theta`** made vLLM serve granite-4 at RoPE 10k instead of 10M.
  Fixed in the export step, but check the exported `config.json` before believing a
  catastrophic row.

## Launch

`bcb-server` must be up first, or `olmes-bigcodebench` fails on all nine rungs:

```bash
gb build start -f recipes/granite4-350m/lsf/bcb-server/build.yaml --space <space>

gb build start -f recipes/granite4-350m/lsf/distill-checkpoint-eval/build.yaml \
  --space <space> --param OE_EVAL_BCB_API_URL=http://<host>:7860/evaluate/
```

**Shake it down first.** The whole graph, the watcher, the mid-run bindings and the roll-up
in minutes instead of a day:

```bash
gb build start -f recipes/granite4-350m/lsf/distill-checkpoint-eval/build.yaml \
  --space <space> \
  --param RUN_NAME=distill-350m-ckpt-eval-shakedown \
  --param EXPERIMENT=distill-350m-ckpt-eval-shakedown \
  --param TARGET_ROWS=2000 --param GOLD_MAX_STEPS=4 \
  --param GOLD_SAVE_STEPS=2 --param CKPT_LADDER=2,4 \
  --param EVAL_CATEGORIES=bfcl
```

`RUN_NAME` **must** differ per launch. gbserver refuses an artifact URI another build in the
space has already registered; the target then reports SUCCESS with an empty output list and
every consumer waits forever — the `b5f030cd` failure mode.

### Reusing a corpus

`sources` + `corpus` are ~50 min on the critical path and are deterministic (three CE_COEF
sweep arms wrote byte-identical files). `--param CORPUS_DIR=<dir>` skips both and reads that
directory directly, with `corpus-pin-check` comparing its `corpus_manifest.json` —
tokenizer identity, `max_length`, think/documents policy, completion boundary, eval
fraction — against this build before any allocation is held. See
[`distill-stage1-v2`](../distill-stage1-v2/README.md#reusing-a-corpus-across-arms).

### Trimming the suite

`EVAL_CATEGORIES` selects from `code,general,math,safety,multilingual,bfcl`. All six is the
default because comparability with the recorded full-suite row is the point.
`--param EVAL_CATEGORIES=bfcl,math,general` is 11 evals per rung instead of 27. The build
drops the exporters for an eval kind that was not selected, so a trimmed run still produces
`combined.csv`.

## Cost

| target | shape | count | time |
|---|---|---|---|
| `sources` | 1 CPU node | 1 | ~5 min at 800k rows; skipped when `CORPUS_DIR` is set |
| `align` | 1 CPU node | 1 | ~10 min |
| `corpus` | 1 CPU node | 1 | 30–45 min; skipped when `CORPUS_DIR` is set |
| `corpus-pin-check` | 1 CPU node | 0–1 | seconds; only when `CORPUS_DIR` is set |
| `train-gold` | 2 × 8 H100 | 1 | **~13h20** for one epoch (~213 GPU-h) |
| `export-<N>` | 1 CPU node | 9 | minutes each, concurrent, starting ~1h40 in |
| `eval-transfer-<N>` | 1 H100 | 9 | minutes each |
| `gen-smoke` | 1 H100 | 1 | ~4 min for all nine rungs |
| sage evals | 1 H100 each | **234** | ~4 GPU-h per rung |
| `eval-bfcl-ck<N>` | 1 H100 | 9 | ~30 min each at `all` |
| CSV exporters + roll-up | 1 CPU node | 19 | minutes |

**285 targets, ~249 GPU-h.** Most of the eval time hides inside the epoch, but these are 243
LSF allocations on the `preemptable` queue — the queue will feel it. Training stays on
`normal` so a 13-hour run cannot be preempted mid-epoch.

## Next

Whichever rung `combined.csv` selects is stage 2's student path:
[`distill-onpolicy-v2`](../distill-onpolicy-v2/README.md).
