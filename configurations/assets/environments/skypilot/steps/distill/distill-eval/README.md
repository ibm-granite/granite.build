# distill-eval

Measures **how far the student's next-token distribution is from the teacher's** on a held-out
corpus, and writes the result as an artifact. This is the read on whether distillation
transferred anything, and unlike a capability benchmark it needs no labels.

| | |
|---|---|
| **Type** | `custom` |
| **Environment** | SkyPilot, LSF only (`subtypes: [lsf]`) |
| **Image** | `docker:us.icr.io/cil15-shared-registry/kd-sandbox-distill:0.1.0-uv` (prebuilt; this step builds none) |
| **GPU** | **yes** — two models' forward passes. Accelerators come from the build.yaml |
| **Emits** | `eval_metrics` (`type: fileset`) → `<output_dir>/{metrics.json,per_sample.jsonl}` |

## What this measures, and why bfcl-eval cannot

`bfcl-eval` scores tool-calling against ground truth: *is the student right?* It structurally
cannot answer *did the student move toward the teacher?*, because that is a **distance between
two models' distributions on the same inputs**, not a property of one model's outputs.

A recipe wires **both** — `bfcl-eval` for capability, this for transfer. They are not
substitutes in either direction. A recipe also wires `space://steps/bfcl-eval` **directly**
rather than having this step wrap it: a wrapper would add a config surface that has to track
bfcl-eval's own, and would hide its artifacts behind a second declaration.

| Metric | Reads as |
|---|---|
| `jsd` | symmetric, bounded by `ln 2`, and the arm's own training objective. The headline. |
| `kld` | forward KL — penalises the student for missing teacher mass (mode-covering). |
| `rkld` | reverse KL — penalises mass where the teacher has none (mode-seeking). |
| `entropy` | the student alone. The only metric computable with **no** teacher, and the one that catches a collapsed student whose divergence happens to look fine. |

All four are reductions over the **same** pair of logit tensors, so asking for four costs
**one** forward pass over the corpus. `kld` and `rkld` are reported together because which one
moved says *which way* the student is wrong, and a single symmetric number cannot.

## `output_dir` is a FILESET

`metrics.json` and `per_sample.jsonl` are a measurement **of** a model, not data anything trains
on. Both `fileset` and `dataset` are real `ArtifactType` members, so the distinction costs
nothing to get right — and an undeclared output would be worse than a mistyped one, because the
resolver drops a `NEWARTIFACT` event whose id is not declared and the target then completes with
no output *and* no error.

## The two keys that break a measurement quietly

**`max_length` should match the training run's.** Too small a value damages the measurement two
ways, one loud and one silent:

- a record whose **prompt alone** exceeds `max_length` is dropped outright;
- a record whose prompt fits but whose **answer** does not is scored on a *prefix* of that
  answer — and this leaves **no trace in `n_samples`**, so a clean-looking `jsd` can be a mean
  over half-read completions.

Only long conversations are hit by either, so a too-small value reports a length-biased estimate
as the divergence. `max_incomplete_fraction` (default `0.25`) is the refusal threshold over both
cases combined; set it to `1.0` to measure a short slice on purpose.

**`corpus` must be the EVAL split**, not the split the student trained on. Measured on training
data, a divergence reports how well the student memorised the teacher's outputs on seen inputs —
the one thing distillation is guaranteed to improve, and therefore the least informative thing
to check. `distill-corpus-prep` writes `eval.jsonl` beside `train.jsonl` whenever
`eval_fraction > 0`; it is deliberately not a declared artifact, so read it from the manifest's
`splits.eval.path` or compose it from the same parameters.

## Config

Source delivery (`code_config`) is identical in every ported distillation step — see
[distill-tokenizer-align's USAGE.md](../../distill-tokenizer-align/skypilot/USAGE.md#source-delivery).

| Key | Default | Notes |
|---|---|---|
| `eval_config.student_model` | `""` | `distill-hf-export`'s `hf_model` post-training, or align's `retagged_student` for the t=0 baseline. |
| `eval_config.teacher_model` | `""` | **Optional, and the emptiness is meaningful**: with no teacher only `entropy` is computable, and asking for `jsd` without one is *refused* rather than defaulted. |
| `eval_config.corpus` | `""` | The eval split — see above. |
| `eval_config.output_dir` | `eval` | Relative resolves against `$GB_BUILD_WORKDIR`. Passed to the script as `--out-dir`. |
| `eval_config.metrics` | `jsd,kld,rkld,entropy` | Comma-separated. |
| `eval_config.max_samples` / `seed` | `256` / `42` | **Sampled** with the seed, not taken as a prefix: corpora are often sorted by source or length, so a prefix measures one slice and the mean moves when the count changes. |
| `eval_config.max_length` / `max_incomplete_fraction` | `4096` / `0.25` | See above. |
| `eval_config.batch_size` | `4` | |
| `eval_config.dtype` | `bfloat16` | Matches training. `float32` doubles memory for a metric whose differences are far larger than bf16's error, but it is there for a suspicious result. |
| `eval_config.allow_tokenizer_mismatch` | `false` | Keep it false: index *i* denotes a different token to each model, so the arithmetic **succeeds and measures nothing**. |

## Where the artifact marker lives

Unlike `distill-corpus-prep` and `distill-hf-export`, the marker is printed by **`src/run-eval.sh`**,
not by the template. The script calls `publish_artifacts()` on both the success path **and** the
already-measured SKIP path, because a resumed recipe whose eval says "already done" and then
publishes nothing has broken whatever reads `eval_metrics`.

Upstream's template echoed the same marker *again* after the script returned, which would
register two `NEWARTIFACT` events for one id. The port drops that duplicate, and a test asserts
the template prints no marker.

## Sanity checks that need no known answer

Worth applying to any result before believing it:

- `jsd < ln 2` (≈ 0.6931). This bound holds for **any** pair of distributions, so exceeding it
  means the mixture term is wrong rather than the models being far apart.
- `entropy < ln(vocab)`.
- `kld ≠ rkld`. If they are equal, the direction argument has not reached the reduction — the
  case that caught a defective upstream *test*, which probed the two directions with a permuted
  pair for which they are equal by symmetry.

## Tests

```bash
make test                                                   # 30 contract tests, no checkout needed
GB_DISTILL_CODE_DIR=/path/to/checkout make test              # + 21 ported (10 skip without torch)
```

Ten of the ported tests need torch and skip without it. That is upstream's design, not an
oversight: the metric arithmetic is worth testing against real tensors, and the image that has
torch is the cluster's, not CI's.
