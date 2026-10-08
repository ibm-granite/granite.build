# eval (SkyPilot / LSF)

Measures **how far the student's next-token distribution is from the teacher's** on a
held-out corpus, and writes the result as an artifact. This is the read on whether
distillation transferred anything, and unlike a capability benchmark it needs no labels.
It runs two models' forward passes on a GPU, in a prebuilt image, on SkyPilot's LSF
backend only.

> **Developing or testing this step?** See `steps/distill/eval/skypilot/README.md` in the
> granite.build repository for how the step is generated, tested, and published —
> including the end-to-end test that needs a BlueVela allocation.

## Referencing the step

Point your build's Space at one that provides the step, then reference it by the stable
`space://steps/distill/eval` URI:

```yaml
steps:
  - step_uri: space://steps/distill/eval
```

## Config contract (`eval_config`)

All fields in the first two tables live under the step's `config.eval_config`.

### Required

| Field | Type | Purpose |
|---|---|---|
| `student_model` | string | Path to the student model directory. The `hf_model` output of the `space://steps/distill/hf-export` step for a post-training measurement, or the `retagged_student` output of the `space://steps/distill/tokenizer-align` step for the t=0 baseline. `src/run-eval.sh` refuses an empty value. |
| `corpus` | string | Path to the JSONL corpus to measure on — the **eval split**, not the training split (see [Notes and limitations](#corpus-must-be-the-eval-split)). `src/run-eval.sh` refuses an empty value. |
| `teacher_model` | string | Path to the teacher model directory, normally the same teacher the trainer used. Required with the default `metrics`: with no teacher only `entropy` is computable, and asking for `jsd` (or `kld`/`rkld`) without one is **refused** rather than defaulted. Leave it empty only together with `metrics: "entropy"`. |

### Optional

| Field | Type | Purpose |
|---|---|---|
| `output_dir` | string | Directory `metrics.json` and `per_sample.jsonl` are written to, and the path registered as `eval_metrics`. A relative value resolves against `$GB_BUILD_WORKDIR`. Default: `eval`. |
| `metrics` | string | Comma-separated subset of `jsd`, `kld`, `rkld`, `entropy`. All four cost one forward pass. Default: `jsd,kld,rkld,entropy`. |
| `max_samples` | integer | Number of corpus records measured, **sampled** with `seed` rather than taken as a prefix. Default: `256`. |
| `seed` | integer | Sampling seed for `max_samples`. Default: `42`. |
| `max_length` | integer | Token budget per record. Should match the training run's. Default: `4096`. |
| `max_incomplete_fraction` | number | Refusal threshold over dropped plus truncated records combined; `1.0` measures a short slice on purpose. Default: `0.25`. |
| `batch_size` | integer | Forward-pass batch size. Default: `4`. |
| `dtype` | string | Model dtype. `bfloat16` matches training; `float32` is there for a suspicious result. Default: `bfloat16`. |
| `allow_tokenizer_mismatch` | boolean | Compute even when the two tokenizers disagree on token ids. Keep it `false`. Default: `false`. |

> **`max_length` and `corpus` break a measurement quietly**, not loudly. Read
> [The two keys that break a measurement quietly](#the-two-keys-that-break-a-measurement-quietly)
> before changing either.

### Source delivery (`code_config`)

The step imports the shared distillation package from a source checkout resolved at run
time. `code_config` is identical in every ported distillation step and is documented once —
see "Source delivery" in the `space://steps/distill/tokenizer-align` step's README. The
defaults clone a public repository at a pinned commit, so a recipe normally sets nothing
here.

| Field | Type | Purpose |
|---|---|---|
| `code_dir` | string | Existing checkout to use instead of cloning. Default: `""` (clone). |
| `expect_ref` | string | Commit `code_dir` must be at; a mismatch fails the step. Default: `a5d59bc45524a8d75706e20d44ae1a254f273f23`. |
| `repo` | string | Repository cloned when `code_dir` is empty. Default: `https://github.com/laminair/gb-steps-distillation.git`. |
| `ref` | string | Branch, tag or commit checked out after the clone. Default: the same commit as `expect_ref`. |
| `workdir` | string | Clone destination, relative to the working directory. Default: `distill-code`. |
| `token_secret` | string | Name of a space secret holding a read credential for a private `repo`. Default: `""` (public clone). |
| `python` | string | Interpreter that runs the workload. Default: `/stage/.venv/bin/python`. |
| `setup_command` | string | Bash run after the source resolves and before the workload. Default: `""` (skipped). |

### Top-level step config

| Field | Type | Purpose |
|---|---|---|
| `compute_config` | mapping | Node and GPU counts, e.g. `{num_nodes: 1, num_gpus_per_node: 1}`. The step pins no accelerators itself. |
| `launcher_config` | mapping | SkyPilot launcher overrides, notably `resources` (`accelerators`, `cluster`, `zone`, `memory`) and `image_id`. |
| `poll_interval_seconds` | integer | Monitor status-poll interval. Default: `300`. |
| `log_retrieval_mode` | string | Monitor log retrieval. Default: `periodic` (the shared monitor's default is `on_completion`). |
| `log_retrieval_interval_seconds` | integer | Interval for `periodic` log retrieval. Default: `300`. |

## Inputs and outputs

### Inputs

The step declares no `inputs:` of its own. It reads three paths from `eval_config` —
`student_model`, `teacher_model` and `corpus` — and a recipe supplies them by declaring
inputs on the target and passing their paths in with `{{ bindings.<name>.binding.path }}`:

- `student` — a `binding:` to `align.retagged_student` (baseline) or `export.hf_model`
  (post-training), passed as `student_model`.
- `teacher_model` — a direct `uri:` with `type: model`. Binding the teacher (rather than
  passing a bare hub id) lets a `hf://` or `s3://` teacher be resolved and cached through
  granite.build's asset stores instead of triggering an uncached fetch inside transformers
  at eval time.
- `corpus` — a `binding:` to `corpus.corpus`, the output of the
  `space://steps/distill/corpus-prep` step. It is bound for the **ordering edge**, not the
  path: `eval.jsonl` is not a declared artifact, so `eval_config.corpus` is composed from
  the same root the corpus target wrote to (or read from the corpus manifest's
  `splits.eval.path`).

### Outputs

| Output | Type | What it is |
|---|---|---|
| `eval_metrics` | `fileset` | The `output_dir` directory, holding `metrics.json` (per-metric summaries and counts) and `per_sample.jsonl`. |

The marker is printed by `src/run-eval.sh`, not by the step template, on both the success
path and the already-measured SKIP path:

```
GB_ARTIFACT_ID:eval_metrics GB_ARTIFACT_PATH:<absolute output_dir>
```

Declare `eval_metrics` on the target (typically `uri: "env://{{ binding.path }}"`,
`type: fileset`). An undeclared output is dropped by the resolver without an error.

## Working directory and paths

- The run block sets `WORK="${GB_BUILD_WORKDIR:-$PWD}"`. A relative `output_dir` is
  absolutised to `$WORK/<output_dir>` before the script runs, because the marker path is
  handed to the `env://` store, possibly from another host.
- `student_model`, `teacher_model` and `corpus` are passed to the script unchanged. Use
  absolute paths; bindings already resolve to absolute paths.
- The source is cloned into `$WORK/<code_config.workdir>` (default `distill-code/`), or
  taken from `code_config.code_dir`, and `$CODE_DIR/src` is prepended to `PYTHONPATH`.
- The step's `src/` directory is mounted at `./src`; the workload is
  `bash ./src/run-eval.sh`, run with `PYBIN` set to `code_config.python`.
- `HF_HOME` is `/opt/hf-cache`, and the Hub stays reachable (see
  [Runtime](#runtime)).

## Example build.yaml

A t=0 baseline and a post-training measurement against the same teacher and corpus.
`align`, `corpus` and `export` are targets running the `space://steps/distill/tokenizer-align`,
`space://steps/distill/corpus-prep` and `space://steps/distill/hf-export` steps; they are
elided here.

```yaml
granite.build:
  name: distill-eval-example
  version: 0.0.1
  targets:
    # align, corpus, train, export: elided.

    eval-transfer-baseline:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      inputs:
        student:
          binding: align.retagged_student
        corpus:
          binding: corpus.corpus        # ordering edge; the path is composed below
        teacher_model:
          uri: "env:///proj/models/teacher"
          type: model
      outputs:
        eval_metrics:
          uri: "env://{{ binding.path }}"
          type: fileset
      steps:
        - step_uri: space://steps/distill/eval
          config:
            compute_config:
              num_nodes: 1
              num_gpus_per_node: 1
            launcher_config:
              resources:
                accelerators: "H100:1"
                cluster: "bluevela"
                zone: "normal"
                memory: 64
            eval_config:
              student_model: "{{ bindings.student.binding.path }}"
              teacher_model: "{{ bindings.teacher_model.binding.path }}"
              corpus: "/proj/run/corpus/eval.jsonl"
              output_dir: "/proj/run/eval-transfer-baseline"
              max_length: 4096          # same budget as corpus prep and training

    eval-transfer:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      inputs:
        student:
          binding: export.hf_model
        corpus:
          binding: corpus.corpus
        teacher_model:
          uri: "env:///proj/models/teacher"
          type: model
      outputs:
        eval_metrics:
          uri: "env://{{ binding.path }}"
          type: fileset
      steps:
        - step_uri: space://steps/distill/eval
          config:
            compute_config:
              num_nodes: 1
              num_gpus_per_node: 1
            launcher_config:
              resources:
                accelerators: "H100:1"
                cluster: "bluevela"
                zone: "normal"
                memory: 64
            eval_config:
              student_model: "{{ bindings.student.binding.path }}"
              teacher_model: "{{ bindings.teacher_model.binding.path }}"
              corpus: "/proj/run/corpus/eval.jsonl"
              output_dir: "/proj/run/eval-transfer"
              max_length: 4096
```

Everything except the student is held fixed between the two targets, which is what makes
the two numbers comparable. The baseline depends only on `align` and `corpus`, so it runs
concurrently with training.

## Notes and limitations

### What this measures, and why bfcl-eval cannot

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

A divergence with no baseline is a number without a direction, which is why a recipe also
measures the `retagged_student` before training.

### The two keys that break a measurement quietly

**`max_length` should match the training run's.** Too small a value damages the measurement two
ways, one loud and one silent:

- a record whose **prompt alone** exceeds `max_length` is dropped outright;
- a record whose prompt fits but whose **answer** does not is scored on a *prefix* of that
  answer — and this leaves **no trace in `n_samples`**, so a clean-looking `jsd` can be a mean
  over half-read completions.

Only long conversations are hit by either, so a too-small value reports a length-biased estimate
as the divergence. `max_incomplete_fraction` (default `0.25`) is the refusal threshold over both
cases combined; set it to `1.0` to measure a short slice on purpose. The step's log prints a
`NOTE:` line with the count of dropped and of truncated records when either is non-zero.

#### `corpus` must be the eval split

`corpus` must be the **eval** split, not the split the student trained on. Measured on training
data, a divergence reports how well the student memorised the teacher's outputs on seen inputs —
the one thing distillation is guaranteed to improve, and therefore the least informative thing
to check. The `space://steps/distill/corpus-prep` step writes `eval.jsonl` beside `train.jsonl`
whenever `eval_fraction > 0`; it is deliberately not a declared artifact, so read it from the
manifest's `splits.eval.path` or compose it from the same parameters.

### Other config notes

- **`max_samples` is sampled**, with `seed`, not taken as a prefix: corpora are often sorted by
  source or length, so a prefix measures one slice and the mean moves when the count changes.
  A divergence mean stabilises well before a corpus is exhausted, and each sample is a teacher
  forward pass, so this is a cost knob.
- **`dtype`**: `bfloat16` matches training. `float32` doubles memory for a metric whose
  differences are far larger than bf16's error, but it is there for a suspicious result.
- **`allow_tokenizer_mismatch`**: keep it `false`. Index *i* denotes a different token to each
  model, so the arithmetic **succeeds and measures nothing**. It is kept as a knob only for a
  deliberate cross-tokenizer experiment, and setting it prints a warning in the step's log.
- **`teacher_model` empty** is "no teacher", not a blank path: the script omits the flag
  entirely rather than passing `""`.

### `output_dir` is a FILESET

`metrics.json` and `per_sample.jsonl` are a measurement **of** a model, not data anything trains
on. Both `fileset` and `dataset` are real `ArtifactType` members, so the distinction costs
nothing to get right — and an undeclared output would be worse than a mistyped one, because the
resolver drops a `NEWARTIFACT` event whose id is not declared and the target then completes with
no output *and* no error.

### Where the artifact marker lives

Unlike the `space://steps/distill/corpus-prep` and `space://steps/distill/hf-export` steps, the
marker is printed by **`src/run-eval.sh`**, not by the template. The script calls
`publish_artifacts()` on both the success path **and** the already-measured SKIP path, because a
resumed recipe whose eval says "already done" and then publishes nothing has broken whatever
reads `eval_metrics`.

Upstream's template echoed the same marker *again* after the script returned, which would
register two `NEWARTIFACT` events for one id. The port drops that duplicate, and a test asserts
the template prints no marker.

### Resume

A restarted recipe re-runs every step, so the script first asks whether this measurement has
already been made (`[0/2] resume check`). Identity is the student's and teacher's content plus
the sampling and length policy, not the `student_model` path.

- Nothing recorded: measure.
- Recorded under the exact same expectation: print the marker and exit 0 (SKIP).
- Recorded under a different expectation, or `metrics.json` / `per_sample.jsonl` is missing:
  **refuse** and fail, naming the key that differs. Nothing is overwritten; use a fresh
  `output_dir` to re-measure.

`batch_size` is not part of the expectation. The completion record is written only after the
post-condition check below passes.

### Sanity checks that need no known answer

The script fails the step when `metrics.json` is missing, holds no metric summaries, or any
metric has `n_samples < 1`, a NaN/infinite or negative mean, or a `jsd` mean above `ln 2`.
Worth applying to any result before believing it:

- `jsd < ln 2` (≈ 0.6931). This bound holds for **any** pair of distributions, so exceeding it
  means the mixture term is wrong rather than the models being far apart.
- `entropy < ln(vocab)`.
- `kld ≠ rkld`. If they are equal, the direction argument has not reached the reduction — the
  case that caught a defective upstream *test*, which probed the two directions with a permuted
  pair for which they are equal by symmetry.

### Runtime

- **Type** `custom`; **environment** SkyPilot, LSF only (`subtypes: [lsf]`).
- **Image** `docker:us.icr.io/cil15-shared-registry/kd-sandbox-distill:0.1.0-uv`, prebuilt; this
  step builds none.
- **GPU required** — two models' forward passes. Accelerators come from the build.yaml, so one
  step serves a 3B-teacher smoke run and a 30B-teacher reference run.
- **No `HF_HUB_OFFLINE`**, unlike the tokenizer-only ported steps. This step loads models, and
  granite 4.x is hybrid Mamba: at load time the `kernels` package resolves
  `kernels-community/causal-conv1d` through the Hub API, so offline mode fails with
  `OfflineModeIsEnabled` after the load has started (build 15267e81). This matches the
  `space://steps/distill/gold` step, which loads a granite 4.x student and a 30B teacher on the
  same image with `HF_HOME` set and the Hub reachable.
- **Periodic log retrieval.** A teacher forward pass over 256 samples is not a seconds-long job,
  and `on_completion` retrieval surfaces nothing until the end, so a stalled run would look
  identical to a slow one.
