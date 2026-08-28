# rl-checkpoint-eval — dynamic checkpoint-fanout RL build

One build that runs **IFRL** or **IdentityRL** training, evaluates **every
checkpoint** the trainer writes, and aggregates the results — per-checkpoint
CSVs plus a combined roll-up across checkpoints. (Issue #45.)

Unlike `sft-eval-full-dataset` (which evaluates a single final checkpoint), this
recipe fans a *selected* eval suite out over *each* checkpoint of the run. Since
the build engine dispatches each downstream target exactly once (keyed by its
binding id), the fanout can't be open-ended: `generate_build.py` computes the
checkpoint schedule up front and emits one training output + one eval
target-set per checkpoint.

## Files

- `generate_build.py` — the generator. Reads `parameters.yaml` + `--param`
  overrides and `eval-catalog.yaml`, writes a plain `build.yaml`.
- `eval-catalog.yaml` — the 27 evals (transcribed from `sft-eval-full-dataset`)
  and named sets (`code-eval`, `general-eval`, `math-eval`, `safety-eval`,
  `multilingual-eval`, `bfcl`, `full-eval`). The generator's source of truth.
- `parameters.yaml` — RL knobs (from `ifrl-smoke`) + eval/resource knobs (from
  `sft-eval-full-dataset`) + eval selection + monitor cadence.

## Usage

1. **Generate** the build. Set the common knobs with flags (or leave them to
   `parameters.yaml`):

   ```shell
   python generate_build.py --workflow ifrl \
     --model-path /proj/.../sft/epoch_hf_2 \
     --output-dir /proj/.../rl-out/ \
     --total-episodes 20480 --save-freq 10 \
     --eval-sets 'bfcl,multilingual-eval' \
     --output build.yaml
   ```

   The generator writes **two** files and prints the checkpoint list, eval list,
   target count, and the exact start command to stderr:
   - `build.yaml` — the generated build.
   - `parameters-resolved.yaml` — the base parameters merged with your flags and
     `--param` overrides. **Pass this to `gb build start`** so those overrides
     are honored when the build's `$${...}` placeholders are resolved (override
     the path with `--params-out`).

   `--workflow identityrl` omits the `code-server` target (IdentityRL trains
   without a code server).

2. **Start** the build with the resolved parameters:

   ```shell
   gb build start -f build.yaml \
     --parameters-path parameters-resolved.yaml --space <your-space>
   ```

### Common parameters

The frequently-changed knobs are collected at the top of `parameters.yaml` and
each has an equivalent generator flag. The flag and `--param` both override the
file; `--param` wins on conflict.

| Concept | Parameter | Flag |
|---|---|---|
| Input checkpoint to train from (SFT ckpt for IFRL; a prior RL ckpt for IdentityRL) | `MODEL_PATH` | `--model-path` |
| Output dir the trainer writes checkpoints to | `OUTPUT_DIR` | `--output-dir` |
| Total training episodes (drives checkpoint count) | `TOTAL_EPISODES` | `--total-episodes` |
| Save a checkpoint every N updates (drives the schedule) | `SAVE_FREQ` | `--save-freq` |
| Trainer's in-loop eval frequency | `EVAL_FREQ` | `--eval-freq` |
| Which evaluations to run | `EVAL_SETS` | `--eval-sets` |
| Experiment namespace for eval/export outputs | `EXPERIMENT` | `--experiment` |
| How often training logs are downloaded + parsed for checkpoints | `RL_LOG_SCRAPE_INTERVAL_SECONDS` | `--log-scrape-interval` |
| How often training job status is checked | `RL_STATUS_POLL_INTERVAL_SECONDS` | `--train-status-interval` |
| How often eval job status is checked | `EVAL_STATUS_POLL_INTERVAL_SECONDS` | `--eval-status-interval` |

**Chaining stages** (e.g. IFRL → IdentityRL): `MODEL_PATH` is a static path, so
run the first build, then generate the second with
`--model-path <first run's final checkpoint>`.

## Selecting evaluations — `EVAL_SETS`

A list of **named sets and/or individual eval names** (see `eval-catalog.yaml`):

- `[full-eval]` — all 27 evaluations.
- `[bfcl, multilingual-eval]` — the single BFCL eval + the 5 multilingual evals.
- `[olmes-gsm8k, math-eval]` — an individual eval plus a whole set (de-duped).

Override at generation time: `--param 'EVAL_SETS=[code-eval]'`.

## How many checkpoints? — the schedule

open-instruct floor-divides:

```
num_updates    = TOTAL_EPISODES // (NUM_UNIQUE_PROMPTS_ROLLOUT * NUM_SAMPLES_PER_PROMPT_ROLLOUT)
checkpoints at = SAVE_FREQ, 2*SAVE_FREQ, …  (≤ num_updates, plus the final update)
```

Smoke default: `20480 // (64*16) = 20` updates, `SAVE_FREQ=10` ⇒ checkpoints at
steps **10** and **20**. Each checkpoint is fanned out to the selected evals, so
`total eval targets = #checkpoints × #evals`. The generator prints this — watch
it before starting a `full-eval` run over many checkpoints (that is a lot of
clusters on the `preemptable` queue).

## Monitor cadence (how soon checkpoints/evals are picked up)

Three knobs control detection latency (all in `parameters.yaml`, overridable):

| Parameter | Controls |
|---|---|
| `RL_CHECKPOINT_WATCH_INTERVAL_SECONDS` | How often the trainer's in-run watcher polls `output_dir` for a newly written checkpoint dir and emits it as an artifact. |
| `RL_LOG_SCRAPE_INTERVAL_SECONDS` | How often the training `skypilot_monitor` pulls + parses logs (in `periodic` mode) — bounds how soon an emitted checkpoint line reaches the build. |
| `RL_STATUS_POLL_INTERVAL_SECONDS` | How often the training job's SkyPilot status is polled. |
| `EVAL_STATUS_POLL_INTERVAL_SECONDS` | How often each eval target's status is polled — bounds how soon an eval's completion is detected so its export runs. |

Lower values detect sooner at the cost of more `sky logs`/status calls.

## Aggregation

- `export-sage-ckpt<step>` / `export-bfcl-ckpt<step>` — one exporter per
  checkpoint, gated on that checkpoint's eval outputs, producing
  `<SAGE_RESULTS_DIR|BFCL_RESULTS_DIR>/<EXPERIMENT>/ckpt_<step>-{sage,bfcl}.csv`.
- `export-combined` — gated on **all** per-checkpoint exports, pivots them into
  `<SAGE_RESULTS_DIR>/<EXPERIMENT>/combined.csv`: a **benchmark × checkpoint**
  table (rows are benchmarks — the sage `model`+`metric`, plus one `BFCL-<expt>`
  row per bfcl eval; columns are `ckpt_<step>`), so each row reads left-to-right
  as a metric's trajectory across the run. It reads whichever of sage/bfcl
  actually ran (either alone is fine).

## Trainer step change

`configurations/.../steps/openinstruct-rl/step.yaml` now runs a background
watcher during training that emits `GB_ARTIFACT_ID:checkpoint_<step>
GB_ARTIFACT_PATH:<dir>` per new checkpoint dir, in addition to the
backward-compatible final `checkpoint` line the older recipes bind to. The
per-step ids match the generated `checkpoint_<step>` training outputs.

## Verification status (issue #45)

Confirmed against real BlueVela grpo_fast runs:

- **Checkpoint dir naming.** grpo_fast writes HF checkpoints as `output_dir/step_<N>`
  (confirmed: `step_10`, `step_20`), which the watcher globs directly.
- **Completeness gating.** The watcher emits a checkpoint only once both
  `model.safetensors` and `tokenizer.json` are present. grpo_fast writes the
  weights early and `tokenizer.json` last (confirmed by mtime), so requiring
  both brackets the whole write sequence — a downstream eval never loads a dir
  that is still mid-write.
- **Mid-run emission.** Confirmed: in one run, training started at 03:16:49, the
  step-10 evals dispatched at 03:36:25, and training finished at 03:47:42 — the
  step-10 fanout began ~11 min *before* training completed. The final (step-20)
  evals dispatched just after completion, as expected.
- **Combined roll-up.** `export-combined` does **not** re-scan a sage result
  tree (sage-eval requires a flat experiment name, so per-checkpoint results are
  siblings, not a nestable tree). It pivots the per-checkpoint CSVs the gates
  already guarantee exist into a benchmark × checkpoint table — see Aggregation.

Operational risk:

- **Cluster leak on a naming mismatch.** If a future grpo_fast change renames
  checkpoint dirs (e.g. `global_step_N`), the watcher emits nothing: eval targets
  bound to a never-emitted `checkpoint_<N>` never dispatch, and `teardown` —
  gated on `training.checkpoint_<last_step>` — never fires, leaking the RL
  cluster (H100:8) + RM/code servers until idle-timeout. The training step now
  guards this by **failing loudly** (non-zero exit) when zero checkpoints are
  emitted, so the build fails cleanly instead of hanging. If you change the
  naming, update `CKPT_GLOB` in the step and `compute_checkpoint_steps` in the
  generator together.
