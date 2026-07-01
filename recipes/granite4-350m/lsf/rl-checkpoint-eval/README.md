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

1. **Generate** the build:

   ```shell
   python generate_build.py --workflow ifrl \
     --param 'EVAL_SETS=[bfcl, multilingual-eval]' \
     --output build.yaml
   ```

   The generator prints the resolved checkpoint list, eval list, and total
   target count to stderr. `--workflow identityrl` omits the `code-server`
   target (IdentityRL trains without a code server).

2. **Start** the build (parameters are substituted at this step, as usual):

   ```shell
   gb build start -f build.yaml \
     --parameters-path parameters.yaml --space <your-space>
   ```

   Keep the same `parameters.yaml` for both steps: the generator only expands
   the knobs it needs (checkpoint schedule, eval selection); everything else
   stays as `$${...}` placeholders resolved by `gb build start`.

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

Smoke default: `2048 // (64*16) = 2` updates, `SAVE_FREQ=1` ⇒ checkpoints at
steps **1** and **2**. Each checkpoint is fanned out to the selected evals, so
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
- `export-combined` — gated on **all** per-checkpoint exports, runs the sage
  exporter once over the whole `<EXPERIMENT>` tree to produce
  `<SAGE_RESULTS_DIR>/<EXPERIMENT>/combined.csv`.

## Trainer step change

`configurations/.../steps/openinstruct-rl/step.yaml` now runs a background
watcher during training that emits `LLMB_ARTIFACT_ID:checkpoint_<step>
LLMB_ARTIFACT_PATH:<dir>` per new checkpoint dir, in addition to the
backward-compatible final `checkpoint` line the older recipes bind to. The
per-step ids match the generated `checkpoint_<step>` training outputs.

## Verification status (issue #45)

Provisional until confirmed against a real BlueVela grpo_fast run:

- **Checkpoint dir naming.** The watcher assumes `output_dir/step_<N>`. If
  grpo_fast names dirs differently (`global_step_N`, `checkpoint-N`), adjust
  `CKPT_GLOB`/parsing in the step and the generator's naming together.
- **Mid-run emission.** Confirm the periodic monitor surfaces each checkpoint
  line while the job is RUNNING (not only at terminal status).
- **Combined exporter.** Confirm `sage/exporters/exporter.py` rolls up multiple
  `ckpt_*` experiment dirs into one CSV distinguishable by checkpoint; if not,
  `export-combined` should post-process the per-checkpoint CSVs (which the gates
  already guarantee exist).
