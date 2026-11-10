# Distillation recipes

Logit distillation of a student model from a teacher, with per-checkpoint exports,
divergence reads and optional benchmark evals.

## What this is

| Path | What it is |
|---|---|
| [`distill/`](distill/build.yaml) | The full pipeline: corpus, alignment, training in one of three λ modes, an export and a divergence read per rung, a repetition check, and optional benchmark evals. |
| [`distill-eval/`](distill-eval/build.yaml) | Benchmark evals of models that are already exported: a ladder of a `distill` build's exports, failed evals re-run, or any model given by URI. Its eval targets are `distill`'s, text for text. |
| [`distill/smoke.yaml`](distill/smoke.yaml) | A parameter layer, not a recipe: shrinks `distill` to a plumbing run (64 rows, 2 steps, 1 node with 2 GPUs) in any λ mode. |
| [`site.example.yaml`](site.example.yaml) | Every site-specific key, with fake values. |
| [`environments/skypilot-slurm.template.yaml`](environments/skypilot-slurm.template.yaml) | The SkyPilot-on-SLURM environment, as a template filled from the site file. |

The recipes need a SkyPilot environment, and are tested against SkyPilot on SLURM (render
tests only; see the last section). They are scheduler-neutral: every launcher uses only SkyPilot's generic keys (`cluster`, `zone`,
`accelerators`, `memory`). Everything SLURM-specific (partitions, account, QOS,
constraint, sbatch options) lives in the environment template.

### The `distill` targets

```
sources ──► corpus ───────────────┐        sources + corpus are replaced by
              ▲                   │        corpus-validate when CORPUS_DIR is set
align ────────┴──► [train-sft] ──►├──► train ──► export-N ──┬──► divergence-N
  │                               │     │                   ├──► repetition-check (every rung)
  │             [rollout-server] ─┘     └──► [teardown]     └──► [eval-*-ckN] ──► [export-sage-ckN,
  │                                                                               export-bfcl-ckN]
  └──► divergence-baseline (the aligned student, before training)                  ──► [export-combined]
```

`corpus` and `corpus-validate` read the retagged tokenizer from `align`. `rollout-server`
serves the same student binding `train` trains (`align` or `train-sft`). Bracketed targets
are conditional:

| Target | Present when |
|---|---|
| `sources`, `corpus` | `CORPUS_DIR` is empty |
| `corpus-validate` | `CORPUS_DIR` is set |
| `train-sft` | `INCLUDE_SFT` is true |
| `rollout-server`, `teardown` | `LMBDA_MODE` is `ramp` or `on-policy` |
| `export-N`, `divergence-N` | one each per `CKPT_LADDER` rung |
| `eval-<name>-ckN`, `eval-bfcl-ckN`, `export-sage-ckN`, `export-bfcl-ckN`, `export-combined` | `SCREEN_EVALS` or `FULL_EVAL_RUNGS` select an eval for rung N |

Every rung's export binds a checkpoint the trainer emits mid-run, so rung N is exported and
measured while training continues.

## Setup

**Requires layered `--parameters-path`** ([#663](https://github.com/ibm-granite/granite.build/issues/663),
PR [#665](https://github.com/ibm-granite/granite.build/pull/665)). Each command below passes
the site file, and the smoke run passes `smoke.yaml` too, as layers over the recipe's own
`parameters.yaml`. Without it, the site file replaces `parameters.yaml` instead of layering
over it.

**1. The site file.** Copy the example to a private location outside this repo (a private
repo, or `~/.granite.build/local-env/`) and fill it in. The real one names your hosts,
users, accounts and paths, so never commit it.

```bash
mkdir -p ~/.granite.build/local-env
cp recipes/distill/site.example.yaml ~/.granite.build/local-env/distill-site.yaml
SITE=~/.granite.build/local-env/distill-site.yaml
```

The file's comments describe each key. The groups:

| Group | Keys describe | Read by |
|---|---|---|
| Environment: identity and scheduler | the SkyPilot cluster name, GPU and CPU partitions, GPU constraint, SLURM account and QOS | the template |
| Environment: filesystems | shared workdir, SkyPilot workdir and tmpdir, container bind mounts, image cache, HF cache | the template |
| Environment: SSH | login nodes, user, and the **name** of the secret holding the SSH key | the template |
| Recipes: where they run | `ENVIRONMENT_URI`, `WORKDIR_ROOT` (the shared root of every output) | the recipes |
| Recipes: models and data | teacher, student, dataset and its splits, smoke dataset, chat template | the recipes |
| Recipes: compute | training and eval accelerator models, `NCCL_IB_HCA` | the recipes |
| Recipes: evaluation | eval and export partitions, eval image registry, sage and BFCL output roots, the BigCodeBench server URL | the recipes |

Secrets are named, never stored: `SSH_KEY_SECRET` is the name of a space secret holding the
private key. Define that secret in the space that runs the environment (see
[Secrets in `cluster_ssh_configs`](../../docs/environments/skypilot-slurm.md#secrets-in-cluster_ssh_configs));
with the space `setup-local-space.sh` writes, that is the `public` space.

**2. The environment.** Render the template with the site file:

```bash
scripts/setup-local-space.sh \
  --template recipes/distill/environments/skypilot-slurm.template.yaml \
  --site "$SITE" --name <env-name>
```

For a standalone gbserver it writes:

- `<space-dir>/overlay/environments/skypilot-local/<env-name>/environment.yaml`, the
  rendered environment (mode 600; it holds site values, so keep it out of git);
- `<space-dir>/space.yaml`, a standalone space named `public` that lists the overlay first,
  then this checkout's `configurations/assets`. An existing `space.yaml` that already lists
  the overlay is kept; one that does not is left unchanged, with a warning.

`<space-dir>` defaults to `$GB_HOME/local-space` (`GB_HOME` is `$GB_HOME_DIR`, or
`~/.granite.build`); `--space-dir DIR` changes it. For a private repo backing a shared
gbserver, pass `--out DIR` instead: it writes only
`DIR/environments/skypilot-local/<env-name>/environment.yaml`, for you to commit there, and
touches neither `GB_HOME` nor `space.yaml`.

Either way it sets `ENVIRONMENT_URI` in the site file to
`space://environments/skypilot-local/<env-name>` and leaves every other line as it was,
including a trailing `# comment` on that line (`--env-key KEY` names a different key). A
site file that is a symlink, say from a private repo into `~/.granite.build/local-env/`,
stays one: the real file is edited. It reads no secret and is safe to re-run. A template
that fails to render writes nothing, and nor does a site file whose key it cannot replace
on one line: a value spanning several lines, or the key given twice.

**3. gbserver.** For the standalone case:

```bash
gbserver standalone --space-dir ~/.granite.build/local-space
```

## Running

Set `RUN_NAME` per run, with a timestamp. Every output path hangs off it, and the rollout
server's `mem://` URIs carry it: two builds may not register the same URI.

```bash
TS=$(date +%Y%m%d_%H%M)
D=recipes/distill/distill/build.yaml
```

| Run | Command |
|---|---|
| Full run (off-policy, the defaults) | `gb build start -f $D --parameters-path $SITE --param RUN_NAME=distill-$TS` |
| Smoke, off-policy | `gb build start -f $D --parameters-path $SITE --parameters-path recipes/distill/distill/smoke.yaml --param RUN_NAME=smoke-off-$TS --param LMBDA_MODE=off-policy` |
| Smoke, ramp | same, with `--param RUN_NAME=smoke-ramp-$TS --param LMBDA_MODE=ramp` |
| Smoke, on-policy | same, with `--param RUN_NAME=smoke-on-$TS --param LMBDA_MODE=on-policy` |
| With an SFT pass first | add `--param INCLUDE_SFT=true` |
| Reusing a corpus | add `--param CORPUS_DIR=<WORKDIR_ROOT>/<RUN_NAME>/<build-id>/corpus` |
| With benchmark evals | add `--param SCREEN_EVALS=salad-bench,attaq,olmes-gsm8k,bfcl --param FULL_EVAL_RUNGS=8000` |

Notes on these:

- **`INCLUDE_SFT=true`** adds a plain-SFT pass between `align` and `train`, and distillation
  continues from its checkpoint. A student that is already an SFT checkpoint needs none.
- **`CORPUS_DIR`** names an earlier build's corpus directory (`train.jsonl`, `eval.jsonl`,
  `corpus_manifest.json`) and skips `sources` and `corpus`. `corpus-validate` fails before any
  GPU is held if the manifest disagrees with this build's tokenizer or prep policies. Reuse
  also keeps the held-out split identical across runs compared on it.
- **Benchmark evals** are off by default. Every rung runs `SCREEN_EVALS`; a rung in
  `FULL_EVAL_RUNGS` runs every eval in `EVAL_CATEGORIES`. Names are `EVAL_CATALOG` names
  (in [`parameters.yaml`](distill/parameters.yaml)) plus `bfcl`. An unknown name, or a rung
  off the ladder, fails the render. The full battery includes `olmes-bigcodebench`, which
  needs a bcb-server answering at the site's `OE_EVAL_BCB_API_URL`. Results land under
  `SAGE_RESULTS_DIR/<EXPERIMENT>/` and `BFCL_RESULTS_DIR/<EXPERIMENT>/`; `EXPERIMENT`
  defaults to `RUN_NAME`.
- **The smoke layer** sets `MAX_RETRIES=0`, so a smoke failure is final.

**`distill-eval` on a list of exports.** `EXPORTS` is `<rung>=<uri>,...`, with full URIs;
the rung becomes the model's `ckpt_<rung>` column. `EVALS` (`<rung>:<eval>,<eval>;<rung>:...`)
names each rung's evals outright, in place of `SCREEN_EVALS` / `FULL_EVAL_RUNGS`.

```bash
gb build start -f recipes/distill/distill-eval/build.yaml --parameters-path $SITE \
  --param RUN_NAME=eval-$TS \
  --param EXPORTS=4000=env://<WORKDIR_ROOT>/<RUN_NAME>/<build-id>/export-4000,8000=env://<WORKDIR_ROOT>/<RUN_NAME>/<build-id>/export-8000 \
  --param SCREEN_EVALS=salad-bench,attaq,bfcl --param FULL_EVAL_RUNGS=8000
```

Set `RERUN_TAG` (letters, digits, `.`, `_`, `-`) whenever the experiment name has been
evaluated before. gbserver refuses an artifact URI another build registered, the target
still reports SUCCESS, and nothing downstream of it dispatches. Every rung must get at
least one eval, or the render fails. `distill-eval` retries up to `MAX_RETRIES=5` times,
because evals run on the site's `EVAL_PARTITION`, which may preempt them.

**Re-running failed evals.** [`rerun_command.py`](distill-eval/rerun_command.py) reads a
finished `distill` or `distill-eval` build and prints the `distill-eval` command that
re-runs exactly its failed evals, under a new experiment name `<EXPERIMENT>-<RERUN_TAG>`.
It needs gbserver up.

```bash
python recipes/distill/distill-eval/rerun_command.py <build-uuid> --site "$SITE" \
  [--rerun-tag TAG] [--whole-rungs] [--space SPACE] [--allow-running]
```

| Argument | Meaning |
|---|---|
| `build_id` | the full UUID of the finished build |
| `--site` | the site file the command layers; read for a render check, never printed. Default `~/.granite.build/local-env/distill-ibm-site.yaml` |
| `--rerun-tag` | the re-run's `RERUN_TAG`; default `rerun-<now>`; must differ from the build's own |
| `--whole-rungs` | re-run every eval of a rung that had a failure, so its re-run column is complete |
| `--space` | passed through to `gb build start` |
| `--allow-running` | plan against a build that has not finished |

The command goes to stdout, notes go to stderr, and the command is render-checked before
it is printed. Exit status: 0 with a command, 0 with nothing when there is nothing to
re-run, 1 on an error, 2 when the build has not finished. A rung whose export failed is
reported, not planned: there is no model to evaluate. "Failed" is gbserver's status only,
so also check a rung's CSV for MISSING cells before trusting a SUCCESS.

## λ modes

`LMBDA_MODE` sets the fraction of student-generated sequences (λ) and the topology. Any
other value fails the render.

| `LMBDA_MODE` | `gold_config` keys set | Rollout server | Typical student |
|---|---|---|---|
| `off-policy` (default) | `lmbda: 0.0`, `vllm_num_servers: 0` | no | a converged SFT checkpoint (the default schedule assumes one) |
| `ramp` | `lmbda` (1.0, or `LMBDA`), `lmbda_schedule: linear`, `lmbda_init: 0.0`, `lmbda_ramp_start_step`, `lmbda_ramp_steps`, `min_completion_length`, `vllm_num_servers: 1`, `vllm_mode: server`, plus the rollout keys (`vllm_sync_frequency`, `top_p`, ...) | yes, plus `teardown` | a good off-policy checkpoint |
| `on-policy` | as `ramp`, but `lmbda_schedule: constant`, with no `lmbda_init` or ramp window | yes, plus `teardown` | a good off-policy checkpoint |

The rollout server runs in its own allocation (`VLLM_NUM_GPUS` GPUs); every trainer node
trains. `teardown` ends the server once `train` emits its final checkpoint.

| Key | Rule |
|---|---|
| `LMBDA` | Empty: the mode's own end value, 1.0. Set it with `ramp` (the ramp stops short of 1) or `on-policy` (a constant fraction). The render fails if it is set with `off-policy`, or is outside (0, 1]. |
| `LMBDA_RAMP_START_STEP`, `LMBDA_RAMP_STEPS` | `ramp` only. λ is 0 before the start step and reaches `LMBDA` after `LMBDA_RAMP_STEPS` steps; `0` / `0` ramps over the whole run. The render fails if either is non-zero in another mode, where it would not be rendered. The step refuses negative values at launch. |
| `MIN_COMPLETION_LENGTH` | The rollouts' generation floor in tokens, sent by the trainer with every request. Empty (the default) means 16 in `ramp` and `on-policy`; `0` turns it off. Off-policy nothing generates, so the render fails if it is set above 0 there. |
| `STOP_AT_STEP` | `0`: train to `GOLD_MAX_STEPS`. Otherwise save and stop there without changing `max_steps`, so a resumed run ends on its original LR schedule. The ladder must then end at `STOP_AT_STEP`; the step refuses a value past `GOLD_MAX_STEPS` at launch. |
| `CKPT_LADDER` | The render fails unless `GOLD_MAX_STEPS` is positive and the rungs are strictly ascending multiples of `GOLD_SAVE_STEPS`, the last equals `GOLD_MAX_STEPS` (or `STOP_AT_STEP`), and `GOLD_SAVE_TOTAL_LIMIT` covers every save the trainer makes, ⌈last rung / `GOLD_SAVE_STEPS`⌉, not only the rungs: the trainer rotates out its oldest checkpoints past the limit, counting the saves between rungs, so a rung could be deleted before its export reads it, and a retry cannot re-emit a deleted rung. A resumed run is held to the same bound. A rung the trainer never writes would be a target that never dispatches. |
| `LOSS_ARM` | Empty keeps jsd at `BETA`. The step validates the arm at launch; `distillm2_like` needs 0 < λ < 1 at some point in the run, which a 0 → 1 ramp satisfies and `on-policy` at 1.0 does not. |

## Failure and resume

Recovery is automatic only.

**A stable `train_dir` per build.** `train` checkpoints to
`WORKDIR_ROOT/RUN_NAME/<build_id>/train`. A build-level retry keeps the build id, so it
finds this directory. The directory lies outside gbserver's per-run tree, so post-build
cleanup leaves it in place. A new build gets a new directory.

**A build-level retry** (up to `MAX_RETRIES`, default 3) re-runs the failed targets and
reuses the ones that succeeded (`target_reuse_enabled`). On the retry, `train`:

1. moves any half-written checkpoint aside to `incomplete-checkpoint-<N>.<timestamp>`;
2. checks progress (below);
3. re-emits every complete checkpoint already in `train_dir`, at the same paths, before
   training starts. Exports and evals that succeeded on the earlier attempt are reused;
   failed ones re-run;
4. resumes from the newest complete checkpoint: weights, optimizer, scheduler, RNG and
   data position.

See the step's [Stable train_dir](../../steps/train/distill/logit-distill/skypilot/USAGE.md#stable-train_dir).

**The progress check.** Each attempt records the highest complete checkpoint it starts from
in `<train_dir>/attempts.jsonl`. A retry that starts no further than the previous attempt is
refused, with exit 3, before training starts:

```
logit-distill: FATAL: no progress since attempt <id> (still at checkpoint-<N>); not retrying
```

The previous attempt failed before its next save, and this one would fail at the same point:
a bad batch, an OOM at a fixed length, a key the trainer rejects. Read that attempt's log,
fix the cause, and start a new build. Everything stays on disk. A deterministic failure
therefore costs one extra attempt, not every retry. See
[No-progress refusal](../../steps/train/distill/logit-distill/skypilot/USAGE.md#no-progress-refusal).

**`MAX_RETRIES`** caps build-level retries: 3 for `distill`, 5 for `distill-eval`, 0 in
`smoke.yaml`. Set 0 when debugging a failure of your own rather than the cluster's.

**The rollout server fails visibly.** A server that dies before `teardown` ends it (idle
kill, OOM, crash, or `VLLM_MAX_LIFETIME_SECONDS` reached) ends FAILED, never SUCCESS. A retry therefore relaunches it along with `train`, rather than
handing the trainer a dead URL. `teardown` is gated on `train`'s final checkpoint, so a
failed trainer never triggers it. Instead, the trainer and the server share a stop file,
`WORKDIR_ROOT/RUN_NAME/<build_id>/rollout-stop`: when `train` exits non-zero its rank 0
writes it, and the server stops within 30 s and fails its target. A rank 0 killed outright
writes nothing; for that, `VLLM_MAX_LIFETIME_SECONDS` (default 12 h, 1 h in `smoke.yaml`)
is the backstop that releases the server's GPUs. The cap also bounds a healthy run: it
counts from the server's health check, which comes right after `align`, so it must cover
the trainer's queue wait plus its run. Raise it for an online run longer than that. See
[When the target ends FAILED](../../steps/train/distill/student-rollout-generator/skypilot/USAGE.md#when-the-target-ends-failed-and-when-success)
and [Stopping when the trainer fails](../../steps/train/distill/student-rollout-generator/skypilot/USAGE.md#stopping-when-the-trainer-fails).

**`RESUME_FROM_CHECKPOINT_DIR`** starts a *new* build from another run's checkpoints. Name
that run's `<train_dir>/checkpoints/<RUN_NAME>_node<N>` directory: every complete
checkpoint is hardlinked in and the trainer resumes from the highest. It seeds only an empty
`train_dir`; on a retry the build's own checkpoints are further along, so it is ignored
(logged). Keep the node and GPU counts, batch sizes and `GOLD_MAX_STEPS`: the step refuses a
different world size, and nothing checks the schedule. With `RESUME_EMIT_SEEDED=true`
(default) the seeded rungs are emitted again and their exports re-run; with `false`,
`CKPT_LADDER` must name only the rungs still to come. Set `GOLD_DATA_SEED` when continuing
over the same corpus, or the run replays the batches the source trained on. See
[Resume](../../steps/train/distill/logit-distill/skypilot/USAGE.md#resume).

**There is no manual restart path.** A retry already resumes from the newest checkpoint and
stops when it stops making progress. A new build starts fresh in its own `train_dir`, so it
can never overwrite another build's checkpoints; continuing a stopped run is the explicit
`RESUME_FROM_CHECKPOINT_DIR` above. Failed evals are re-run with a new `distill-eval` build
(`rerun_command.py`), not a build restart: a restart of a build whose exports are reused
fails on them.

## The on-policy study arms as `--param` sets

The G4-350m on-policy switch study ran every arm as `--param` overrides of one LSF recipe.
Here each arm is `--param` overrides of `distill`. Its defaults, arm A2, differ from this
recipe's full off-policy defaults:

| Key | Study default (A2) | `distill` default |
|---|---|---|
| `LMBDA_MODE` | `ramp` (linear 0 → 1 over the run) | `off-policy` |
| `STUDENT_MODEL_URI` | the export of off-policy ck2000 | the site's student |
| `CORPUS_DIR` | a pinned corpus | empty (build one) |
| `GOLD_MAX_STEPS` / `CKPT_LADDER` / `GOLD_SAVE_STEPS` | 2500 / `1000,1500,2000,2500` / 500 | 8000 / `1000,...,8000` / 1000 |
| `GOLD_DATA_SEED` | 43 | -1 |
| `GOLD_LEARNING_RATE` | `1.0e-06` | `2.0e-06` |
| `WARMUP_RATIO` | 0.4 | 0.05 |
| `SCREEN_EVALS` | 8 evals (below) | empty |
| `FULL_EVAL_RUNGS` | 2500 | empty |
| `MAX_RETRIES` | 1 | 3 |

Fill in the placeholders, then define the base and the helper:

```bash
CK2000='env://<export of off-policy ck2000>'
CK2500='env://<export of the first on-policy run at step 2500>'
SFT='env://<the SFT checkpoint>'
CORPUS='<the study corpus dir>'
FRONT='<the front-loaded safety corpus dir>'
SEED='<a dir holding only checkpoint-2000 of the ck2000 run>'
SCREEN=salad-bench,attaq,olmes-gsm-symbolic,olmes-minerva-math,olmes-ifeval,olmes-mmlu-mc,global-mmlu,bfcl

STUDY="--param LMBDA_MODE=ramp --param STUDENT_MODEL_URI=$CK2000 --param CORPUS_DIR=$CORPUS
  --param GOLD_MAX_STEPS=2500 --param CKPT_LADDER=1000,1500,2000,2500 --param GOLD_SAVE_STEPS=500
  --param GOLD_DATA_SEED=43 --param GOLD_LEARNING_RATE=1.0e-06 --param WARMUP_RATIO=0.4
  --param SCREEN_EVALS=$SCREEN --param FULL_EVAL_RUNGS=2500"
SHORT="--param CKPT_LADDER=250,500,1000,1500,2000,2500 --param GOLD_SAVE_STEPS=250"
A5="--param ADAM_BETA2=0.95 --param WARMUP_RATIO=0.02 $SHORT"
FRONTP="--param CORPUS_DIR=$FRONT --param TRAIN_SAMPLING_STRATEGY=sequential"
B="--param STUDENT_MODEL_URI=$SFT --param GOLD_DATA_SEED=42 --param WARMUP_RATIO=0.4 $FRONTP"
arm() { A=$1; shift; N=study-$A-$TS
  gb build start -f recipes/distill/distill/build.yaml --parameters-path "$SITE" $STUDY \
    --param RUN_NAME=$N --param EXPERIMENT=$N "$@"; }
```

Later `--param`s win, so an arm's overrides replace the base's.

| Arm | Command | Differences from the study |
|---|---|---|
| A0 | `arm A0 --param LMBDA_MODE=off-policy` | none; off-policy drops the server |
| A1 | `arm A1 --param RESUME_FROM_CHECKPOINT_DIR=$SEED --param RESUME_EMIT_SEEDED=false --param GOLD_MAX_STEPS=<source max_steps> --param STOP_AT_STEP=4500 --param GOLD_DATA_SEED=-1 --param GOLD_LEARNING_RATE=2.0e-06 --param WARMUP_RATIO=0.05 --param LMBDA_RAMP_START_STEP=2000 --param LMBDA_RAMP_STEPS=2500 --param CKPT_LADDER=2250,2500,3000,3500,4000,4500 --param GOLD_SAVE_STEPS=250 --param GOLD_SAVE_TOTAL_LIMIT=18 --param FULL_EVAL_RUNGS=4500` | `GOLD_SAVE_TOTAL_LIMIT=18`: the trainer saves every 250 steps up to 4,500, 18 saves, and the default 12 would rotate rungs out. The study passed `GOLD_MAX_STEPS=0`; this recipe refuses 0, so give the length of the source run's schedule, 8,150 steps per the study. `SEED` must hold checkpoint-2000 alone, since the trainer resumes from the highest. |
| A2 | `arm A2` | none |
| A3 | `arm A3 --param WARMUP_RATIO=0.05 $SHORT` | none |
| A4 | `arm A4 --param LMBDA_MODE=on-policy --param LMBDA=0.25` | none |
| A5 | `arm A5 $A5` | none |
| A6 | `arm A6 $A5 $FRONTP` (if A5 dipped less) or `arm A6 $FRONTP` (if A2 did) | The front-loaded corpus was built by a study recipe that was not ported; build it elsewhere. `corpus-validate` checks its manifest. |
| B0 | `arm B0 $B --param LMBDA_MODE=off-policy --param GOLD_LEARNING_RATE=2.0e-06` | as A6, for the corpus |
| B1 | `arm B1 $B --param LMBDA_RAMP_STEPS=1250` | as A6 |
| B2 | `arm B2 $B --param LMBDA_MODE=on-policy` | as A6 |
| C1 | `arm C1 $W --param LOSS_ARM=distillm2_like` (`W`: the winner's overrides) | refused at launch if the winner is `on-policy` at λ 1 |
| C2 | `arm C2 $W --param BETA=0.9` | none |
| D1–D3 | not expressible | The study ramped from `LMBDA_INIT=0.25`; this recipe always ramps from 0 (`lmbda_init: 0.0`) and has no such parameter. The rest of D (`STUDENT_MODEL_URI=$CK2500`, `GOLD_DATA_SEED=44`, `WARMUP_RATIO=0.05`, `MIN_COMPLETION_LENGTH=0`, `CKPT_LADDER=500,1000,1500,2000,2500`, and D1 `LMBDA=0.5 LOSS_ARM=distillm2_like`, D2 `LMBDA=0.5 BETA=0.9`, D3 `LMBDA=1.0`) is expressible, but it ramps from 0, which is a different experiment. |

The study's eval steps map onto `distill-eval`:

| Study step | Here |
|---|---|
| Track 0 checks 3 and 4: SALAD and AttaQ on the off-policy step-1,000 export and on the teacher | `--param 'EXPORTS=1000=<that export>,1=<teacher URI>' --param 'EVALS=1000:salad-bench,attaq;1:salad-bench,attaq'` |
| Track 0 check 7: the screening subset on ck2000 | `--param EXPORTS=2000=$CK2000 --param SCREEN_EVALS=$SCREEN` |
| The full battery on an arm's best rung | `--param EXPERIMENT=<arm RUN_NAME>-ck<N>-rest --param EXPORTS=<N>=env://<WORKDIR_ROOT>/<arm RUN_NAME>/<build-id>/export-<N> --param EVALS=<N>:<every EVAL_CATALOG eval not in $SCREEN>` |
| Re-running an arm's failed evals | `rerun_command.py <build-uuid> --site "$SITE"` |
| Track 0 check 2: interpolated checkpoints | not supported: neither recipe interpolates models |
| Track 0 checks 6 and 6b: the safety corpora | not ported; build them elsewhere and pass them as `CORPUS_DIR` |

The study's patched trainer checkout (its step 0a) is no longer needed: the λ ramp window,
the generation floor and `stop_at_step` are in the vendored trainer.

## What is site-specific and where it lives

| File | Holds | In this repo |
|---|---|---|
| your site file (from [`site.example.yaml`](site.example.yaml)) | every host, user, secret name, partition, account, QOS, path, model, dataset, accelerator model and registry | no; only the fake example |
| [`environments/skypilot-slurm.template.yaml`](environments/skypilot-slurm.template.yaml) | how the site's SLURM values become an environment: sbatch options, the image cache, container mounts, SSH config | yes, with placeholders only |
| [`distill/parameters.yaml`](distill/parameters.yaml), [`distill-eval/parameters.yaml`](distill-eval/parameters.yaml) | the experiment's knobs; no key may also be a site key | yes |

[`test_distill_site.py`](../../test/unit/recipes/test_distill_site.py) scans every file
under `recipes/distill/` for site-specific tokens.

## Not yet verified on a cluster

These recipes and the step changes behind them are unit-tested and render-tested only. The
cluster checks still to run are listed in the PR.
