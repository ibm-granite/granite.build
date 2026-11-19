# lm-engine pretraining

Pretrain a model from scratch with [lm-engine](https://github.com/open-lm-engine/lm-engine),
export its checkpoints to Hugging Face format while the run continues, and score
them. The defaults are the granite-4 350M muP model at about 500B tokens; a
smoke layer runs the same graph in GPU-minutes, and a 165M example shows the
other parametrization.

| Path | What it is |
|---|---|
| [`pretrain/build.yaml`](pretrain/build.yaml) | The recipe: corpus check, training, the checkpoint ladder, the exports, the evals and the roll-up. |
| [`pretrain/parameters.yaml`](pretrain/parameters.yaml) | The experiment's knobs — 100 of them, in 13 groups. Every model knob is an upper-cased knob of the training step's `render_config.DEFAULT_KNOBS`. |
| [`pretrain/smoke.yaml`](pretrain/smoke.yaml) | A layer that shortens a run to 4 optimizer steps on one 2-GPU pod. |
| [`pretrain/models/165m-sp-768x17.yaml`](pretrain/models/165m-sp-768x17.yaml) | A layer: a 165M standard-parametrization model (768 x 17), instead of the 350M muP default. |
| [`pretrain-eval/`](pretrain-eval/) | The eval half alone, pointed at exports a previous run wrote. |
| [`environments/skypilot-k8s.template.yaml`](environments/skypilot-k8s.template.yaml) | The cluster side, as a template: a SkyPilot-on-Kubernetes environment with an RWX volume, RoCE and a large `/dev/shm`. |
| [`site.example.yaml`](site.example.yaml) | Every value a cluster owns, with fake values. Copy it, fill it in, keep the copy out of this repository. |
| [`report.py`](report.py) | The roll-up: one results table per rung. The `eval-combined` target ships this file inline. |

The steps the recipe runs live in [`steps/train/lm-engine/`](../../steps/train/lm-engine/)
(the image, the pretraining step and the export step) and
[`steps/eval/lmeval/`](../../steps/eval/lmeval/) (the benchmarks).

## The graph

```
data-check ──▶ train ──▶ export-N ──▶ eval-<entry>-ckN ──▶ eval-combined
   (CPU)    (multi-node    (CPU)           (1 GPU)             (CPU)
                GPU)
```

`data-check` runs the training step with `phase: data-check` on one cheap CPU
pod. It reads schemas only, and it fails in seconds on a mis-mounted volume, a
misspelled sub-dataset or a tokenizer whose special-token ids disagree with the
model knobs — before `train` asks for `NUM_NODES x GPUS_PER_NODE` accelerators
and dies building its dataloader. `train` binds it for ordering only; no
artifact flows between them.

Everything to the right of `train` exists because of the **rungs**. A rung is a
checkpoint the trainer writes mid-run: the rank-0 watcher prints
`GB_ARTIFACT_ID:checkpoint_<N>` once checkpoint N is completely written, so
gbserver dispatches work on rung N while training continues.

| Target | Present when |
|---|---|
| `data-check` | always |
| `train` | always |
| `export-N` | N is in `EXPORT_RUNGS` **or** `EVAL_RUNGS` |
| `eval-<entry>-ckN` | N is in `EVAL_RUNGS`, and `<entry>` is in `SCREEN_EVALS` — or N is in `FULL_EVAL_RUNGS`, which runs the whole `EVAL_CATALOG` |
| `eval-combined` | any `eval-` target exists |

The two rung lists are a **union**, not a subset: an eval rung implies an export
rung, because an eval scores a Hugging Face directory. Naming a rung in both
lists still produces one checkpoint and one export. Every rung must sit on the
save grid (`SAVE_INTERVAL`) and inside the run; a rung that does not fails the
render rather than becoming a target that never dispatches.

## Setup

Once per cluster:

1. Copy [`site.example.yaml`](site.example.yaml) to
   `~/.granite.build/local-env/pretrain-k8s.yaml` and fill in your own values.
   Keep it out of this repository.
2. Build the two images and push them to a registry the cluster can pull from.
   The training image: `make -C steps/train/lm-engine image check-image
   publish-image REGISTRY=...`. The lmeval image: `make -C steps/eval
   image-lmeval publish-image-lmeval REGISTRY=...`. Pin both in the site file
   by tag or digest, never `latest`.
3. Render the environment and register it:

```bash
scripts/setup-local-space.sh \
  --template recipes/pretrain/environments/skypilot-k8s.template.yaml \
  --site ~/.granite.build/local-env/pretrain-k8s.yaml \
  --name pretrain-k8s
```

That writes the rendered environment into a standalone space and sets
`ENVIRONMENT_URI` in the site file to the URI it registered. The recipe reads
nothing else about the cluster.

## Running it

The site file is always the first `--param-file`. Give every run its own
`RUN_NAME`, with a timestamp: `RUN_DIR` is `WORKDIR_ROOT/EXP/RUN_NAME`, and two
builds may not register one artifact URI — the second target reports SUCCESS
with an empty output list, and everything downstream of it waits forever.

**Smoke: the whole graph in GPU-minutes.** Run this first on a new cluster.

```bash
gb build start -f recipes/pretrain/pretrain/build.yaml \
  --param-file ~/.granite.build/local-env/pretrain-k8s.yaml \
  --param-file recipes/pretrain/pretrain/smoke.yaml \
  --param RUN_NAME=smoke-$(date +%Y%m%d-%H%M)
```

**A full run with a ladder.** Rungs every 10,000 steps, the screening evals on
each, the whole catalog on the last:

```bash
gb build start -f recipes/pretrain/pretrain/build.yaml \
  --param-file ~/.granite.build/local-env/pretrain-k8s.yaml \
  --param RUN_NAME=run-$(date +%Y%m%d-%H%M) \
  --param EXPORT_RUNGS=10000,20000,40000,80000,120000 \
  --param EVAL_RUNGS=20000,40000,80000,120000 \
  --param FULL_EVAL_RUNGS=120000
```

**Score exports that already exist.** No training, no export; one CPU pod per
rung adopts the existing directory, then the same evals and the same roll-up
run. Use a new `RUN_NAME`: the results of a re-score are new artifacts.

```bash
gb build start -f recipes/pretrain/pretrain-eval/build.yaml \
  --param-file ~/.granite.build/local-env/pretrain-k8s.yaml \
  --param EXP=<the run's EXP> \
  --param EXPORT_DIR=<WORKDIR_ROOT>/<EXP>/<run>/export \
  --param RUNGS=20000,40000 \
  --param RUN_NAME=rescore-$(date +%Y%m%d-%H%M)
```

**A different architecture.** Layer a model file, and put `smoke.yaml` last if
you want both:

```bash
gb build start -f recipes/pretrain/pretrain/build.yaml \
  --param-file ~/.granite.build/local-env/pretrain-k8s.yaml \
  --param-file recipes/pretrain/pretrain/models/165m-sp-768x17.yaml \
  --param-file recipes/pretrain/pretrain/smoke.yaml \
  --param RUN_NAME=sp165m-$(date +%Y%m%d-%H%M)
```

A list or a mapping — `EVAL_CATALOG`, `KERNELS`, `NCCL_ENV`, `EXCLUDE_NODES` —
can only arrive in a `--param-file`. `--param KEY=VALUE` hands every value
through as a string, so a list passed that way renders as the text of a list.

## Resuming

`MAX_RETRIES` (2 by default) lets gbserver retry a failed build in place,
re-running only the failed targets. `train` resumes from its own build-stable
`train_dir`, which sits outside the per-target workdir that teardown deletes, so
a fault does not re-train from step 0. The step refuses an attempt that makes no
optimizer progress, so a deterministic failure does not spend every retry
holding the allocation.

Once the retries are spent, continue the same build by hand. It reuses the
targets that succeeded and re-runs the rest:

```bash
gb build restart <build-id>
```

To continue a run in a **new** build, point `RESUME_FROM_CHECKPOINT_DIR` at the
old run's `train` directory. The recipe seeds the new run from it and then
ignores it, so once the new run has written a checkpoint of its own it resumes
from that instead.

## Reading the results

Each eval target writes one `results.json` under
`EVAL_ROOT/EXP/RUN_NAME/ckpt_<N>/<entry>/`. `eval-combined` joins them into
`report/`:

- `summary.csv` — one row per rung, one column per entry, the cell being that
  run's headline value.
- `per_task.csv` — long form: rung, entry, task, metric, value, n.
- `summary.md` — the same headline table, the metric each entry is measured in,
  and a Notes section listing every gap.

Nothing in the roll-up fails on missing data: a missing or unreadable
`results.json` is an empty cell and a note. That matters because a failed eval
target leaves the rest of the ladder worth reading. To run the roll-up by hand
over a tree of results, use [`report.py`](report.py) directly:

```bash
python report.py <EVAL_ROOT>/<EXP>/<RUN_NAME>
```

## What the cluster owns

The recipe's `parameters.yaml` holds no path, registry reference, accelerator
model or cluster name; a unit test holds the parameter names and the site keys
disjoint, and a second scans every file here for a real cluster's names. The
site file supplies the kube context, the RWX claim and its mount, the priority
class, excluded nodes, the RoCE network and VF count, `/dev/shm`, the NCCL
facts, both accelerator models, both images, the three roots
(`WORKDIR_ROOT`, `EVAL_ROOT`, `HF_CACHE`), the corpus and the tokenizer.

## Not yet verified on a cluster

Everything here is covered by unit tests — the renders, the two shared blocks,
the lm-engine config against a golden fixture captured from the Ansible role
this recipe replaces, and the export step's parity check. None of it has run on
a cluster yet. Specifically:

- **The eval targets cannot run against any published image.** They need an
  lmeval image carrying the generic `lm-eval` bridge
  ([`steps/eval/lmeval/src/generic.py`](../../steps/eval/lmeval/src/generic.py)),
  which this tree has but no published tag does. Build and push the lmeval
  family image yourself, or run with `EVAL_RUNGS` empty, as `smoke.yaml` does.
- The environment's pod spec is untested against an admission controller. A
  cluster that admits only AppWrappers or Jobs will reject these plain pods.
- The in-image RDMA packages are unverified: the reference image carried none
  and relied on the host, so the first multi-node run is the test of whether
  NCCL reports `NET/IB` rather than `NET/Socket`.
- `require_progress` cannot catch a run that dies before its first checkpoint;
  it has no baseline to compare against.
- A seeded resume symlink dangles if the source run's directory is reaped.
