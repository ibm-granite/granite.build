# On-cluster filesystem map & cleanup (BlueVela SLURM)

Where every tree a distill build touches on BlueVela lives, what writes it, and what
is safe to delete. All paths are under the shared root `/proj/granite-build/g4os/`
(call it `$R`). Two config layers decide the locations:

- **Environment** (`configurations/assets/environments/skypilot/slurm/ibm-bluevela/environment.yaml`):
  - `config.shared_workdir: /proj/granite-build/g4os/gbtest/builds` → per-run `$GB_BUILD_WORKDIR`
  - `config.cloud_config.slurm.cluster_configs.bluevela.workdir: /proj/granite-build/g4os` → SkyPilot `workdir`
  - `...bluevela.tmpdir: /proj/granite-build/g4os/sky-runtime` → SkyPilot SLURM `tmpdir`
  - `assetstores...hf.cache_path: /proj/granite-build/g4os/gbtest/hfcache`
- **Recipe** (`recipes/granite4-gold-distillation/slurm/distill-gold*/parameters.yaml`):
  - `WORKDIR_ROOT: /proj/granite-build/g4os/gbtest/distill` + `RUN_NAME` + `BUILD_SUBDIR` → step artifact dirs

---

## SkyPilot control/runtime trees (under the SkyPilot `workdir`/`tmpdir`)

| Path | What it holds | Written by | Reused? | Cleanup |
|---|---|---|---|---|
| `$R/.sky_provision/` | Per-launch sbatch scripts `gb-*.sh` + their stdout `slurm-<jobid>.out`. One pair per cluster launch, forever. | SkyPilot SLURM backend | No | **Delete when idle.** Pure launch artifacts, regenerated each launch. Keep the newest if debugging a live job. |
| `$R/.sky_clusters/<cluster>-<hash>/` | Per-cluster home + `sky_logs/<n>-<task>/run.log` (the streamed workload log — the one to read for a step's real stdout). One dir per known cluster. | SkyPilot | Per cluster | **Delete dirs of torn-down clusters.** Keep live ones and any you still want logs from. |
| `$R/skypilot/` | SkyPilot **`workdir`** — one `gb-<step>-<hash>/` staging dir **per task launch**. The biggest accumulator (thousands of dirs: every eval/export/merge/train launch ever). | SkyPilot | No | **Delete old `gb-*` entries when idle.** Never reused across builds; this is usually the bulk of the clutter. |
| `$R/sky-runtime/` | SkyPilot SLURM **`tmpdir`**: the shared `skypilot-runtime` venv + `uv-python`, plus per-cluster `gb-<cluster>` skylet state. | SkyPilot | **venv reused** | **Keep `skypilot-runtime` + `uv-python`** (rebuild is slow). Per-cluster `gb-*` subdirs: delete once that cluster is down. |
| `$R/enroot/` | Pre-staged `.sqsh` container images (incl. `us.icr.io-...-kd-sandbox-distill-0.1.0-uv.sqsh` the recipe mounts via `IMAGE_ID`). | manual pre-staging | Yes | **Keep.** Deleting forces a ~20-min registry re-import per job. |

> **Why `.sky_provision/` and `skypilot/` live under `/proj/...g4os`:** that dir is the
> SkyPilot `workdir`, which the SLURM backend bind-mounts into the enroot container.
> Deleting entries for a *running* job can break it — scope cleanup to old mtimes.

---

## gbserver build & step trees

| Path | What it holds | Written by | Cleanup |
|---|---|---|---|
| `$R/gbtest/builds/builds/<build_id>/runs/<targetrun_id>/` | Per-run `$GB_BUILD_WORKDIR`: the trainer-code `distill-code/` clone (re-cloned every run) + a working `checkpoints/` dir. | gbserver (`shared_workdir`) | **Delete per finished build.** The clone is disposable; the working checkpoints are duplicated into the artifact tree below. |
| `$R/gbtest/distill/<RUN_NAME>/<build_id>/` | **The step artifacts** — `align/retagged_student/`, `corpus/{train,eval}.jsonl`, `checkpoints/<run_name>_node<N>/` (the final model). This is `WORKDIR_ROOT`. | distill steps (`out_dir`) | **Selective.** These are the real outputs. Delete old `<build_id>` dirs you don't need, but **keep any build whose align/corpus/checkpoint you reuse** (e.g. `distill-gold-slurm/cc33e2e2.../` feeds `distill-gold-trainonly`). |
| `$R/gbtest/hfcache/` | HF assetstore pull cache. | hfpull / assetstore | Keep if still pulling those models; else deletable (re-downloads). |

### Why align+corpus re-run every build
`BUILD_SUBDIR = {{ run_metadata.build_id }}`, so each build gets its own
`<build_id>/` subtree under both `gbtest/builds` and `gbtest/distill`. Nothing is
shared across builds (that is deliberate — it avoids artifact-URI collisions). The
`distill-gold-trainonly` recipe is the way to *reuse* a prior build's align/corpus
instead of recomputing them.

---

## Cleanup recipe (run only when no build is active)

`-mtime +2` keeps anything from the last 2 days (and thus anything a live job touches):

```sh
R=/proj/granite-build/g4os
# Biggest win: old per-launch staging dirs (thousands)
find $R/skypilot             -maxdepth 1 -name 'gb-*' -mtime +2 -exec rm -rf {} +
# Per-launch sbatch scripts + stdout
find $R/.sky_provision       -maxdepth 1 -mtime +2 -delete
# Torn-down cluster homes + their run.logs
find $R/.sky_clusters        -maxdepth 1 -name 'gb-*' -mtime +2 -exec rm -rf {} +
# Stale per-cluster runtime (NOT skypilot-runtime / uv-python)
find $R/sky-runtime          -maxdepth 1 -name 'gb-*' -mtime +2 -exec rm -rf {} +
# Finished-build workdirs (code clones + working checkpoints)
find $R/gbtest/builds/builds -maxdepth 1 -mtime +2 -exec rm -rf {} +
```

**Never delete:** `enroot/` (images), `sky-runtime/{skypilot-runtime,uv-python}` (shared
venv), and any `gbtest/distill/<run>/<build_id>` whose outputs you still reuse.

**Before cleaning:** confirm no live clusters —
`sky status | grep -E 'UP|INIT'` and `squeue --me` should be empty of your jobs — then
the `-mtime +2` filter is belt-and-suspenders.

To find space hogs first:
```sh
du -sh /proj/granite-build/g4os/{skypilot,.sky_provision,.sky_clusters,sky-runtime,gbtest/distill,gbtest/builds} 2>/dev/null
```
(Recursive `du` on GPFS is slow for large checkpoint trees; run it in the background.)
