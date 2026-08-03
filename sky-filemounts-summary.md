# Support `step.yaml`-relative `file_mounts` in the SkyPilot environment

Branch: `feat/sky-step-doc`

## Motivation

In the SkyPilot environment a step's `file_mounts` **local source** was passed
verbatim to SkyPilot's `Task.set_file_mounts()`, which resolves a *relative*
source against the gbserver process's working directory — effectively undefined
and unusable. Unlike the bash / docker / k8s launchers, the SkyPilot launcher
received `targetsteprun_asset_dir` (the per-run directory that holds the rendered
`step.yaml` and its sibling files) but **ignored it**. As a result a step author
could not mount a file or directory that ships alongside their `step.yaml`.

## What changed

A **relative** `file_mounts` source now resolves against the **`step.yaml`
directory**, so files that ship next to a step are copied to the cluster.
Absolute paths and remote URIs (`s3://`, `gs://`, `file://`, …) are used
unchanged, so the change is backward-compatible.

### Launcher (`src/gbserver/environment/skypilot.py`)

- Added `_resolve_local_mount_source(source, asset_dir)` — joins a relative local
  source onto the asset dir; returns URIs and absolute paths unchanged.
- Added `_build_skypilot_mounts(file_mounts_raw, asset_dir)` — shared builder that
  splits a raw `file_mounts` mapping into string file-mounts and `sky.Storage`
  storage-mounts, resolving relative local sources. Bucket-URI sub-path extraction
  is preserved and now gated on a URI scheme, which also fixes a latent bug where a
  scheme-less **absolute** dict source was mangled into `"://"`.
- Wired the builder into `_launch_skypilot_inner`, and added
  `targetsteprun_asset_dir` to the retry-replay stash so a relaunch re-resolves
  relative mounts (mirrors how k8s stashes it).

### Managed launcher (`src/gbserver/environment/skypilot_managed.py`)

- Reuses the shared `_build_skypilot_mounts`, removing the duplicated `file_mounts`
  block (and now-unused imports).

### Docs (`docs/environments/skypilot.md`)

- Expanded the `step.yaml` field reference (`image_id`, `run`/`setup`, `resources`,
  the `config`/`docker` overrides, `file_mounts`, `envs`, `post_launch_task`,
  `idle_minutes_to_autostop`) and the `skypilot_monitor` config
  (`poll_interval_seconds`, `log_retrieval` modes).
- Documented `file_mounts` source resolution and added a
  "copying a directory that ships with the step" example, including the caveat that
  the mount **destination** must be an absolute (or `~`) remote path — it cannot be
  the dynamic `run` working directory.

## Tests

### Unit — `test/unit/environment/test_skypilot.py`

- `_resolve_local_mount_source`: relative → `asset_dir/<rel>`; absolute and `s3://`
  unchanged; `None` asset dir leaves the source unresolved; `file://` asset dir
  tolerated.
- `_build_skypilot_mounts`: string relative source resolved; `s3://bucket/prefix`
  still split into `source=s3://bucket` + `_bucket_sub_path=prefix`; local dict
  source resolved without a bucket sub-path.
- Launch-level: `launch_skypilot(..., targetsteprun_asset_dir=...)` calls
  `set_file_mounts` with the resolved path, and stashes the asset dir for retry.

### Build test — `test/integration/standalone/buildrunner/skypilot_slurm/test_skypilot_filemount.py`

Mirrors the sibling `1step` test but adds a **co-located test space**
(`test-data/.../skypilot_slurm/filemount/space/`) containing a custom
`filemount-check` step — the `command` step plus a `file_mounts` key — with a
`payload/` directory next to its `step.yaml`. The step's `run` asserts the mounted
directory is present (failing the build if absent), so a SUCCESS build proves the
mount landed. The build is HF-free (`env://` input/output) so it is fast and needs
no `HF_TOKEN`, and it runs only the basic build variant (no cancellation).

## Verification

- Unit suites pass (`test_skypilot.py`, `test_skypilot_hfstore.py`); typecheck adds
  no new errors.
- Build test on the local Docker SLURM cluster: **passed** (~67s).

## Files

| File | Change |
|------|--------|
| `src/gbserver/environment/skypilot.py` | Mount-resolution helpers; launcher + retry-stash wiring |
| `src/gbserver/environment/skypilot_managed.py` | Reuse shared mount builder (de-dup) |
| `docs/environments/skypilot.md` | `step.yaml` field reference + `file_mounts` resolution/example |
| `test/unit/environment/test_skypilot.py` | Unit tests for the new helpers + launch behavior |
| `test/integration/standalone/buildrunner/skypilot_slurm/test_skypilot_filemount.py` | Build test |
| `test-data/.../skypilot_slurm/filemount/**` | Build test fixtures (step, payload dir, test space, build.yaml, buildtest.yaml) |

## Note (out of scope)

The passing build logs a swallowed cleanup warning during shared-workdir teardown:
`teardown_skypilot rm -rf … failed: Invalid task name gb-td-<id>-`. `teardown_skypilot`
builds a cluster name that can end in a dash, which SkyPilot rejects. It is
pre-existing, non-fatal, and unrelated to `file_mounts` — left for a follow-up.
