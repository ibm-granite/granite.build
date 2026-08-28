# Job backends

## Overview

`POST /jobs` never blocks on training. When a job is accepted it is validated,
persisted as `pending`, and handed to a **`JobRunner`** off the request path —
the HTTP response returns immediately. The job then stays `pending` until a
runner picks it up and moves it through its six-state lifecycle:
`pending → running | completed | error | terminated`;
`running → paused | completed | error | terminated`;
`paused → running | terminated | error` (see `ALLOWED_JOB_TRANSITIONS` in
`models/job.py` for the exact transitions). A `pending` job may jump **straight
to a terminal state**, and that is what makes the reconcile loop below
restart-safe: a sweep can observe a build that already finished while the service
was not watching, and record what genuinely happened instead of leaving the job a
permanent zombie.

Which runner is built is chosen by a single setting,
`AUTOTUNEX_JOB_BACKEND`. There is **no external broker or task queue yet** — the
`AUTOTUNEX_QUEUE_URL` setting is reserved for a future queue-backed runner but is
inert today, so every runner that ships now executes **in-process** in the API
server. The runner is a clean seam (`JobRunner`, a `typing.Protocol`): a future
backend (for example a queue worker) attaches by adding one more `JobRunner`
implementation and wiring it into the single provider `get_job_runner` — nothing
else in the codebase changes.

## Backends at a glance

| Backend (`AUTOTUNEX_JOB_BACKEND`) | What it does | Requirements | Models | When to use |
| --- | --- | --- | --- | --- |
| **`none`** (default) | No-op runner: accepts the job and does nothing — it stays `pending` forever. | None. | Any (nothing runs). | Exploring the API, or when an external system executes jobs and writes results back to the database. |
| **`local`** | Runs the `autotune` HPO pipeline **in-process via Ray Tune**, with no external build system. Persists trials, results, and logs live and drives the job to `completed`/`error` itself. | The optional `autotune` trainer package installed; the dataset's files present on local disk. | `huggingface`, `custom_path`. | Self-contained HPO on one host (or against an existing Ray cluster). |
| **`llmb`** | Submits a [granite.build](https://github.com/ibm-granite/granite.build) build (a `build.yaml` spec) via the `llmb` CLI; a reconcile loop then polls a granite.build server and advances the job. | The `llmb` CLI; a reachable granite.build server; an auth token in the environment. | `huggingface`, `custom_path` (the local-bash variant is `huggingface`-only). | Offloading builds to a granite.build server — a remote cluster, or a local standalone machine. |

The sections below describe each backend in detail.

## `none` (default)

Selected when `AUTOTUNEX_JOB_BACKEND` is unset or `none`. The runner is a no-op:
it logs a warning that the job was accepted but will not be executed, and leaves
it in `pending`. Nothing advances it — there is no runner to move its status.

```
AUTOTUNEX_JOB_BACKEND=none
```

Use it to explore the API without standing up any training machinery, or when
jobs are executed by an external system (for example the tuning pipeline writing
trials, results, and status directly into the database).

## `local` — in-process HPO via Ray Tune

Selected with `AUTOTUNEX_JOB_BACKEND=local`. This runner executes the HPO
pipeline **itself, in the API process**, with no granite.build and no cluster
submission. On submit it:

1. flips the job `pending → running`;
2. runs the `autotune` search under Ray Tune on a worker thread (so the event
   loop stays free to service the database writes the run drives);
3. persists trials, results, and log entries **live** as the run progresses;
4. drives the job to `completed` on success, or to `error` on any failure —
   sweeping any still-`running` trials to `error` so none is left dangling.

Because the run happens in-process, no reconcile loop is started for this backend.

### Requirements

- **The optional `autotune` trainer package.** It is vendored in-tree at
  `src/fm-tune/`, so it needs no credentials and no network fetch: `make install`
  installs the slim, torch-free catalog from that path, and `make install-training`
  installs the heavy training stack (`src/fm-tune[full,mlx]`), which is what pulls
  in Ray and Torch. Whether it is installed is a **runtime** concern, not a startup
  gate: a `local`-configured deployment starts fine even without it, and a submitted
  job fails cleanly (the job lands in `error`) only when a run is attempted and the
  package cannot be imported.
- **The dataset's files on local disk.** This runner reads training data from
  disk, never from a remote host. Files are resolved at
  `<AUTOTUNEX_DATASET_STORAGE_DIR>/<dataset_id>/<name>_<split>.<ext>`; a missing
  training file fails the run.
- **Model source** must be `huggingface` or `custom_path`.

### Relevant settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `AUTOTUNEX_LOCAL_RAY_ADDRESS` | unset | Ray cluster address. Unset → a local `ray.init()`. Set it (e.g. `ray://host:10001`) to run against an existing Ray cluster. |
| `AUTOTUNEX_LOCAL_OUTPUT_DIR` | `<ARTIFACT_DIR>/local` | Root for a run's output; each run writes under `<dir>/<job_id>/`. Follows `AUTOTUNEX_ARTIFACT_DIR` unless set explicitly (resolved to an absolute path — Ray Tune's storage path cannot be relative). |
| `AUTOTUNEX_LOCAL_CANCEL_TIMEOUT_SECONDS` | `30.0` | How long a cancel waits for the cooperative in-process stop to finish. On timeout the request reports a 409 (`JobCancellationInProgressError`, "still stopping") — the signal stays latched, so the run still stops and a retry succeeds. |

```
AUTOTUNEX_JOB_BACKEND=local
AUTOTUNEX_DATASET_STORAGE_BACKEND=local
AUTOTUNEX_DATASET_STORAGE_DIR=artifacts/datasets
# AUTOTUNEX_LOCAL_RAY_ADDRESS=ray://localhost:10001   # unset → local Ray
# AUTOTUNEX_LOCAL_OUTPUT_DIR=artifacts/local
```

A ready-to-copy example lives at `envs/local.env.example`.

## `llmb` — granite.build builds via the `llmb` CLI

Selected with `AUTOTUNEX_JOB_BACKEND=llmb`. This runner does **not** train
in-process. On submit it assembles a granite.build `build.yaml` spec, writes it
to disk, authenticates the CLI, and submits it with `llmb build start`. A
successful submission records the returned build handle and **leaves the job
`pending`** — submitted is not running. A separate **reconcile loop** (a
background task started for this backend only) then periodically polls a
granite.build server for each build's status and advances the job
(`pending → running → completed | error | …`). Only a *submission* failure writes
status directly (`pending → error`).

The generated spec is written to `<AUTOTUNEX_JOB_SPEC_DIR>/<job_id>/build.yaml`
(default `AUTOTUNEX_JOB_SPEC_DIR=tmp`) and is **kept** after submission — even on
failure — so it can be inspected or replayed by hand.

There are three spec variants, chosen by granite.build's own `GB_ENVIRONMENT`
variable and, within `standalone`, by `AUTOTUNEX_LSF_CLUSTER`.

### Remote (`custom_code`) — the default

When `GB_ENVIRONMENT` is unset (or anything other than `standalone`), the
launcher emits the remote `custom_code` spec, which runs the tuning build on a
cluster. HuggingFace-hosted datasets are pulled by URI; models may be
`huggingface` or `custom_path`.

Required settings (startup **fails fast** if any is missing):

| Setting | Meaning |
| --- | --- |
| `AUTOTUNEX_JOB_RUNTIME_IMAGE` | Container image the cluster runs the tuning build in. |
| `AUTOTUNEX_JOB_TRAINER_REPO` | Trainer source repository the build checks out. |
| `AUTOTUNEX_JOB_OUTPUT_URI_ROOT` | Root URI for run artifacts; each run writes under a subpath. |
| `AUTOTUNEX_GB_SERVER_URL` | Base URL of the granite.build server the reconcile loop polls, e.g. `https://gb.example.com`. |

Optional in this mode:

| Setting | Default | Meaning |
| --- | --- | --- |
| `AUTOTUNEX_JOB_TRAINER_REF` | `main` | Branch/tag/commit of the trainer repo to check out. Not part of the fail-fast set. |

```
AUTOTUNEX_JOB_BACKEND=llmb
# GB_ENVIRONMENT left UNSET → remote custom_code spec

AUTOTUNEX_JOB_RUNTIME_IMAGE=registry.example.com/tuner:1
AUTOTUNEX_JOB_TRAINER_REPO=https://git.example.com/org/trainer.git
AUTOTUNEX_JOB_TRAINER_REF=main
AUTOTUNEX_JOB_OUTPUT_URI_ROOT=s3://your-bucket/runs
AUTOTUNEX_GB_SERVER_URL=https://gb.example.com
```

A ready-to-copy example lives at `envs/remote.env.example`.

### Local bash

When granite.build's own `GB_ENVIRONMENT=standalone` is set, the launcher emits a
local-bash spec instead, which runs granite.build **and** AutoTuneX together on a
single machine (for example a laptop or Mac, on MPS/CPU via `mlx`).

> **`GB_ENVIRONMENT` is granite.build's own variable and is read *without* the
> `AUTOTUNEX_` prefix.** The prefixed `AUTOTUNEX_GB_ENVIRONMENT` is deliberately
> ignored — set the bare `GB_ENVIRONMENT` that granite.build already uses.

This variant drops the remote-only inputs (`AUTOTUNEX_JOB_RUNTIME_IMAGE`,
`AUTOTUNEX_JOB_TRAINER_REPO`, `AUTOTUNEX_JOB_OUTPUT_URI_ROOT`) — the bash spec
carries none of them. It still needs `AUTOTUNEX_GB_SERVER_URL` (the reconcile loop
polls the local server) plus:

| Setting | Default | Meaning |
| --- | --- | --- |
| `AUTOTUNEX_BASH_FM_TUNE_ROOT` | unset | The fm-tune checkout *or* repo injected into the bash spec's environment — either a local path or a repo URL; see below. |
| `AUTOTUNEX_BASH_FM_TUNE_REF` | unset | Branch/tag/commit of the trainer to check out; unset uses the repo's default branch. Leave unset when `FM_TUNE_ROOT` is a local checkout, since the tree is already at the right commit. |
| `AUTOTUNEX_BASH_FM_TUNE_EXTRA` | `full,mlx` | The extras to install. |
| `AUTOTUNEX_BASH_BACKEND` | `mlx` | `mlx` for Apple Silicon, or `torch`. |

`AUTOTUNEX_BASH_FM_TUNE_ROOT` accepts **either** a repo URL **or** a local checkout path.
With fm-tune vendored in-tree (`src/fm-tune/`), point it at the local checkout instead of
cloning:

```
AUTOTUNEX_BASH_FM_TUNE_ROOT=/abs/path/to/granite.build/autotunex/src/fm-tune
```

The local-bash build references its dataset by URI (its `artifact_url` must be
set), but the builder is **scheme-agnostic**: either an `hf://` artifact, which
gbserver pulls, or the absolute `file://` locator that standalone dataset upload
writes, which the same-host gbserver mounts as the build's `dataset_files` input.
Its model is HuggingFace-only, though — the spec emits `hf:///<model>`, so a
`custom_path` model cannot be launched by this variant.

```
AUTOTUNEX_JOB_BACKEND=llmb
GB_ENVIRONMENT=standalone            # unprefixed — granite.build's own variable

AUTOTUNEX_GB_SERVER_URL=https://gb.example.com
# AUTOTUNEX_BASH_FM_TUNE_ROOT=https://git.example.com/org/trainer
# AUTOTUNEX_BASH_FM_TUNE_REF=main
AUTOTUNEX_BASH_FM_TUNE_EXTRA=full,mlx
AUTOTUNEX_BASH_BACKEND=mlx           # or: torch
AUTOTUNEX_DATASET_STORAGE_BACKEND=auto
```

`auto` is deliberate here, and forcing `huggingface` instead **fails startup**: the
HuggingFace push runs `llmb artifact push`, which granite.build disables in
standalone mode, so settings validation refuses that combination outright for the
same-host bash variant (`gb_environment="standalone"` with no `lsf_cluster`). `auto`
resolves to local storage and emits the `file://` locator described above; `local`
works too.

A ready-to-copy example lives at `envs/bash.env.example`.

### LSF / SkyPilot

When granite.build's own `GB_ENVIRONMENT=standalone` is set **and**
`AUTOTUNEX_LSF_CLUSTER` names a cluster, the launcher emits the LSF/SkyPilot spec
instead of the local-bash spec, submitting the tuning build to an LSF cluster via
SkyPilot. `AUTOTUNEX_LSF_CLUSTER` is the discriminator between the two standalone
specs: leave it unset for the bash spec above.

Required settings (startup **fails fast** if any is missing):

| Setting | Meaning |
| --- | --- |
| `AUTOTUNEX_LSF_ENVIRONMENT_URI` | granite.build space environment for the LSF build, e.g. `space://environments/skypilot/lsf/<cluster>`. |
| `AUTOTUNEX_LSF_IMAGE` | Runtime container image the LSF step runs in. |
| `AUTOTUNEX_JOB_TRAINER_REPO` | Trainer source repository the build checks out. |
| `AUTOTUNEX_GB_SERVER_URL` | Base URL of the granite.build server the reconcile loop polls. |

`AUTOTUNEX_LSF_CLUSTER` itself is **not** in that fail-fast set: it is the variant
discriminator, so leaving it unset does not fail startup — it simply selects the bash
spec above.

The remaining `AUTOTUNEX_LSF_*` knobs (accelerators, queue, memory, CPUs and memory
per node, venv path, CUDA home, poll interval) are optional and documented in
[configuration.md](configuration.md#execution-job-launch). Like the bash spec, the
LSF build references its dataset by URI, so it **requires an HF-hosted dataset**.

```
AUTOTUNEX_JOB_BACKEND=llmb
GB_ENVIRONMENT=standalone            # unprefixed — granite.build's own variable

AUTOTUNEX_LSF_CLUSTER=example-cluster
AUTOTUNEX_LSF_ENVIRONMENT_URI=space://environments/skypilot/lsf/example-cluster
AUTOTUNEX_LSF_IMAGE=registry.example.com/tuner:1
AUTOTUNEX_JOB_TRAINER_REPO=https://git.example.com/org/trainer.git
AUTOTUNEX_GB_SERVER_URL=https://gb.example.com
AUTOTUNEX_DATASET_STORAGE_BACKEND=huggingface
```

A ready-to-copy example lives at `envs/lsf.env.example`.

### Auth and reconcile (all variants)

**Authentication.** `AUTOTUNEX_GB_TOKEN_ENV` (default `GB_TOKEN`) names the
environment variable that holds the granite.build token — it does not hold the
token itself. Before submitting, the launcher runs a CLI login with that token;
the reconcile loop authenticates to the granite.build server with it over HTTP.
The token **value** is read only at the subprocess/HTTP boundary and is never
loaded into settings and never logged. For the `llmb` backend the token
environment variable must be present at startup (there is no ambient credential
fallback over HTTP for the reconcile loop) — export it into the shell, do not
merely write it into `.env`:

```bash
export GB_TOKEN=<your granite.build token>
```

**Reconcile cadence.** The loop sweeps non-terminal jobs on an interval and
bounds how many status reads it issues per sweep:

| Setting | Default | Meaning |
| --- | --- | --- |
| `AUTOTUNEX_JOB_RECONCILE_INTERVAL_SECONDS` | `30` | How often the loop sweeps non-terminal jobs for status changes. |
| `AUTOTUNEX_JOB_RECONCILE_CONCURRENCY` | `5` | Upper bound on simultaneous status reads per sweep. |

The loop is restart-safe: its whole working set is one query per sweep, so a
process restart resumes where it left off. Repeated `401`/`403` responses (almost
always one expired token affecting every read) are logged once per sweep rather
than once per job.

**Fail-fast.** Startup refuses to run if a required `llmb` setting — or the token
environment variable — is missing, so a misconfiguration surfaces at boot rather
than on the first submitted job or the first reconcile sweep.

## Choosing a backend

- **`none`** — for exploring the API, or when an external system executes jobs
  and writes results back to the database. Nothing runs on its own.
- **`local`** — for self-contained, in-process HPO on a single host (or against
  an existing Ray cluster). No build server, no external tokens; needs the
  `autotune` package and the dataset on local disk.
- **`llmb`** — to offload builds to a granite.build server, either a remote
  cluster (`custom_code`) or a local standalone machine
  (`GB_ENVIRONMENT=standalone`, the bash spec).
