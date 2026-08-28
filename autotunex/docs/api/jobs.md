# Jobs API

A **job** is one automated tuning run. It references a configuration (whose search
space it optimizes) and a dataset, snapshots that configuration at submit time, and
searches by running **trials**. This page documents the job endpoints under the
`/api/v1/jobs` prefix.

See [overview.md](overview.md) for shared conventions (pagination, the `ProblemDetail`
error shape, base URL), [authentication.md](authentication.md) for how a caller is
resolved to an owner, and [../concepts.md](../concepts.md) for the domain model.

## Ownership and scope

Every read is owner-scoped. By default a caller — admin included — sees only its own
jobs. An admin widens to all owners per request by passing `scope=all`; a non-admin who
passes `scope=all` gets a **403**. This `scope` query parameter (`own` | `all`, default
`own`) is accepted on every owner-scoped endpoint below; `POST /jobs`, the admin-only
`reconcile` endpoint, and the `estimate-usages`/`generate-test-solutions` endpoints do
not take it. Submission is always own-scoped — even an admin submits against its own
configuration and dataset.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/jobs` | List jobs (lean summaries), newest first |
| `POST` | `/api/v1/jobs` | Submit a new tuning job |
| `POST` | `/api/v1/jobs/estimate-usages` | Estimate resource usage for a tuning run |
| `POST` | `/api/v1/jobs/generate-test-solutions` | Generate sample reward test solutions (online-RL) |
| `GET` | `/api/v1/jobs/{job_id}` | Get one job with full detail |
| `GET` | `/api/v1/jobs/by-build-id/{build_id}` | Get one job by its granite.build build id |
| `POST` | `/api/v1/jobs/{job_id}/cancel` | Cancel a live job |
| `DELETE` | `/api/v1/jobs/{job_id}` | Delete a job |
| `POST` | `/api/v1/jobs/{job_id}/reconcile` | Force-reconcile a job's status with granite.build (admin only) |
| `GET` | `/api/v1/jobs/{job_id}/result-report` | List a job's downloadable output assets |
| `GET` | `/api/v1/jobs/{job_id}/result-report/file` | Download one of a job's output files |
| `GET` | `/api/v1/jobs/{job_id}/result-report/archive` | Download all of a job's output files as a ZIP |
| `GET` | `/api/v1/jobs/{job_id}/logs` | Get a job's log lines (keyset paginated) |
| `GET` | `/api/v1/jobs/{job_id}/trials/{trial_id}/logs` | Get one trial's log lines |
| `GET` | `/api/v1/jobs/{job_id}/gb-logs` | Get live build logs for the job |

---

## GET /api/v1/jobs

List jobs as compact `JobSummary` rows, newest first. Returns a `Page<JobSummary>`.

### Query parameters

| Name | Type | Default | Constraints |
| --- | --- | --- | --- |
| `limit` | int | `20` | 1–100 |
| `offset` | int | `0` | ≥ 0 |
| `scope` | string | `own` | `own` \| `all` (admin only for `all`) |
| `q` | string | `none` | Case-insensitive substring filter. Matches the experiment name, model, or status. |

### Response `200` — `Page<JobSummary>`

The page wrapper carries the requested window plus the total count:

| Field | Type | Notes |
| --- | --- | --- |
| `items` | `JobSummary[]` | The page of jobs |
| `total` | int | Total matching jobs, ignoring pagination |
| `limit` | int | Echoes the requested limit |
| `offset` | int | Echoes the requested offset |

`JobSummary` is deliberately lean — it carries identity, status, the owner/config/dataset
labels, the model, and timestamps. Heavier detail (nested tasks, trials, JSON blobs) is
only on the detail endpoint.

| Field | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Job id |
| `user_id` | string | Owner's id |
| `status` | string | One of the six run states (see below) |
| `seed` | int \| null | Random seed |
| `config_id` | UUID | Referenced configuration |
| `config_name` | string | Configuration's name |
| `dataset_id` | UUID | Referenced dataset |
| `dataset` | string | Dataset's name |
| `model` | string | Model being fine-tuned |
| `experiment_name` | string | Human-readable run label |
| `user` | string | Owner's email |
| `created_at` | datetime | ISO 8601 |
| `updated_at` | datetime | ISO 8601 |
| `finished_at` | string \| null | The latest `gb_tasks.updated_at` for the job — a string rather than a datetime because the column is `VARCHAR(255)`; `null` when the job has no build tasks (e.g. the `local` backend, or a job still `pending`) |

```json
{
  "items": [
    {
      "id": "6b1f...",
      "user_id": "a2c9...",
      "status": "running",
      "seed": 42,
      "config_id": "d4e1...",
      "config_name": "granite-sft-sweep",
      "dataset_id": "9f30...",
      "dataset": "support-tickets",
      "model": "example-org/base-model-1b",
      "experiment_name": "sweep-2026-08",
      "user": "you@example.com",
      "created_at": "2026-08-11T09:00:00Z",
      "updated_at": "2026-08-11T09:12:00Z",
      "finished_at": null
    }
  ],
  "total": 1,
  "limit": 20,
  "offset": 0
}
```

### Notable statuses

`403` if a non-admin requests `scope=all`.

---

## POST /api/v1/jobs

Submit a new tuning job owned by the calling principal. Returns `201` with the full
`JobRead`. The accepted job is handed to the runner seam and starts as `pending`.

Ownership is taken from the caller (there is **no** `user_id` in the body), and
`tuning_type` is derived server-side from the referenced configuration. Unknown fields
are rejected (`extra="forbid"`).

### Request body — `JobCreate`

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `config_id` | UUID | yes | — | Configuration to optimize; must be owned by the caller |
| `dataset_id` | UUID | yes | — | Dataset to train on; must be owned by the caller and `ready` |
| `model` | string | yes | — | Non-empty; model to fine-tune |
| `model_source` | string | no | `huggingface` | `huggingface` \| `custom_path` |
| `experiment_name` | string | yes | — | Non-empty; human-readable run label |
| `autotune` | bool | no | `true` | Whether to run the HPO search |
| `seed` | int | no | `42` | Random seed |
| `reward_function_code` | string \| null | no | `null` | Reward function source; required for online-RL configs |
| `reward_function_name` | string \| null | no | `null` | Reward function entry-point name (defaults to `compute_score` when code is supplied) |

```bash
curl -X POST https://example.com/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "d4e1...",
    "dataset_id": "9f30...",
    "model": "example-org/base-model-1b",
    "model_source": "huggingface",
    "experiment_name": "sweep-2026-08",
    "autotune": true,
    "seed": 42
  }'
```

### Validation rules

- The caller must own the referenced configuration **and** dataset.
- The dataset must be in status `ready`.
- An **online-RL** configuration (its `rl_tuner_type` is one of `ppo` / `grpo` / `dapo`,
  compared case-insensitively) requires a non-empty `reward_function_code`.

### Response `201` — `JobRead`

See [the `JobRead` shape](#the-jobread-shape) below.

### Notable statuses

| Status | When |
| --- | --- |
| `403` | Caller has no resolvable owner (unprovisioned) |
| `404` | The referenced configuration or dataset does not exist or is not the caller's |
| `409` | The dataset is not `ready`, or a referenced row was deleted mid-submission |
| `422` | Body fails validation, or an online-RL config is missing its reward function |

---

## POST /api/v1/jobs/estimate-usages

Estimate the GPU/CPU memory and GPU count a tuning run would need, for a **saved**
configuration (`config_id`) or an **unsaved** one supplied inline (`config_data`) — the
inline path is what the start-tuning wizard (Step 3) uses before a configuration is
persisted. Returns `200` with an `EstimateUsagesResponse`. The estimate is a planning-time
heuristic, not a measured profile, and this endpoint takes **no** `scope` query parameter.

### Request body — `EstimateUsagesRequest`

Exactly one of `config_id` / `config_data` must be supplied. Unknown fields are rejected
(`extra="forbid"`).

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `model_name` | string | yes | — | Model to size the run for; its parameter count is parsed from the name |
| `gpu_memory` | int | no | `80` | Per-GPU memory in GB; must be ≥ 1 |
| `config_id` | UUID \| null | no | `null` | A saved, caller-owned configuration; mutually exclusive with `config_data` |
| `config_data` | object \| null | no | `null` | An inline configuration; mutually exclusive with `config_id` |
| `tuner_type` | string \| null | no | `null` | Tuner variant override |
| `rl_tuner_type` | string \| null | no | `null` | RL tuner variant override |

```bash
curl -X POST https://example.com/api/v1/jobs/estimate-usages \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "example-org/base-model-7b",
    "gpu_memory": 80,
    "config_id": "d4e1..."
  }'
```

### Response `200` — `EstimateUsagesResponse`

| Field | Type | Notes |
| --- | --- | --- |
| `model_size_billion_params` | float | Estimated model size, in billions of parameters |
| `gpu_memory_gb` | float | Estimated total GPU memory, in GB |
| `cpu_memory_gb` | float | Estimated CPU memory, in GB |
| `num_gpus` | int | Estimated number of GPUs |
| `weights_memory` | float | GPU memory attributed to model weights, in GB |
| `optimizer_memory` | float | GPU memory attributed to optimizer state, in GB |
| `gradients_memory` | float | GPU memory attributed to gradients, in GB |
| `activations_memory` | float | GPU memory attributed to activations, in GB |

### Notable statuses

| Status | When |
| --- | --- |
| `404` | The referenced `config_id` does not exist or is not the caller's |
| `422` | Body fails validation (neither or both of `config_id`/`config_data`, `gpu_memory` < 1), or `model_name` has no parseable parameter count |

---

## POST /api/v1/jobs/generate-test-solutions

Generate one sample model answer per prompt, so the online-RL reward step can preview how
a reward function scores realistic outputs — the reward-seed half of the wizard's reward
step. Each prompt is sent to the configured LLM seam. Returns `200` with a
`GenerateTestSolutionsResponse`. A prompt whose completion fails degrades to an empty
string (index-aligned) rather than failing the whole request; this endpoint takes **no**
`scope` query parameter.

### Request body — `GenerateTestSolutionsRequest`

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `prompts` | `ChatMessage[][]` | yes | — | Each prompt is a chat-message array (a VERL prompt); a `ChatMessage` is `{"role": ..., "content": ...}` |

```bash
curl -X POST https://example.com/api/v1/jobs/generate-test-solutions \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      [{"role": "user", "content": "What is 2 + 2?"}]
    ]
  }'
```

### Response `200` — `GenerateTestSolutionsResponse`

| Field | Type | Notes |
| --- | --- | --- |
| `solutions` | `string[]` | One solution string per input prompt, index-aligned; `""` for a prompt that failed |

### Notable statuses

| Status | When |
| --- | --- |
| `503` | No LLM provider is configured on this server |

---

## GET /api/v1/jobs/{job_id}

Return one job with its current status and full detail. Returns `JobRead`.

### Path & query parameters

| Name | In | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `job_id` | path | UUID | — | Job id |
| `scope` | query | string | `own` | `own` \| `all` (admin only for `all`) |

### The `JobRead` shape

`JobRead` includes every `JobSummary` field **plus** the following:

| Field | Type | Notes |
| --- | --- | --- |
| `model_source` | string | `huggingface` or `custom_path` |
| `tuning_type` | string \| null | Derived from the configuration at submit |
| `rl_tuner_type` | string \| null | RL tuner variant, if any |
| `ray_address` | string \| null | Address of the compute cluster, if set |
| `cleanup` | bool \| null | Whether artifacts are cleaned up after the run |
| `autotune` | bool \| null | Whether the HPO search runs |
| `num_trials` | int | Trial count for this job (≥ 0) |
| `tasks` | `GbTaskRead[]` | Build/deploy steps attached to the job (may be empty) |
| `config_snapshot` | object \| null | The configuration captured at submit time |
| `output_artifacts` | object \| null | Free-form artifact descriptor written by the pipeline |
| `trials` | `TrialRead[]` | The job's trials (may be empty) |
| `is_stale` | bool | `true` when the live configuration's behavioural settings no longer match what the job snapshotted at submit; detail-only, never on `JobSummary` |

```json
{
  "id": "6b1f...",
  "user_id": "a2c9...",
  "status": "completed",
  "seed": 42,
  "config_id": "d4e1...",
  "config_name": "granite-sft-sweep",
  "dataset_id": "9f30...",
  "dataset": "support-tickets",
  "model": "example-org/base-model-1b",
  "experiment_name": "sweep-2026-08",
  "user": "you@example.com",
  "model_source": "huggingface",
  "tuning_type": "sft",
  "rl_tuner_type": null,
  "ray_address": null,
  "cleanup": true,
  "autotune": true,
  "num_trials": 8,
  "tasks": [],
  "config_snapshot": { "name": "granite-sft-sweep", "config_data": { "...": "..." } },
  "output_artifacts": { "best_trial": "a1b2c3" },
  "trials": [],
  "is_stale": false,
  "created_at": "2026-08-11T09:00:00Z",
  "updated_at": "2026-08-11T10:30:00Z",
  "finished_at": "2026-08-11 10:28:14"
}
```

### Nested: `TrialRead`

One training run inside a job, evaluating a single concrete parameter assignment.

| Field | Type | Notes |
| --- | --- | --- |
| `id` | string | Short opaque id (up to 16 chars), assigned by the pipeline — not a UUID |
| `job_id` | UUID | Parent job |
| `status` | string | One of the six run states |
| `config` | object \| null | The concrete parameter assignment this trial tested |
| `metric` | string \| null | Name of the objective metric |
| `metrics` | object | Reported metrics, e.g. `{"eval_loss": 0.42}`; empty until the trial reports |
| `created_at` | datetime \| null | ISO 8601; may be absent for pipeline-written rows |
| `updated_at` | datetime \| null | ISO 8601; may be absent for pipeline-written rows |

### Nested: `tasks[]` element

Each entry describes a build or deployment step attached to the job (for example a
tuning build or an artifact download). Tasks are nested rather than flattened.

| Field | Type | Notes |
| --- | --- | --- |
| `task_id` | UUID | Task id |
| `build_id` | UUID \| null | Underlying build id, if any |
| `task_status` | string | One of the six run states |
| `task_type` | string | Kind of step; one of the `GbTaskType` values `RITS`, `TUNING`, or `DOWNLOAD` |
| `github_pr_url` | string \| null | Associated pull-request URL, if any |
| `artifact_id` | UUID \| null | Produced artifact id |
| `artifact_uri` | string \| null | Produced artifact location |
| `build_status` | object \| null | Free-form status detail from the build backend |
| `task_started_at` | string \| null | Free-text start time (stored as a string, not a timestamp) |
| `task_updated_at` | string \| null | Free-text update time |
| `rits_url` | string \| null | Deployment-endpoint URL, populated when the step deploys a served model |

### Notable statuses

`403` (non-admin requesting `scope=all`), `404` (no such job, or not the caller's).

---

## GET /api/v1/jobs/by-build-id/{build_id}

Return one job located by its granite.build **build id** instead of its job id. Same
`JobRead` payload and owner-scoping as `GET /jobs/{job_id}` — it differs only in the lookup
key. Returns `JobRead`.

### Path & query parameters

| Name | In | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `build_id` | path | UUID | — | granite.build build id carried by one of the job's tasks |
| `scope` | query | string | `own` | `own` \| `all` (admin only for `all`) |

### Response `200` — `JobRead`

See [the `JobRead` shape](#the-jobread-shape) above.

### Notable statuses

`403` (non-admin requesting `scope=all`), `404` (no job carries this build id, or it is not
the caller's — `BuildNotFoundError`).

---

## POST /api/v1/jobs/{job_id}/cancel

Drive a live job (`pending`/`running`/`paused`) to `terminated`, cancelling any live backend
work. Owner-scoped exactly like `GET`/`DELETE`, and **idempotent** for an already-
`terminated` job (it is returned unchanged). Returns `JobRead`.

### Path & query parameters

| Name | In | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `job_id` | path | UUID | — | Job id |
| `scope` | query | string | `own` | `own` \| `all` (admin only for `all`) |

### Response `200` — `JobRead`

See [the `JobRead` shape](#the-jobread-shape) above.

### Notable statuses

| Status | When |
| --- | --- |
| `403` | Non-admin requesting `scope=all` |
| `404` | No such job, or it belongs to someone else |
| `409` | The job is `completed` or `error` and cannot be cancelled (`JobNotCancellableError`) |
| `409` | A local in-process run is still stopping (`JobCancellationInProgressError`) — retry |
| `502` | The backend cancel failed upstream (`BuildCancelUpstreamError`) |

---

## DELETE /api/v1/jobs/{job_id}

Delete a job. Returns `204` with an empty body. A live job (`pending`/`running`/`paused`)
is auto-cancelled first — its backend work is stopped via the runner — before the row is
removed; a job with no live work deletes directly. The delete cascades to the job's trials,
results, log entries, and build tasks.

### Path & query parameters

| Name | In | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `job_id` | path | UUID | — | Job id |
| `scope` | query | string | `own` | `own` \| `all` (admin only for `all`) |

### Notable statuses

| Status | When |
| --- | --- |
| `403` | Non-admin requesting `scope=all` |
| `404` | No such job, or it belongs to someone else |
| `409` | A local in-process run is still stopping (`JobCancellationInProgressError`) — retry |
| `502` | The backend cancel failed upstream (`BuildCancelUpstreamError`) |

---

## POST /api/v1/jobs/{job_id}/reconcile

Force one job to re-sync with granite.build, rewriting its `build_status` and output
artifacts and forcing `status` to what the build backend reports. **Admin only**, and it
takes **no** `scope` query parameter. Returns `JobRead`.

### Path parameters

| Name | In | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `job_id` | path | UUID | — | Job id |

### Response `200` — `JobRead`

See [the `JobRead` shape](#the-jobread-shape) above.

### Notable statuses

| Status | When |
| --- | --- |
| `403` | Caller is not an admin |
| `404` | No such job |
| `409` | The job cannot be reconciled (`JobNotReconcilableError`) |
| `502` | The build backend returned an error (`BuildReconcileUpstreamError`) |
| `503` | The build backend is unavailable (`BuildReconcileUnavailableError`) |

---

## GET /api/v1/jobs/{job_id}/result-report

List the job's downloadable output files (the Results panel). Owner-scoped like the rest of
the read path. The list is computed **on read** from the job's artifact source — the request
never writes server state. If the job's `output_artifacts` column has already been populated
(the tuning pipeline may write it), that is served directly; otherwise the files are listed
from the job's produced-model location, resolved from its `TUNING` task's `artifact_uri`:

- `hf://…` — the produced HuggingFace model repo (listed via the HuggingFace Hub API);
- `file://…` — a directory on the server (granite.build standalone output);
- no `TUNING` task at all, and the job is `completed` — a local run's output directory
  (`local_output_dir/<job_id>/results`).

A job that *does* have a `TUNING` task whose `artifact_uri` is missing, or whose build has not
succeeded, is a **409** — it never falls back to the local output directory.

A job that has produced artifacts returns a JSON array of `AssetSummary` (possibly empty when
the source is readable but holds no files). A job that has not produced artifacts yet, or
whose source cannot be read, returns an RFC 9457 problem detail rather than an empty array —
see *Notable statuses*.

### Path & query parameters

| Name | In | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `job_id` | path | UUID | — | Job id |
| `scope` | query | string | `own` | `own` \| `all` (admin only for `all`) |

### Response `200` — `list[AssetSummary]`

Each element describes one output file:

| Field | Type | Notes |
| --- | --- | --- |
| `filename` | string | File name |
| `size` | int | File size in bytes |
| `modified` | datetime \| null | Last-modified time, if known |
| `path` | string \| null | Path within the artifact source, relative to its root, if known |
| `file_hash` | string \| null | Content hash, if known |
| `published` | bool \| null | Whether the asset has been published, if known |

```json
[
  { "filename": "best_model.safetensors", "size": 4194304,
    "modified": "2026-08-11T10:30:00Z", "path": "best_model.safetensors", "file_hash": null, "published": null }
]
```

### Notable statuses

`403` (non-admin requesting `scope=all`), `404` (no such job, not the caller's, or no artifact
source could be located), `409` (the job has not produced artifacts yet — it is not complete,
or its build has not succeeded), `502` (the artifact source exists but could not be read).

---

## GET /api/v1/jobs/{job_id}/result-report/file

Download a single output file listed by the result-report endpoint. Owner-scoped like the
rest of the read path. The bytes are always streamed from the job's **physical** artifact
source (they cannot be served from a metadata-only `output_artifacts` column). The response
is the raw file, **not** JSON: it is streamed with `Content-Disposition: attachment` and a
`Content-Type` guessed from the filename (falling back to `application/octet-stream`), plus a
`Content-Length` when the size is known.

The required `path` query parameter is the file's path **as returned in the result-report
list** — the `path` field, relative to the artifact source's root, not the bare filename. The
list keys on the path rather than the basename because filenames can repeat across directories
(for example, several `adapters.safetensors` files under different trial subdirectories).

### Path & query parameters

| Name | In | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `job_id` | path | UUID | — | Job id |
| `path` | query | string | — | Required (non-empty); the file's relative path from the result-report list |
| `scope` | query | string | `own` | `own` \| `all` (admin only for `all`) |

### Response `200` — the raw file

The body is the file's bytes, streamed as an attachment. There is no JSON envelope.

```
Content-Disposition: attachment; filename="adapters.safetensors"; filename*=UTF-8''adapters.safetensors
Content-Type: application/octet-stream
Content-Length: 4194304
```

### Notable statuses

`403` (non-admin requesting `scope=all`), `404` (no such job, not the caller's, no artifact
source could be located, the file does not exist under it, or the `path` escapes the source
root), `409` (the job has not produced artifacts yet — it is not complete, or its build has
not succeeded), `502` (the artifact source exists but could not be read).

---

## GET /api/v1/jobs/{job_id}/result-report/archive

Download **all** of a job's output files as a single ZIP archive. Owner-scoped like the rest
of the read path, and — like the single-file endpoint — the bytes are streamed from the job's
physical artifact source. The archive is built on the fly (no temporary file on the server)
and its entry names are the assets' relative paths, so directory structure is preserved and
same-named files across directories stay distinct.

The response is streamed with media type `application/zip` and
`Content-Disposition: attachment; filename="<experiment_name>_assets.zip"`, where the
experiment name is reduced to a filesystem/header-safe stem.

### Path & query parameters

| Name | In | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `job_id` | path | UUID | — | Job id |
| `scope` | query | string | `own` | `own` \| `all` (admin only for `all`) |

### Response `200` — a ZIP archive

The body is the ZIP bytes, streamed as an attachment. There is no JSON envelope.

```
Content-Disposition: attachment; filename="sweep-2026-08_assets.zip"; filename*=UTF-8''sweep-2026-08_assets.zip
Content-Type: application/zip
```

### Notable statuses

`403` (non-admin requesting `scope=all`), `404` (no such job, not the caller's, or no artifact
source could be located), `409` (the job has not produced artifacts yet — it is not complete,
or its build has not succeeded), `502` (the artifact source exists but could not be read).

---

## GET /api/v1/jobs/{job_id}/logs

Return one keyset page of the job's job-level log lines, **newest first**. Returns a
`LogPage`. Logs are an append stream read backward by cursor, so this uses keyset
pagination (`before_id`) rather than `offset`.

### Path & query parameters

| Name | In | Type | Default | Constraints |
| --- | --- | --- | --- | --- |
| `job_id` | path | UUID | — | Job id |
| `before_id` | query | int | `0` | ≥ 0; return lines older than this id (`0` = newest page) |
| `limit` | query | int | `50` | 1–500 |
| `scope` | query | string | `own` | `own` \| `all` (admin only for `all`) |

### Response `200` — `LogPage`

| Field | Type | Notes |
| --- | --- | --- |
| `logs` | `LogEntryRead[]` | Log lines, newest first |
| `has_more` | bool | Whether an older page exists |
| `next_before_id` | int \| null | Pass as `before_id` for the next (older) page; `null` when `has_more` is false |

**`LogEntryRead`:**

| Field | Type | Notes |
| --- | --- | --- |
| `id` | int | Line id (the keyset cursor) |
| `level` | string \| null | Log level |
| `filename` | string \| null | Source file, if recorded |
| `message` | string \| null | Log message |
| `iteration` | int \| null | Training iteration, if recorded |
| `epoch` | float \| null | Training epoch, if recorded |
| `timestamp` | datetime \| null | Stored as a naive (unzoned) datetime |

```json
{
  "logs": [
    { "id": 812, "level": "INFO", "filename": "train.py", "message": "epoch complete",
      "iteration": 400, "epoch": 2.0, "timestamp": "2026-08-11T10:15:00" }
  ],
  "has_more": true,
  "next_before_id": 812
}
```

### Notable statuses

`403` (non-admin requesting `scope=all`), `404` (no such job, or not the caller's).

---

## GET /api/v1/jobs/{job_id}/trials/{trial_id}/logs

Same `LogPage` shape and pagination as the job-logs endpoint, scoped to a single trial.

### Path & query parameters

| Name | In | Type | Default | Constraints |
| --- | --- | --- | --- | --- |
| `job_id` | path | UUID | — | Job id |
| `trial_id` | path | string | — | Trial id (short opaque string) |
| `before_id` | query | int | `0` | ≥ 0 |
| `limit` | query | int | `50` | 1–500 |
| `scope` | query | string | `own` | `own` \| `all` (admin only for `all`) |

### Notable statuses

`403`, `404`.

---

## GET /api/v1/jobs/{job_id}/gb-logs

Return the job's **live build logs** from the configured build backend, oldest first.
The response is a plain JSON array of strings.

### Path & query parameters

| Name | In | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `job_id` | path | UUID | — | Job id |
| `all` | query | bool | `false` | Page through all logs, not just the first page |
| `scope` | query | string | `own` | `own` \| `all` (admin only for `all`) |

### Response `200` — `list[str]`

```json
["Starting build...", "Fetching base image...", "Build succeeded."]
```

### Notable statuses

| Status | When |
| --- | --- |
| `403` | Non-admin requesting `scope=all` |
| `404` | No such job, or not the caller's |
| `502` | The build backend returned an error |
| `503` | The build backend is unavailable |

---

## Run status vocabulary

Jobs, trials, and build tasks share one six-value status enum:

`pending → running | completed | error | terminated`,
`running → paused | completed | error | terminated`,
`paused → running | terminated | error`. `pending` may go straight to a terminal state
because the reconcile loop can observe a build that already finished; `paused` never goes
directly to `completed`. The terminal states (`completed`, `error`, `terminated`) have no
outgoing transitions.

## See also

- [overview.md](overview.md) — base URL, pagination, error shape
- [authentication.md](authentication.md) — owners, admin, and `scope`
- [../concepts.md](../concepts.md) — jobs, trials, tasks, and the domain model
- [../operations/configuration.md](../operations/configuration.md) — runner and build-backend settings
