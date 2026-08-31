# Getting started

This guide takes you from a fresh clone to a running AutoTuneX service and a
complete round trip through the API — create a configuration, register and
upload a dataset, submit a job, and read it back. Every command is meant to be
copy-pasted against a local server.

AutoTuneX is a FastAPI service for automated fine-tuning and hyperparameter
optimization (HPO) of large language models. A **job** describes what to
optimize; the tuning pipeline searches by running **trials** and reports which
configuration performed best. For the vocabulary behind those terms, see
[concepts.md](concepts.md).

## Prerequisites

- **Python 3.11+**. AutoTuneX uses `StrEnum`, `Self`, `datetime.UTC`, and PEP
  604 unions, so 3.11 is the floor.
- **[uv](https://docs.astral.sh/uv/)** is recommended but not required — any
  tool that can create a virtual environment and install a package works. The
  commands below use `uv`.

No database server is needed for local development: the default configuration
uses a file-backed SQLite database created on first run.

## Install and run

```bash
git clone https://github.com/ibm-granite/granite.build.git
cd granite.build/autotunex

uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install -e "./src/fm-tune"

cp .env.example .env          # defaults work as-is for local development
uvicorn autotunex.main:app --reload
```

The second install supplies the slim, torch-free `autotune` catalog that the
configuration-template and dataset-format endpoints need; `make install` runs
both steps for you.

The server starts on <http://127.0.0.1:8000> and creates a local SQLite database
(`autotunex.db`) in the working directory on first run. No database server
needed.

`.env.example` is the full configuration reference; copying it unchanged runs
the service with the no-op job backend (see
[Why is my job still pending?](#why-is-my-job-still-pending) below). For the
full list of settings and how to point at PostgreSQL or MySQL instead, see
[operations/configuration.md](operations/configuration.md).

## The interactive API docs

Once the server is running, open any of these in a browser. `/` issues a
temporary redirect to `/autotune`, where the SPA mounts by default, so it 404s
unless the SPA is built and `AUTOTUNEX_FRONTEND_DIR` is configured. That target
is hard-coded: changing `AUTOTUNEX_FRONTEND_BASE_PATH` moves the SPA but not this
redirect. The interactive docs are at `/docs`.

| URL | What it is |
| --- | --- |
| <http://127.0.0.1:8000/docs> | Swagger UI — try every endpoint from the browser |
| <http://127.0.0.1:8000/redoc> | ReDoc reference |
| <http://127.0.0.1:8000/openapi.json> | Raw OpenAPI schema |

A liveness probe lives at <http://127.0.0.1:8000/health>, outside the versioned
API path, so you can check the process is up without authenticating:

```bash
curl -s http://127.0.0.1:8000/health
```

```json
{ "status": "ok", "service": "AutoTuneX API", "version": "0.3.5" }
```

## Default mode: standalone, no authentication

Out of the box the service runs in **standalone mode** — authentication is
disabled. Every caller is treated as the same principal, and every write
(configurations, datasets, jobs) is attributed to a single default owner,
`standalone@autotunex.local`, which is provisioned automatically on the first
request that needs it. That means **all of the examples below work with no
credentials** — no tokens, no headers.

This is meant for local development and single-user deployments. To enable real
authentication — API keys, OIDC bearer tokens, or browser sessions — see
[api/authentication.md](api/authentication.md). Production deployments must
configure a real provider; startup refuses to run with authentication disabled
in a `prod` environment unless `AUTOTUNEX_ALLOW_INSECURE_NO_AUTH=true` is set,
which logs a loud startup warning.

The versioned API is mounted under **`/api/v1`** by default. That prefix is
configurable through the `AUTOTUNEX_API_PREFIX` setting; this guide assumes the
default. Health (`/health`) and the auth routes (`/auth/*`) sit outside it.

## A guided walkthrough

The steps below use `curl` against `http://127.0.0.1:8000` and pipe responses
through `python -m json.tool` for readable output. Each request produces JSON;
copy the `id` from one response into the next step. If you have
[`jq`](https://jqlang.github.io/jq/) installed you can extract ids
automatically, but it is not required.

The flow is: **configuration → dataset (create, then upload) → job → read
back → logs**. A job references an existing configuration and dataset, so those
must exist first, and the dataset must be `ready` before a job can use it.

### 1. Create a configuration

A configuration is a named, reusable set of tuning settings, stored in the
schema-less `config_data` JSON column. Its shape is **not** validated — the only
requirement is that `config_data` be a non-empty JSON object. Send whatever
structure your tuning pipeline expects.

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/configurations \
  -H 'Content-Type: application/json' \
  -d '{
        "name": "lora-sweep",
        "config_data": {
          "learning_rate": {"type": "float", "min_val": 1e-5, "max_val": 1e-3},
          "num_epochs": {"type": "int", "min_val": 1, "max_val": 3}
        }
      }' | python -m json.tool
```

Returns `201 Created` with the stored configuration. A brand-new configuration
is not yet referenced by any job, so `associated_jobs` is always empty here.

```json
{
  "id": "8f14e45f-ceea-467d-9a1e-4b2c3d4e5f60",
  "user_id": "00000000-0000-4000-8000-000000000000",
  "name": "lora-sweep",
  "tuner_type": null,
  "rl_tuner_type": null,
  "config_data": {
    "learning_rate": {"type": "float", "min_val": 1e-05, "max_val": 0.001},
    "num_epochs": {"type": "int", "min_val": 1, "max_val": 3}
  },
  "associated_jobs": [],
  "created_at": "2026-08-11T10:00:00Z",
  "updated_at": "2026-08-11T10:00:00Z"
}
```

Save the returned `id` for step 3:

```bash
CONFIG_ID=8f14e45f-ceea-467d-9a1e-4b2c3d4e5f60   # paste your own id here
```

> **Tip:** `GET /api/v1/configurations/template` returns a starter template you
> can adapt instead of writing `config_data` from scratch (it requires the
> `autotune` catalog from the second install step above, and returns `503`
> otherwise).

### 2. Register a dataset

Registering a dataset happens in two parts: first create the metadata record,
then upload the training file to it.

**Create the record.** Only a `name` and a `data_format` (`jsonl`, `csv`, or
`parquet`) are needed; `description` is optional.

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/datasets \
  -H 'Content-Type: application/json' \
  -d '{ "name": "alpaca-sample", "data_format": "jsonl" }' | python -m json.tool
```

Returns `201 Created` with `status: "empty"` — the record exists but has no file
yet.

```json
{
  "id": "9f41b2c3-1d2e-4a5b-8c7d-6e5f4a3b2c1d",
  "user_id": "00000000-0000-4000-8000-000000000000",
  "name": "alpaca-sample",
  "description": null,
  "data_format": "jsonl",
  "status": "empty",
  "status_detail": null,
  "train_file": "alpaca-sample_train",
  "train_records": null,
  "train_file_size": null,
  "validation_file": "alpaca-sample_validation",
  "validation_records": null,
  "validation_file_size": null,
  "artifact_id": null,
  "artifact_url": null,
  "associated_jobs": [],
  "created_at": "2026-08-11T10:01:00Z",
  "updated_at": "2026-08-11T10:01:00Z",
  "preview": null
}
```

```bash
DATASET_ID=9f41b2c3-1d2e-4a5b-8c7d-6e5f4a3b2c1d   # paste your own id here
```

**Upload the file.** Create a small training file, then upload it as
multipart form data in a `train_file` field:

```bash
cat > train.jsonl <<'EOF'
{"input": "Translate to French: Hello", "output": "Bonjour"}
{"input": "Translate to French: Goodbye", "output": "Au revoir"}
EOF

curl -s -X POST "http://127.0.0.1:8000/api/v1/datasets/${DATASET_ID}/upload" \
  -F train_file=@train.jsonl | python -m json.tool
```

Returns `202 Accepted` with `status: "uploading"`: the file is accepted and
processed off-request.

```json
{
  "id": "9f41b2c3-1d2e-4a5b-8c7d-6e5f4a3b2c1d",
  "user_id": "00000000-0000-4000-8000-000000000000",
  "name": "alpaca-sample",
  "description": null,
  "data_format": "jsonl",
  "status": "uploading",
  "status_detail": null,
  "train_file": "alpaca-sample_train",
  "train_records": null,
  "train_file_size": null,
  "validation_file": "alpaca-sample_validation",
  "validation_records": null,
  "validation_file_size": null,
  "artifact_id": null,
  "artifact_url": null,
  "associated_jobs": [],
  "created_at": "2026-08-11T10:01:00Z",
  "updated_at": "2026-08-11T10:02:00Z",
  "preview": null
}
```

Processing happens in the background, so poll the dataset until its status
becomes `ready` before moving on — **a job cannot use a dataset that is not
`ready`**:

```bash
curl -s "http://127.0.0.1:8000/api/v1/datasets/${DATASET_ID}" \
  | python -m json.tool | grep '"status"'
```

When it reports `"status": "ready"`, continue.

### 3. Submit a job

A job ties together the configuration, the dataset, and the model to tune.
`model_source` is `huggingface` (the default) or `custom_path`. Ownership,
`tuning_type`, and the configuration snapshot are all filled in server-side.

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d "{
        \"config_id\": \"${CONFIG_ID}\",
        \"dataset_id\": \"${DATASET_ID}\",
        \"model\": \"Qwen/Qwen2.5-0.5B-Instruct\",
        \"model_source\": \"huggingface\",
        \"experiment_name\": \"my-first-tuning\"
      }" | python -m json.tool
```

Returns `201 Created` with the job in status `pending`:

```json
{
  "id": "0b1e7a2c-3d4e-4f5a-9b8c-7d6e5f4a3b2c",
  "user_id": "00000000-0000-4000-8000-000000000000",
  "status": "pending",
  "seed": 42,
  "config_id": "8f14e45f-ceea-467d-9a1e-4b2c3d4e5f60",
  "config_name": "lora-sweep",
  "dataset_id": "9f41b2c3-1d2e-4a5b-8c7d-6e5f4a3b2c1d",
  "dataset": "alpaca-sample",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "model_source": "huggingface",
  "experiment_name": "my-first-tuning",
  "tuning_type": null,
  "rl_tuner_type": null,
  "autotune": true,
  "ray_address": null,
  "cleanup": null,
  "user": "standalone@autotunex.local",
  "num_trials": 0,
  "tasks": [],
  "config_snapshot": {
    "name": "lora-sweep",
    "tuner_type": null,
    "rl_tuner_type": null,
    "config_data": {
      "learning_rate": {"type": "float", "min_val": 1e-05, "max_val": 0.001},
      "num_epochs": {"type": "int", "min_val": 1, "max_val": 3}
    }
  },
  "output_artifacts": null,
  "trials": [],
  "is_stale": false,
  "created_at": "2026-08-11T10:05:00Z",
  "updated_at": "2026-08-11T10:05:00Z",
  "finished_at": null
}
```

```bash
JOB_ID=0b1e7a2c-3d4e-4f5a-9b8c-7d6e5f4a3b2c   # paste your own id here
```

If the referenced configuration or dataset does not exist, the request is
rejected with a `404`; if the dataset exists but is not `ready`, a `409`; and if
an online-RL configuration is missing a reward function, a `422` problem detail
instead. See [api/jobs.md](api/jobs.md) for the full submission contract.

### 4. Read the job back

Fetch one job for full detail — including the nested `tasks`, the `trials`
list, `num_trials`, and the two JSON blobs:

```bash
curl -s "http://127.0.0.1:8000/api/v1/jobs/${JOB_ID}" | python -m json.tool
```

List jobs for the lean page shape — identity, status, and the
owner/config/dataset labels only, newest first. It supports `limit` (1–100,
default 20) and `offset` pagination:

```bash
curl -s "http://127.0.0.1:8000/api/v1/jobs?limit=20&offset=0" | python -m json.tool
```

```json
{
  "items": [
    {
      "id": "0b1e7a2c-3d4e-4f5a-9b8c-7d6e5f4a3b2c",
      "status": "pending",
      "user_id": "00000000-0000-4000-8000-000000000000",
      "user": "standalone@autotunex.local",
      "config_id": "8f14e45f-ceea-467d-9a1e-4b2c3d4e5f60",
      "config_name": "lora-sweep",
      "dataset_id": "9f41b2c3-1d2e-4a5b-8c7d-6e5f4a3b2c1d",
      "dataset": "alpaca-sample",
      "model": "Qwen/Qwen2.5-0.5B-Instruct",
      "experiment_name": "my-first-tuning",
      "seed": 42,
      "created_at": "2026-08-11T10:05:00Z",
      "updated_at": "2026-08-11T10:05:00Z",
      "finished_at": null
    }
  ],
  "total": 1,
  "limit": 20,
  "offset": 0
}
```

`model_source`, `tuning_type`, `num_trials`, the nested `tasks`, the `trials`,
`is_stale`, and the JSON blobs live only on the detail response above — fetch
them per-job after showing the list.

`is_stale` is `true` when the live configuration's behavioural settings no longer
match what the job snapshotted at submit; it is detail-only, never on
`JobSummary`.

`finished_at` is the latest `gb_tasks.updated_at` for the job, a **string**
rather than a datetime because the column is `VARCHAR(255)`, and `null` when the
job has no build tasks — which is the case here.

### 5. Read the logs

Job-level log lines are read newest-first, by keyset cursor:

```bash
curl -s "http://127.0.0.1:8000/api/v1/jobs/${JOB_ID}/logs" | python -m json.tool
```

```json
{
  "logs": [],
  "has_more": false,
  "next_before_id": null
}
```

To page backward through older lines, pass the returned `next_before_id` as
`?before_id=` on the next request (`limit` is 1–500, default 50). On the default
setup this list stays empty — the next section explains why.

## Why is my job still pending?

**Submitting a job accepts it, but does not run it.** With the default job
backend (`AUTOTUNEX_JOB_BACKEND=none`), an accepted job is recorded and then
handed to a no-op runner that does nothing. The job stays `pending`, no trials
are ever created, and `GET /api/v1/jobs/{id}/logs` returns an empty list. This
is expected: the default install lets you explore and drive the full API, but
**execution requires configuring a backend**.

To actually run tuning, choose a job backend:

- **`AUTOTUNEX_JOB_BACKEND=local`** runs the HPO pipeline in-process. It needs
  the optional `autotune` trainer package installed and the dataset's files
  available on local disk, and it drives the job to a terminal state itself,
  persisting trials, results, and logs as it goes.
- Other backends submit the run to an external builder rather than executing
  in-process.

Each backend has its own required settings. See
[operations/job-backends.md](operations/job-backends.md) for how to pick and
configure one.

## Next steps

- **[concepts.md](concepts.md)** — the domain model: jobs, configurations,
  datasets, trials, results, and how they relate.
- **[api/overview.md](api/overview.md)** — the API surface, pagination, the
  owner-scoping model, and the RFC 9457 problem-detail error shape.
- **[api/jobs.md](api/jobs.md)** — the full job submission and read contract.
- **[api/authentication.md](api/authentication.md)** — enabling API keys, OIDC
  bearer tokens, or browser sessions in place of standalone mode.
- **[operations/job-backends.md](operations/job-backends.md)** — configuring a
  backend so submitted jobs actually run.
- **[operations/configuration.md](operations/configuration.md)** — the full
  settings reference, including pointing at PostgreSQL or MySQL.
