# AutoTuneX

**Automated fine-tuning and hyperparameter optimization for large language models.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

AutoTuneX is a research platform from IBM Research for automated fine-tuning and
hyperparameter optimization (HPO) of large language models. A **job** searches a
hyperparameter space — drawn from a reusable **configuration** — by running one training
**trial** per candidate, and reports which one won.

Jobs can be submitted through the API, and are also created by the tuning pipeline writing
to the real MySQL database directly:

```
POST /api/v1/jobs        →  submit a tuning run (references an existing config + dataset)
GET  /api/v1/jobs        →  a lean page of jobs — identity, status, owner/config/dataset labels
GET  /api/v1/jobs/{id}   →  one job, full detail: tasks, trials, per-trial metrics, JSON blobs
```

> ### Project status: submission + read path implemented; execution is backend-gated
>
> `GET /jobs`, `GET /jobs/{id}`, and `POST /jobs` are real and tested against the production
> MySQL schema (`resources/autotunex_schema.sql`). `POST /jobs` submits a run referencing an
> existing configuration and dataset (see the [Jobs API](docs/api/jobs.md)). **Job execution
> is implemented** for the `local` and `llmb` backends — `local` runs the `autotune` HPO
> in-process via Ray and drives the job to `completed`/`error`, and `llmb` submits a
> granite.build build (see [Job backends](#job-backends)). Only the default `none` backend
> accepts a job and leaves it `pending`; the seam where execution attaches is
> [services/runner.py](src/autotunex/services/runner.py).

## Quickstart

Requires Python 3.11+. [uv](https://docs.astral.sh/uv/) is recommended but not required.

```bash
git clone https://github.com/ibm-granite/granite.build.git
cd granite.build/autotunex

uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install -e "./src/fm-tune"   # the slim, torch-free autotune catalog

cp .env.example .env          # defaults work as-is for local development
uvicorn autotunex.main:app --reload
```

`make install` runs both installs for you. The second one supplies the slim, torch-free
`autotune` catalog that the configuration-template and dataset-format endpoints need —
without it they return `503`.

The server starts on <http://127.0.0.1:8000> and creates a local SQLite database
(`autotunex.db`) on first run. No database server needed.

The root `.env.example` is the full reference and runs the no-op backend as-is.
To configure a specific job runner, copy the matching focused example from
`envs/` instead — `envs/local.env.example` (in-process HPO), `envs/bash.env.example`
(granite.build on a laptop/Mac), `envs/lsf.env.example` (the LSF/SkyPilot variant,
selected by `AUTOTUNEX_LSF_CLUSTER`), or `envs/remote.env.example` (granite.build
cluster). Each carries only the variables that runner needs.

Then open the auto-generated interactive docs:

| URL | What it is |
| --- | --- |
| <http://127.0.0.1:8000/docs> | Swagger UI — try every endpoint from the browser |
| <http://127.0.0.1:8000/redoc> | ReDoc reference |
| <http://127.0.0.1:8000/openapi.json> | Raw OpenAPI schema |

For a full round trip — create a configuration, register and upload a dataset, submit a job,
and read it back with `curl` — follow the
**[getting-started guide](docs/getting-started.md)**.

## Run with Docker

A single, self-sufficient container runs the API **and** the web UI with no
external services. It defaults to standalone mode (no auth), an embedded SQLite
database, and the `local` job backend — all persisted under a `/data` volume.
Every default is overridable at run time with `-e AUTOTUNEX_...`.

```bash
# Build
docker build -t autotunex:local .

# Run (data persists in the named volume)
docker run --rm -p 8000:8000 -v autotunex-data:/data autotunex:local
```

Then open:

- **Web UI** — http://localhost:8000/autotune
- **Health** — http://localhost:8000/health
- **API** — http://localhost:8000/api/v1/...

Or with Compose:

```bash
docker compose up --build
```

> **Podman works too.** Every command above runs unchanged under Podman — it is a
> drop-in replacement (`podman build`, `podman run`, `podman compose up`).

**Overriding defaults** — for example, to attribute writes to a named owner:

```bash
docker run --rm -p 8000:8000 -v autotunex-data:/data \
  -e AUTOTUNEX_STANDALONE_EMAIL=you@example.com \
  autotunex:local
```

To point at an external database instead of the embedded SQLite file, set
`-e AUTOTUNEX_DATABASE_URL=...` (the `postgres`/`mysql` drivers are **not** in
this image — installing those extras is a separate build).

> **Training execution.** This image installs the lean `fm-tune[core]` training
> stack — `torch` plus `ray[tune,default]`, transformers/trl/peft — from the
> in-tree `src/fm-tune/`, and defaults `AUTOTUNEX_JOB_BACKEND=local`, so tuning
> runs in-container on CPU. Only the GPU / online-RL extras (`[full]`:
> verl/vLLM, flash-attn, deepspeed) and `[mlx]` (macOS-arm64 only) are left out.
> Locally, `make install` installs just the slim, torch-free catalog, so wizard
> endpoints work out of the box, and those heavy extras stay a separate, opt-in
> install — `make install-training` (`pip install -e "./src/fm-tune[full,mlx]"`).
> Job submission, all CRUD, dataset upload/preview, auth, and the full UI work
> as well. See
> [job backends](docs/operations/job-backends.md) for the `local` backend's
> requirements and [production deployment](docs/operations/deployment.md#container-image)
> for the image's baked defaults.

## Core concepts

| Term | Meaning |
| --- | --- |
| **Configuration** | A named, reusable set of tuning settings, stored as a schema-less JSON blob (`config_data`) and validated only as a non-empty object — the tuning pipeline writes a rich, evolving structure the API does not pin to a fixed schema. |
| **Job** | One optimization run — a model, a dataset reference, a configuration reference, and experiment metadata. Created via `POST /jobs` (owned by the caller), or by the tuning pipeline writing to the database directly. |
| **Trial** | One training run inside a job, evaluating a single concrete point from the job's configuration. |
| **Result** | The metrics a trial's training run reported, one-to-one with the trial. |

For the full domain model — how jobs, configurations, datasets, trials, results, users, and
tasks relate, plus the job lifecycle — see [concepts](docs/concepts.md).

## API surface

Resource endpoints are mounted under `/api/v1`; `/auth/*`, `/health` and `/mcp` are
unprefixed. Reads are own-data by default; an admin widens to every owner's rows per request
with `?scope=all` (a non-admin who asks gets a `403`). Every endpoint is runnable from the
browser at [Swagger UI](http://127.0.0.1:8000/docs), and the full reference lives in
[`docs/api/`](docs/api/).

| Resource | Endpoints | Reference |
| --- | --- | --- |
| **Jobs** | `POST`/`GET` `/jobs`, `GET`/`DELETE` `/jobs/{id}`, `POST /jobs/{id}/cancel`, `POST /jobs/{id}/reconcile`, `POST /jobs/estimate-usages`, `POST /jobs/generate-test-solutions`, `GET /jobs/by-build-id/{build_id}`, result-report (list/file/archive), job/trial/gb logs | [jobs.md](docs/api/jobs.md) |
| **Reward functions** | `POST /reward-functions/validate` (validate an online-RL reward function, sandboxed) | [reward-functions.md](docs/api/reward-functions.md) |
| **Configurations** | full CRUD `/configurations` (+ `GET /configurations/template`) | [configurations.md](docs/api/configurations.md) |
| **Datasets** | full CRUD `/datasets`, `POST /datasets/{id}/upload`, `?preview=true` | [datasets.md](docs/api/datasets.md) |
| **Dataset intelligence** | `POST /datasets/intelligence/{parse-strategy,suggest-mapping,validate-strategy}`, `GET .../formats` | [datasets.md](docs/api/datasets.md) |
| **Users** | `GET /users`, `GET /users/{id}`, `PATCH /users/{id}` (all admin-only); `GET /users/me/metadata` (open to any authenticated caller) | [users.md](docs/api/users.md) |
| **Chat & MCP** | `POST /chat`, `POST /chat/stream`, `/mcp` (unprefixed) | [chat.md](docs/api/chat.md), [mcp.md](docs/api/mcp.md) |
| **Auth** | `GET /auth/{login,callback,me}`, `POST /auth/logout`, `POST /auth/assume/{user_id}` (admin), `POST /auth/unassume` | [authentication.md](docs/api/authentication.md) |
| **Health** | `GET /health`, `GET /health/live` (liveness alias), `GET /health/ready` (DB-gated readiness; `503` when the database is unreachable) | [overview.md](docs/api/overview.md) |
| **App config** | `GET /app-config` (unauthenticated; the upload cap and client gzip/preview knobs the web UI reads at boot) | [overview.md](docs/api/overview.md) |

For the conventions shared by every endpoint — pagination, ownership scoping, and the RFC 9457
error shape — see the [API overview](docs/api/overview.md).

## Job backends

An accepted job is handed to a `JobRunner` off the request path; `POST /jobs` never blocks on
training. Which runner is built is chosen by `AUTOTUNEX_JOB_BACKEND`:

| `AUTOTUNEX_JOB_BACKEND` | What it does |
| --- | --- |
| `none` (default) | Accepts the job and does nothing — it stays `pending`. |
| `local` | Runs the `autotune` HPO pipeline in-process via Ray Tune, driving the job to `completed`/`error` itself. |
| `llmb` | Submits a granite.build build via the `llmb` CLI; a reconcile loop then polls for status. |

There is no external broker yet — every runner ships in-process. See
[job backends](docs/operations/job-backends.md) for each backend's requirements and settings,
and the `custom_code`/`bash`/LSF-SkyPilot spec variants of the `llmb` runner (the last
selected by `AUTOTUNEX_LSF_CLUSTER`).

## Chat assistant and MCP server

AutoTuneX ships a natural-language **assistant** that can query and act on your jobs,
configurations, datasets, and account, and — optionally — a standalone **MCP server** exposing
the same operations to external MCP clients (Claude Desktop, Cursor). Both run in-process as the
authenticated caller, so a tool only ever sees your own data. They reuse the same
OpenAI-compatible LLM gateway as dataset intelligence (`AUTOTUNEX_LLM_BASE_URL`,
`AUTOTUNEX_LLM_API_KEY`, `AUTOTUNEX_LLM_MODEL`) and return `503` when it is unset.

The MCP server is off by default and lives behind an optional dependency:

```
pip install -e ".[mcp]"       # installs fastmcp
AUTOTUNEX_ENABLE_MCP=true      # mount it at /mcp, authenticated via X-API-Key
```

See [chat](docs/api/chat.md) and [MCP](docs/api/mcp.md) for the full contract.

## Authentication

Every request resolves to a `Principal` (an email, a provider, and whether the caller is an
admin); what a caller can see and do follows from that. Which credential kinds the service
accepts is set by `AUTOTUNEX_AUTH_PROVIDERS` (a JSON list): `"disabled"` cannot combine with
any other provider, while `"api_key"`, `"oidc"`, and `"session"` may combine freely.

| Mode | Provider value | For | Credential |
| --- | --- | --- | --- |
| Standalone (default) | `["disabled"]` | Local dev / single-user | none — every caller is one principal |
| API key | `["api_key"]` | Machine callers (CI, monitors) | `X-API-Key: <raw-key>` |
| OIDC bearer token | `["oidc"]` | CLI / service callers | `Authorization: Bearer <token>` |
| Browser session (BFF) | `["session"]` | Browser UIs | httpOnly session cookie (server-set) |

By default the API runs in **standalone mode**: every caller is the same unrestricted principal
and writes are attributed to a lazily-provisioned default owner (`standalone@autotunex.local`,
or `AUTOTUNEX_STANDALONE_EMAIL` if set). This is meant for local development and single-user
deployments — **production must configure a real provider**; startup refuses to run in
`AUTOTUNEX_ENVIRONMENT=prod` with auth disabled unless `AUTOTUNEX_ALLOW_INSECURE_NO_AUTH=true`
is also set (which logs a loud warning at startup).

See **[authentication](docs/api/authentication.md)** for configuring each provider — key
minting, OIDC and audience handling, the backend-for-frontend flow, and just-in-time
provisioning — and **[authentication testing](docs/authentication-testing.md)** for a runbook
to exercise each provider against a running server.

## Development

```bash
make install     # install the package with dev dependencies
make dev         # run the API with autoreload
make test        # pytest
make lint        # ruff check + format check
make format      # apply ruff formatting and autofixes
make typecheck   # mypy (strict)
make check       # lint + typecheck + test — the three CI jobs you can run locally
                 # (CI also runs the migrations matrix, DCO, OSS-compliance and gitleaks)
make migrate     # alembic upgrade head — see the warning below before using this on a real database
```

Those are the common ones. `make help` (the default goal, so a bare `make` works too)
lists every target, and CLAUDE.md documents what each one is for — including
the training-install, coverage, audit, api-bridge and migration-authoring targets.

Migrations are verified against SQLite, PostgreSQL 16 and MySQL 8.4 in CI.

## Deploying against an existing database

> **Do not run `make migrate` against a database that already has the AutoTuneX
> schema.** The baseline revision would try to create tables that exist and fail.

For an established deployment, *stamp* the baseline revision as already-applied and then upgrade
only the later revisions, rather than building the schema from scratch. The full procedure — the
exact `alembic stamp` / `alembic upgrade` commands, and the one post-baseline revision that
migrates data — is in [database & migrations](docs/operations/database.md). For a fresh, empty
database (local development, tests, CI) none of this applies: `alembic upgrade head` builds the
whole schema normally.

## Documentation

Full documentation lives in **[`docs/`](docs/README.md)** — [concepts](docs/concepts.md), a
[getting-started walkthrough](docs/getting-started.md), the [API reference](docs/api/overview.md),
and operations guides for [configuration](docs/operations/configuration.md),
[job backends](docs/operations/job-backends.md), [database](docs/operations/database.md), and
[deployment](docs/operations/deployment.md).

## Contributing

Contributions are welcome from anyone — this project accepts external pull requests.
Start with **[CONTRIBUTING.md](../CONTRIBUTING.md)**, which covers filing issues, the PR
workflow, commit conventions, and the required **DCO `Signed-off-by`** line on every commit.

- [CONTRIBUTING.md](../CONTRIBUTING.md) — how to contribute
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) — Contributor Covenant v2.1
- [SECURITY.md](../SECURITY.md) — how to report a vulnerability privately
- CLAUDE.md — architecture, conventions, and domain model (written for AI coding agents, useful for humans too)

## License

Apache License 2.0 — see [LICENSE](../LICENSE) and [NOTICE](../NOTICE).
