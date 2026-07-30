# AGENTS.md

This file provides guidance to coding agents (Claude Code and others) when working with code in this repository.

## Git Commits

Always create commits with the `-s` (sign-off) flag — e.g. `git commit -s -m "..."`. The upstream repo (`ibm-granite/granite.build`) enforces the [DCO check](https://developercertificate.org/), which requires a `Signed-off-by:` trailer on every commit. Commits without it will fail CI on the PR.

## Project Overview

gbserver is the build orchestration server for LLM.Build (Granite.Build). It manages model build pipelines — watching PRs and repos for build configurations, executing multi-step builds on Kubernetes/LSF clusters, and exposing a REST API for build management. Written in Python 3.11+, it uses Click for CLI, FastAPI for the REST API, and SQLAlchemy with PostgreSQL for metadata storage.

## Common Commands

### Virtual Environment Setup
```shell
# Requires ARTIFACTORY_USER and ARTIFACTORY_API_KEY env vars
make venv
source .venv/bin/activate
```

### Running Tests
```shell
# Run all tests (requires GBTEST_SPS_IBMCLOUD_API_KEY for secrets)
pytest -s test

# Run a specific test directory
pytest -s test/gbserver_test/api

# Run a specific test file
pytest -s test/gbserver_test/api/test_artifacts.py

# Run a single test method
pytest -s test/gbserver_test/api/test_artifacts.py::TestArtifactAPI::test_artifact_get

# CI test suites (creates venv, runs with coverage and parallel execution)
make cicd-pr-test     # abbreviated test set
make cicd-merge-test  # extended test set (the `extended` marker)

# Local test suites (the `-setup` target provisions the venv and any infra first)
make quick-tests-setup quick-tests        # fast suite: GBTEST_MODE=mock, -m "not ibm and not extended"
make extended-tests-setup extended-tests  # full suite: GBTEST_MODE=live, -m "not ibm" (includes `extended` tests); setup also brings up MinIO + SLURM
```

### Formatting and Linting
```shell
make format        # isort + black on everything
make staticcheck   # pylint + mypy on src/gbserver/
```

### Docker Images
```shell
# Build container image (requires clean git status)
make image      # native platform
make imagex     # cross-platform (for Mac ARM → linux/x86_64)
# DOCKER defaults to podman; override with DOCKER=docker
```

### CLI Usage
```shell
gbserver --help
gbserver rest-server --help
gbserver build-watch --build-dir <dir>
gbserver build-runner ...
```

## Architecture

### Source Layout (`src/gbserver/`)

- **cli.py** — Click-based CLI root. Dynamically discovers subcommands from `commands/command_*.py` files (filename maps to CLI command: `command_build_watch.py` → `build-watch`).
- **commands/** — CLI subcommand implementations. Each file exports a `cli` Click command.
- **api/** — FastAPI REST API for build management. Routes prefixed with `/api/v1`.
- **build/** — Core build execution engine. Key classes: `Build`, `BuildRun`, `Target`, `TargetRun`, `Step`, `TargetStepRun`. Represents the hierarchy: a Build contains Targets, each Target has Steps, each Step produces a TargetStepRun.
- **buildwatcher/** — Watches for pending builds (from PRs or local directories) and dispatches build runners. Can run builds as k8s jobs, processes, or threads (controlled by `GBSERVER_DEFAULT_BUILDRUNNER_TYPE`).
- **storage/** — Data persistence layer with multiple backends:
  - `sql/` — Primary backend using SQLAlchemy with PostgreSQL
  - `sqlite/` — SQLite backend for local/testing use
  - `singleton_storage.py` — Global storage access point
  - `storage_factory.py` — Backend selection based on `GBSERVER_METADATA_STORAGE` env var
- **types/** — Pydantic models and configuration types. `constants.py` is the central env var registry — almost all `GBSERVER_*` env vars are defined here. `gbserverenvconfig.py` handles per-environment (DEV/STAGING/PROD) configuration.
- **spacesecretmanager/** — Secret management abstraction with IBM Cloud, local, hybrid, and env-based implementations.
- **github/** — GitHub Enterprise API integration for PR operations and repo access.
- **messaging/** — RabbitMQ/AMQP messaging integration (aio-pika).
- **resilience/** — Retry strategies and resilience patterns (uses tenacity).
- **metrics/** — Metrics collection and push to metrics endpoint.
- **monitoring/** — Health checks and sidecar monitoring.
- **environment/** — Compute environment abstractions (Kubernetes, LSF).
- **builtins/** — Built-in step implementations (gbstep, hfpull, lhpull, lhpush, cosrclone).

### Test Layout (`test/`)

- **conftest.py** — Session-level fixture that fetches test secrets from IBM Cloud Secret Manager (SPS) using `GBTEST_SPS_IBMCLOUD_API_KEY`. Also hooks into pytest failure reporting to dump build state for debugging.
- **gbserver_test/** — Mirrors source structure. Tests marked `secret_manager` require real IBM Cloud connections and are excluded by default.
- **sidecar_test/** — Tests for the monitoring sidecar container.
- Test parallelism: uses `pytest-xdist` with `--dist=loadgroup` mode.

### Key Dependencies
- **dmf-lib** (v1.10.2) — Data Model Factory library for Lakehouse integration. Installed from IBM Artifactory.
- **kubernetes_asyncio** — Async Kubernetes client for job management.
- **SQLAlchemy + psycopg2** — PostgreSQL storage backend.
- **aio-pika** — AMQP messaging.

## Environment Variables

The central registry is `src/gbserver/types/constants.py`. All gbserver env vars use the `GBSERVER_` prefix. Key ones for development:

- `GB_ENVIRONMENT` — DEV, STAGING, PROD, or STANDALONE (controls cluster, namespace, Lakehouse config, and standalone-mode defaults)
- `GBSERVER_GITHUB_TOKEN` — GitHub Enterprise access token
- `GBSERVER_DEFAULT_BUILDRUNNER_TYPE` — `job` (k8s), `process`, or `thread` (useful for local dev: set to `thread` to avoid needing a cluster)
- `GBSERVER_METADATA_STORAGE` — Storage backend selection (default: `sql`)
- `GBTEST_SPS_IBMCLOUD_API_KEY` — IBM Cloud API key for test secret retrieval
- `ARTIFACTORY_USER` / `ARTIFACTORY_API_KEY` — Required for `make venv` (dmf-lib installation)

## Code Style

- Formatting: **black** (default config) + **isort** (profile: black)
- Linting: **pylint** (config in `.pylintrc`) + **mypy** (`--disable-error-code=import-untyped`)
- The `xformat`/`xcheck` targets diff against the `dev` branch, not `main`
- Python 3.11+ required (3.12 for pylint target)
- Apache License 2.0

## Frontend (gb-ui)

The `frontend/` directory contains the gb-ui Next.js dashboard and `src/gb_ui_backend/` is its analytics service. Both are part of this repo after the gb-ui migration.

### Frontend commands

```shell
# Compile and sync to src/gbserver/static/ui/ (incremental — reuses .next/ cache)
make build-frontend

# Full clean rebuild (wipes frontend/out/, frontend/.next/, src/gbserver/static/ui/)
make clean-frontend && make build-frontend

# Wipe all build artifacts without rebuilding
make clean-frontend
```

`yarn build` produces a static export and removes `out/404.html` via a postbuild script so the SPA fallback handler works correctly. `build-frontend` does not call `clean-frontend` — run them together for a guaranteed fresh compile.

### Running modes

The frontend has two modes:

**Standalone mode** — gbserver serves the compiled static files and REST API from the same origin. This is the default for end users.

```shell
make build-frontend           # compile once (or after any frontend change)
gbserver standalone           # serves UI + API at http://localhost:8080
```

API calls use relative paths (`/api/v1`, `/api/analytics`) — no extra configuration needed. To point the frontend at a different gbserver, set `GBSERVER_API_URL` at build time:

```shell
GBSERVER_API_URL=http://other-host:8080 make build-frontend
gbserver standalone
```

**Dev mode** — Next.js dev server at `:3000` with hot reload. Useful when iterating on UI changes without rebuilding the static export.

```shell
cd frontend && yarn dev       # UI at http://localhost:3000, no backend required
```

Without a backend, the UI loads but all data pages show empty states. To connect to a running gbserver:

```shell
# frontend/.env.local
GBSERVER_API_URL=http://localhost:8080
```

```shell
cd frontend && yarn dev       # proxies /api/* to gbserver at :8080 (no CORS)
```

`GBSERVER_API_URL` in `.env.local` sets the proxy destination — the browser always uses relative paths, so no CORS configuration is needed on gbserver.

### Running with the analytics service

`gb_ui_backend` is bundled with the `standalone` extra. If installed, gbserver includes its routers directly into its own process at startup — no separate process or port. When `GB_UI_DATABASE_URL` is unset, gbserver derives it from the main store's own backend config (`GBSERVER_METADATA_STORAGE`) instead of an independent default — see `derive_analytics_database_url()` in `src/gbserver/types/constants.py`:

```shell
pip install -e ".[standalone]"
gbserver standalone
# Analytics routes are served at /api/analytics/* on gbserver's own port
# GBSERVER_METADATA_STORAGE=sqlite (standalone default): analytics uses its own
# ~/.granite.build/dashboard-analytics.db (auto-created on first run).
# GBSERVER_METADATA_STORAGE=sql: analytics inherits GBSERVER_SQL_* automatically,
# translated to a postgresql+asyncpg:// URL (including the SQL TLS cert, if any) —
# it connects to the same Postgres as the main store, not a separate database.
```

To override the derived default explicitly:

```shell
GB_UI_DATABASE_URL="postgresql+asyncpg://user:pass@host/db" gbserver standalone
```

### Frontend source layout

| Path | Description |
|------|-------------|
| `frontend/app/` | Next.js App Router pages |
| `frontend/components/` | Shared React components (Carbon Design System) |
| `frontend/api/` | API clients — `gbserver.ts`, `analytics.ts`, `dataProcessing.ts` |
| `frontend/api/client.ts` | `apiBase()` helper — handles `GBSERVER_API_URL` override |
| `frontend/next.config.ts` | Build config — static export in standalone mode, rewrite proxy in dev |
| `frontend/.env.local.example` | Dev environment template — copy to `.env.local` |
| `src/gb_ui_backend/` | Analytics service — FastAPI routers for charts, AI analysis; included directly into gbserver |
| `src/gb_ui_backend/config.py` | Pydantic settings — all `GB_UI_*` env vars |
| `src/gbserver/api/root_api.py` | Includes gb_ui_backend's routers under `/api/analytics/*` and calls its startup init |
| `src/gbserver/static/ui/` | Compiled frontend served by gbserver at runtime |

### Key env vars (frontend / analytics)

| Variable | Where set | Description |
|---|---|---|
| `GBSERVER_API_URL` | `frontend/.env.local` or build env | API base URL. Dev: sets the rewrite proxy target. Standalone: baked into the bundle at `make build-frontend` time. Unset = same-origin default. |
| `GBSERVER_UI_DIR` | gbserver env | Override path to compiled frontend (default: `src/gbserver/static/ui/`) |
| `GB_UI_DATABASE_URL` | gbserver env | Analytics DB. Auto-derived from the main store's own backend (`GBSERVER_METADATA_STORAGE`) when unset — see `derive_analytics_database_url()` in `src/gbserver/types/constants.py`. `sql` mode inherits `GBSERVER_SQL_*` as a `postgresql+asyncpg://` URL; `sqlite` mode uses its own `dashboard-analytics.db` under `GB_HOME_DIR`. |
| `GB_UI_DATABASE_CONNECT_ARGS` | gbserver env (internal) | JSON-encoded `create_async_engine()` connect args, set by gbserver when the main SQL store requires TLS — carries the cert file path across the env-var boundary since an `ssl.SSLContext` isn't JSON-serializable. Not meant to be hand-set. |
| `GB_UI_GBSERVER_DB_URL` | gbserver env | gbserver's own DB for richer analytics. Auto-set to gbserver's SQLite file when unset and storage is sqlite. |
| `GB_UI_GBSERVER_URL` | analytics env | gbserver URL, used for the standalone dev-mode startup banner (default: `http://localhost:8080`) |
| `GB_UI_LLM_BASE_URL` | analytics env | OpenAI-compatible endpoint for AI analysis |
| `GB_UI_LLM_API_KEY` | analytics env | API key for the LLM endpoint |

## Deployment

- Container images built on UBI 9 + Python 3.12
- Three environments: dev, staging, prod — each with its own IBM Container Registry namespace (`us.icr.io/cil15-shared-registry/gb-{dev,staging,prod}`)
- Kubernetes deployments managed via Helm charts in `k8s/chart/`
- CI via Travis CI on `dev` and `main` branches
- Image tags derived from git commit SHA (`commit-<hash>`)