# AGENTS.md

This file provides guidance to coding agents (Claude Code and others) when working with code in this repository.

## Git Commits

Always create commits with the `-s` (sign-off) flag — e.g. `git commit -s -m "..."`. The upstream repo (`ibm-granite/granite.build`) enforces the [DCO check](https://developercertificate.org/), which requires a `Signed-off-by:` trailer on every commit. Commits without it will fail CI on the PR.

## GitHub PRs and Issues

When writing PR or issue descriptions, comments, and reviews, do **not** use bare `#`-number notation (`#1`, `#2`, …) to enumerate list items or steps. GitHub auto-links `#N` to the issue or PR with that number, so a list numbered this way turns into a set of misleading cross-references. Use plain numbers (`1.`, `2.`), letters (`(a)`, `(b)`), or descriptive names instead. Reserve `#N` for genuine references to a specific issue or PR (e.g. "builds on #257").

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
`pytest -s test` runs the suite (requires `GBTEST_SPS_IBMCLOUD_API_KEY` for secret retrieval); narrow with the usual `test/dir`, `file.py`, or `file.py::Class::method` args. The non-obvious make targets:
```shell
make cicd-pr-test     # abbreviated CI set (coverage + parallel)
make cicd-merge-test  # extended CI set (the `extended` marker)
# `-setup` targets provision the venv and infra first:
make quick-tests-setup quick-tests        # fast: GBTEST_MODE=mock, -m "not ibm and not extended"
make extended-tests-setup extended-tests  # full: GBTEST_MODE=live, -m "not ibm"; setup also brings up MinIO + SLURM
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

Most directory names are self-describing; these carry non-obvious behavior worth knowing up front:

- **cli.py** — Click CLI root. Discovers subcommands from `commands/command_*.py` (`command_build_watch.py` → `build-watch`).
- **build/** — Core execution engine. Hierarchy: a `Build` contains `Target`s, each has `Step`s, each `Step` produces a `TargetStepRun`.
- **buildwatcher/** — Watches for pending builds (PRs or local dirs) and dispatches runners as k8s jobs, processes, or threads (`GBSERVER_DEFAULT_BUILDRUNNER_TYPE`).
- **storage/** — Persistence layer; `storage_factory.py` selects the backend (`sql/` PostgreSQL, `sqlite/` local) from `GBSERVER_METADATA_STORAGE`. `singleton_storage.py` is the global access point.
- **types/** — Pydantic models. `constants.py` is the central `GBSERVER_*` env-var registry; `gbserverenvconfig.py` handles DEV/STAGING/PROD config.
- **builtins/** — Built-in step implementations (gbstep, hfpull, lhpull, lhpush, cosrclone).
- **api/** (FastAPI, routes under `/api/v1`), **spacesecretmanager/**, **github/**, **messaging/**, **resilience/**, **metrics/**, **monitoring/**, **environment/** (k8s, LSF) — self-explanatory by name.

### Test Layout (`test/`)

- **conftest.py** — Session fixture fetching test secrets from IBM Cloud Secret Manager via `GBTEST_SPS_IBMCLOUD_API_KEY`; also dumps build state on pytest failure.
- **gbserver_test/** — Mirrors source structure. `secret_manager`-marked tests need real IBM Cloud and are excluded by default.
- Parallelism via `pytest-xdist` `--dist=loadgroup`.

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

`gb_ui_backend` ships in the `standalone` extra (`pip install -e ".[standalone]"`). When installed, gbserver mounts its routers at `/api/analytics/*` in its own process — no separate port. The analytics DB (`GB_UI_DATABASE_URL`, see table below) is auto-derived from `GBSERVER_METADATA_STORAGE` when unset.

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
| `GBSERVER_API_URL` | `frontend/.env.local` or build env | API base URL. Dev: rewrite-proxy target. Standalone: baked into the bundle at `make build-frontend` time. Unset = same-origin default. |
| `GBSERVER_UI_DIR` | gbserver env | Override path to compiled frontend (default: `src/gbserver/static/ui/`) |
| `GB_UI_DATABASE_URL` | gbserver env | Analytics DB. Auto-derived from `GBSERVER_METADATA_STORAGE` when unset (`derive_analytics_database_url()` in `constants.py`): `sql` inherits `GBSERVER_SQL_*` as a `postgresql+asyncpg://` URL (with TLS cert if any), connecting to the same Postgres; `sqlite` uses its own `dashboard-analytics.db` under `GB_HOME_DIR`. |
| `GB_UI_DATABASE_CONNECT_ARGS` | gbserver env (internal) | JSON `create_async_engine()` connect args, set by gbserver when the SQL store needs TLS (carries the cert path — `ssl.SSLContext` isn't JSON-serializable). Not hand-set. |
| `GB_UI_GBSERVER_DB_URL` | gbserver env | gbserver's own DB for richer analytics. Auto-set to gbserver's SQLite file when unset and storage is sqlite. |
| `GB_UI_GBSERVER_URL` | analytics env | gbserver URL for the dev-mode startup banner (default: `http://localhost:8080`) |
| `GB_UI_LLM_BASE_URL` / `GB_UI_LLM_API_KEY` | analytics env | OpenAI-compatible endpoint + key for AI analysis |

## Deployment

- Container images built on UBI 9 + Python 3.12
- Three environments: dev, staging, prod — each with its own IBM Container Registry namespace (`us.icr.io/cil15-shared-registry/gb-{dev,staging,prod}`)
- Kubernetes deployments managed via Helm charts in `k8s/chart/`
- CI via Travis CI on `dev` and `main` branches
- Image tags derived from git commit SHA (`commit-<hash>`)