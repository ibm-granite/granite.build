# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
make cicd-merge-test  # extended test set (GBTEST_ENABLE_EXTENDED_TESTS=true)
```

### Formatting and Linting
```shell
# Format and lint only files changed vs dev branch (use before PRs)
make xformat    # runs isort + black on changed .py files
make xcheck     # runs pylint + mypy, filters output to changed files

# Format/lint entire codebase
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
gbserver pr-watch --gh-token $TOKEN --config <config.yaml>
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

## Deployment

- Container images built on UBI 9 + Python 3.12
- Three environments: dev, staging, prod — each with its own IBM Container Registry namespace (`us.icr.io/cil15-shared-registry/gb-{dev,staging,prod}`)
- Kubernetes deployments managed via Helm charts in `k8s/chart/`
- CI via Travis CI on `dev` and `main` branches
- Image tags derived from git commit SHA (`commit-<hash>`)
