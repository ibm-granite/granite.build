# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoTuneX is an IBM Research platform for automated fine-tuning and hyperparameter optimization of LLMs. It consists of three independently running services plus a shared MySQL database:

| Service | Directory | Port | Stack | Entry Point |
|---------|-----------|------|-------|-------------|
| API Server | `api/` | 8000 | FastAPI + FastMCP + LangChain agent | `python server.py` |
| API Bridge | `api-bridge/` | 8001 | FastAPI (logging microservice) | `python server.py` |
| Frontend | `ux/` | 5173 (dev) / 3400 (preview) | SvelteKit + Carbon Design System | `npm run dev` |

The API server and API bridge share the same MySQL database but run as separate processes.

## Development Commands

### API Server (run from `api/`)
```bash
pip install -e .                    # Install with editable mode
python server.py                    # Start server (port 8000)
# Set DEV_MODE=True in .env for uvicorn auto-reload
# Set ENABLE_MCP=true to mount the MCP server at /mcp (SSE)
```

### API Bridge (run from `api-bridge/`)
```bash
pip install .                       # Install dependencies
python server.py                    # Start server (port 8001)
```

### Frontend (run from `ux/`)
```bash
npm install                         # Install dependencies
npm run dev                         # Dev server at http://localhost:5173/autotune
npm run build                       # Production build to ux/build/
npm run preview                     # Preview production build (port 3400)
npm run check                       # TypeScript/Svelte type checking
npm run lint                        # Prettier + ESLint
npm run format                      # Prettier auto-format
```

### Database
```bash
mysql -u <user> -p <db_name> < api/data_models/autotune_latest.sql   # Initialize schema
# Migrations live in api/data_models/migrations/
```

### Docker
```bash
# API: from api/
docker build -f Dockerfile -t autotune-api . && docker run -p 8000:8000 --env-file .env autotune-api

# Bridge: from api-bridge/
docker build -f Dockerfile -t autotune-bridge . && docker run -p 8001:8001 --env-file .env autotune-bridge

# Frontend: from ux/
docker build -f Dockerfile -t autotune-ux . && docker run -p 3000:3000 autotune-ux
```

The API directory also ships additional runtime Dockerfiles: `Dockerfile.runtime` (Ray workers for SFT), `Dockerfile.rl_runtime` (Ray workers with `verl` for RL), and `Dockerfile.gb` (Granite Build runtime).

## Architecture

### API Server (`api/`)
- **server.py**: Monolithic FastAPI app (~2270 lines). Contains all route handlers for jobs, configs, datasets, DMF, auth, users, and the `/api/chat` endpoint. Mounts the MCP server at `/mcp` when `ENABLE_MCP=true`. Startup runs DB test, job monitoring thread, orphan cleanup, and GB login.
- **routes/**: Supplementary routers (`gb_routes.py` for Granite Build, `health_checker.py`, `utils_routes.py`) mounted under the `/fmtune` prefix.
- **services/**: Business logic layer. Key services: `job_service`, `config_service`, `dataset_service`, `dmf_service`, `user_service`, `gb_service`, `db_service`, `yaml_service`, `logging_service`, `file_service`, `chat_service`.
  - `services/impl/runner.py` + `services/runners/` — Job execution layer with `local_runner.py` (Ray Tune + `verl` for RL) and `gb_runner.py` (Granite Build remote execution).
  - **Dataset storage is pluggable.** `services/storage/` defines a `StorageBackend`
    ABC (`persist`/`preview`/`delete`) selected by `get_storage_backend(is_gb_enabled())`;
    built-ins are `LocalStorageBackend` (disk under `UPLOAD_DIR`) and `GBStorageBackend`
    (`gb artifact push`, imported lazily). Add a backend by implementing the ABC + a
    factory branch — see `services/storage/README.md`. The locator reuses the
    `artifact_id`/`artifact_url` columns (no new schema).
  - `dataset_service.py` and `file_service.py` are backward-compatible **shims**; the real
    code lives in `services/datasets/` (`service.py` = upload/view/delete, `intelligence.py`
    = LLM parse/mapping) and `services/file/` (`validation`/`parsing`/`streaming`/`reads`).
  - **Resumable uploads (Phase 2, tus).** Browser dataset uploads go through an embedded
    `tuspyserver` router mounted at `/fmtune/api/datasets/tus`. The completion hook
    (`services/datasets/tus_app.py`) decodes `Upload-Metadata`, coordinates multi-file uploads
    via a per-`dataset_id` rendezvous (`tus_rendezvous.py`), and the LAST file of an upload
    group to complete runs the unchanged `_finalize_upload` seam (`tus_finalize.py`) — reusing
    the same `_DiskBackedUpload` → `upload_*` path as chunked uploads (tus replaces transport,
    not the pipeline). Metadata decode/validation lives in `tus_metadata.py`. tus staging is
    `<UPLOAD_DIR>/.tus/`; retention via `UPLOAD_STAGING_TTL_MINUTES` (default 360 min, ceil'd to
    `days_to_keep`). The legacy `PUT /api/datasets` route is retained as a compatibility shim
    (verified: MCP tools + API Bridge `dataset_service` touch DB metadata only, never upload
    file bytes — nothing is orphaned). Client transport is `tus-js-client` in
    `ux/src/lib/api.ts` (`uploadDatasetChunked`, signature unchanged); after upload the wizard
    re-fetches the dataset list (finalization is server-side, so the helper returns no record).
    The rendezvous is in-process state (single-process assumption, same as the chunked
    `_finalizing_uploads` guard) — a DB-backed completion record is the documented fix if the
    API is ever run multi-replica (deliberately out of scope: no new DB columns).
- **mcp_server.py**: FastMCP server exposing the platform as MCP tools (list/get/create/delete for jobs, configs, datasets, DMF models, users). Mounted at `/mcp` with SSE transport.
- **dependencies.py**: FastAPI dependency injection — all service instantiation via `Depends()` goes through here.
- **models.py**: Pydantic models for the entire API.
- **utils.py**: Shared helper functions.
- **auth.py**: OIDC / OAuth flow with IBM w3id (`/login`, `/callback`, token introspection).
- **config/**: YAML configuration templates for tuning jobs (`autotune.yaml`, `autotune_new.yaml`, `autotune_debug.yaml`, `autotune_debug_rl.yaml`, `rl_config.json`).
- **utilites/**: Converter utilities and Granite Build helpers (note: directory is intentionally misspelled as `utilites`).

All API routes are prefixed with `/fmtune/api`. Swagger docs at `/fmtune/try`, ReDoc at `/fmtune/docs`.

### AI Layer
- **`api/services/chat_service.py`** — LangChain ReAct agent (via `langgraph`) on Claude Sonnet 4.5, routed through a LiteLLM proxy (AWS Bedrock). Auto-discovers tools from the built-in MCP server using `MultiServerMCPClient` (SSE transport). Tool results over ~50K chars are truncated.
- **`/fmtune/api/chat`** — endpoint that drives the agent; consumed by the frontend's `ChatBox.svelte` (built on `@carbon/ai-chat`).
- **MCP tools** — defined in `mcp_server.py`, each tool creates its own short-lived service bundle via `_get_services()` (mirrors `dependencies.py` without FastAPI `Depends()`).
- **Environment**: `ENABLE_MCP`, `MCP_SERVER_URL`, `LITELLM_URL`, `LITELLM_API_KEY`, `LITELLM_MODEL`.

### Execution Layer
- **Local Runner** (`api/services/runners/local_runner.py`) — Ray Tune-based distributed HPO. Supports SFT (via the `autotune` library's `AutotunePipeline`) and RL (via `verl` 0.7.0). For RL jobs, the user's `reward_function.py` is written to the output directory and `NaiveRewardManager` invokes `compute_score()` against trajectories. **Local runs write logs/trials directly to MySQL in-process** via `logging_service` + `db_service` — they do NOT go through the API Bridge.
- **GB Runner** (`api/services/runners/gb_runner.py`) — Generates a Granite Build YAML spec and delegates build submission, polling, and log streaming to `gb_service.py`, which uses the `gbcli` Python API directly (no subprocess). The runner passes `AUTOTUNE_SERVER_BRIDGE_URL` into the remote build so Granite Build jobs can report back through the API Bridge.
- **Runner ABC** (`services/impl/runner.py`) — Abstract base class; `job_service` dispatches to the appropriate runner.

### API Bridge (`api-bridge/`)
Lightweight logging/trial-tracking microservice. **Only remote Granite Build jobs use the bridge** — the local runner writes directly to MySQL through `logging_service` + `db_service` in-process.

Core endpoints:
- `POST /fmtune/api/record_logs` — Record training log entries
- `POST /fmtune/api/record_trial` — Insert trial data
- `POST /fmtune/api/update_status` — Update job/trial status
- `POST /fmtune/api/insert_trial_result` — Store trial results

The bridge also hosts a **sub-service layer** (`api-bridge/services/`: `config_service`, `dataset_service`, `github_service`, `job_service`, `user_service`) that remote Granite Build jobs use to push configs, datasets, and job-record updates back without going through the main API. These are exposed as additional endpoints under `/fmtune/api/*` on port 8001.

**Observability:** `logging_config.py` configures the root logger once at startup (`setup_logging()` is called at the top of `server.py`) — controlled by `LOG_LEVEL` and `LOG_FORMAT` (`text`/`json`). `middleware.py` adds `RequestLoggingMiddleware`, which logs one line per request (method, path, status, latency, caller email) and attaches an `X-Request-ID` response header; unhandled route exceptions are logged with that context before propagating. All logs go to stdout, which supervisord forwards to container logs. Use module loggers via `logging.getLogger(__name__)` — do NOT add per-module `setLevel` or `basicConfig` calls, and do NOT use `print()`.

### Frontend (`ux/`)
- SvelteKit with `adapter-static` (SPA mode, fallback to `index.html`).
- Base path: `/autotune` (configured in `svelte.config.js`).
- **src/lib/api.ts**: API client class — all backend calls go through here using `PUBLIC_AUTOTUNEX_API_URL`.
- **src/lib/store.ts**: Svelte writable stores for global state (tunings, datasets, configs, published models, logs, notifications).
- **src/lib/components/**: UI components organized as:
  - `views/` — Page-level views (Start, Tunings, Settings, Resources, Users)
  - `forms/` — Create/edit forms + wizards (`StartTuningWizard`, `CreateConfigForm`, `IntelligentDatasetWizard`, `CreateTuningForm`, and `steps/` for the wizard sub-steps)
  - `displays/` — Detail views for configs, datasets, tunings, DMF models
  - `tables/` — Data tables for listing entities (Trials, Datasets, Configurations, DmfTable, Tasks)
  - `tabs/` — Tab sub-components
  - `ChatBox.svelte` — AI assistant widget built on `@carbon/ai-chat`, posts to `/api/chat`.
- **vite.config.ts**: Proxy config for development — `/local` proxies to `localhost:8000`, `/stage` and `/prod` proxy to deployed IBM servers.

### Key Environment Variables
- **API** (`.env` in `api/`):
  - DB: `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`
  - Datasets/paths: `AUTOTUNE_DATASETS_PATH`, `AUTOTUNE_RESULTS_PATH`, `LOG_PATH`, `AUTOTUNEX_MODELS_DIR` — all default to subfolders under `AUTOTUNEX_DATA_DIR` (default `<cwd>/AUTOTUNEX_DATA`)
  - OIDC: `OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET`, `OIDC_SECURITY_ENDPOINT`
  - Granite Build: `GB_TOKEN`
  - DMF: `LAKEHOUSE_TOKEN`, `DMF_CACHE`
  - MCP + Chat: `ENABLE_MCP`, `MCP_SERVER_URL`, `LITELLM_URL`, `LITELLM_API_KEY`, `LITELLM_MODEL`
  - Development: `DEV_MODE`, `LOG_LEVEL`
- **Frontend** (`.env` in `ux/`): `PUBLIC_AUTOTUNEX_API_URL` (use `/local/fmtune/api` for local dev); `PUBLIC_DMF_UI_URL`, `PUBLIC_RITS_UI_URL` (optional "Open in…" links). Authentication is server-side (see below) — the frontend has no bypass variable.
- **Bridge** (`.env` in `api-bridge/`): Same DB connection vars as API server. The bridge historically used `DB_SCHEMA` for the database name; it now also accepts `DB_NAME` (falls back `DB_SCHEMA` → `DB_NAME`), so both services can share a single deployment secret. The bridge requires TLS — it always connects with `ssl_verify_identity=True`, and `DB_KEY` (optional) supplies a client key path. Logging: `LOG_LEVEL` (DEBUG/INFO/WARNING/ERROR, default INFO) and `LOG_FORMAT` (`text` default, or `json` for log aggregators).
- **GB Runner** (injected into remote builds): `AUTOTUNE_SERVER_BRIDGE_URL` — points remote jobs back at the API Bridge.

### Authentication
OAuth/OIDC flow with IBM w3id. Authentication is resolved server-side: when `OIDC_CLIENT_ID`/`OIDC_CLIENT_SECRET`/`OIDC_SECURITY_ENDPOINT` are all unset (or `AUTOTUNEX_AUTH=dev`), the API falls back to a dev provider that authenticates every request as an admin — for local development only. The API validates tokens via cookies (`email`, `token`, `role`).

### Database Schema
8 tables + 1 view defined in `api/data_models/autotune_latest.sql`:
- `users`, `configurations`, `datasets`, `jobs`, `trials`, `log_entries`, `results`, `gb_tasks`
- View: `autotunex_jobs`

## Conventions
- Python 3.10+. No formal test framework is set up for the API — test manually via Swagger at `/fmtune/try`.
- Frontend uses Carbon Design System components (`carbon-components-svelte`). Follow existing Carbon patterns for new UI work.
- Database changes: add migration SQL files to `api/data_models/migrations/`, update `autotune_latest.sql` to match.
- The `autotune` core training library is vendored in-tree at `fm-tune/` (package `autotune`) and installed from there in the Docker build (`pip install ./fm-tune[core]`).
- When adding new MCP tools in `mcp_server.py`, create short-lived service bundles via `_get_services()` — do NOT reach for FastAPI `Depends()` (the MCP tools run outside the request lifecycle).
- Keep `chat_service.py` provider-agnostic where reasonable — it speaks to LiteLLM as an OpenAI-compatible proxy, which happens to front Claude.

## Diagrams
- `autotunex-architecture.html` — static system overview (single SVG).
- `autotunex-mindmap.html` — interactive drag-enabled component mindmap with layer toggles, search, comments, and generated prompt.

Keep both in sync when architecture changes.

## Resources
- **Carbon Components Svelte docs**: https://svelte.carbondesignsystem.com — Use this to check component APIs, props, events, and examples for `carbon-components-svelte`.
- **`@carbon/ai-chat`**: used by `ChatBox.svelte` for the assistant widget.
- **FastMCP**: https://github.com/jlowin/fastmcp — MCP server library used in `api/mcp_server.py`.
- **langgraph** / **langchain-mcp-adapters**: used in `chat_service.py` for the ReAct agent + MCP tool discovery.
