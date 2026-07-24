# AutoTuneX

**Interactive Automated Fine-Tuning Platform for Large Language Models**

AutoTuneX is an IBM Research platform that provides an intuitive UI and a comprehensive REST + MCP API for automated hyperparameter optimization and fine-tuning of Large Language Models. It streamlines the end-to-end workflow — dataset preparation, configuration, distributed HPO, remote execution, and model publication — and includes an AI assistant that can operate the platform via natural language.

<!-- TODO: re-host splash screen on public GitHub (commit the image under
     docs/assets/ or upload it as a github.com user-attachment) and restore
     the <img> here. The previous src pointed at an internal github.ibm.com
     attachment that will not render publicly. -->
<!-- <img width="1709" alt="AutoTuneX splash screen" src="docs/assets/autotunex-splash.png" /> -->


## Features

- **Interactive Web Interface** — SvelteKit SPA built on IBM Carbon Design System with wizard-driven job creation.
- **AI Assistant (ChatBox)** — In-app assistant powered by a LangChain ReAct agent running Claude Sonnet 4.5 (via LiteLLM). The agent auto-discovers tools from the built-in MCP server, so it can list, create, and delete jobs, configs, datasets, and models without hand-written API glue.
- **MCP Server** — FastMCP server mounted at `/mcp` (SSE transport) exposing the platform as first-class MCP tools for external agents.
- **Automated Hyperparameter Tuning** — Multiple search algorithms (Random, LDS, HyperOpt, BayesOpt, BOHB) and schedulers (Hyperband, FIFO) driven by Ray Tune.
- **Multiple Fine-Tuning Methods** — LoRA, ALoRA, LoHa, LoKR, P-Tuning, Prefix Tuning, Prompt Tuning, VERA, and full SFT.
- **Reinforcement Learning Support** — RL fine-tuning via `verl` 0.7.0 with user-supplied `reward_function.py` (GSM8K-style `compute_score` evaluated by `NaiveRewardManager`).
- **Pluggable Runners** — Local execution via Ray Tune or remote execution on Granite Build (via the programmatic `gbcli` API — no subprocess).
- **Dataset Management** — Upload via the Intelligent Dataset Wizard, LLM-assisted parsing strategy generation, format validation.
- **DMF Integration** — Publish models to IBM Data Model Factory (Lakehouse / Iceberg) with versioning and model cards.
- **Flexible Configuration** — YAML-based configuration templates (`autotune.yaml`, debug variants, `rl_config.json`).
- **REST API** — FastAPI backend with ~50 endpoints, Pydantic models, and OAuth/OIDC authentication.
- **Real-time Monitoring** — Training logs and trial metrics streamed back from Ray workers through the API Bridge.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Architecture](#architecture)
- [Installation](#installation)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [API Bridge Setup](#api-bridge-setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [Docker Deployment](#docker-deployment)
  - [Quick Start with Docker Compose](#quick-start-with-docker-compose)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Contact](#contact)
- [License](#license)

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python**: 3.10 or higher (< 4.0)
- **Node.js**: 16.x or higher
- **npm / yarn / pnpm**: Package manager for Node.js
- **MySQL**: Database for storing configurations and job metadata
- **Git**: For cloning the repository
- **CUDA-capable GPU**: Recommended for training (optional for development)

## Architecture

AutoTuneX is composed of three independently deployable services plus a shared MySQL database:

| Service | Directory | Port | Stack |
|---------|-----------|------|-------|
| API Server | `api/` | 8000 | FastAPI + MCP + LangChain agent |
| API Bridge | `api-bridge/` | 8001 | FastAPI (logging microservice) |
| Frontend | `ux/` | 5173 (dev) / 3400 (preview) | SvelteKit + Carbon Design System |

```
┌───────────────┐        ┌────────────────────────────┐        ┌──────────────┐
│   Frontend    │◄──────►│         API Server         │◄──────►│    MySQL     │
│  (SvelteKit)  │  REST  │ FastAPI · /mcp · /api/chat │  SQL   │  (shared DB) │
└───────┬───────┘        └────────────┬───────────────┘        └──────▲───────┘
        │                             │                                │
        │ ChatBox                     │ dispatches                     │
        │ (@carbon/ai-chat)           ▼                                │
        │                  ┌───────────────────┐                       │
        │                  │  Job Runners      │                       │
        │                  │  Local (Ray Tune) │───────────────────────┤ local logs
        │                  │  GB  (gbcli API)  │                       │ direct to DB
        │                  └─────────┬─────────┘                       │
        │                            │ dispatch remote builds          │
        │                            ▼                                 │
        │                  ┌───────────────────┐                       │
        │                  │ Remote GB jobs    │                       │
        │                  │ (AUTOTUNE_SERVER_ │                       │
        │                  │  BRIDGE_URL)      │                       │
        │                  └─────────┬─────────┘                       │
        │                            │ logs / trials / status          │
        │                            ▼                                 │
        │                  ┌───────────────────┐                       │
        │                  │    API Bridge     │───────────────────────┘
        │                  │ FastAPI · :8001   │   record logs & trials
        │                  └───────────────────┘
```

See `autotunex-architecture.html` (static overview) and `autotunex-mindmap.html` (interactive, drag-enabled) for detailed component maps.

### AI Layer

- `api/services/chat_service.py` runs a LangChain ReAct agent (via `langgraph`) on Claude Sonnet 4.5, routed through a LiteLLM proxy (AWS Bedrock).
- `api/mcp_server.py` (FastMCP) is mounted at `/mcp` when `ENABLE_MCP=true`, exposing list/get/create/delete tools for jobs, configs, datasets, DMF models, and users.
- The ChatBox (`ux/src/lib/components/ChatBox.svelte`) uses `@carbon/ai-chat` and posts to `/fmtune/api/chat`.

### Execution Layer

- **Local Runner** (`api/services/runners/local_runner.py`) — Ray Tune-based distributed HPO. Supports SFT and RL jobs. For RL jobs, the user's `reward_function.py` is written to the output directory and `verl` invokes `compute_score()` against trajectories.
- **GB Runner** (`api/services/runners/gb_runner.py`) — Generates a Granite Build YAML spec and delegates build submission, polling, and log streaming to `gb_service.py`, which uses the `gbcli` Python API directly (no subprocess). The runner also passes `AUTOTUNE_SERVER_BRIDGE_URL` into the remote build, so Granite Build jobs report back through the same API Bridge as local runs.
- **API Bridge** — Remote Granite Build jobs POST training logs, trials, status updates, and results to the bridge (port 8001). The local runner does **not** use the bridge — it writes logs and trials directly to MySQL through the in-process `logging_service` + `db_service`. The bridge also hosts a sub-service layer (`config / dataset / github / job / user`) that GB build runners use to push configs, datasets, and job updates back without going through the main API.

## Installation

### Backend Setup

1. **Clone the repository**:

   AutoTuneX lives in the [`granite.build`](https://github.com/ibm-granite/granite.build) monorepo, under the `autotunex/` directory.
   ```bash
   git clone https://github.com/ibm-granite/granite.build.git
   cd granite.build/autotunex
   ```

2. **Navigate to the API directory**:
   ```bash
   cd api
   ```

3. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -e .
   ```

5. **Set up environment variables**:
   Create a `.env` file in the `api/` directory:
   ```env
   # Database
   DB_HOST=localhost
   DB_PORT=3306
   DB_USER=your_db_user
   DB_PASSWORD=your_db_password
   DB_NAME=autotune

   # Datasets
   AUTOTUNE_DATASETS_PATH=path_to_store_datasets

   # Authentication (optional — leave these unset to run without OIDC as a local dev admin)
   OIDC_CLIENT_ID=your_client_id
   OIDC_CLIENT_SECRET=your_client_secret
   OIDC_SECURITY_ENDPOINT=https://your-oidc-provider.com/oauth2

   # Granite Build (optional — required for GB runner)
   GB_TOKEN=your_github_access_token

   # DMF / Lakehouse (optional)
   LAKEHOUSE_TOKEN=your_dmf_lakehouse_token
   DMF_CACHE=path_to_store_dmf_cache

   # MCP server + AI chat (optional)
   ENABLE_MCP=true
   MCP_SERVER_URL=http://localhost:8000/mcp
   LITELLM_URL=https://your-litellm-proxy
   LITELLM_API_KEY=your_litellm_key
   LITELLM_MODEL=aws/claude-sonnet-4-6

   # Development
   LOG_LEVEL=INFO
   DEV_MODE=True
   ```

6. **Initialize the database**:
   ```bash
   mysql -u your_db_user -p autotune < data_models/autotune_latest.sql
   ```

7. **Start the API server**:
   ```bash
   python server.py
   ```

   The API will be available at `http://localhost:8000/fmtune/api`.

### Frontend Setup

1. **Navigate to the UX directory**:
   ```bash
   cd ../ux
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Configure the API endpoint** — create a `.env` file in `ux/`:
   ```env
   PUBLIC_AUTOTUNEX_API_URL=/local/fmtune/api   # uses Vite dev proxy → localhost:8000

   # Or point directly at a deployed environment:
   # PUBLIC_AUTOTUNEX_API_URL=/stage/fmtune/api
   # PUBLIC_AUTOTUNEX_API_URL=/prod/fmtune/api
   ```

   Authentication is handled server-side — the frontend trusts the API's
   `/auth/validate` response, so there is no frontend bypass variable. To skip
   OIDC for local work, leave the API server's `OIDC_*` vars unset (or set
   `AUTOTUNEX_AUTH=dev`); see the API setup above.

4. **Start the development server**:
   ```bash
   npm run dev
   ```

   The UI will be available at `http://localhost:5173/autotune`.

5. **Open in browser**:
   ```bash
   npm run dev -- --open
   ```

### API Bridge Setup

1. **Navigate to the API Bridge directory**:
   ```bash
   cd ../api-bridge
   ```

2. **Install dependencies**:
   ```bash
   pip install .
   ```

3. **Configure environment variables** — set the same DB connection vars as the API server in `api-bridge/.env`.

4. **Start the API bridge server**:
   ```bash
   python server.py
   ```

   The bridge will be available at `http://localhost:8001/fmtune/api`.

## Configuration

### AutoTune YAML Templates

Configuration templates live in `api/config/`:

| File | Purpose |
|------|---------|
| `autotune.yaml` | Default SFT / HPO template used for new jobs. |
| `autotune_new.yaml` | Newer template variant. |
| `autotune_debug.yaml` | Minimal config for local debugging. |
| `autotune_debug_rl.yaml` | Debug template for RL jobs. |
| `rl_config.json` | RL hyperparameter defaults consumed by the local runner. |

Each YAML template defines:

- **System Configuration**: CPU/GPU allocation, Ray worker settings
- **Training Parameters**: Learning rates, batch sizes, epochs, gradient accumulation
- **Hyperparameter Search Spaces**: Ranges and distributions for tuning
- **Model Settings**: Precision (fp32, bf16, int8, int4), checkpoint intervals
- **Search Algorithms**: Random, LDS, HyperOpt, BayesOpt, BOHB
- **Schedulers**: Hyperband, FIFO

Example:
```yaml
system_config:
  sys_cpus:
    default: 16
    min_val: 1
    max_val: 32
  sys_gpus:
    default: 2
    min_val: 1
    max_val: 32
  num_gpus_per_worker:
    default: 1
```

### Dataset Configuration

Datasets are uploaded through the UI's Intelligent Dataset Wizard, which generates a parsing strategy via LLM, validates the file, and stores the artifact under `AUTOTUNE_DATASETS_PATH`. For direct use, JSONL is the canonical format:

```json
{"input": "Your input text", "output": "Expected output"}
```

Sample datasets are bundled in `api/datasets/` and cover finance, law, climate, sentiment analysis, emotion detection, summarization (CNN, DailyMail, Reddit, SAMSum, DialogSum), and tweets.

## Usage

### Web Interface

1. **Open the UI**: `http://localhost:5173/autotune`
2. **Create a fine-tuning job** via the Start Tuning wizard:
   - Select a base model (HuggingFace or DMF)
   - Pick a dataset (existing or upload via the wizard)
   - Choose a configuration template, adjust the search space
   - For RL: supply a `reward_function.py` with a `compute_score(...)` function
   - Review and launch
3. **Monitor progress** under **Tunings** — real-time trials, metrics, and logs
4. **Publish** a completed model to DMF / RITS from the tuning detail view
5. **Chat** with the AI assistant (bottom-right ChatBox) to list, create, or inspect resources in natural language

### REST API

**Create a fine-tuning job**:
```bash
curl -X POST http://localhost:8000/fmtune/api/job \
  -H "Content-Type: application/json" \
  -b 'email="<email>"; token=<token>; role=<role>' \
  -d '{
    "model": "ibm-granite/granite-3.3-8b-instruct",
    "dataset_id": "dataset-uuid",
    "config_id": "config-uuid",
    "experiment_name": "autotune-test"
  }'
```

**Check job status**:
```bash
curl -b 'email="<email>"; token=<token>; role=<role>' \
  http://localhost:8000/fmtune/api/jobs/{job_id}
```

**List all jobs**:
```bash
curl -b 'email="<email>"; token=<token>; role=<role>' \
  http://localhost:8000/fmtune/api/jobs
```

**Chat with the assistant**:
```bash
curl -X POST http://localhost:8000/fmtune/api/chat \
  -H "Content-Type: application/json" \
  -b 'email="<email>"; token=<token>; role=<role>' \
  -d '{"messages":[{"role":"user","content":"list my datasets"}]}'
```

### API Documentation

When the server is running, interactive API documentation is available at:

- **Swagger UI**: `http://localhost:8000/fmtune/try`
- **ReDoc**: `http://localhost:8000/fmtune/docs`
- **Health checks**: `/fmtune/health`, `/fmtune/health/live`, `/fmtune/health/ready`
- **MCP endpoint** (SSE): `http://localhost:8000/mcp` (when `ENABLE_MCP=true`)

## Development

### Building for Production

**Frontend**:
```bash
cd ux
npm run build      # outputs to ux/build/
npm run preview    # serves the production build on :3400
```

### Code Quality

```bash
npm run lint       # Prettier + ESLint
npm run format     # Prettier auto-format
npm run check      # svelte-check / TypeScript
```

There is no formal test suite for the API yet — test manually via Swagger at `/fmtune/try` or through the UI.

## Docker Deployment

Docker configurations are available for each component. The API ships multiple runtime Dockerfiles:

| File | Purpose |
|------|---------|
| `api/Dockerfile` | API server image |
| `api/Dockerfile.runtime` | SFT training runtime (Ray workers) |
| `api/Dockerfile.rl_runtime` | RL training runtime with `verl` |
| `api/Dockerfile.gb` | Granite Build runtime |

### Quick Start with Docker Compose

The fastest way to run the whole stack (API + UI, API Bridge, and MySQL) is the
root `docker-compose.yml`. It builds everything from source and creates the
database schema automatically on first boot — no manual SQL, no external
credentials.

```bash
cp .env.example .env          # then edit the passwords + SESSION_SECRET
docker compose up -d --build  # builds & starts mysql, app (API+UI), bridge
# open http://localhost:8000/autotune  (log in with the dev user)
docker compose logs -f app    # follow API/UI logs
docker compose down           # stop; add -v to also drop the DB + data volumes
```

Three services come up:

| Service  | What it is                    | Host port |
|----------|-------------------------------|-----------|
| `mysql`  | MySQL 8.4 (internal only)     | —         |
| `app`    | API + SvelteKit UI (combined) | `8000`    |
| `bridge` | API Bridge logging service    | `8001`    |

**What "out of the box" means here.** The default profile is credential-free and
**local-only**: local job runner, dev auth (default user `dev@example.com` /
`admin`), local dataset storage, local model registry. The UI, auth, dataset and
config flows, MCP, and the chat assistant (if you set `LITELLM_*`) all work. No
`github.ibm.com` access, OIDC, Granite Build, or DMF is required.

**Schema is auto-created.** `api/data_models/autotune_latest.sql` is mounted into
MySQL's init directory and runs once, the first time the DB volume is empty. It
creates the schema in the `autotune` database, so keep `DB_NAME=autotune`. To
re-run it after schema changes, recreate the volume: `docker compose down -v`.

**Configuration** lives in the root `.env` (copied from `.env.example`); Compose
passes each service only the variables it needs. See the comments in
`.env.example` for every option.

**Optional: IBM extras (Granite Build, DMF, local training).** The default image
is credential-free and CPU-only. Optional features come from `pip` extras
(`api/setup.py`), selected at build time with the `APP_EXTRAS` build arg — no
Dockerfile edits needed:

| To enable | `.env` settings |
|-----------|-----------------|
| **Granite Build** runner | `APP_EXTRAS=[granite-build]`, `GH_USER`, `GH_TOKEN` (for the `github.ibm.com` dependency), plus `GB_TOKEN` at runtime. Optionally `AUTOTUNEX_RUNNER=gb`. |
| **Full IBM profile** (GB + DMF + training core) | `APP_EXTRAS=[ibm]`, `GH_USER`, `GH_TOKEN` |
| **Local GPU training** | the above *and* `BASE_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu22.04` (requires the NVIDIA container toolkit on the host) |

Then rebuild: `docker compose up -d --build`. Granite Build activates only when the
image was built with the `granite-build` extra **and** `GB_TOKEN` is set — setting
one without the other leaves the stack safely in local mode.

Design notes: `docs/architecture-review/2026-07-03-docker-compose-stack-design.md`.

### API Server
```bash
cd api
docker build -f Dockerfile -t autotune-api .
docker run -p 8000:8000 --env-file .env autotune-api
```

### Frontend
```bash
cd ux
docker build -f Dockerfile -t autotune-ux .
docker run -p 3000:3000 autotune-ux
```

### API Bridge
```bash
cd api-bridge
docker build -f Dockerfile -t autotune-bridge .
docker run -p 8001:8001 --env-file .env autotune-bridge
```

## Project Structure

```
AutoTuneX/
├── api/                         # Backend API server (FastAPI)
│   ├── config/                  # YAML / JSON tuning templates
│   ├── data_models/             # SQL schema + migrations
│   ├── datasets/                # Sample datasets
│   ├── routes/                  # Sub-routers (gb_routes, health_checker, utils_routes)
│   ├── services/                # Business logic
│   │   ├── impl/runner.py       # Runner ABC
│   │   └── runners/             # local_runner (Ray Tune), gb_runner (Granite Build)
│   ├── utilites/                # Converter + GB helper utilities (sic — dir is intentionally misspelled)
│   ├── auth.py                  # OIDC / OAuth flow
│   ├── dependencies.py          # FastAPI dependency injection
│   ├── mcp_server.py            # FastMCP tools (mounted at /mcp)
│   ├── models.py                # Pydantic models
│   ├── server.py                # FastAPI app (~2000 lines)
│   └── utils.py
├── api-bridge/                  # Logging microservice (FastAPI, :8001)
│   ├── services/                # config / dataset / github / job / user (used by GB runners)
│   ├── database.py
│   ├── log_service.py
│   └── server.py
├── ux/                          # Frontend (SvelteKit)
│   ├── src/
│   │   ├── lib/
│   │   │   ├── api.ts           # Central API client
│   │   │   ├── store.ts         # Svelte writable stores
│   │   │   └── components/      # views, forms, displays, tables, tabs, ChatBox
│   │   └── routes/              # Page routes (under /autotune)
│   ├── static/
│   └── vite.config.ts
├── autotunex-architecture.html  # Static system diagram
├── autotunex-mindmap.html       # Interactive component mindmap
├── cache/                       # DMF cache directory
└── README.md                    # This file
```

## Contributing

We welcome contributions. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

For questions, issues, or collaboration:

- **Daniel Karl** — daniel.karl@ibm.com
- **Priyanshu Rai** — priyanshu.rai@ibm.com
- **Radu Marinescu** — radu.marinescu@ie.ibm.com

## License

Copyright IBM Research. Licensed under the Apache License 2.0.

---

**Note**: This project is part of IBM Research initiatives. Access to certain features (DMF integration, OIDC authentication, Granite Build execution, LiteLLM proxy) may require IBM-specific credentials and infrastructure.
