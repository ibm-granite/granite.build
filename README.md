# Granite.Build

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Build orchestration for LLM pipelines. Define multi-step model workflows in YAML — download, fine-tune, evaluate, and deploy — and run them locally or on cloud infrastructure.

## What is Granite.Build?

Granite.Build orchestrates LLM build pipelines. You describe your workflow in a `build.yaml` file — which models to download, how to fine-tune them, what evaluations to run — and Granite.Build executes each step in the environment you choose: a local Docker container, a Kubernetes cluster, a cloud GPU instance, or a plain bash process on your laptop.

The system has three main components:

- **gbserver** — the orchestration server. It provides a REST API (`/api/v1`) for build management and a build watcher that polls for pending builds and dispatches them to execution environments. It stores build metadata in SQLite (standalone) or PostgreSQL (production).

- **gb** (gbcli) — the command-line client. It talks to the server's REST API to submit builds, list status, manage artifacts, and more.

- **build.yaml** — the pipeline definition. Each file declares a set of named **targets** (logical stages like "download", "fine-tune", "evaluate"). Each target specifies an execution environment, input/output artifacts, and one or more **steps** to run. Targets can depend on each other through artifact **bindings** — when an upstream target produces an output, downstream targets that reference it are automatically dispatched.

### How the pieces fit together

```
build.yaml ──→ gb build start ──→ gbserver REST API
                                       │
                                  BuildWatcher
                                       │
                                  BuildRunner
                                       │
                          ┌────────────┼────────────┐
                          │            │            │
                       Docker     Kubernetes     Bash
                       RunPod      SkyPilot
                          │            │            │
                          └────────────┼────────────┘
                                       │
                              Artifact stores
                          (HuggingFace, file://, git://)
```

The **BuildWatcher** polls storage for pending builds and creates a **BuildRunner** for each one. The runner walks the target graph, resolving dependencies and launching steps through the configured **Environment** (Docker, Kubernetes, Bash, RunPod, or SkyPilot). Each step can pull inputs from and push outputs to **artifact stores** selected by URI scheme (`hf://`, `file://`, `git://`, `cos://`).

### Example build.yaml

A minimal pipeline that runs a single step in a Docker container:

```yaml
granite.build:
  name: my-build
  targets:
    download:
      environment_uri: space://environments/docker
      inputs:
        model:
          uri: hf://ibm-granite/granite-3.3-2b-instruct
      outputs:
        model:
          uri: file:workspace/model
      steps:
        - step_uri: space://steps/hfpull
```

A multi-target pipeline chains stages through bindings:

```yaml
granite.build:
  name: tune-and-eval
  targets:
    download:
      environment_uri: space://environments/docker
      outputs:
        model: { uri: file:workspace/model }
      steps:
        - step_uri: space://steps/hfpull
    fine-tune:
      environment_uri: space://environments/docker
      inputs:
        model: { binding: download.model }
      outputs:
        checkpoint: { uri: file:workspace/checkpoint }
      steps:
        - step_uri: space://steps/sft
    evaluate:
      environment_uri: space://environments/docker
      inputs:
        model: { binding: fine-tune.checkpoint }
      steps:
        - step_uri: space://steps/eval
```

## Repository Layout

| Directory | Description |
|-----------|-------------|
| `src/gbserver/` | Build orchestration server (REST API, build engine, storage) |
| `src/gbcli/` | CLI client for interacting with gbserver |
| `src/gbcommon/` | Shared types and utilities |
| `test/` | Test suites for all components |
| `samples/` | Sample build configs, environments, and steps |
| `scripts/` | Helper scripts including the standalone demo |

## Features

- **Multi-environment execution** — Docker, Kubernetes, RunPod, SkyPilot/AWS, or local bash
- **HuggingFace Hub integration** — download and push models and datasets via `hf://` URIs
- **Pipeline orchestration** — chain steps with artifact bindings in a single `build.yaml`
- **CLI client** — `gb` command for build management, artifact handling, model operations, and more
- **REST API** — FastAPI-based build management at `/api/v1`
- **Standalone mode** — SQLite + thread-based execution, no external services needed
- **Lineage tracking** — records data provenance of builds, targets, and artifacts

## Standalone Setup Guide

### Prerequisites

- Python 3.11+ (3.12 or 3.13 recommended)
- Docker or Podman with a running daemon (for container-based steps)

### Installation

```bash
# Clone the repository
git clone https://github.com/ibm-granite/granite.build.git
cd granite.build

# Create a virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[standalone,docker]"
```

This installs both the server (`gbserver`) and the CLI client (`gb`) with standalone (SQLite + NATS) and Docker execution support.

### Starting the server

```bash
gbserver standalone --space-dir ./my-builds
```

This starts the REST API on port 8080 and the build watcher in a single process. The `--space-dir` flag points to a directory containing your build configurations, environments, and steps.

### Using the CLI

With the server running, use `gb` to manage builds:

```bash
# Submit a build
gb build start build.yaml

# List all builds
gb build list

# Get build details
gb build get <build-id>

# Cancel a running build
gb build cancel <build-id>

# View artifacts
gb artifact list <build-id>
```

Run `gb --help` for the full command reference.

## Running the Demo

The repository includes an end-to-end demo that runs TRL fine-tuning and unitxt evaluation in Docker containers.

### Prerequisites

- Docker or Podman with a running daemon
- For macOS with Podman: the VM needs at least 4 GB of RAM (`podman machine set --memory 4096`)

### Setup

```bash
make demo-venv PYTHON=python3.13
source .venv/bin/activate
```

### Run

```bash
# Run both TRL fine-tuning and unitxt evaluation
bash scripts/demo-standalone.sh

# TRL fine-tuning only
bash scripts/demo-standalone.sh --trl-only

# unitxt evaluation only (lighter, good for low-memory systems)
bash scripts/demo-standalone.sh --unitxt-only

# Force CPU mode (skip GPU auto-detection)
GBSERVER_DEMO_CPU=1 bash scripts/demo-standalone.sh
```

The demo starts a standalone server, builds a container image (on first run), submits the builds, and streams progress to the terminal.

### SLURM Demo (via SkyPilot)

Runs the same TRL fine-tuning workload on a local Docker-based SLURM cluster via SkyPilot, with artifact push to MinIO (S3-compatible object storage).

#### Prerequisites

- Docker (or Podman) with a running daemon
- Python 3.11+ (3.12 or 3.13 recommended)
- No cloud credentials needed — everything runs locally

#### Setup (from scratch)

```bash
# 1. Create virtual environment with SkyPilot support
make g4os-skypilot-venv PYTHON=python3.13
source .venv/bin/activate

# 2. Start MinIO (S3-compatible artifact store)
make minio-setup

# 3. Start the Docker SLURM cluster (slurmctld + 2 compute nodes)
#    This also connects MinIO to the SLURM network
make slurm-setup

# 4. Verify SkyPilot sees the SLURM cluster
sky check slurm
```

#### Run

```bash
# Run both TRL fine-tuning and unitxt evaluation on SLURM
bash scripts/demo-slurm.sh

# TRL fine-tuning only
bash scripts/demo-slurm.sh --trl-only

# unitxt evaluation only
bash scripts/demo-slurm.sh --unitxt-only
```

The demo submits builds that run on the SLURM cluster via SkyPilot. When training completes, an `s3push` step automatically uploads the checkpoint to MinIO. First run takes 5-10 minutes (SkyPilot installs dependencies on the SLURM nodes).

#### Verify artifacts in MinIO

```bash
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

# Fine-tuning checkpoint
aws --endpoint-url http://localhost:9000 s3 ls s3://gb-checkpoints/outputs/trl-finetune/ --recursive

# Evaluation results
aws --endpoint-url http://localhost:9000 s3 ls s3://gb-checkpoints/outputs/unitxt-eval/ --recursive
```

#### Teardown

```bash
make slurm-teardown
make minio-teardown
```

#### How it works

```
build.yaml ──→ gbserver ──→ SkyPilot ──→ SLURM (sbatch)
                                              │
                                    TRL trains on compute node
                                              │
                                    Artifact signal emitted
                                              │
                              pushasset_cosstore auto-queues s3push
                                              │
                                    s3push uploads to MinIO
                                              │
                                    Build completes SUCCESS
```

## Supported Environments

| Environment | Platform | GPU Support | Status |
|-------------|----------|-------------|--------|
| Docker | Linux, macOS | Yes (nvidia-container-toolkit) | Stable |
| Bash | macOS / Linux | CPU only | Stable |
| Kubernetes | Linux | Yes | Stable |
| SLURM (via SkyPilot) | Linux | Yes (auto-detected) | Beta |
| RunPod | Cloud | Yes | Beta |
| SkyPilot / AWS | Cloud | Yes | Beta |

## CLI Reference

### gb (client)

The CLI client is available as multiple equivalent entry points: `gb`, `gbcli`, `llmbuild`, `llmb`, `lamb`.

```
gb build       Build management (start, list, get, cancel, delete)
gb model       Model operations
gb artifact    Artifact management
gb step        Step operations
gb space       Space management
gb template    Template operations
gb auth        Authentication
gb secret      Secret management
gb admin       Administrative commands
gb version     Show version
```

### gbserver (server)

```
gbserver standalone   Start all-in-one server (API + build watcher)
gbserver rest-server  Start only the REST API
gbserver build-watch  Start only the build watcher
gbserver build        Run a build directly
```

## API

The REST API is available at `/api/v1` when the server is running. Start with `gbserver standalone` or `gbserver rest-server` and visit `http://localhost:8080/docs` for the interactive OpenAPI documentation.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and pull request guidelines.

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).

## Security

To report a vulnerability, see [SECURITY.md](SECURITY.md).

## License

[Apache License 2.0](LICENSE)
