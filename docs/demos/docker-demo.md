# Standalone Docker demo

Runs TRL fine-tuning and unitxt evaluation in Docker containers via the
standalone server.

## Prerequisites

- Docker or Podman with a running daemon
- For macOS with Podman: the VM needs at least 4 GB of RAM (`podman machine set --memory 4096`)

## Setup

```bash
make demo-venv PYTHON=python3.13
source .venv/bin/activate
```

## Run

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

The demo starts a standalone server, builds a container image (on first run),
submits the builds, and streams progress to the terminal.

## See also

- [Demos overview](README.md)
- [SLURM demo (via SkyPilot)](skypilot-slurm-demo.md) — the same workload on a local SLURM cluster
- [Docker environment](../environments/docker.md)
