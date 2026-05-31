# Running gbserver on RunPod as a Persistent Orchestrator

## Overview

gbserver can run on a RunPod CPU pod as an always-on remote orchestrator, launching GPU pods on demand for compute-intensive build steps (model fine-tuning, evaluation, data processing). This gives you a cloud-hosted build server accessible from anywhere, with GPU costs incurred only during active builds.

```
User's laptop                    RunPod (CPU pod, ~$0.10/hr)           RunPod (GPU pod, on-demand)
┌──────────┐                    ┌────────────────────────┐            ┌──────────────────┐
│  gbcli   │──HTTPS proxy──────│ gbserver standalone    │            │ Fine-tuning job   │
│          │                    │   REST API :8080       │──RunPod──▶│ (A100-80GB)       │
│          │                    │   BuildWatcher         │   API     │ image: fms-tuning │
│          │                    │   SQLite @ /workspace  │           │ s3pull → train →  │
│          │                    │   nats-server -js      │◀──poll────│ s3push            │
└──────────┘                    └────────────────────────┘            └──────────────────┘
                                    Network Volume
                                    /workspace/
                                    ├── gbserver.db (SQLite)
                                    ├── .gbserver/nats-data/
                                    └── spaces/standalone/
```

## Prerequisites

- A RunPod account with API key
- `gbcli` installed locally
- A RunPod Network Volume (for persistent storage)

## How It Works

The `gbserver standalone` command runs the full gbserver stack in a single process:

- **REST API** (FastAPI + uvicorn) — accepts build requests from gbcli
- **BuildWatcher** — monitors pending builds and dispatches runners
- **Build Runner** (thread-based) — executes build steps using configured environments
- **SQLite** — metadata storage, persisted on network volume
- **NATS + JetStream** — event messaging (embedded nats-server subprocess)

When a build step is configured with the `Runpod` environment, the build runner calls the RunPod API to create a GPU pod, polls its status, and cleans up when done. The GPU pod runs the specified Docker image with the build step's command.

## Setup

### 1. Create a Network Volume

In the RunPod dashboard, create a Network Volume (100GB recommended):
- Region: choose one with good GPU availability
- Name: `gbserver-data`

### 2. Create the Orchestrator Pod

Create a CPU pod (or minimal GPU pod) with:
- **Image**: A Docker image containing gbserver + nats-server (see [Building the Image](#building-the-image))
- **Network Volume**: Mount `gbserver-data` at `/workspace`
- **Exposed Ports**: `8080/http`
- **Environment Variables**:
  ```
  RUNPOD_API_KEY=<your-runpod-api-key>
  GBSERVER_API_KEY=<choose-a-secret-key>
  GBSERVER_NATS_EMBEDDED=true
  ```

### 3. Initialize the Space Directory

SSH into the pod and set up the space:

```bash
mkdir -p /workspace/spaces/standalone/environments /workspace/spaces/standalone/steps

# Create a minimal space.yaml
cat > /workspace/spaces/standalone/space.yaml << 'EOF'
name: standalone
description: Standalone space on RunPod
EOF

# Copy or create environment configs for RunPod
# (see samples/standalone/ for examples)
```

### 4. Start gbserver

```bash
gbserver standalone \
  --host 0.0.0.0 \
  --port 8080 \
  --space-dir /workspace/spaces/standalone
```

The `--host 0.0.0.0` flag is required so the API is accessible via RunPod's HTTPS proxy.

### 5. Connect gbcli

On your laptop, configure gbcli to point at the RunPod proxy URL:

```bash
export GB_ENVIRONMENT=STANDALONE
export GBSERVER_HOST=https://<pod-id>-8080.proxy.runpod.net
export GBSERVER_API_KEY=<same-key-as-server>

# Verify connectivity
gb build list --space standalone
```

## Building the Image

The orchestrator pod needs a Docker image with gbserver and nats-server installed:

```dockerfile
FROM python:3.12-slim

# Install nats-server
RUN apt-get update && apt-get install -y curl && \
    curl -L https://github.com/nats-io/nats-server/releases/download/v2.10.24/nats-server-v2.10.24-linux-amd64.tar.gz | \
    tar xz -C /usr/local/bin --strip-components=1 nats-server-v2.10.24-linux-amd64/nats-server && \
    apt-get remove -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Install gbserver with standalone + runpod extras
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir ".[standalone,runpod]"

EXPOSE 8080

CMD ["gbserver", "standalone", "--host", "0.0.0.0", "--port", "8080", "--space-dir", "/workspace/spaces/standalone"]
```

## Cost Estimate

### Fixed Infrastructure (Always On)

| Component | Hourly | Monthly |
|-----------|--------|---------|
| Orchestrator CPU pod | ~$0.10/hr | ~$72 |
| Network Volume (100GB) | — | ~$7 |
| **Total fixed** | | **~$79/month** |

### GPU Compute (Pay Per Use)

| GPU Type | Per GPU/hr (Community) | Per GPU/hr (Secure) |
|----------|----------------------|---------------------|
| H100 SXM 80GB | ~$3.89 | ~$4.69 |
| H100 PCIe 80GB | ~$2.39 | ~$3.49 |
| A100 80GB PCIe | ~$1.64 | ~$2.21 |

### Example: 8x H100 SXM Fine-Tuning Job for 1 Week

| Component | Hours | Rate | Cost |
|-----------|-------|------|------|
| Orchestrator (CPU pod) | 168 (1 week) | $0.10/hr | $17 |
| 8x H100 SXM (Community Cloud) | 168 (1 week) | $3.89/hr/GPU | $5,228 |
| 8x H100 SXM (Secure Cloud) | 168 (1 week) | $4.69/hr/GPU | $6,303 |
| Network Volume (100GB) | — | — | ~$2 |

**Total for 1 week of 8x H100 SXM training:**
- Community Cloud: **~$5,247**
- Secure Cloud: **~$6,322**

For comparison, equivalent 8x H100 on-demand pricing:
- AWS p5.48xlarge: ~$98.32/hr → **~$16,518/week**
- GCP a3-highgpu-8g: ~$84.48/hr → **~$14,193/week**
- Lambda Labs 8x H100 SXM: ~$22.88/hr → **~$3,844/week**

RunPod Community Cloud is competitive with Lambda Labs and significantly cheaper than hyperscalers. GPU costs dominate — the orchestrator overhead ($17/week) is negligible.

## Limitations

- **Single container per pod**: RunPod pods run one Docker image. The orchestrator image must include all required software (gbserver, nats-server). GPU compute pods each run a single step image.
- **Single-user**: The standalone mode is designed for a single user. Multiple users share the `"standalone"` space.
- **No Docker-in-Docker**: The orchestrator pod cannot run `DockerEnvironment` steps. Use the `Runpod` environment to launch separate GPU pods instead.
- **Pod preemption**: Use on-demand pricing (not spot) for the orchestrator to avoid surprise termination. Spot pricing is fine for GPU compute pods if your builds are idempotent.

## Auto-Restart

RunPod pods do not auto-restart applications. Wrap gbserver in a restart loop for resilience:

```bash
#!/bin/bash
while true; do
  echo "Starting gbserver standalone..."
  gbserver standalone --host 0.0.0.0 --port 8080 --space-dir /workspace/spaces/standalone
  echo "gbserver exited, restarting in 5s..."
  sleep 5
done
```

Or use `supervisord` in the Docker image for more robust process management.

## SkyPilot Alternative

If you prefer multi-cloud GPU orchestration, gbserver also supports the SkyPilot environment backend (`src/gbserver/environment/skypilot.py`). SkyPilot can launch GPU jobs across AWS, GCP, Azure, Lambda, and RunPod — with automatic spot instance management and failover. See the SkyPilot environment design doc for details.
