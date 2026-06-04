# gbserver Standalone Deployment

Run gbserver locally without IBM Cloud dependencies.

## Prerequisites

- Python 3.11+
- gbserver installed (`make venv && source .venv/bin/activate`)
- gbcli installed (for REST API mode)

## Quick Start: REST API Mode (works with gbcli)

This is the recommended approach. Start the standalone server, then use `gb` (gbcli) to submit and monitor builds.

**Terminal 1 — Start the server:**

```bash
gbserver standalone --space-dir samples/standalone/standalone-quickstart
```

This starts the REST API on port 8080 with a BuildWatcher, using SQLite storage and thread-based execution. A `standalone` space is auto-created from the provided directory.

**Terminal 2 — Submit a build:**

```bash
export GB_ENVIRONMENT=STANDALONE
gb build start -f samples/standalone/standalone-quickstart/build.yaml
```

The command returns a build ID. Use it to check status and logs:

```bash
gb build status <build-id>
gb build log <build-id>
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GBSERVER_API_KEY` | _(empty)_ | API key for auth. If empty, localhost access is allowed without auth |
| `GBSERVER_HOST` | `http://localhost:8080` | Override the server URL in gbcli |
| `GBSERVER_METADATA_STORAGE` | `sql` | Storage backend: `sqlite`, `sql`, `lakehouse` |
| `GBSERVER_DEFAULT_BUILDRUNNER_TYPE` | `job` | Runner type: `thread` or `process` for local, `job` for K8s |

### Authentication

Standalone mode uses `GBSERVER_AUTH_MODE=apikey` (set automatically by the `standalone` command). There are two modes of operation:

**Localhost-only (default, no key required):**

When `GBSERVER_API_KEY` is not set, unauthenticated access is allowed from localhost (`127.0.0.1` / `::1`) only. Remote requests receive a 401. This is the simplest path for local development — no configuration needed.

**API key auth (for remote or shared access):**

Set the same `GBSERVER_API_KEY` on both the server and client:

```bash
# Terminal 1 — server
export GBSERVER_API_KEY="my-secret-key"
gbserver standalone --space-dir samples/standalone/standalone-quickstart

# Terminal 2 — client (gbcli)
export GB_ENVIRONMENT=STANDALONE
export GBSERVER_API_KEY="my-secret-key"
gb build start -f samples/standalone/standalone-quickstart/build.yaml
```

The client sends the key as a `Bearer` token in the `Authorization` header. The server validates it using a timing-safe comparison.

> **Note:** `GBSERVER_API_USER` is a server-side only variable (not used by gbcli). It controls the username of the synthetic user created during API key auth and defaults to `standalone`. You shouldn't need to change it.

## Quick Start: CLI-Only Mode (no server needed)

For simple one-off builds without the REST API:

```bash
export GBSERVER_METADATA_STORAGE=sqlite
export GBSERVER_DEFAULT_BUILDRUNNER_TYPE=process
gbserver build run \
  --space-config-uri "file://$(pwd)/samples/standalone/standalone-quickstart" \
  samples/standalone/standalone-quickstart
```

## Compute Backends

The `standalone-quickstart` sample includes environment configurations for 4 compute backends. Edit `build.yaml` and uncomment the desired `environment_uri` line.

### Bash (default)

Local process execution. No extra dependencies.

```yaml
environment_uri: space://environments/bash
```

### Docker

Runs the build step inside a container. Requires Docker or Podman running locally.

```yaml
environment_uri: space://environments/docker
```

For Podman, set the Docker-compatible socket before starting the server:

```bash
export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock
```

The `build.yaml` includes commented-out resource constraints (`num_cpus_per_node`, `total_memory_per_node`, `num_gpus_per_node`) that map directly to Docker resource limits and GPU device requests.

### RunPod

GPU cloud execution on RunPod. Requires a RunPod API key.

```bash
export RUNPOD_API_KEY="your-runpod-api-key"
```

```yaml
environment_uri: space://environments/runpod
```

The RunPod environment defaults to an NVIDIA A100 80GB GPU. Edit `environments/runpod/environment.yaml` to change the GPU type. Supported types: A100-80GB, A100-40GB, H100-80GB, H100-SXM, L40S, RTX-4090, RTX-A6000, A40.

### SkyPilot

Cloud execution via SkyPilot. Supports AWS (EC2 direct) or Kubernetes backends.

**AWS (default):** Requires AWS credentials at `~/.aws/credentials`.

**Kubernetes:** Edit `environments/skypilot/environment.yaml` and change `default_cloud` from `aws` to `kubernetes`. Requires kubeconfig at `~/.kube/config`. See `docs/operators/setup/skypilot-kubernetes-setup.md` for K8s cluster setup.

```yaml
environment_uri: space://environments/skypilot/kubernetes
```

## Storage Backends

The sample includes 3 asset store configurations under `assetstores/`. Environments reference specific stores in their `environment.yaml`. The Bash and Docker environments default to local storage; RunPod and SkyPilot default to S3.

### Local Filesystem (default for Bash/Docker)

Uses `file:` URIs. No configuration needed.

### S3 (default for RunPod/SkyPilot)

S3-compatible object storage (AWS S3, MinIO, etc.).

```bash
export COS_ACCESS_KEY_ID="your-access-key"
export COS_SECRET_ACCESS_KEY="your-secret-key"
```

Edit `assetstores/s3/store.yaml` to change the endpoint or region.

### HuggingFace Hub

For loading models and datasets from HuggingFace.

```bash
export HF_TOKEN="your-hf-token"  # optional for public repos
```

To use HuggingFace as a store in an environment, add it to the environment's `assetstores` list in its `environment.yaml`.

## Sample Structure

```
standalone-quickstart/
  space.yaml                              # Space config (env secret manager)
  build.yaml                              # Build definition (1 target, swap environment_uri)
  environments/
    bash/environment.yaml                 # Local process execution
    docker/environment.yaml               # Container execution (alpine:latest)
    runpod/environment.yaml               # RunPod GPU cloud
    skypilot/environment.yaml             # SkyPilot (AWS or K8s)
  assetstores/
    local/store.yaml                      # file: URIs
    s3/store.yaml                         # S3-compatible storage
    hf/store.yaml                         # HuggingFace Hub
  steps/
    hello/
      step.yaml                           # Step with per-environment launcher configs
      bash_scripts/hello/command.sh       # Script for Bash/Docker backends
```

## Architecture

Standalone deployment uses local-friendly backends for every component:

| Component | Standalone Backend | Cloud Backend |
|-----------|--------------------|---------------|
| Storage | SQLite | PostgreSQL / Lakehouse |
| Execution | Bash / Docker / RunPod / SkyPilot | Kubernetes / LSF |
| Asset Store | FileStore / S3 / HuggingFace | COS / Lakehouse |
| Secrets | Env / Hybrid | IBM Cloud Secret Manager |
| Messaging | NATS (optional) | RabbitMQ (optional) |
| Auth | API key / localhost | GitHub Enterprise |

No special "standalone mode" flag is needed. Each component is independently configured via the existing plugin architecture.

## What's Next

See `docs/plans/2026-02-18-standalone-deployment-design.md` for the full roadmap including multi-process deployments.
