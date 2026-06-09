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
gbserver standalone --space-dir configurations/spaces/local
```

This starts the REST API on port 8080 with a BuildWatcher, using SQLite storage and thread-based execution. A `local` space (aliased as `public` and `standalone`) is auto-created from the in-repo [`configurations/spaces/local`](../../configurations/spaces/local/) directory, whose `base_uris` chain resolves the sample's `space://` URIs into the shared [`configurations/assets/`](../../configurations/assets/) tree.

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
gbserver standalone --space-dir configurations/spaces/local

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
  --space-config-uri "file://$(pwd)/configurations/spaces/local" \
  samples/standalone/standalone-quickstart
```

The first positional argument is the directory containing the `build.yaml` (the sample); `--space-config-uri` points at the canonical space, which resolves the build's `space://` URIs through `configurations/assets/`.

## Compute Backends

The sample's `build.yaml` can target any compute-backend environment defined in the shared [`configurations/assets/environments/`](../../configurations/assets/environments/) tree. Edit `build.yaml` and uncomment the desired `environment_uri` line.

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

The RunPod environment defaults to an NVIDIA A100 80GB GPU. Edit [`configurations/assets/environments/runpod/environment.yaml`](../../configurations/assets/environments/runpod/environment.yaml) to change the GPU type. Supported types: A100-80GB, A100-40GB, H100-80GB, H100-SXM, L40S, RTX-4090, RTX-A6000, A40.

### SkyPilot

Cloud execution via SkyPilot. Separate environments are provided for AWS (EC2 direct) and Kubernetes backends.

**AWS:** Requires AWS credentials at `~/.aws/credentials`.

```yaml
environment_uri: space://environments/skypilot/aws
```

**Kubernetes:** Requires a kubeconfig at `~/.kube/config`. See `docs/operators/setup/skypilot-kubernetes-setup.md` for K8s cluster setup.

```yaml
environment_uri: space://environments/skypilot/kubernetes
```

## Storage Backends

Three asset store configurations live under [`configurations/assets/assetstores/`](../../configurations/assets/assetstores/). Environments reference specific stores in their `environment.yaml`. The Bash and Docker environments bind local + HuggingFace storage; RunPod and SkyPilot bind S3.

### Local Filesystem (default for Bash/Docker)

Uses `file:` URIs. No configuration needed.

### S3 (default for RunPod/SkyPilot)

S3-compatible object storage (AWS S3, MinIO, etc.).

```bash
export COS_ACCESS_KEY_ID="your-access-key"
export COS_SECRET_ACCESS_KEY="your-secret-key"
```

Edit [`configurations/assets/assetstores/s3/store.yaml`](../../configurations/assets/assetstores/s3/store.yaml) to change the endpoint or region.

### HuggingFace Hub

For loading models and datasets from HuggingFace.

```bash
export HF_TOKEN="your-hf-token"  # optional for public repos
```

The Bash and Docker environments already bind HuggingFace; to add it elsewhere, append it to that environment's `assetstores` list in `configurations/assets/environments/<env>/environment.yaml`.

## Sample Structure

The sample is now just a build definition. It runs against the in-repo canonical space
[`configurations/spaces/local`](../../configurations/spaces/local/), whose `base_uris` chain
resolves the build's `space://` URIs into the shared [`configurations/assets/`](../../configurations/assets/) tree.

```
samples/standalone/standalone-quickstart/
  build.yaml                              # Build definition (1 target, swap environment_uri)

configurations/assets/                    # Shared, canonical definitions (resolved via the space)
  environments/
    bash/        environment.yaml + steps/hello/   # Local process execution
    docker/      environment.yaml + steps/hello/   # Container execution (alpine:latest)
    runpod/      environment.yaml + steps/hello/   # RunPod GPU cloud
    skypilot/aws/        environment.yaml + steps/hello/   # SkyPilot on AWS
    skypilot/kubernetes/ environment.yaml                  # SkyPilot on K8s (Skypilot hello resolves via env-class match)
  assetstores/
    local/store.yaml                      # file: URIs
    s3/store.yaml                         # S3-compatible storage
    hf/store.yaml                         # HuggingFace Hub
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
