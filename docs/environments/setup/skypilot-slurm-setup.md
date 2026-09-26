# SkyPilot SLURM setup (local Docker cluster + S3)

This guide covers setting up a local Docker SLURM cluster and a local S3-compatible
store (SeaweedFS) for development and integration testing with SkyPilot. For the SkyPilot-on-SLURM
configuration that runs against this cluster, see [skypilot-slurm.md](../skypilot-slurm.md).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [SLURM Cluster](#slurm-cluster)
- [S3 Storage](#s3-storage)
- [Running Integration Tests](#running-integration-tests)
- [Teardown](#teardown)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- **Docker** or **Podman** with the `compose` plugin
- **Python 3.11+** with a virtual environment (`make standalone-venv` or `make venv`)
- **SSH client** (for verifying SLURM connectivity)
- **nvidia-container-toolkit** (optional, for GPU passthrough)

## Quick Start

```bash
# Bring up SLURM cluster (auto-detects GPU)
make slurm-setup

# Bring up local S3 storage (only needed for S3 artifact push, e.g. the SLURM demo)
make s3-setup

# Run integration tests
make integration-test

# Tear everything down
make slurm-teardown
make s3-teardown
```

## SLURM Cluster

### What gets deployed

The SLURM cluster runs as a set of Docker containers:

| Container        | Role                                  |
|------------------|---------------------------------------|
| `slurm-slurmctld`| Controller + login node (SSH target) |
| `slurm-slurmdbd` | Database daemon                      |
| `slurm-mysql`    | MariaDB for accounting               |
| `slurm-c1`       | Compute node 1 (GPU if available)    |
| `slurm-c2`       | Compute node 2 (CPU only)            |

### Setup

```bash
make slurm-setup
```

Or invoke the script directly with custom options:

```bash
SLURM_SSH_PORT=2222 SLURM_VERSION=25.11.4 bash scripts/slurm/setup-slurm.sh
```

The script:
1. Generates an SSH key pair at `~/.ssh/slurm_docker_key`
2. Detects whether a GPU is available on the host
3. Starts the Docker Compose stack
4. Waits for all nodes to register with the controller
5. Verifies SSH connectivity

It does **not** write the `slurm-docker` entry into `~/.slurm/config`. The SLURM
SSH reachability config is inlined in the Skypilot `environment.yaml`
(`cluster_ssh_configs.slurm`, see
[skypilot-slurm.md](../skypilot-slurm.md#cluster_ssh_configsslurm--reachability))
and materialized into `~/.slurm/config` by gbserver at build launch time. The
script only provisions the cluster and the SSH key that the inline `identity_file`
references.

Each run also **clears any stale gbserver-managed or legacy `setup-slurm.sh`
block** from `~/.slurm/config` (preserving unrelated `Host` entries), so a
leftover from an older run can't trip gbserver's refuse-on-conflict — re-running
`make slurm-setup` is enough to refresh config; you never need `make slurm-teardown`
just for that.

### GPU support

GPU support is auto-detected. When `nvidia-smi` is available on the host, the script:
- Applies `docker-compose.gpu.yml` overlay to pass the GPU through to `c1`
- Writes `AutoDetect=nvidia` to `gres.conf`
- Adds `Gres=gpu:1` to the `c1` node definition

To force CPU-only mode even when a GPU is present:

```bash
SLURM_NO_GPU=1 make slurm-setup
```

### Environment variables

| Variable          | Default   | Description                            |
|-------------------|-----------|----------------------------------------|
| `SLURM_SSH_PORT`  | `2222`    | Host port for SSH to slurmctld         |
| `SLURM_VERSION`   | `25.11.4` | SLURM image tag                        |
| `SLURM_NO_GPU`    | `0`       | Set to `1` to disable GPU detection    |
| `DOCKER`          | auto      | Container runtime (`docker` or `podman`) |

### Verification

```bash
# SSH to the login node
ssh -i ~/.ssh/slurm_docker_key -p 2222 root@localhost

# Check cluster status
ssh -i ~/.ssh/slurm_docker_key -p 2222 root@localhost sinfo

# Submit a test job
ssh -i ~/.ssh/slurm_docker_key -p 2222 root@localhost sbatch --wrap 'hostname'

# Test GPU (if available)
ssh -i ~/.ssh/slurm_docker_key -p 2222 root@localhost srun --gres=gpu:1 nvidia-smi
```

## S3 Storage

### What gets deployed

A single [SeaweedFS](https://github.com/seaweedfs/seaweedfs) container (`gb-s3`,
running `weed mini`) with:
- S3-compatible API on port 9000 (reachable from the SLURM containers as `gb-s3:9000`)
- A `gb-checkpoints` bucket pre-created (via `aws s3 mb`, so the AWS CLI must be on
  `PATH` — the repo `.venv` provides it)

### Setup

```bash
make s3-setup
```

Or invoke the script directly:

```bash
GB_S3_PORT=9000 bash scripts/s3/setup-s3.sh
```

### Environment variables

| Variable               | Default          | Description               |
|------------------------|------------------|---------------------------|
| `GB_S3_PORT`           | `9000`           | S3 API port (host)        |
| `GB_S3_ACCESS_KEY`     | `gbadmin`        | Admin access key          |
| `GB_S3_SECRET_KEY`     | `gbadmin`        | Admin secret key          |
| `GB_S3_BUCKET`         | `gb-checkpoints` | Default bucket name       |
| `GB_S3_IMAGE`          | `docker.io/chrislusf/seaweedfs:4.47` | Container image |
| `GB_S3_CONTAINER_NAME` | `gb-s3`          | Container name            |
| `GB_S3_DATA_VOLUME`    | `gb-s3-data`     | Persistent data volume    |

### Verification

```bash
export AWS_ACCESS_KEY_ID=gbadmin
export AWS_SECRET_ACCESS_KEY=gbadmin
export AWS_ENDPOINT_URL=http://localhost:9000

# List buckets
aws s3 ls

# Upload a test file
echo "hello" | aws s3 cp - s3://gb-checkpoints/test.txt
aws s3 ls s3://gb-checkpoints/
```

> **Upgrading from MinIO:** earlier versions ran a `gb-minio` container on port 9000.
> Remove it first (`docker rm -f gb-minio`; `docker volume rm gb-minio-data` to drop
> its data) or `make s3-setup` will fail on the port conflict.

## Running Integration Tests

Tests that require local SLURM infrastructure use the `skypilot_integration` pytest marker:

```bash
# Run only integration tests
make integration-test

# Or directly with pytest
pytest -s -m skypilot_integration --strict-markers test
```

These tests are excluded from the default test run (`make py-test`) and CI test suites.

> **Note:** If the infrastructure is not running, tests marked `skypilot_integration` should skip
> gracefully (e.g., via a fixture that checks connectivity).

## Teardown

```bash
# Stop and remove SLURM cluster (preserves volumes)
make slurm-teardown

# Stop and remove the S3 store (preserves data volume)
make s3-teardown
```

To also remove persistent data:

```bash
bash scripts/slurm/teardown-slurm.sh --remove-volumes
bash scripts/s3/teardown-s3.sh --remove-data
```

## Troubleshooting

### Port conflicts

If port 2222 (SLURM SSH) or 9000 (S3) are already in use:

```bash
SLURM_SSH_PORT=2223 make slurm-setup
GB_S3_PORT=9010 make s3-setup
```

### GPU not detected

1. Verify `nvidia-smi` works on the host
2. Verify nvidia-container-toolkit is installed: `nvidia-ctk --version`
3. Check Docker runtime config: `docker info | grep -i nvidia`
4. Force CPU-only mode: `SLURM_NO_GPU=1 make slurm-setup`

### Container runtime not found

The scripts auto-detect `docker` or `podman`. To force one:

```bash
DOCKER=docker make slurm-setup
```

### SLURM nodes stuck in UNKNOWN/DOWN state

Wait 1-2 minutes after setup for nodes to register. If they remain down:

```bash
# Check controller logs
docker logs slurm-slurmctld

# Check compute node logs
docker logs slurm-c1
docker logs slurm-c2
```

### S3 bucket creation fails

Bucket creation runs `aws s3 mb` from the host. Check that the AWS CLI is on `PATH`
(`source .venv/bin/activate`) and that the store is healthy:

```bash
curl -sf http://localhost:9000/healthz && echo OK
docker logs gb-s3
```

## Configuration Reference

For the full list of fields supported in a Skypilot `environment.yaml`, the
per-step `step.yaml` `environment_configs.Skypilot.*` block, and the
`build.yaml` step `config:` fields the SkyPilot launcher reads, see:

- [SkyPilot overview](../skypilot.md) — compute model, launcher/monitor types, and config common to all clouds
- [SkyPilot on SLURM](../skypilot-slurm.md) — the SLURM-specific config and a bare-host example matching this cluster

The bare-host SLURM example documented there matches the cluster you bring
up with `make slurm-setup`.
