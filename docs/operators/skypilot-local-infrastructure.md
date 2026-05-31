# Local Infrastructure Setup (SLURM + MinIO)

This guide covers setting up a local Docker SLURM cluster and MinIO S3-compatible
storage for development and integration testing with SkyPilot.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [SLURM Cluster](#slurm-cluster)
- [MinIO S3 Storage](#minio-s3-storage)
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

# Bring up MinIO S3 storage
make minio-setup

# Run integration tests
make integration-test

# Tear everything down
make slurm-teardown
make minio-teardown
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
6. Configures `~/.sky/config.yaml` for SkyPilot

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

## MinIO S3 Storage

### What gets deployed

A single MinIO container (`gb-minio`) with:
- S3-compatible API on port 9000
- Web console on port 9001
- A `gb-checkpoints` bucket pre-created

### Setup

```bash
make minio-setup
```

Or invoke the script directly:

```bash
MINIO_API_PORT=9000 MINIO_CONSOLE_PORT=9001 bash scripts/minio/setup-minio.sh
```

### Environment variables

| Variable              | Default         | Description               |
|-----------------------|-----------------|---------------------------|
| `MINIO_API_PORT`      | `9000`          | S3 API port               |
| `MINIO_CONSOLE_PORT`  | `9001`          | Web console port          |
| `MINIO_ROOT_USER`     | `minioadmin`    | Root access key           |
| `MINIO_ROOT_PASSWORD` | `minioadmin`    | Root secret key           |
| `MINIO_BUCKET`        | `gb-checkpoints`| Default bucket name       |
| `MINIO_IMAGE`         | `quay.io/minio/minio:latest` | Container image |

### Verification

```bash
# Set AWS credentials for MinIO
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_ENDPOINT_URL=http://localhost:9000

# List buckets
aws s3 ls

# Upload a test file
echo "hello" | aws s3 cp - s3://gb-checkpoints/test.txt
aws s3 ls s3://gb-checkpoints/
```

> **Note:** The MinIO web console is available at http://localhost:9001 (login: `minioadmin`/`minioadmin`).

## Running Integration Tests

Tests that require local SLURM and MinIO infrastructure use the `skypilot_integration` pytest marker:

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

# Stop and remove MinIO (preserves data volume)
make minio-teardown
```

To also remove persistent data:

```bash
bash scripts/slurm/teardown-slurm.sh --remove-volumes
bash scripts/minio/teardown-minio.sh --remove-data
```

## Troubleshooting

### Port conflicts

If port 2222 (SLURM SSH) or 9000/9001 (MinIO) are already in use:

```bash
SLURM_SSH_PORT=2223 make slurm-setup
MINIO_API_PORT=9010 MINIO_CONSOLE_PORT=9011 make minio-setup
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

### MinIO bucket creation fails

The `mc` CLI runs inside the container. If it fails, verify MinIO is healthy:

```bash
curl -sf http://localhost:9000/minio/health/ready && echo OK
```

## Configuration Reference

For the full list of fields supported in a Skypilot `environment.yaml`, the
per-step `step.yaml` `environment_configs.Skypilot.*` block, and the
`build.yaml` step `config:` fields the SkyPilot launcher reads, see the
Skypilot sections of
[`docs/operators/environment-yaml-config.md`](environment-yaml-config.md):

- [`Skypilot` environment config](environment-yaml-config.md#skypilot-environment-config)
- [Skypilot launcher and monitor types](environment-yaml-config.md#skypilot-launcher-and-monitor-types)
- [Skypilot-specific top-level config fields](environment-yaml-config.md#skypilot-specific-top-level-config-fields)
- [Annotated Skypilot example (bare-host SLURM)](environment-yaml-config.md#skypilot-environmentyaml-bare-host-slurm)

The bare-host SLURM example documented there matches the cluster you bring
up with `make slurm-setup`.
