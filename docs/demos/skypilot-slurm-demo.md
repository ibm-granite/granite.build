# SLURM demo (via SkyPilot)

Runs TRL fine-tuning and unitxt evaluation on a local Docker-based SLURM cluster
via SkyPilot, with artifact push to MinIO (S3-compatible object storage).

## Prerequisites

- Docker (or Podman) with a running daemon
- Python 3.11+ (3.12 or 3.13 recommended)
- No cloud credentials needed — everything runs locally

## Setup (from scratch)

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

See [SkyPilot SLURM setup](../environments/setup/skypilot-slurm-setup.md) for details on the local
Docker SLURM cluster and MinIO, and [SkyPilot on SLURM](../environments/skypilot-slurm.md) for the
environment configuration.

## Run

```bash
# Run both TRL fine-tuning and unitxt evaluation on SLURM
bash scripts/demo-slurm.sh

# TRL fine-tuning only
bash scripts/demo-slurm.sh --trl-only

# unitxt evaluation only
bash scripts/demo-slurm.sh --unitxt-only
```

The demo submits builds that run on the SLURM cluster via SkyPilot. When
training completes, an `s3push` step automatically uploads the checkpoint to
MinIO. First run takes 5-10 minutes (SkyPilot installs dependencies on the
SLURM nodes).

## Verify artifacts in MinIO

```bash
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

# Fine-tuning checkpoint
aws --endpoint-url http://localhost:9000 s3 ls s3://gb-checkpoints/outputs/trl-finetune/ --recursive

# Evaluation results
aws --endpoint-url http://localhost:9000 s3 ls s3://gb-checkpoints/outputs/unitxt-eval/ --recursive
```

## Teardown

```bash
make slurm-teardown
make minio-teardown
```

## How it works

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

## See also

- [Demos overview](README.md)
- [Standalone Docker demo](docker-demo.md) — the same workload without a cluster
- [SkyPilot SLURM setup](../environments/setup/skypilot-slurm-setup.md) · [SkyPilot on SLURM](../environments/skypilot-slurm.md)
