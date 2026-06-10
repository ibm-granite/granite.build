# DiGiT Synthetic Data Generation — SkyPilot with S3 file_mounts

This build template generates synthetic data using DiGiT on a SkyPilot environment. Input seed data and documents are mounted from an S3/COS bucket via SkyPilot's `file_mounts`.

## How It Works

| Target | Description |
|---|---|
| `syntheticdatageneration` | Runs DiGiT with a RITS teacher model to generate synthetic Q&A data from seed data and documents. |

| Artifact | Direction | Description |
|---|---|---|
| `digit_input` | Input | S3 bucket containing `seeds.jsonl` and `documents/*.md`. Mounted to `/data/input/` via file_mounts. |
| `digit_output` | Output | S3 location where generated synthetic data is written. |

## Prerequisites

1. **S3/COS bucket** with seed data uploaded:
   ```
   s3://your-bucket/digit/input/seeds.jsonl
   s3://your-bucket/digit/input/documents/*.md
   ```
2. **AWS/COS credentials** as environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID=<hmac_access_key>
   export AWS_SECRET_ACCESS_KEY=<hmac_secret_key>
   export AWS_ENDPOINT_URL=https://s3.xxx.xxx.xxx.cloud  # for COS
   ```
3. **RITS API key:**
   ```bash
   export RITS_API_KEY=<your_key>
   ```
4. **SkyPilot** installed and configured (`sky check kubernetes` passes).

## Configuration

Edit `configurations/spaces/local/space.yaml` to set your S3 bucket URIs:

```yaml
variables:
  S3_DIGIT_INPUT_URI: "s3://your-bucket/digit/input"
  S3_DIGIT_OUTPUT_URI: "s3://your-bucket/digit/output"
```

To use SkyPilot managed mode instead of unmanaged, edit `build.yaml` and change:

```yaml
environment_uri: space://environments/skypilot-managed/kubernetes
```

## Running

```bash
gbserver standalone --space-dir configurations/spaces/local &
gbserver build run-and-monitor configurations/assets/templates/DiGiT_Skypilot \
  --space-name public
```

## Setup Reference

See [SkyPilot Kubernetes Setup](../../../../docs/operators/setup/skypilot-kubernetes-setup.md) for cluster configuration, image pull secrets, and SkyPilot API server deployment.
