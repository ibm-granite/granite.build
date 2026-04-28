# Sage LLM Evaluation — SkyPilot with S3 file_mounts

This build template runs a Sage LLM evaluation on a SkyPilot environment. Evaluation results are written to an S3/COS bucket via SkyPilot's `file_mounts`.

## How It Works

| Target | Description |
|---|---|
| `evaluation` | Runs `sage lm-eval` with a HuggingFace model on a specified benchmark. Results are written to the mounted S3 output path. |

| Artifact | Direction | Description |
|---|---|---|
| `sage_output` | Output | JSON results from the evaluation, written to S3 via file_mounts at `/data/output`. |

## Prerequisites

1. **HuggingFace token** — required for model downloads:
   ```bash
   export HF_TOKEN=hf_xxxxx
   bash docs/setup/create-hf-secret.sh
   ```
2. **GitHub Enterprise token** — required for sage to clone IBM-internal repos at runtime:
   ```bash
   export GHE_TOKEN=ghp_xxxxx
   bash docs/setup/create-ghe-secret.sh
   ```
3. **S3/COS bucket** for evaluation results.
4. **AWS/COS credentials** as environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID=<hmac_access_key>
   export AWS_SECRET_ACCESS_KEY=<hmac_secret_key>
   export AWS_ENDPOINT_URL=https://s3.xxx.xxx.xxx.cloud  # for COS
   ```
5. **SkyPilot** installed and configured (`sky check kubernetes` passes).

## Configuration

Edit `space/space.yaml` to set your S3 output URI:

```yaml
variables:
  S3_SAGE_OUTPUT_URI: "s3://your-bucket/sage/output"
```

The default evaluation runs `piqa` (10 samples) with `ibm-granite/granite-4.0-350m` on CPU. Override via `sage_config` in `build.yaml`.

To use SkyPilot managed mode instead of unmanaged, edit `build.yaml` and change:

```yaml
environment_uri: space://environments/skypilot-managed
```

## Running

```bash
gbserver standalone --space-dir space/ &
gbserver build run-and-monitor assets/templates/Sage_Skypilot \
  --space-name standalone
```

## Setup Reference

See [SkyPilot Kubernetes Setup](../../docs/setup/skypilot-kubernetes-setup.md) for cluster configuration, image pull secrets, and SkyPilot API server deployment.
