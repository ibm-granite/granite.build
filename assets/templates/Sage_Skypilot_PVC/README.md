# Sage LLM Evaluation — SkyPilot with PVC

Runs a Sage LLM evaluation on SkyPilot using a PersistentVolumeClaim (PVC) for output data. Results are written to the PVC, then uploaded to S3 via `s3push`.

## How It Works

| Target | Description |
|---|---|
| `evaluation` | Runs `sage lm-eval` with a HuggingFace model. Results written to PVC at `/gb-data/outputs/sage`. |
| `upload_output` | Uploads results from PVC to S3 via `s3push`. |

## Prerequisites

1. **HuggingFace token** — required for model downloads:
   ```bash
   export HF_TOKEN=hf_xxxxx
   bash docs/operators/setup/create-hf-secret.sh
   ```

2. **GitHub Enterprise token** — required for sage to clone IBM-internal repos at runtime:
   ```bash
   export GHE_TOKEN=ghp_xxxxx
   bash docs/operators/setup/create-ghe-secret.sh
   ```

3. **PVC and secrets configured in `~/.sky/config.yaml`** — The cluster-level SkyPilot config must mount the PVC at `/gb-data` and inject credentials from K8s secrets:

   ```yaml
   kubernetes:
     pod_config:
       spec:
         volumes:
           - name: gb-data
             persistentVolumeClaim:
               claimName: gb-data-pvc
         containers:
           - volumeMounts:
               - name: gb-data
                 mountPath: /gb-data
             env:
               - name: HF_TOKEN
                 valueFrom:
                   secretKeyRef:
                     name: hf-credentials
                     key: token
               - name: AWS_ACCESS_KEY_ID
                 valueFrom:
                   secretKeyRef:
                     name: s3-credentials
                     key: access-key-id
               - name: AWS_SECRET_ACCESS_KEY
                 valueFrom:
                   secretKeyRef:
                     name: s3-credentials
                     key: secret-access-key
               - name: AWS_ENDPOINT_URL
                 valueFrom:
                   secretKeyRef:
                     name: s3-credentials
                     key: endpoint-url
   ```

4. **S3 URI variables in `space/space.yaml`** — Set `S3_SAGE_OUTPUT_URI` to your bucket path.

## Usage

```bash
gbserver standalone --space-dir space/ &
gbserver build run-and-monitor assets/templates/Sage_Skypilot_PVC \
  --space-name standalone
```

## Switching to Skypilot_managed

Edit `build.yaml` and change `environment_uri` from `space://environments/skypilot` to `space://environments/skypilot-managed`. Requires a running SkyPilot API Server.

## See Also

- [SkyPilot Kubernetes Setup](../../docs/operators/setup/skypilot-kubernetes-setup.md)
- [Sage Skypilot S3 Template](../Sage_Skypilot/) — S3 file_mounts variant
