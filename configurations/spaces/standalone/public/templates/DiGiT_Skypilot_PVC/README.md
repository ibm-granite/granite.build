# DiGiT Skypilot PVC Template

Runs DiGiT synthetic data generation on SkyPilot using a PersistentVolumeClaim (PVC) for input data. The PVC is mounted cluster-wide via `~/.sky/config.yaml` `pod_config`, so no `file_mounts` are needed in this template.

## Prerequisites

1. **Kubernetes secrets** — Create secrets for S3/COS access and the RITS API key:

   ```bash
   kubectl create secret generic s3-credentials \
     --from-literal=access-key-id=<YOUR_ACCESS_KEY> \
     --from-literal=secret-access-key=<YOUR_SECRET_KEY> \
     --from-literal=endpoint-url=<YOUR_S3_ENDPOINT>

   kubectl create secret generic rits-credentials \
     --from-literal=api-key=<YOUR_RITS_API_KEY>
   ```

2. **PVC and secrets configured in `~/.sky/config.yaml`** — The cluster-level SkyPilot config must mount the PVC at `/gb-data` and inject credentials from K8s secrets:

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
               - name: RITS_API_KEY
                 valueFrom:
                   secretKeyRef:
                     name: rits-credentials
                     key: api-key
   ```

3. **Seed data staged on the PVC** — Copy seed data to the PVC before running:
   - `/gb-data/inputs/digit/seeds.jsonl`
   - `/gb-data/inputs/digit/documents/*.md`

4. **S3 URI variables in `configurations/spaces/standalone/public/space.yaml`** — Set `S3_DIGIT_INPUT_URI` and `S3_DIGIT_OUTPUT_URI` to your bucket paths.

## Usage

```bash
gbserver standalone --space-dir configurations/spaces/standalone/public &
gbserver build run-and-monitor configurations/spaces/standalone/public/templates/DiGiT_Skypilot_PVC \
  --space-name public
```

## Switching to Skypilot_managed

Edit `build.yaml` and change `environment_uri` from `space://environments/skypilot` to `space://environments/skypilot-managed/kubernetes`. Requires a running SkyPilot API Server.

## See Also

- [SkyPilot Kubernetes Setup](../../docs/operators/setup/skypilot-kubernetes-setup.md)
- [DiGiT Skypilot S3 Template](../DiGiT_Skypilot/) — S3 file_mounts variant
