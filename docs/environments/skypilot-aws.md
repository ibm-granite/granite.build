# SkyPilot on AWS

> **Audience:** operators configuring a `Skypilot` environment whose `default_cloud` is `aws`.
> Read [skypilot.md](skypilot.md) first for the compute model and config common to all clouds; this
> page covers only what is AWS-specific.

## Compute environment

With `default_cloud: aws`, SkyPilot **provisions EC2 instances** in your AWS account for each step,
runs the job, and tears them down on cleanup. Unlike the SSH-provisioned HPC backends
([SLURM](skypilot-slurm.md), [LSF](skypilot-lsf.md)), there is no SSH reachability file — AWS is
API-provisioned and reached through AWS credentials.

## AWS-specific configuration

### Credentials: `aws_credentials`

SkyPilot's API server uses boto3, which reads `~/.aws/credentials`. Inline credential profiles and
gbserver materializes that file (INI, mode `0600`) at launch; SkyPilot then uploads the file to the
provisioned nodes so they can reach S3.

```yaml
config:
  default_cloud: aws
  aws_credentials:
    - profile: default              # The INI [section] name.
      aws_access_key_id: AWS_KEY_ID_SECRET      # Secret name or literal — keep these as secret names.
      aws_secret_access_key: AWS_SECRET_SECRET
      # aws_session_token: AWS_TOKEN_SECRET     # Optional.
```

Each value is resolved by exact-name lookup against the environment's secrets, falling back to the
literal; only secret *names* appear in the asset. Profiles merge by section name — an identical
pre-existing profile is a no-op, a conflicting one raises `SkypilotConfigCollisionError`, and foreign
profiles are preserved. See the shared rules in
[skypilot.md](skypilot.md#inline-skypilot-config-cluster_ssh_configs--cloud_config--aws_credentials).

> If the gbserver host already has working `~/.aws/credentials` (e.g. an instance role or pre-provisioned
> profile), you can omit `aws_credentials` entirely — the inline block is optional.

### Region and other AWS settings: `cloud_config.aws`

`aws_credentials` is **credentials only**. Region and other behavioral AWS settings go in a
`cloud_config` `aws:` block (deep-merged into `~/.sky/config.yaml`) or via `AWS_DEFAULT_REGION`:

```yaml
config:
  cloud_config:
    aws:
      # SkyPilot aws: settings, e.g. security groups, VPC, etc.
```

### Resources: instance type, spot, accelerators

AWS-relevant launcher `resources` fields:

```yaml
launchers:
  train:
    type: skypilot
    monitors:
      - skypilot_monitor
    config:
      resources:
        accelerators: A100:8       # SkyPilot picks a matching instance type (e.g. p4d).
        instance_type: p4d.24xlarge  # Optional. Pin a specific EC2 instance type.
        use_spot: true             # Optional. Use spot instances.
        disk_size: 200             # Optional. Root disk GB.
        zone: us-east-1a           # Optional. AWS availability zone.
      image_id: docker:nvcr.io/nvidia/pytorch:24.01-py3   # Containers run natively on AWS.
      run: |
        python train.py
```

### `shared_workdir`

For cross-step state, point `shared_workdir` at a path backed by **EFS / FSx** mounted on every worker
(e.g. `/mnt/efs`). See [skypilot.md](skypilot.md#shared_workdir).

## Example `environment.yaml`

```yaml
name: skypilot-aws
type: Skypilot
config:
  default_cloud: aws
  idle_minutes_to_autostop: 5       # Safety net; per-step cleanup already runs `sky down`.
  aws_credentials:
    - profile: default
      aws_access_key_id: AWS_KEY_ID_SECRET
      aws_secret_access_key: AWS_SECRET_SECRET
assetstores:
  - store_uri: space://assetstores/hf
    load:
      - mode: hf_pull
    push:
      - mode: hf_push
```

## See also

- [SkyPilot overview](skypilot.md) — compute model, launcher fields, inline-config rules
- [SkyPilot on Kubernetes](skypilot-kubernetes.md) · [SLURM](skypilot-slurm.md) · [LSF](skypilot-lsf.md)
