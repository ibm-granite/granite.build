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

> **Not here: `profile`.** Selecting a named AWS profile is *not* a valid key in this
> global `aws:` block — SkyPilot rejects it with `Found unsupported field 'profile'` and the
> API server fails to start. Profile selection is workspace-scoped; see the runbook below.

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

## Runbook: use a non-default AWS profile via the local secret store

Use this when the gbserver host **already has a working `~/.aws/credentials` `[default]`** whose
identity differs from the one you want SkyPilot to use. Materializing into `[default]` would raise
`SkypilotConfigCollisionError`; instead materialize a **named** profile and select it. This keeps
one `environment.yaml` usable in both standalone and shared deployments — only the secret backend
differs (a local file here, a server-managed store in shared).

### 1. Declare a named profile + select it in `environment.yaml`

```yaml
config:
  default_cloud: aws
  # Materialize a NON-default profile (values are secret NAMES, resolved by the
  # space's secret_manager — never commit literals).
  aws_credentials:
    - profile: gb-skypilot
      aws_access_key_id: GB_AWS_ACCESS_KEY_ID
      aws_secret_access_key: GB_AWS_SECRET_ACCESS_KEY
  # Select that profile. `profile` is ONLY valid under workspaces.<name>.aws —
  # NOT the global aws: block (that form crashes the API server). `default` is
  # SkyPilot's default active workspace.
  cloud_config:
    workspaces:
      default:
        aws:
          profile: gb-skypilot
```

### 2. Seed the local secret store (standalone)

With `secret_manager: type: local`, secrets are read from `$GB_HOME_DIR/space_secrets/` (default
`~/.granite.build/space_secrets/`). Files may be `.json`/`.yaml`/`.env`; **values are base64-encoded**
and looked up by exact name. Write a file whose keys match the secret names above:

```bash
mkdir -p ~/.granite.build/space_secrets
python3 - <<'PY'
import os, base64, json, pathlib
d = pathlib.Path.home() / ".granite.build" / "space_secrets"; d.mkdir(parents=True, exist_ok=True)
enc = lambda v: base64.b64encode(v.encode()).decode()
p = d / "aws.json"
p.write_text(json.dumps({
    "GB_AWS_ACCESS_KEY_ID":     enc(os.environ["AWS_ACCESS_KEY_ID"]),
    "GB_AWS_SECRET_ACCESS_KEY": enc(os.environ["AWS_SECRET_ACCESS_KEY"]),
}, indent=2))
p.chmod(0o600)
PY
```

> The `env` secret manager (`type: env`) is an alternative: it reads `GBSERVER_SECRET_<NAME>`
> env vars (prefixed, upper-cased) — note it does **not** read a bare `AWS_ACCESS_KEY_ID`.

### 3. Verify cheaply (no EC2)

Confirm the profile is materialized and actually used, **with the ambient AWS env vars unset** so a
pass proves the profile — not your shell — supplied the credentials:

```bash
grep -A2 '^\[gb-skypilot\]' ~/.aws/credentials    # written by gbserver at launch
sky api stop
env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY sky check aws   # expect: AWS: enabled
```

### 4. Run the build standalone

```bash
export GB_ENVIRONMENT=STANDALONE GBTEST_MODE=live
# e.g. the fixture test that provisions one t3.medium in us-east-2:
pytest -s -m extended --strict-markers \
  test/integration/ibm/buildrunner/skypilot/aws/test_1step_image.py
```

### How it works / gotchas

- **Explicit profile wins over env vars.** `cloud_config.workspaces.default.aws.profile` makes
  SkyPilot call `boto3.Session(profile_name=…)`, which (being an *explicit* profile) removes the env
  provider from botocore's chain — so `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` in the shell are
  ignored while the profile is set.
- **`profile` is workspace-scoped only.** The global `aws:` block rejects it
  (`unsupported field 'profile'`) and the API server exits on startup.
- **Stale `~/.sky/config.yaml`.** `cloud_config` is deep-merged, not replaced. If a bad global
  `aws: {profile: …}` was written by an earlier attempt, delete `~/.sky/config.yaml` so it is
  regenerated cleanly from `environment.yaml`.
- **Collision safety.** Materialization refuses to overwrite an existing profile whose values
  differ; pick a name (e.g. `gb-skypilot`) you don't already have in `~/.aws/credentials`.
- **Field names are exact; typos are silent.** `aws_credentials` entries are validated
  permissively, so a misspelled key (e.g. `access_key_id` instead of `aws_access_key_id`) is
  dropped rather than rejected and the value stays unset — surfacing later as a confusing
  credential-resolution failure, not a config error. Use the field names exactly: `profile`,
  `aws_access_key_id`, `aws_secret_access_key`, `aws_session_token`.
- **Region is separate.** Placement comes from the launcher `resources.infra` (e.g. `aws/us-east-2`),
  not from this profile.

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
    pull:
      - mode: default
    push:
      - mode: default
```

## See also

- [SkyPilot overview](skypilot.md) — compute model, launcher fields, inline-config rules
- [SkyPilot on Kubernetes](skypilot-kubernetes.md) · [SLURM](skypilot-slurm.md) · [LSF](skypilot-lsf.md)
