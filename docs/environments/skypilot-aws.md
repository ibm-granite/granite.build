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

> **`sbatch_options` is a no-op on AWS.** The per-step `sbatch_options` field
> ([skypilot.md](skypilot.md#config-overrides-docker-sbatch_options)) is a
> **SLURM-only** knob; SkyPilot exposes no per-task equivalent on AWS, so a value
> set here is ignored (a WARNING is logged). Bound a job's runtime inside the
> `run:` command instead, with `idle_minutes_to_autostop` as a crash safety net.

### `shared_workdir`

For cross-step state, point `shared_workdir` at a path backed by **EFS / FSx** mounted on every worker
(e.g. `/mnt/efs`). See [skypilot.md](skypilot.md#shared_workdir).

Containers run natively on AWS (Docker on the VM), so a step with an `image_id` still needs the shared
mount visible **inside** the container, not just on the host — otherwise its output lands in the
container's ephemeral layer and the downstream `hfpush` can't see it (the general caveat in
[skypilot.md](skypilot.md#containerized-steps-must-also-see-the-shared-workdir-inside-the-container)).
Ensure the EFS/FSx mount is exposed to the container (e.g. as a Docker bind/volume) so the per-run
workdir resolves the same path on the host and in the container.

## `shared_filesystem` (auto-mounting EFS)

`shared_workdir` above assumes *you* mount the EFS on every worker. `shared_filesystem` (issue #378)
instead has **gbserver mount** a BYO, pre-provisioned EFS at launch (at `mount_point`) — the per-step
EC2 instances SkyPilot allocates then share state with no manual mount. It is **aws-only**.
`shared_filesystem` defines **only the mount**; you must *also* set `shared_workdir`, which is
**required** here and must be an absolute path equal to `mount_point` or a subdirectory of it. The
workdir lives at that path on the mount, and (per
[#404](https://github.com/ibm-granite/granite.build/issues/404)) its prefix will later select which
filesystem once multiple are supported. `EnvironmentConfig` validation rejects a `shared_filesystem`
with no `shared_workdir`, or a `shared_workdir` outside `mount_point`. For the common-schema view see
[skypilot.md](skypilot.md#shared_filesystem).

```yaml
config:
  default_cloud: aws
  shared_workdir: /mnt/gb-shared/gbroot   # Required with shared_filesystem; must be mount_point or a subdir of it.
  shared_filesystem:
    provider: efs
    mount_point: /mnt/gb-shared   # Mounted on each worker.
    efs:
      file_system_id: fs-0abc123
      region: us-east-1           # Must match the workers' region (mount targets are AZ-scoped).
      tls: true
      # cleanup_zone: us-east-1a  # Optional. Pin the teardown VM to a mount-target AZ.
assetstores:
  - store_uri: space://assetstores/hf
    pull:  [{ mode: default, config: {} }]   # no cache_path/inline: cache on the shared FS
    push:  [{ mode: default, config: {} }]
```

The filesystem itself is **bring-your-own** — provision it *once* with the
[runbook below](#runbook-provision-a-shared-efs-filesystem) (SG for NFS 2049, elastic-throughput
`create-file-system`, a mount target per worker AZ, and the one-time `chmod 1777` root bootstrap).
gbserver never creates or deletes the filesystem; this section covers only how it is *used* at build
time.

### aws-only gate

`shared_filesystem` is supported only on a `Skypilot`/`aws` environment. A `shared_filesystem` block on
any non-`aws` (or non-`Skypilot`) environment is **rejected at config load** — `EnvironmentConfig`
validation raises a `ValueError` rather than silently ignoring it at runtime. It also requires
`config.default_cloud: aws` (the value the per-run mount and teardown VM actually key on, defaulting to
`k8s` when unset) — a `subtype: aws` env whose `default_cloud` is anything else is rejected too, so the
mount never targets a cloud without a mount target. On other backends use an operator-mounted
[`shared_workdir`](skypilot.md#shared_workdir) instead.

### Multiple shared filesystems

`shared_filesystem` may be a **list** of mounts as well as a single object (a lone object is treated
as a one-element list, so existing configs keep working — [#404](https://github.com/ibm-granite/granite.build/issues/404)).
Each entry is an independent `{ provider, mount_point, efs, local_scratch? }` mount; gbserver mounts
every one on each worker. Cross-mount rules, enforced at config load:

- `mount_point`s must be **unique** and **non-nested** (no mount under another).
- At most **one** mount may set `local_scratch` (it is instance-local, not per-filesystem).
- `shared_workdir` must sit under **exactly one** mount's `mount_point`; that mount hosts the per-run
  workdir (and its `1777` walk is bounded by *its* root). The other mounts are just mounted, for the
  step to read/write as it likes.

```yaml
config:
  default_cloud: aws
  shared_workdir: /mnt/gb-shared/gbroot      # under the first mount below
  shared_filesystem:
    - provider: efs                          # BYO durable share (hosts the workdir)
      mount_point: /mnt/gb-shared
      efs: { file_system_id: fs-0abc123, region: us-east-1 }
    - provider: efs                          # a second, e.g. ephemeral, mount
      mount_point: /mnt/gb-scratch
      efs: { provision: ephemeral, region: us-east-1 }
```

### Ephemeral auto-provisioned EFS (`provision: ephemeral`)

Set `efs.provision: ephemeral` (default is `byo`) to have **gbserver create the filesystem at
target-run setup and destroy it at teardown** ([#391](https://github.com/ibm-granite/granite.build/issues/391)),
instead of referencing a pre-provisioned one. An ephemeral `efs` block **must not** set
`file_system_id`/`dns_name` (there is nothing to reference yet) and **requires `region`**; optional
`vpc_id`, `subnets`, and `security_group_id` pin the networking (otherwise the region's default VPC,
all its subnets, and a fresh gbserver-created security group are used). At setup gbserver creates an
encrypted, elastic-throughput filesystem tagged `app=granite.build`, `gb-ephemeral=true`,
`gb-build-id`, `gb-targetrun-id`, `gb-created-at`, a mount target per subnet, and (unless a BYO
`security_group_id` is given) a security group allowing NFS (TCP 2049) from the VPC CIDR; the runtime
DNS name is threaded to the workers' mount. Because a fresh EFS root is `root:root 0755`, the ephemeral
mount prologue `chmod 1777`s it so non-root steps can create the per-run workdir.

**IAM split.** BYO needs **no** gbserver AWS permissions (a plain NFS mount). Ephemeral needs the
gbserver identity (the `cloud_config`/`aws_credentials` profile) to hold:
`elasticfilesystem:{Create,Delete,Describe}{FileSystem,MountTarget}`,
`ec2:{CreateSecurityGroup,DeleteSecurityGroup,AuthorizeSecurityGroupIngress,DescribeVpcs,DescribeSubnets,DescribeSecurityGroups}`,
plus resource tagging. Grant these only where ephemeral EFS is used.

**Leak / cost / reclamation.** Teardown deletes the mount targets, the filesystem, and (only) a
gbserver-created security group. A crashed/killed gbserver, or a teardown-time API failure, can
**orphan** the filesystem — teardown logs a `WARNING` naming the `fsid`/`sg`/tags so it can be found.
There is **no automatic TTL sweeper** (deferred); reclaim orphans manually by the `gb-ephemeral=true`
tag, e.g. `aws efs describe-file-systems` filtered on that tag, then delete mount targets → filesystem.
Elastic throughput bills per byte moved, so an orphan costs until reaped — keep hot IO on
`GB_LOCAL_SCRATCH`, not EFS.

**Region pin.** As with BYO, an ephemeral EFS is region/VPC-scoped: pin the env (and `efs.region`) to
the region the workers run in (e.g. `us-east-1`) so the mount targets are reachable from the instances'
AZs.

### Per-run workdir and permissions (`1777`)

gbserver mounts the EFS at `mount_point` on every worker and creates the same per-target-run subdir it
would under any `shared_workdir` — `${shared_workdir}/builds/<build_id>/runs/<targetrun_id>/` (with
`shared_workdir` under `mount_point`, e.g. `/mnt/gb-shared/gbroot`) — as the CWD of
each step's `setup`/`run`. Because steps run as a non-root user, the **EFS root must be `chmod 1777`**
(sticky, like `/tmp`); gbserver then makes every level from the root down to the per-run dir `1777` too
(guarded, so a step running as a different uid than the one that created a parent does not EPERM/abort),
so a later step on a separate instance — as any uid — can create and traverse its own per-run dir. A
producer step's output is world-readable to the consumer (default umask), and the sticky bit prevents
cross-deletion. Two steps that *rewrite the same file* as different uids still need a shared uid/gid.

> **Trust boundary.** A `1777` root plus a `1777` per-run tree means every concurrent build on this
> filesystem can **read and write** every other build's per-run tree; the sticky bit stops only
> cross-*deletion*, not cross-read/write. That is a shared-scratch model appropriate for a single trusted
> team/space. For stronger isolation, provision the EFS with **access points** (`PosixUser` +
> `RootDirectory` per space), which also removes the `chmod` bootstrap entirely.
>
> This `1777` shared-scratch model applies only to mounts without an
> `access_point_id`. Configure a per-space EFS access point (see
> "Per-space isolation with EFS access points" below) to give each space its own
> uid/gid-pinned root with no world-writable bootstrap. An access point alone does
> not *enforce* the boundary: a client that can reach a mount target can still
> mount the filesystem root over NFS or name another space's access point. See the
> enforcement note in that section.

### Per-space isolation with EFS access points (recommended for multi-tenant BYO)

By default a BYO EFS mount uses a world-writable, sticky (`1777`) per-run tree so
cross-uid, cross-instance steps can create their workdirs. On a filesystem shared
by many builds that is a shared-scratch trust model: every build can read and
write every other build's per-run tree (the sticky bit only blocks cross-deletion).

To isolate each space, provision one **EFS access point** per space and name it in
that space's `environment.yaml`:

1. Create an access point on the shared filesystem with a fixed `PosixUser`
   (a uid/gid dedicated to the space), a per-space `RootDirectory` (e.g.
   `/spaces/<space>`), and `CreationInfo` (owner uid/gid + `0700`/`0770`
   permissions) so EFS auto-creates the root owned by that uid/gid:

   ```
   aws efs create-access-point \
     --file-system-id fs-0abc123 \
     --posix-user Uid=<uid>,Gid=<gid> \
     --root-directory 'Path=/spaces/<space>,CreationInfo={OwnerUid=<uid>,OwnerGid=<gid>,Permissions=0770}' \
     --tags Key=space,Value=<space>
   ```

2. Put the returned `AccessPointId` in the space's `environment.yaml`:

   ```yaml
   shared_filesystem:
     provider: efs
     mount_point: /mnt/gb-shared
     efs:
       file_system_id: fs-0abc123
       region: us-east-1
       access_point_id: fsap-0abc123
       tls: true
   ```

**Requirements and behavior**

- The worker host/image MUST ship `amazon-efs-utils` (`mount.efs`): access points
  are mounted with `mount -t efs -o accesspoint=<id>,tls`, and plain `nfs4` cannot
  select an access point. If `mount.efs` is absent the step fails fast with a clear
  message (there is no `nfs4`/cleartext fallback). Containerized steps therefore
  need an image that bundles `amazon-efs-utils`.
- The access point's fixed `PosixUser` pins a stable uid/gid across steps and
  instances, so gbserver no longer applies the `chmod 1777` bootstrap or per-run
  chmod walk for that mount.
- IAM: creating/managing access points is an operator action
  (`elasticfilesystem:CreateAccessPoint`/`DeleteAccessPoint`/`DescribeAccessPoints`);
  gbserver needs no AWS permissions for BYO access points — it only mounts through
  the configured id.
- `access_point_id` is BYO-only; it is rejected for `provision: ephemeral`
  (ephemeral EFS is single-tenant and needs no access point). It must also match
  the AWS `fsap-<hex>` format (lowercase hex), or validation rejects it.
- `tls` must stay `true` (the default). amazon-efs-utils refuses `accesspoint`
  without `tls`, so validation rejects `tls: false` together with
  `access_point_id`.
- Teardown: the throwaway cleanup VM that reaps the per-run tree runs SkyPilot's
  default image, which has no `mount.efs`. For an access-point mount gbserver
  therefore installs amazon-efs-utils v1.35.2 on that VM before reaping: built
  from `github.com/aws/efs-utils` on apt hosts (the clone must resolve to the
  pinned v1.35.2 commit, or the reap aborts before building), or via `yum
  install amazon-efs-utils` otherwise. That VM needs outbound access to the
  distro mirrors and `github.com`. This applies to the gbserver-owned cleanup VM
  only; worker steps never auto-install.
- **Not yet supported: hidden pull/push steps.** buildrunner auto-queues hidden
  steps for non-environment inputs/outputs (e.g. `hf://` → hfpull/hfpush,
  `s3://` → s3pull/s3push). These run on images without `mount.efs` (the bare
  default image, or `amazon/aws-cli`), so with an access point on the workdir
  mount they fail fast. Until that is addressed
  ([#458](https://github.com/ibm-granite/granite.build/issues/458)), use access
  points only for builds whose steps all run in an image that bundles
  amazon-efs-utils.
- **Enforcing the boundary.** An access point pins the uid/gid and the root
  directory for clients that mount *through it*. By itself it does not stop a
  client that can reach a mount target from mounting the filesystem root over
  NFS, or from naming another space's access point. gbserver mounts without IAM
  authorization today (no `-o iam`), so limit who can reach the filesystem:
  - restrict the mount targets' security group to NFS (TCP 2049) from the build
    workers' security group only. This keeps hosts outside the build fleet from
    mounting the filesystem, but not other spaces' builds on the same fleet;
  - keep each space's `environment.yaml` operator-controlled, so a space cannot
    point itself at another space's access point.

  AWS documents the `elasticfilesystem:AccessPointArn` file-system-policy
  condition (allow `ClientMount`/`ClientWrite` only through a given access point)
  for IAM-authorized clients. Enforcing per-space access with a file-system
  policy therefore needs IAM mount authorization (`-o iam`) with per-space roles,
  which gbserver does not emit yet
  ([#458](https://github.com/ibm-granite/granite.build/issues/458)).

### Containerized steps

A step with an `image_id` runs in a container on the EC2 host, yet still sees the EFS mount because
SkyPilot launches its containers with host networking, `--cap-add=SYS_ADMIN`, `--device=/dev/fuse`, and
`--security-opt apparmor:unconfined` — so an in-container `mount -t nfs4` is permitted (past both seccomp
*and* AppArmor) and reaches the mount target as the host IP (covered by the VPC-CIDR SG rule). gbserver
does **not** add any container `run_options` of its own — it relies on those SkyPilot defaults (pinning
`--net=host` would duplicate SkyPilot's and make `docker run` fail; the flag that actually clears the
mount, `--security-opt apparmor:unconfined`, is SkyPilot's too). If a future SkyPilot bump drops them,
pin the needed dup-tolerant ones (not `--net=host`) in `docker.run_options`. The in-container mount runs
sudo-free as root. Image
requirements: an **NFS client** (`nfs-common`/`nfs-utils`) and `mountpoint` (util-linux) — gbserver
installs the NFS client best-effort via the image's package manager (using `sudo` only when not root),
so a slim image needs a package manager; prefer an image that already ships the client for
offline/locked-down bases. **Validated on real AWS** (a 2-step producer→consumer build in a container).

### `GB_LOCAL_SCRATCH` and hot-path staging

EFS is the durable **hand-off medium** between steps, not fast scratch — every byte read/written bills
under elastic throughput. Each step also gets an instance-local `GB_LOCAL_SCRATCH`: **stage hot paths
(checkpoints, decompress/scratch) there** and copy only the durable result back to the per-run workdir.
It defaults to `/tmp/gb-scratch`, which on stock AWS/DLAMI images is the **EBS root volume, not
instance-store NVMe** — set `shared_filesystem.local_scratch` (validated, must be absolute) to the
image's NVMe mount (e.g. `/opt/dlami/nvme/...`) if you want true local-NVMe scratch. Keeping churn off
EFS bounds latency and cost.

### hf cache

When `shared_filesystem` is enabled, drop `cache_path: /tmp/hf_cache` and `inline: true` from the hf
assetstore (use `config: {}`, as above), so `hfpull` runs as its own step and caches to
`${mount_point}/hf_cache` — otherwise it caches instance-locally and the model never reaches EFS.

### Cleanup-zone pinning (`efs.cleanup_zone`)

At target-run teardown gbserver launches a small VM to `rm -rf` the per-run tree. That VM must land in
an AZ that has a mount target; set `efs.cleanup_zone` to a mount-target AZ (e.g. `us-east-1a`) to pin
it when the default placement might pick an AZ without one.

### GC / quota / cost

- **What teardown reaps.** Teardown runs **per target-run**: gbserver `rm -rf`'s that run's per-run dir,
  then does a best-effort `rmdir` of the now-empty `runs/` and `builds/<build_id>/` parents — a parent
  is removed **only if it is now empty** (never a build-completion `rm -rf` of the whole build tree), so
  concurrent runs under the same build are left intact. Teardown waits for the reap job to finish
  (up to 5 min, or 10 min for access-point mounts, which build amazon-efs-utils first) so a failed
  reap is logged as an `ORPHANED` tree; the build status is already final by then. A reap still running
  at the limit logs a `WARNING` that the tree may be orphaned. Retries get a fresh dir. Crashes or
  killed servers can still orphan trees, and `hf_cache/` is intentionally **not** reaped (it is a
  shared cache), so it grows unbounded.
- **Operator hygiene.** Run a **TTL sweeper** over `builds/<id>/` for crash-orphans, cap per-space
  usage, and **monitor `hf_cache/`** size — none of these are automatic.
- **Cost.** Empty ≈ $0, but Elastic throughput bills **per byte moved** (~$0.03/GB read, ~$0.06/GB
  write) on top of per-GB storage, so a leaked ~200 GB tree runs roughly **$60/mo** until swept. This
  is why hot IO belongs on `GB_LOCAL_SCRATCH`, not EFS. See the runbook's
  [Notes](#notes) for storage/throughput rates and the lifecycle-policy knob.

## Runbook: provision a shared EFS filesystem

A one-time, **bring-your-own** setup that creates the EFS the workers mount for cross-step state.
gbserver does **not** create or mount the filesystem — this is admin infra you provision once and then
reference from `environment.yaml`. It is the prerequisite for an EFS-backed `shared_workdir` (and for
the auto-mounting `shared_filesystem` provider, [issue #378](https://github.com/ibm-granite/granite.build/issues/378)).

**Prerequisites.** The AWS CLI configured with an identity that can create EFS + EC2 networking
resources — `elasticfilesystem:{Create,Describe}FileSystem/MountTarget`, `ec2:CreateSecurityGroup`,
`ec2:AuthorizeSecurityGroupIngress`, `ec2:Describe{Vpcs,Subnets}`. This is heavier than the build-time
mount (a plain NFS mount needs **no** AWS creds), so use an operator/admin identity. Pick the region
your workers run in and pin it:

```bash
export REGION=us-east-1 PROFILE=gb-skypilot   # PROFILE = an identity with the perms above
```

> **The EFS is region- and VPC-scoped.** An instance can only mount a target in **its own AZ**, and
> the mount targets live in one VPC. So the workers must launch in the *same region and VPC* — pin the
> env to `$REGION` (SkyPilot is otherwise unpinned and may pick another region, where the mount targets
> are unreachable).

### 1. Find the worker VPC + one subnet per AZ

```bash
aws ec2 describe-vpcs --region "$REGION" --profile "$PROFILE" --filters Name=isDefault,Values=true \
  --query 'Vpcs[].{VpcId:VpcId,Cidr:CidrBlock}' --output table
aws ec2 describe-subnets --region "$REGION" --profile "$PROFILE" \
  --filters Name=vpc-id,Values=vpc-XXXX \
  --query 'Subnets[].{SubnetId:SubnetId,AZ:AvailabilityZone}' --output table
```

### 2. Create a security group allowing NFS (TCP 2049)

The EFS mount-target SG must accept 2049 from the workers. SkyPilot creates its own per-launch SG we
can't predict, so allow the VPC CIDR (simple, VPC-scoped):

```bash
SG=$(aws ec2 create-security-group --region "$REGION" --profile "$PROFILE" \
       --group-name gb-efs-sg --description "granite.build shared EFS (NFS 2049)" \
       --vpc-id vpc-XXXX --query GroupId --output text)
aws ec2 authorize-security-group-ingress --region "$REGION" --profile "$PROFILE" \
  --group-id "$SG" --protocol tcp --port 2049 --cidr 172.31.0.0/16      # <- your VPC CIDR
```

### 3. Create the EFS filesystem

Use **elastic** throughput so an idle FS has no standing throughput charge (never
`--throughput-mode provisioned` for this use); encrypt at rest:

```bash
FS=$(aws efs create-file-system --region "$REGION" --profile "$PROFILE" \
       --performance-mode generalPurpose --throughput-mode elastic --encrypted \
       --tags Key=Name,Value=gb-shared Key=app,Value=granite.build \
       --query FileSystemId --output text)
# wait until available (a few seconds)
until [ "$(aws efs describe-file-systems --file-system-id "$FS" --region "$REGION" --profile "$PROFILE" \
             --query 'FileSystems[0].LifeCycleState' --output text)" = available ]; do sleep 3; done
```

### 4. Create one mount target per worker AZ

Mount targets are free; covering all AZs means any worker AZ can mount (no AZ pinning needed):

```bash
for s in subnet-AZ1 subnet-AZ2 subnet-AZ3 ... ; do
  aws efs create-mount-target --region "$REGION" --profile "$PROFILE" \
    --file-system-id "$FS" --subnet-id "$s" --security-groups "$SG" \
    --query '[MountTargetId,AvailabilityZoneName,LifeCycleState]' --output text
done
```

### 5. Wait until every mount target is `available` (async, ~1–2 min)

```bash
aws efs describe-mount-targets --file-system-id "$FS" --region "$REGION" --profile "$PROFILE" \
  --query 'MountTargets[].[AvailabilityZoneName,LifeCycleState]' --output text
```

### 6. Bootstrap the root permissions to `1777`

A fresh EFS root is `root:root 0755`, so a non-root step's `mkdir` of the per-run workdir would fail
with `EACCES`. Fix it once by mounting from an **in-VPC instance** and `chmod 1777` (like `/tmp`: any
uid can create the per-run dir, the sticky bit prevents cross-deletion). Your laptop can't reach the
mount target, so use a one-off `sky launch` (auto-terminates):

```bash
sky launch -c gb-efs-bootstrap --cloud aws --region "$REGION" --instance-type t3.small -y --down \
  "sudo mkdir -p /mnt/gb-shared \
   && { command -v mount.nfs4 >/dev/null 2>&1 || { sudo apt-get update -qq && sudo apt-get install -y -qq nfs-common; }; } \
   && sudo mount -t nfs4 -o nfsvers=4.1 ${FS}.efs.${REGION}.amazonaws.com:/ /mnt/gb-shared \
   && sudo chmod 1777 /mnt/gb-shared \
   && ls -ld /mnt/gb-shared"          # expect: drwxrwxrwt
```

### 7. Validate cross-instance access (optional but recommended)

Prove the hand-off works across *separate* instances — write on one, read on another:

```bash
sky launch -c gb-efs-w --cloud aws --region "$REGION" --instance-type t3.small -y --down \
  "sudo mkdir -p /mnt/gb-shared && sudo mount -t nfs4 -o nfsvers=4.1 ${FS}.efs.${REGION}.amazonaws.com:/ /mnt/gb-shared \
   && echo hello | tee /mnt/gb-shared/probe"
sky launch -c gb-efs-r --cloud aws --region "$REGION" --instance-type t3.small -y --down \
  "sudo mkdir -p /mnt/gb-shared && { command -v mount.nfs4 || { sudo apt-get update -qq && sudo apt-get install -y -qq nfs-common; }; } \
   && sudo mount -t nfs4 -o nfsvers=4.1 ${FS}.efs.${REGION}.amazonaws.com:/ /mnt/gb-shared \
   && test \"\$(cat /mnt/gb-shared/probe)\" = hello && echo CROSS_INSTANCE_OK"
sky status --refresh    # confirm no clusters remain (each used --down); sky down <name> if any linger
```

> **Watch for leaked EC2.** The `--down` flag terminates each probe cluster after its job; still run
> `sky status --refresh` and `sky down <name>` for any that linger — a leaked instance bills per hour
> (far more than the near-$0 empty EFS).

### 8. Reference it from `environment.yaml`

- **EFS-backed `shared_workdir`** (today): mount the EFS on every worker at, say, `/mnt/gb-shared`, and
  set `shared_workdir: /mnt/gb-shared` (see the [`shared_workdir`](#shared_workdir) note above).
- **Auto-mounting `shared_filesystem`** (issue #378): once landed, reference the FS directly and drop
  the instance-local hf cache so `hfpull` caches to the shared FS:

  ```yaml
  config:
    shared_workdir: /mnt/gb-shared/gbroot   # Required; must be mount_point or a subdir of it.
    shared_filesystem:
      provider: efs
      mount_point: /mnt/gb-shared
      efs: { file_system_id: fs-XXXX, region: us-east-1, tls: true }
  assetstores:
    - store_uri: space://assetstores/hf
      pull:  [{ mode: default, config: {} }]   # no cache_path/inline: cache on the shared FS
      push:  [{ mode: default, config: {} }]
  ```

### Decommission

Delete mount targets first, then the filesystem (also delete `gb-efs-sg` if unused):

```bash
for mt in $(aws efs describe-mount-targets --file-system-id "$FS" --region "$REGION" --profile "$PROFILE" \
              --query 'MountTargets[].MountTargetId' --output text); do
  aws efs delete-mount-target --region "$REGION" --profile "$PROFILE" --mount-target-id "$mt"
done
aws efs delete-file-system --region "$REGION" --profile "$PROFILE" --file-system-id "$FS"
```

### Notes

- **Cost.** Empty ≈ $0. EFS bills per **GB stored** (~$0.30/GB-mo Standard; ~$0.016/GB-mo IA) and
  Elastic throughput bills **per byte moved** (~$0.03/GB read, ~$0.06/GB write). Mount targets are
  free. Keep hot IO (checkpoints, scratch) on instance-local scratch (`GB_LOCAL_SCRATCH`; the EBS root
  volume by default, configurable via `local_scratch`) and use EFS only for the durable hand-off; set a
  lifecycle policy (`put-lifecycle-configuration TransitionToIA=AFTER_30_DAYS`) and sweep stale
  `builds/<id>/` trees to avoid a leaked large tree costing indefinitely.
- **Containerized steps: validated on real AWS.** SkyPilot runs containers with host networking,
  `--cap-add=SYS_ADMIN`, `--device=/dev/fuse`, and `apparmor:unconfined`, so an in-container
  `mount -t nfs4` is permitted and reaches the mount target as the host IP (covered by the VPC-CIDR SG
  rule). The image needs an NFS client (`nfs-common`/`nfs-utils`) + `mountpoint`; the mount runs
  sudo-free as root.
- **uid stability & trust.** gbserver makes the per-run tree `1777` from the EFS root down, so a
  different-uid step on a separate instance can create/traverse its own per-run dir; producer→consumer
  works (outputs world-readable via the default umask; sticky prevents cross-delete). Two steps that
  *rewrite the same file* as different uids still need a shared uid/gid. Trust boundary: every build on
  the filesystem can read/write every other's per-run tree — use EFS **access points** per space for
  stronger isolation.

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

## Quickstart: run the aws step tests for the first time

The per-cluster **step tests** (`steps/{byoc,eval,dpk}/skypilot/test/aws/…`) provision a real
EC2 instance and are gated so they never launch without credentials. Starting from just an AWS
access-key pair, with the repo `.venv` built (`make venv` at the repo root):

1. **Give the build its launch credentials.** The committed `environment.yaml` selects the
   `gb-skypilot` profile, so your key pair needs to reach that profile. Pick one path:

   **A — simplest (local/standalone): add the profile to `~/.aws/credentials` by hand.**

   ```ini
   [gb-skypilot]
   aws_access_key_id = AKIA...
   aws_secret_access_key = ...
   ```

   gbserver leaves an existing `gb-skypilot` profile as-is (the `GB_AWS_*` secrets are lenient
   when absent) and SkyPilot reads it. Do **not** *also* seed the secret store (path B) with
   different values — a mismatch raises `SkypilotConfigCollisionError`.

   **B — portable (standalone *and* shared): seed the secret store.** gbserver materializes the
   `gb-skypilot` profile from the `GB_AWS_*` secrets at launch, so the *same* `environment.yaml`
   also works in a server deployment (where the server-managed store supplies them):

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
   }, indent=2)); p.chmod(0o600)
   PY
   ```

2. **Set the skip-gate** so the test runs instead of self-skipping — export **either**
   `AWS_PROFILE=gb-skypilot` **or** the `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` pair. A bare
   `~/.aws/credentials` `[default]` does *not* satisfy the gate on its own.

3. **Verify the credentials without provisioning EC2:**

   ```bash
   sky api stop
   env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY sky check aws   # expect: AWS: enabled
   ```

   Stripping the env vars proves the `gb-skypilot` profile — not your shell — supplies the creds.

4. **Run a fixture** — this provisions a real EC2 instance, runs the build, and tears it down:

   ```bash
   AWS_PROFILE=gb-skypilot make -C steps/dpk/skypilot test TEST_DIR=test/aws-tok
   # or byoc, and the heavier dpk pii fixture:
   AWS_PROFILE=gb-skypilot make -C steps/dpk/skypilot test TEST_DIR=test/aws-pii
   ```

   `make test` forces pytest's `-s` (required — a second SkyPilot launch in a captured process
   hits `OSError: [Errno 9] Bad file descriptor`). Reaching SUCCESS proves the step ran end to
   end on EC2. (`eval` is a custom-image step: publish its image first — see
   `steps/eval/skypilot/README.md`.)

5. **Confirm nothing leaked:**

   ```bash
   sky status --refresh    # expect: No existing clusters
   ```

Region/zone come from the launcher `resources` (`infra`, e.g. `aws/us-east-2`) or
`AWS_DEFAULT_REGION`, not from the profile.

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
