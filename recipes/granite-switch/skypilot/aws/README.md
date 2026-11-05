# Running the granite-switch recipes on AWS

The recipes in this folder train on AWS EC2 GPUs through SkyPilot:

| Recipe | What it builds |
|---|---|
| [`answerability-lora`](answerability-lora/README.md) | LoRA and/or aLoRA answerability adapters (`hf-peft-sft`). |
| [`answerability-sr`](answerability-sr/README.md) | A Shadow Residual answerability adapter, trained and scored. |

Each target, and each `hf://` download, runs on its own EC2 instance. The
instances share data through an **EFS filesystem** that you provision once in
your own AWS account. This page covers the one-time setup: your credentials,
an EFS, and a private copy of the environment that holds your EFS id.

Nothing in this setup goes into git. Credentials and resource ids live under
`~/.granite.build/` on the machine that runs gbserver (`$GB_HOME_DIR` if you
set it).

## What you need

- **An AWS account** with:
  - an access key for an IAM user or role that can launch EC2 instances in the
    region you pick, and create EFS filesystems, security groups and mount
    targets there (for the one-time setup below);
  - GPU quota in that region: *Running On-Demand G and VT instances* for
    L4/L40S, *Running On-Demand P instances* for H100/H200 (192 vCPUs for one
    8-GPU p5 instance). Check under Service Quotas → Amazon EC2.
- **A Hugging Face token** with read access to the datasets the recipe pulls
  (the recipes' README says which, and whether they are private).
- **granite.build installed** with a standalone gbserver
  ([Getting started](../../../../docs/getting-started.md)), and the AWS CLI.

## 1. Pick a region

The EFS lives in one region, and every step must run there. The repo ships an
environment template for two regions; any other region works the same way.

| Template | Region |
|---|---|
| [`skypilot/aws/gb-shared`](../../../../configurations/assets/environments/skypilot/aws/gb-shared/environment.yaml) | `us-east-1` |
| [`skypilot/aws/gb-shared-usw2`](../../../../configurations/assets/environments/skypilot/aws/gb-shared-usw2/environment.yaml) | `us-west-2` |

Pick the region where you have GPU quota and capacity. GPU capacity varies by
region and changes by the hour: when testing these recipes, single-L4
instances were available in us-west-2 while us-east-1 had none of any type.

## 2. Create the EFS (once per region)

Follow [Runbook: provision a shared EFS filesystem](../../../../docs/environments/skypilot-aws.md#runbook-provision-a-shared-efs-filesystem)
with your AWS profile and region. In short:

1. a security group allowing NFS (TCP 2049) from your VPC's CIDR;
2. an encrypted EFS filesystem with elastic throughput;
3. one mount target **in every AZ** of the region, since SkyPilot may place a
   step in any of them;
4. the filesystem root `chmod 1777`, done once from a small instance in the VPC
   (the runbook's `sky launch ... --down` command), so non-root steps can
   create their directories.

Record the ids the runbook prints, in a file only you can read:

```bash
mkdir -p ~/.granite.build/local-env && chmod 700 ~/.granite.build/local-env
umask 077
cat > ~/.granite.build/local-env/gb-shared-<region>.ids <<EOF
REGION=<region>
VPC=<vpc-id>
SG=<security-group-id>
FS=<file-system-id>
EOF
```

These ids are what you need to find, clean up or delete the resources later.
They are not credentials, but they name resources in your account, so keep
them out of git and out of shared logs.

## 3. Store your credentials

Two kinds, both in the local secret store, never in a recipe file:

**AWS key** (used by gbserver to launch instances). The environment reads the
secret names `GB_AWS_ACCESS_KEY_ID` and `GB_AWS_SECRET_ACCESS_KEY` from the
space `public`, and writes them to the `gb-skypilot` profile of
`~/.aws/credentials` at launch. Write them as in
[Runbook: use a non-default AWS profile via the local secret store](../../../../docs/environments/skypilot-aws.md#runbook-use-a-non-default-aws-profile-via-the-local-secret-store),
which puts them, base64-encoded, in
`~/.granite.build/space_secrets/public/aws.json`.

**Hugging Face token**, under the name `HF_TOKEN`, with gbserver running (step 5):

```bash
export GB_ENVIRONMENT=STANDALONE GBSERVER_HOST=http://127.0.0.1:8080
gb secret create HF_TOKEN --personal --value "$HF_TOKEN" --skip-version-check
gb secret list --personal --skip-version-check       # -> HF_TOKEN
```

`--value "$HF_TOKEN"` reads the token from your shell's environment, so it is
not typed on the command line or kept in shell history.

## 4. Make a private copy of the environment

`environment.yaml` is plain YAML, not a template, so the EFS id has to be in an
environment file. The shipped templates hold `fs-REPLACE-ME`. Put a copy with
your id in a **personal space** outside the repo:

```bash
S=~/.granite.build/local-space
REPO=<path to your granite.build checkout>
T=gb-shared          # the template: gb-shared (us-east-1) or gb-shared-usw2 (us-west-2)
mkdir -p "$S/overlay/environments/skypilot-local/aws-$T"
chmod 700 "$S"

# The environment, with your EFS id.
. ~/.granite.build/local-env/gb-shared-<region>.ids
sed -e "s/fs-REPLACE-ME/$FS/" -e "s/^name: skypilot-aws-$T\$/name: skypilot-local-aws-$T/" \
  "$REPO/configurations/assets/environments/skypilot/aws/$T/environment.yaml" \
  > "$S/overlay/environments/skypilot-local/aws-$T/environment.yaml"
chmod 600 "$S/overlay/environments/skypilot-local/aws-$T/environment.yaml"
grep -nE '^name:|file_system_id:' "$S/overlay/environments/skypilot-local/aws-$T/environment.yaml"

# The space: your overlay first, then the repo's assets for everything else.
cat > "$S/space.yaml" <<EOF
name: public
secret_manager:
  type: local
  config: {}
base_uris:
  - file://$S/overlay
  - file://$REPO/configurations/assets
EOF
```

- Use a name under `skypilot-local/` for your copy, not the template's own
  name, so it never shadows a shipped environment. The repo's `.gitignore`
  also excludes `environments/skypilot-local/`, in case a copy lands in a
  checkout.
- If you chose a region other than the two templates', also change the
  template's `region:` and every `infra: aws/<region>` line in your copy.
- The space is named `public` so it reads the same `space_secrets/public/`
  secrets as the default standalone space.

## 5. Start gbserver on that space and run

```bash
gbserver standalone --space-dir ~/.granite.build/local-space   # 127.0.0.1:8080
```

Then run a recipe with `ENVIRONMENT_URI` pointing at your copy:

```bash
cd answerability-lora
gb build start -f build.yaml --skip-version-check \
  --param ENVIRONMENT_URI=space://environments/skypilot-local/aws-gb-shared \
  --param RUN_ID=smoke1 --param ACCELERATORS=L4:1 --param INSTANCE_TYPE=g6.4xlarge \
  --param DATASET_SPLIT='train[50:66]' --param MAX_STEPS=4
```

For the us-west-2 copy, the URI ends in `aws-gb-shared-usw2`. Each recipe's
README has its own smoke run and full-run parameters. Without
the `ENVIRONMENT_URI` override, a build uses the shipped template, whose
placeholder id cannot be mounted.

## Costs and cleanup

- **EFS** bills for the bytes stored (about $0.30 per GB-month standard), not
  per hour. The hf cache holds the base model once (about 17.6 GB for
  granite-4.2-8b); each run adds its prepared data and adapter.
- **EC2** bills while instances run. gbserver tears down each step's
  instance when the step ends, and `idle_minutes_to_autostop: 5` stops an
  idle one. After a cancelled or failed build, check that nothing is left:
  `sky status` and `aws ec2 describe-instances --filters
  Name=instance-state-name,Values=running`.
- A failed teardown can leave a run's directory under `gbroot/builds/` on the
  EFS; the gbserver log names it (`ORPHANED at ...`). Remove it from any
  instance that mounts the EFS.
- To remove everything, delete the mount targets, then the filesystem, then
  the security group (the ids are in your `.ids` file).

## Limits

- EFS traffic is not encrypted in transit unless amazon-efs-utils is
  installed where it is mounted; the recipes' images and the bare instances
  do not have it, so they mount over plain NFSv4 inside the VPC. Data at rest
  is encrypted.
- The EFS root is world-writable (`1777`), so every build that mounts it can
  read every run's data. Use one EFS per team, or EFS access points per space.
- `file:.` environments next to `build.yaml` do not work through
  `gb build start` yet (#628), which is why the environment lives in a space.
