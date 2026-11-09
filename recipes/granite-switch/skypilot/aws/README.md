# Running the granite-switch recipes on AWS

The recipes in this folder train on AWS EC2 GPUs through SkyPilot:

| Recipe | What it builds |
|---|---|
| [`answerability-lora`](answerability-lora/README.md) | LoRA and/or aLoRA answerability adapters (`hf-peft-sft`). |
| [`answerability-sr`](answerability-sr/README.md) | A Shadow Residual answerability adapter, trained and scored. |

Each target, and each `hf://` download, runs on its own EC2 instance. The
instances share data through an **EFS filesystem** that you provision once in
your own AWS account. This page covers the one-time setup: your credentials,
an EFS, and a private copy of the environment that holds your EFS id, which
[`setup-local-space.sh`](setup-local-space.sh) writes (step 4).

Nothing in this setup goes into git. Credentials and resource ids live under
`~/.granite.build/` on the machine that runs gbserver (`$GB_HOME_DIR` if you
set it).

## What you need

- **An AWS account** with:
  - an access key for an IAM user or role that can launch EC2 instances in the
    region you pick, and create EFS filesystems, security groups and mount
    targets there (for the one-time setup below);
  - for the AWS CLI profile that runs step 4, `ssm:GetParameter` on
    `arn:aws:ssm:<region>::parameter/aws/service/deeplearning/*`, to look up the
    GPU host image (or pin it in the `.ids` file instead; see step 4);
  - GPU quota in that region: *Running On-Demand G and VT instances* for
    L4/L40S, *Running On-Demand P instances* for H100/H200 (192 vCPUs for one
    8-GPU p5 instance). Check under Service Quotas → Amazon EC2.
- **Your data.** The recipes are templates and ship without data; each
  recipe's README gives the format it needs and the parameter that names it.
- **A Hugging Face token**, if your data (or a gated model) is in a private
  Hugging Face repo.
- **granite.build installed** with a standalone gbserver
  ([Getting started](../../../../docs/getting-started.md)), and the AWS CLI.

## 1. Pick a region

The EFS lives in one region, and every step must run there. One environment
template, [`environment-template.yaml`](environment-template.yaml), serves every
region; step 4 fills in your region and EFS id, and names the result:

| Region | Environment the recipes use (`ENVIRONMENT_URI`) |
|---|---|
| `us-east-1` | `space://environments/skypilot-local/aws-gb-shared` (the recipes' default) |
| `us-west-2` | `space://environments/skypilot-local/aws-gb-shared-usw2` |
| any other | `space://environments/skypilot-local/aws-gb-shared-<region>` |

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
environment file. The template holds the placeholders `fs-REPLACE-ME`,
`REGION-REPLACE-ME` and `AMI-REPLACE-ME` and is in no space, so the recipes
cannot use it as is. Let the script fill them in: one left in a hand-made copy
fails the launch.
Until you run this step, a build fails before it is submitted, as gbserver
finds no environment at the recipes' `ENVIRONMENT_URI`:

```
Unresolvable space uri : space://environments/skypilot-local/aws-gb-shared
```

Run the setup script once per region, from your checkout:

```bash
recipes/granite-switch/skypilot/aws/setup-local-space.sh <region>
```

It reads `FS` from your `.ids` file (step 2), looks up the GPU host image
(below), and writes, outside the repo and readable only by you:

| File | What |
|---|---|
| `~/.granite.build/local-space/overlay/environments/skypilot-local/aws-gb-shared*/environment.yaml` | The template with your EFS id, region and GPU host AMI, named for the region as in step 1. Under `skypilot-local/`, so it never shadows a shipped environment. |
| `~/.granite.build/local-space/space.yaml` | A standalone space named `public` (so it reads the `space_secrets/public/` secrets): the overlay first, then this checkout's `configurations/assets`. An existing one that already lists the overlay is kept. |
| `~/.granite.build/local-env/granite-switch.params` | `ENVIRONMENT_URI=space://environments/skypilot-local/<name>`. Other lines in the file are kept. |

It reads no secret; it is safe to re-run, and `ENVIRONMENT_URI` points at the
region set up last. It ends by printing what is left to do, and warns about
any environment written before the GPU host image existed (re-run it for that
region).

**The GPU host image.** The GPU steps (`hf-peft-sft`, and `byoc` for
answerability-sr) run on the AWS
[Deep Learning Base GPU AMI](https://aws.amazon.com/releasenotes/aws-deep-learning-base-gpu-ami-ubuntu-22-04/)
("Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)"). On SkyPilot's
default AMI an 8-GPU H200 instance does not see its GPUs: the GPUs are joined
through NVSwitch, which needs the DLAMI's NVIDIA Fabric Manager before CUDA can
start. The environment sets it as `docker.host_image_id`: the host a container
runs on, or the AMI itself for a step without one. Either way a step keeps its
own image, and the CPU-only steps (`prepare`, the `hf://` pulls) keep
SkyPilot's default AMI. Each region's environment holds that region's AMI, so
changing `ENVIRONMENT_URI` is all it takes to run in another region.

AMI ids differ by region and AWS publishes a new build every few weeks, so the
script reads the current one for your region from AWS's public SSM parameter
`/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id`.
That is its one AWS call, made with your AWS CLI as the profile in
`AWS_PROFILE` (or the CLI's default), which needs the permission above:

```bash
AWS_PROFILE=<your profile> recipes/granite-switch/skypilot/aws/setup-local-space.sh <region>
```

To pin a build, or to run without the AWS CLI, add an `AMI=ami-...` line to the
region's `.ids` file. Re-run the script to pick up a newer build.

## 5. Add your data, start gbserver and run

The recipes ship without data. Add yours to the same params file, one
`KEY=VALUE` per line, so that private repo names stay out of your shell
history and out of the checkout:

```bash
cat >> ~/.granite.build/local-env/granite-switch.params <<'EOF'
SOURCE_DATA_URI=hf:///datasets/<org>/<repo>/<revision>/<dir>
DATA_URI=hf:///datasets/<org>/<repo>
EOF
```

`SOURCE_DATA_URI` is for `answerability-lora` and `DATA_URI` for
`answerability-sr`; each recipe's README gives the format. One file serves
both: a recipe ignores `--param` keys it does not use. Values must not contain
spaces.

Then start gbserver on the space, and expand the file into `--param` flags:

```bash
gbserver standalone --space-dir ~/.granite.build/local-space   # 127.0.0.1:8080

cd recipes/granite-switch/skypilot/aws/answerability-lora
gb build start -f build.yaml --skip-version-check \
  $(sed 's/^/--param /' ~/.granite.build/local-env/granite-switch.params) \
  --param RUN_ID=smoke1 --param ACCELERATORS=L4:1 --param INSTANCE_TYPE=g6.4xlarge \
  --param DATASET_SPLIT='train[:16]' --param MAX_STEPS=4
```

A `--param` given later on the command line wins, so add any override after
the file's. Don't use `--parameters-path` for the file: it replaces the
recipe's `parameters.yaml` rather than adding to it. Each recipe's README has
its own smoke run and full-run parameters.

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
