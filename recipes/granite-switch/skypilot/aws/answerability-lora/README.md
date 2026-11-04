# Answerability LoRA / aLoRA adapter (hf-peft-sft, AWS EC2)

The AWS version of [`../../slurm/answerability-lora`](../../slurm/answerability-lora/README.md).
It trains the same answerability adapter on `ibm-granite/granite-4.2-8b`, a
regular LoRA (`METHOD=lora`, the default) or an activated LoRA
(`METHOD=alora`), with the [`hf-peft-sft`](../../../../../steps/train/hf-peft-sft/skypilot/USAGE.md)
step. The training runs on EC2 GPUs through SkyPilot, and the build's data
moves between instances through a shared EFS filesystem. It registers the
**adapter**, not a merged model, as the `adapter` artifact.

```bash
gb build start -f build.yaml --param RUN_ID=<unique>
```

`parameters.yaml` beside `build.yaml` is picked up automatically.

The data, the `prepare` conversion, what is trained, and aLoRA are the same as
in the SLURM recipe, and its README documents them:
[The data](../../slurm/answerability-lora/README.md#the-data),
[What is trained](../../slurm/answerability-lora/README.md#what-is-trained),
[aLoRA](../../slurm/answerability-lora/README.md#alora). This README covers
what differs on AWS.

## How it runs

```
prepare ──(dataset)──> train
```

Every step runs on its own EC2 instance in `us-east-1`, the region the EFS is
in:

| Step | Instance | What it does |
|---|---|---|
| hfpull × 2 (for `prepare`) | small CPU instance | Downloads the answerability parquet directory and the base model into the EFS hf cache. |
| `prepare` | ≥ 4 vCPU / 16 GB | Writes the prompt/completion dataset to `$DATA_ROOT/$RUN_ID` on the EFS. |
| hfpull (for `train`) | small CPU instance | Finds the base model already in the cache. |
| `train` | `ACCELERATORS` (8×H200 by default) | Trains and writes the adapter to `$ADAPTER_ROOT/answerability-$METHOD/$RUN_ID` on the EFS. |

The hf cache, the dataset and the adapter are on the EFS, outside the per-run
workdirs that gbserver removes when a target ends. They stay there after the
build; later runs reuse the cached model.

## Prerequisites

### Secrets

Register these once in the space you build in:

| Secret | What |
|---|---|
| `HF_TOKEN` | A read token for an account with access to the private `ibm-research/touma` dataset repo. See [the SLURM recipe](../../slurm/answerability-lora/README.md#the-one-prerequisite-an-hf_token-secret). |
| `GB_AWS_ACCESS_KEY_ID`, `GB_AWS_SECRET_ACCESS_KEY` | The AWS key the instances are launched with. gbserver writes them to the `gb-skypilot` profile of `~/.aws/credentials` (see [the AWS environment docs](../../../../../docs/environments/skypilot-aws.md#credentials-aws_credentials)). |

```bash
gb secret create HF_TOKEN --personal --value "$HF_TOKEN" --skip-version-check
gb secret create GB_AWS_ACCESS_KEY_ID --personal --value "$AWS_ACCESS_KEY_ID" --skip-version-check
gb secret create GB_AWS_SECRET_ACCESS_KEY --personal --value "$AWS_SECRET_ACCESS_KEY" --skip-version-check
```

The account needs a P-instance vCPU quota of at least 192 in `us-east-1`
(one 48xlarge) for the default `H200:8`.

### The EFS filesystem

[`environment.yaml`](environment.yaml) mounts the EFS `fs-REPLACE-ME`
in `us-east-1` at `/mnt/gb-shared`. From another AWS account, replace
`file_system_id` (and `region`, if it is not `us-east-1`) with your own
filesystem's. It needs:

- a mount target in every availability zone of the region, since SkyPilot may
  place a step in any of them;
- a security group on the mount targets allowing NFS (TCP 2049) from the
  workers (for example from the default VPC's CIDR);
- its root `chmod 1777`, so the non-root steps can create their directories.

The [runbook in the AWS environment docs](../../../../../docs/environments/skypilot-aws.md#runbook-provision-a-shared-efs-filesystem)
provisions one.

`environment.yaml` is selected with `ENVIRONMENT_URI: "file:."`, which only a
standalone gbserver accepts.

## GPUs

`ACCELERATORS` picks the GPUs; SkyPilot picks the instance type. AWS has no
4-GPU H100, H200 or A100 instance, so the BlueVela setup (4×H200, effective
batch 4 × 1 × 8 = 32) becomes 8 GPUs with `GRAD_ACCUM=4`:

| `ACCELERATORS` | `GRAD_ACCUM` | Instance | On-demand $/h | Notes |
|---|---|---|---|---|
| `H200:8` (default) | `4` | p5en.48xlarge | ~63 | Same 141 GB GPUs as BlueVela, so aLoRA (no gradient checkpointing) fits as it did there. |
| `H100:8` | `4` | p5.48xlarge | ~55 | 80 GB GPUs. LoRA fits; aLoRA at 8,192 tokens without gradient checkpointing is untested. |
| `H100:1` | (smoke) | p5.4xlarge | ~7 | For a smoke run. |

Prices are SkyPilot's catalog prices for `us-east-1`. The hf-peft-sft step
saves no checkpoints, so a spot instance that is reclaimed restarts training
from the start; the recipe uses on-demand instances.

p5-family capacity is often short. When there is none, gbserver retries the
launch with backoff for about 2 hours (`retry` in `environment.yaml`) before
the build fails.

A smoke run, to try the plumbing on one GPU before the full run:

```bash
gb build start -f build.yaml --param RUN_ID=smoke1 \
  --param ACCELERATORS=H100:1 --param DATASET_SPLIT='train[:64]' --param MAX_STEPS=4
```

## Parameters

The parameters shared with the SLURM recipe (`SOURCE_DATA_URI`,
`DATASET_SPLIT`, `MAX_LENGTH`, `METHOD`, `ALORA_INVOCATION_STRING`, `LORA_R`,
`LORA_ALPHA`, `LEARNING_RATE`, `NUM_TRAIN_EPOCHS`, `MAX_STEPS`, `CLEARML`)
mean the same here; see [its table](../../slurm/answerability-lora/README.md#parameters).
What differs:

| Parameter | Default | Meaning |
|---|---|---|
| `ENVIRONMENT_URI` | `file:.` | The EFS-backed environment beside `build.yaml`. |
| `DATA_ROOT` | `/mnt/gb-shared/granite-switch/answerability-lora/data` | Where `prepare` writes, on the EFS. As on SLURM, it does not include the method, so builds running at the same time need different `RUN_ID`s. |
| `ADAPTER_ROOT` | `/mnt/gb-shared/granite-switch/answerability-lora/adapters` | Where the adapter is written, on the EFS, under `answerability-$METHOD/$RUN_ID`. |
| `ACCELERATORS` | `H200:8` | See [GPUs](#gpus). |
| `GRAD_ACCUM` | `4` | `gradient_accumulation_steps`. Keep GPUs × 1 × `GRAD_ACCUM` = 32 for runs comparable with BlueVela's. |
| `PREP_IMAGE` | the hf-peft-sft `peft-v0.21.2-6` image | `prepare` runs in the step image, for its pyarrow and transformers. |
| `PREP_CPUS` / `PREP_MEMORY` | `4` / `16` | The minimum vCPUs and GB of memory of the `prepare` instance. |

## Getting the adapter

The adapter is on the EFS, at
`/mnt/gb-shared/granite-switch/answerability-lora/adapters/answerability-<METHOD>/<RUN_ID>`.
The build does not copy it anywhere else. To use it outside AWS, copy it from
an instance that mounts the EFS, or add an `hf://` or `s3://` output to the
`train` target.

## Limits

- **EFS traffic is not encrypted in transit.** `tls: true` needs
  amazon-efs-utils (`mount.efs`) where the filesystem is mounted, and neither
  the hf-peft-sft image nor the bare workers have it. The mounts fall back to
  plain NFSv4 inside the VPC, and the step log warns about it. The filesystem
  itself is encrypted at rest.
- **The EFS root is world-writable (`1777`).** Any build that mounts it can
  read the adapters and the dataset. Use an EFS
  [access point](../../../../../docs/environments/skypilot-aws.md#per-space-isolation-with-efs-access-points-recommended-for-multi-tenant-byo)
  per space for isolation. That needs `mount.efs` in every step image, so it
  is not done here.
- **Standalone only**, because of `ENVIRONMENT_URI: "file:."`.
