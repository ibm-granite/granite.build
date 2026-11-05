# Answerability LoRA / aLoRA adapter (hf-peft-sft, AWS EC2)

The AWS version of [`../../slurm/answerability-lora`](../../slurm/answerability-lora/README.md).
It trains the same answerability adapter on `ibm-granite/granite-4.2-8b`, a
regular LoRA, an activated LoRA (aLoRA), or both in one build (see
[Methods](#methods)), with the [`hf-peft-sft`](../../../../../steps/train/hf-peft-sft/skypilot/USAGE.md)
step. The training runs on EC2 GPUs through SkyPilot, and the build's data
moves between instances through a shared EFS filesystem. Each train registers
its **adapter**, not a merged model, as its `adapter` artifact.

```bash
gb build start -f build.yaml --param RUN_ID=<unique>                          # LoRA
gb build start -f build.yaml --param RUN_ID=<unique> --param METHODS=alora,lora   # both
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
prepare ──(dataset)──> train-<m1> ──(order)──> train-<m2>
```

Every step runs on its own EC2 instance in `us-east-1`, the region the EFS is
in:

| Step | Instance | What it does |
|---|---|---|
| hfpull × 2 (for `prepare`) | small CPU instance | Downloads the answerability parquet directory and the base model into the EFS hf cache. |
| `prepare` | ≥ 4 vCPU / 16 GB | Writes the prompt/completion dataset to `$DATA_ROOT/$RUN_ID` on the EFS. |
| hfpull (per train) | small CPU instance | Finds the base model already in the cache. |
| `train-<method>` | `ACCELERATORS` (8×H200 by default) | Trains and writes the adapter to `$ADAPTER_ROOT/answerability-<method>/$RUN_ID` on the EFS. |

The hf cache, the dataset and the adapters are on the EFS, outside the per-run
workdirs that gbserver removes when a target ends. They stay there after the
build; later runs reuse the cached model.

## Methods

`METHODS` is a comma-separated list of `lora` and `alora`, in the order to
train them; the default is `lora`. The build has one `prepare` and one
`train-<method>` target per entry, and every train reads the same prepared
dataset, so the methods are trained on the same rows.

The trains run **one after another**, not at the same time: each train after
the first binds the previous train's adapter as an `after` input. The step
does not read it; the binding only makes the train wait. Only one GPU instance
is up at a time, which is what a single reserved instance needs, and it never
pays for two 8-GPU instances at once.

| Train | `gradient_checkpointing` | Notes |
|---|---|---|
| `train-lora` | on | |
| `train-alora` | off | Recomputing activations loses peft's aLoRA offsets and zeroes the gradients (peft #2826), so aLoRA needs more GPU memory: fits on H200 (141 GB); on 80 GB GPUs, untested at 8,192 tokens. Invocation: `ALORA_INVOCATION_STRING` ([why this one](../../slurm/answerability-lora/README.md#alora)). |

Put the method most likely to fail first (aLoRA, for its memory), so that a
failure shows up before the other train has spent its GPU hours. If a train
fails, the trains after it do not start; rerun them with `METHODS` set to the
rest. A rerun with the same `RUN_ID` re-runs `prepare` into the same dataset
directory and replaces it with the same rows.

Entries must be distinct and each `lora` or `alora`; anything else fails the
build before it is submitted, with
`'METHODS_must_be_distinct_entries_of_lora_or_alora' is undefined`.

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

The build runs in `space://environments/skypilot/aws/gb-shared`
([`environment.yaml`](../../../../../configurations/assets/environments/skypilot/aws/gb-shared/environment.yaml)),
which mounts the EFS `fs-REPLACE-ME` in `us-east-1` at `/mnt/gb-shared`.
From another AWS account, copy that environment into your space with your own
filesystem's `file_system_id` (and `region`, if it is not `us-east-1`), and
point `ENVIRONMENT_URI` at it. The filesystem needs:

- a mount target in every availability zone of the region, since SkyPilot may
  place a step in any of them;
- a security group on the mount targets allowing NFS (TCP 2049) from the
  workers (for example from the default VPC's CIDR);
- its root `chmod 1777`, so the non-root steps can create their directories.

The [runbook in the AWS environment docs](../../../../../docs/environments/skypilot-aws.md#runbook-provision-a-shared-efs-filesystem)
provisions one.

## GPUs

`ACCELERATORS` picks the GPUs; SkyPilot picks the instance type. AWS has no
4-GPU H100, H200 or A100 instance, so the BlueVela setup (4×H200, effective
batch 4 × 1 × 8 = 32) becomes 8 GPUs with `GRAD_ACCUM=4`:

| `ACCELERATORS` | `GRAD_ACCUM` | Instance | On-demand $/h | Notes |
|---|---|---|---|---|
| `H200:8` (default) | `4` | p5en.48xlarge | ~63 | Same 141 GB GPUs as BlueVela, so aLoRA (no gradient checkpointing) fits as it did there. |
| `H100:8` | `4` | p5.48xlarge | ~55 | 80 GB GPUs. LoRA fits; aLoRA at 8,192 tokens without gradient checkpointing is untested. |
| `H100:1` | (smoke) | p5.4xlarge | ~7 | For a smoke run. |
| `L40S:1` | (smoke) | g6e.xlarge | ~1.9 | For a smoke run; 48 GB, so not for the full run. |

Prices are SkyPilot's catalog prices for `us-east-1`. The hf-peft-sft step
saves no checkpoints, so a spot instance that is reclaimed restarts training
from the start; the recipe uses on-demand instances.

GPU capacity in `us-east-1` is often short. When there is none, gbserver
retries the launch with backoff for about 2 hours (the environment's `retry`)
before the build fails. SkyPilot only asks for the cheapest instance type with
the requested GPUs, and AWS stocks each type separately, so when that one has
none, set `INSTANCE_TYPE` to a larger size with the same GPUs (for `L40S:1`,
`g6e.2xlarge` or `g6e.4xlarge`; for `H200:8`, `p5e.48xlarge` or
`p5en.48xlarge`).

A smoke run, to try the plumbing on one GPU before the full run:

```bash
gb build start -f build.yaml --param RUN_ID=smoke1 \
  --param ACCELERATORS=L40S:1 --param INSTANCE_TYPE=g6e.2xlarge \
  --param DATASET_SPLIT='train[:64]' --param MAX_STEPS=4
```

## Parameters

The parameters shared with the SLURM recipe (`SOURCE_DATA_URI`,
`DATASET_SPLIT`, `MAX_LENGTH`, `ALORA_INVOCATION_STRING`, `LORA_R`,
`LORA_ALPHA`, `LEARNING_RATE`, `NUM_TRAIN_EPOCHS`, `MAX_STEPS`, `CLEARML`)
mean the same here; see [its table](../../slurm/answerability-lora/README.md#parameters).
They apply to every train in the build. What differs:

| Parameter | Default | Meaning |
|---|---|---|
| `METHODS` | `lora` | Replaces the SLURM recipe's `METHOD`: a comma-separated list of `lora` / `alora`, trained in that order. See [Methods](#methods). Also names the build (`answerability-<methods>-<RUN_ID>`). |
| `ENVIRONMENT_URI` | `space://environments/skypilot/aws/gb-shared` | SkyPilot on AWS with the `gb-shared` EFS. |
| `DATA_ROOT` | `/mnt/gb-shared/granite-switch/answerability-lora/data` | Where `prepare` writes, on the EFS. It does not include the method (every train in a build reads it), so builds running at the same time need different `RUN_ID`s. |
| `ADAPTER_ROOT` | `/mnt/gb-shared/granite-switch/answerability-lora/adapters` | Where each adapter is written, on the EFS, under `answerability-<method>/$RUN_ID`. |
| `ACCELERATORS` | `H200:8` | See [GPUs](#gpus). |
| `INSTANCE_TYPE` | empty | An EC2 instance type with the `ACCELERATORS` GPUs, for when SkyPilot's pick has no capacity. Empty lets SkyPilot pick. |
| `GRAD_ACCUM` | `4` | `gradient_accumulation_steps`. Keep GPUs × 1 × `GRAD_ACCUM` = 32 for runs comparable with BlueVela's. |
| `PREP_IMAGE` | the hf-peft-sft `peft-v0.21.2-6` image | `prepare` runs in the step image, for its pyarrow and transformers. |
| `PREP_CPUS` / `PREP_MEMORY` | `4` / `16` | The minimum vCPUs and GB of memory of the `prepare` instance. |

## Getting the adapter

Each adapter is on the EFS, at
`/mnt/gb-shared/granite-switch/answerability-lora/adapters/answerability-<method>/<RUN_ID>`.
The build does not copy it anywhere else. To use it outside AWS, copy it from
an instance that mounts the EFS, or add an `hf://` or `s3://` output to the
`train-<method>` target.

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
