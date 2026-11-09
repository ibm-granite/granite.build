# Answerability LoRA / aLoRA adapter (hf-peft-sft, AWS EC2)

A recipe **template**: it trains a classification adapter on
`ibm-granite/granite-4.2-8b`, a regular LoRA, an activated LoRA (aLoRA), or
both in one build (see [Methods](#methods)), with the
[`hf-peft-sft`](../../../../../steps/train/hf-peft-sft/skypilot/USAGE.md)
step. The training runs on EC2 GPUs through SkyPilot, and the build's data
moves between instances through a shared EFS filesystem. Each train registers
its **adapter**, not a merged model, as its `adapter` artifact.

It ships **without data**: you bring a dataset in the format below (see
[Your data](#your-data)). The hyperparameters were set for one answerability
task (does a set of documents answer the user's question?); see
[What to expect](#what-to-expect) before relying on them for another.

First run the one-time AWS setup in [Running the granite-switch recipes on
AWS](../README.md): it ends with `setup-local-space.sh`, which writes your
environment copy and a params file holding its `ENVIRONMENT_URI`; add
`SOURCE_DATA_URI` there. Then:

```bash
P="$(sed 's/^/--param /' ~/.granite.build/local-env/granite-switch.params)"
gb build start -f build.yaml $P --param RUN_ID=<unique>                          # LoRA
gb build start -f build.yaml $P --param RUN_ID=<unique> --param METHODS=alora,lora   # both
```

`parameters.yaml` beside `build.yaml` is picked up automatically. The build
fails before it is submitted, naming what is missing, without
`SOURCE_DATA_URI` (`'SOURCE_DATA_URI' is undefined`) or without the
environment the setup writes
(`Unresolvable space uri : space://environments/skypilot-local/aws-gb-shared`).

## How it runs

```
prepare ──(dataset)──> train-<m1> ──(order)──> train-<m2>
```

Every step runs on its own EC2 instance in the region of the EFS
(`us-east-1` for the default environment template):

| Step | Instance | What it does |
|---|---|---|
| hfpull × 2 (for `prepare`) | small CPU instance | Downloads your parquet directory and the base model into the EFS hf cache. |
| `prepare` | ≥ 4 vCPU / 16 GB | Writes the prompt/completion dataset to `$DATA_ROOT/$RUN_ID` on the EFS. |
| hfpull (per train) | small CPU instance | Finds the base model already in the cache. |
| `train-<method>` | `ACCELERATORS` (8×H200 by default) | Trains and writes the adapter to `$ADAPTER_ROOT/answerability-<method>/$RUN_ID` on the EFS. |

The hf cache, the dataset and the adapters are on the EFS, outside the per-run
workdirs that gbserver removes when a target ends. They stay there after the
build; later runs reuse the cached model.

## Your data

`SOURCE_DATA_URI` names a **directory** holding `train.parquet` and
`test.parquet`. Each needs two columns of chat messages, in TRL's
conversational prompt/completion format:

| Column | Holds |
|---|---|
| `prompt` | a list of `{"role": ..., "content": ...}` messages that **ends in a user turn** |
| `completion` | **one** assistant message: the label |

One row, shown as JSON (in the parquet file each column is a list of
`{content, role}` structs). Here the documents are part of the user turn; the
hf-peft-sft step has no separate `documents` column:

```json
{
  "prompt": [
    {"role": "system", "content": "Decide whether the documents contain an answer to the user's last question."},
    {"role": "user", "content": "Document 1: The Eiffel Tower is 330 m tall.\n\nQuestion: When was the Eiffel Tower built?"}
  ],
  "completion": [
    {"role": "assistant", "content": "\"unanswerable\""}
  ]
}
```

Granite 4.2's chat template renders this row as
`...<|im_start|>assistant\n<think></think>"unanswerable"<|im_end|>\n`, and the
loss falls on `</think>"unanswerable"<|im_end|>\n` (see
[What is trained](#what-is-trained)). The system turn is optional. Use the
same label strings in every row, quotes included if you use them: the adapter
learns the exact text.

To write a split from Python rows like the one above:

```python
import pyarrow as pa, pyarrow.parquet as pq
pq.write_table(pa.Table.from_pylist(rows), "train.parquet")
```

Other columns are dropped. For an `hf://` dataset repo, put the directory in
the URI after the revision, so that only it is downloaded:
`hf:///datasets/<org>/<repo>/<revision>/<dir>` (three slashes; `datasets/` is
needed for a dataset repo). A private repo needs an `HF_TOKEN` secret with
read access (setup guide, step 3). An `env://` path already on the EFS works
too.

### What `prepare` does

- **Keeps exactly `prompt` and `completion`.** In particular it drops
  `input_ids`: TRL treats a dataset that has `input_ids` as already tokenized
  and skips its chat-template tokenization and completion-only masking, so the
  run would train on the whole sequence.
- **Fails on a schema problem:** a `prompt` that is not a non-empty message
  list, or a `completion` that is not one assistant message.
- **Drops rows whose prompt does not end in a user turn**, since the
  completion must answer one. More than 1% of a split dropped this way fails
  the run: that points to a data problem, not stray rows.
- **Drops rows longer than `MAX_LENGTH` tokens**, counted as TRL counts them
  (prompt + completion through the base model's chat template). TRL keeps the
  start of a row, so in a longer row the label at the end is cut off and TRL
  would drop the row anyway, silently. More than 10% of a split too long fails
  the run: `MAX_LENGTH` is then too short for the data.
- Writes `data/{train,test}.parquet` and a `README.md` listing the source
  hashes, every dropped row by its source row number, and the count of each
  completion.

`train` uses only `train.parquet`; see [Limits](#limits).

### What is trained

The loss is on the completion only (TRL's `completion_only_loss`, on by
default for prompt/completion data). Granite 4.2's chat template ends a
generation prompt with `<|im_start|>assistant\n<think>\n`, but renders a
finished assistant turn as `<|im_start|>assistant\n<think></think>...<|im_end|>`.
TRL starts the completion where the two diverge, so the trained span is
`</think>`, the completion's content, and `<|im_end|>\n`.

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
| `train-alora` | off | Recomputing activations loses peft's aLoRA offsets and zeroes the gradients (peft #2826), so aLoRA needs more GPU memory: it has fit on H200 (141 GB) at 8,192 tokens; on 80 GB GPUs it is untested at that length. |

Put the method most likely to fail first (aLoRA, for its memory), so that a
failure shows up before the other train has spent its GPU hours. If a train
fails, the trains after it do not start; rerun them with `METHODS` set to the
rest. A rerun with the same `RUN_ID` re-runs `prepare` into the same dataset
directory and replaces it with the same rows.

Entries must be distinct and each `lora` or `alora`; anything else fails the
build before it is submitted, with
`'METHODS_must_be_distinct_entries_of_lora_or_alora' is undefined`.

### aLoRA

An activated LoRA acts only from its invocation on, so a server can reuse the
base model's cache for the prompt and switch the adapter on just for the
answer. It cannot be merged into the base model; load it as an adapter.

- **Invocation: `<|im_start|>assistant`** (`ALORA_INVOCATION_STRING`),
  Granite 4.2's ChatML assistant marker, tokens `[100256, 78191]`. The last
  occurrence in each row opens the answer turn, which is where peft activates
  the adapter. The step re-checks the first 32 tokenized rows and fails before
  training if the invocation is in none of them. Granite 4.1's
  `<|start_of_role|>assistant<|end_of_role|>` does not work on 4.2: 4.2 has no
  `<|start_of_role|>` token.
- **No trailing newline.** gbserver renders every step config string as a
  Jinja template, and Jinja drops one trailing newline, so
  `<|im_start|>assistant\n` would reach the trainer as `<|im_start|>assistant`
  anyway. The default is the form that arrives.
- **Hyperparameters are the LoRA ones** (`LORA_R` 16, `LORA_ALPHA` 32,
  `LEARNING_RATE` 5e-5, 3 epochs), so an aLoRA run is directly comparable
  with a LoRA run. None is tuned for aLoRA.

## What to expect

- **Transfers to other data:** the parts tied to the model and the method:
  the Granite 4.2 chat-template handling, the aLoRA invocation, LoRA rank 16 /
  alpha 32 on all linear layers, learning rate 5e-5 with a cosine schedule,
  and an effective batch of 32.
- **Tied to the answerability task:** 3 epochs over about 12,000
  single-label rows, `MAX_LENGTH` 8,192 for long multi-document prompts, and
  `prepare`'s 1% / 10% failure thresholds. Revisit them for your data.
- **Not evaluated.** No LoRA or aLoRA adapter from this recipe has been
  scored yet. With a one-label completion the training loss falls to near zero
  quickly, so it shows the plumbing works and says nothing about held-out
  accuracy. Score the adapter on held-out data before using it.

## Prerequisites

The one-time AWS setup is shared by the recipes in this folder and is in
[Running the granite-switch recipes on AWS](../README.md): your AWS key and
GPU quota, an EFS in your region, where to store the EFS id and the
credentials under `~/.granite.build/`, and the private copy of the
environment you point `ENVIRONMENT_URI` at. For this recipe specifically:

- your data, in the [format above](#your-data), and an `HF_TOKEN` secret if
  it is in a private Hugging Face repo;
- for the default `H200:8`, a P-instance vCPU quota of at least 192 (one
  48xlarge) in your region.

### The EFS filesystem

Every step mounts an EFS at `/mnt/gb-shared`. The repo ships one environment
template, [`../environment-template.yaml`](../environment-template.yaml), with
placeholders for the EFS id and region; it is in no space.
[Setup steps 2 and 4](../README.md#2-create-the-efs-once-per-region) create the
EFS and write your copy into your own space. `ENVIRONMENT_URI` defaults to the
name that copy gets for us-east-1,
`space://environments/skypilot-local/aws-gb-shared`; for us-west-2 it is
`...-usw2`, and `setup-local-space.sh` records the right one in your params
file.

## GPUs

`ACCELERATORS` picks the GPUs; SkyPilot picks the instance type. AWS has no
4-GPU H100, H200 or A100 instance, so the default is 8 GPUs with
`GRAD_ACCUM=4`, an effective batch of 8 × 1 × 4 = 32:

| `ACCELERATORS` | `GRAD_ACCUM` | Instance | On-demand $/h | Notes |
|---|---|---|---|---|
| `H200:8` (default) | `4` | p5en.48xlarge | ~63 | 141 GB GPUs, so aLoRA (no gradient checkpointing) fits at 8,192 tokens. |
| `H100:8` | `4` | p5.48xlarge | ~55 | 80 GB GPUs. LoRA fits; aLoRA at 8,192 tokens without gradient checkpointing is untested. |
| `H100:1` | (smoke) | p5.4xlarge | ~7 | For a smoke run. |
| `L40S:1` | (smoke) | g6e.xlarge | ~1.9 | For a smoke run; 48 GB, so not for the full run. |
| `L4:1` | (smoke) | g6.xlarge | ~0.8 | For a smoke run with short rows; 24 GB. |

Prices are SkyPilot's catalog prices for `us-east-1`. The hf-peft-sft step
saves no checkpoints, so a spot instance that is reclaimed restarts training
from the start; the recipe uses on-demand instances.

GPU capacity is often short. When there is none, gbserver retries the launch
with backoff for about 2 hours (the environment's `retry`) before the build
fails. SkyPilot only asks for the cheapest instance type with the requested
GPUs, and AWS stocks each type separately, so when that one has none, set
`INSTANCE_TYPE` to a larger size with the same GPUs (for `L40S:1`,
`g6e.2xlarge` or `g6e.4xlarge`; for `L4:1`, `g6.4xlarge`; for `H200:8`,
`p5e.48xlarge` or `p5en.48xlarge`).

A smoke run, to try the plumbing on one GPU before the full run:

```bash
gb build start -f build.yaml $(sed 's/^/--param /' ~/.granite.build/local-env/granite-switch.params) --param RUN_ID=smoke1 \
  --param ACCELERATORS=L40S:1 --param INSTANCE_TYPE=g6e.2xlarge \
  --param DATASET_SPLIT='train[:64]' --param MAX_STEPS=4
```

## Parameters

They apply to every train in the build.

| Parameter | Default | Meaning |
|---|---|---|
| `RUN_ID` | `it1-r1` | Names the build (`answerability-<methods>-<RUN_ID>`), the dataset dir, the adapter dirs and the ClearML tasks. Use a new one per run: re-using one replaces that run's outputs. |
| `ENVIRONMENT_URI` | `space://environments/skypilot-local/aws-gb-shared` | Your copy of the EFS environment, as `setup-local-space.sh` names it for us-east-1; it records the one for your region in your params file. See [The EFS filesystem](#the-efs-filesystem). |
| `SOURCE_DATA_URI` | none (required) | Your parquet directory; see [Your data](#your-data). |
| `DATASET_SPLIT` | `train` | `load_dataset` split; slicing works (`train[:64]` for a smoke run). |
| `MAX_LENGTH` | `8192` | Tokens per row: `prepare` drops longer rows, and the train uses it as `max_length`. |
| `METHODS` | `lora` | A comma-separated list of `lora` / `alora`, trained in that order. See [Methods](#methods). |
| `ALORA_INVOCATION_STRING` | `<\|im_start\|>assistant` | Only for an `alora` entry. See [aLoRA](#alora). |
| `LORA_R` / `LORA_ALPHA` / `LEARNING_RATE` | `16` / `32` / `5e-5` | Adapter rank, scale and learning rate. |
| `NUM_TRAIN_EPOCHS` / `MAX_STEPS` | `3` / `-1` | A positive `MAX_STEPS` caps the run. |
| `BASE_MODEL_URI` / `BASE_MODEL_ID` | `hf:///ibm-granite/granite-4.2-8b` / `ibm-granite/granite-4.2-8b` | The public base model, and its Hub id written into `adapter_config.json`. Keep the two naming the same weights. |
| `DATA_ROOT` | `/mnt/gb-shared/granite-switch/answerability-lora/data` | Where `prepare` writes, on the EFS. It does not include the method (every train in a build reads it), so builds running at the same time need different `RUN_ID`s. |
| `ADAPTER_ROOT` | `/mnt/gb-shared/granite-switch/answerability-lora/adapters` | Where each adapter is written, on the EFS, under `answerability-<method>/$RUN_ID`. |
| `ACCELERATORS` | `H200:8` | See [GPUs](#gpus). |
| `INSTANCE_TYPE` | empty | An EC2 instance type with the `ACCELERATORS` GPUs, for when SkyPilot's pick has no capacity. Empty lets SkyPilot pick. |
| `GRAD_ACCUM` | `4` | `gradient_accumulation_steps`. Keep GPUs × 1 × `GRAD_ACCUM` = 32 to keep the effective batch the other hyperparameters were set with. |
| `PREP_IMAGE` | the hf-peft-sft `peft-v0.21.2-6` image | `prepare` runs in the step image, for its pyarrow and transformers. |
| `PREP_CPUS` / `PREP_MEMORY` | `4` / `16` | The minimum vCPUs and GB of memory of the `prepare` instance. |
| `CLEARML` / `CLEARML_PROJECT` | `false` / `granite-switch/answerability` | `true` tracks in ClearML and needs space secrets `CLEARML_API_HOST`, `CLEARML_API_ACCESS_KEY` and `CLEARML_API_SECRET_KEY`. |
| `POLL_INTERVAL_SECONDS` | `120` | How often gbserver polls each step. |

## Getting the adapter

Each adapter is on the EFS, at
`/mnt/gb-shared/granite-switch/answerability-lora/adapters/answerability-<method>/<RUN_ID>`.
The build does not copy it anywhere else. To use it outside AWS, copy it from
an instance that mounts the EFS, or add an `hf://` output to the
`train-<method>` target (the environment pins its hidden `hfpush` step to the
EFS region). `s3://` outputs do not work on SkyPilot yet: the built-in s3push
step's default path does not resolve there.

## Limits

- **No evaluation, and no checkpoint per epoch.** hf-peft-sft does not
  evaluate and passes `--save_strategy no`, so one adapter is registered at
  the end. `prepare` converts `test.parquet`, but the train does not use it.
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
