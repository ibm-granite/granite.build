# Shadow Residual answerability adapter (AWS EC2)

The AWS version of [`../../lsf-bluevela/answerability-sr`](../../lsf-bluevela/answerability-sr/README.md).
It trains the same [Shadow Residual](https://github.com/generative-computing/shadow-residual)
(SR) answerability adapter on `ibm-granite/granite-4.2-8b` and scores it, on
EC2 GPUs through SkyPilot, with the build's data moving between instances
through a shared EFS filesystem.

```bash
gb build start -f build.yaml --param RUN_ID=<unique>
```

`parameters.yaml` beside `build.yaml` is picked up automatically.

The data, the generated SR config, the load-bearing parameters, the expected
scores and how to check that a run did real work are the same as in the LSF
recipe, and its README documents them:
[Data](../../lsf-bluevela/answerability-sr/README.md#data),
[load-bearing parameters](../../lsf-bluevela/answerability-sr/README.md#the-parameters-that-are-load-bearing-in-a-non-obvious-way),
[Checking the run did real work](../../lsf-bluevela/answerability-sr/README.md#checking-the-run-did-real-work).
This README covers what differs on AWS.

## How it runs

```
train ──(adapter)──> eval
```

Every step runs on its own EC2 instance in `us-east-1`, the region the EFS is
in (the environment pins the `byoc` and `hfpull` steps there):

| Step | Instance | What it does |
|---|---|---|
| hfpull × 2 (per target) | small CPU instance | Downloads the `ibm-research/touma` dataset repo (~270 MB) and the base model into the EFS hf cache; later pulls find them there. |
| `train` | `ACCELERATORS` (1×H100 by default) | Clones shadow-residual, writes the SR config, runs `sr-train`, writes the adapter to `$ADAPTER_ROOT/$RUN_ID`. |
| `eval` | `EVAL_ACCELERATORS` (1×H100) | Scores that adapter with `answerability_eval --mode peft`, writes the report to `$REPORT_ROOT/$RUN_ID`. |

The adapter and the report are on the EFS, outside the per-run workdirs that
gbserver removes when a target ends. That is what lets `eval`, on a different
instance, read the adapter `train` wrote. On LSF both targets share `/proj`, so
the LSF recipe can keep them in the workdir.

A re-run with the same `RUN_ID` clears and replaces that run's adapter and
report.

## Prerequisites

The same as [the LoRA recipe on AWS](../answerability-lora/README.md#prerequisites):
the `HF_TOKEN`, `GB_AWS_ACCESS_KEY_ID` and `GB_AWS_SECRET_ACCESS_KEY` secrets,
the P-instance quota, and an EFS in your own copy of the
`skypilot/aws/gb-shared` (us-east-1) or `skypilot/aws/gb-shared-usw2`
(us-west-2) template, set as `ENVIRONMENT_URI` (see
[The EFS filesystem](../answerability-lora/README.md#the-efs-filesystem)).
`HF_TOKEN` needs read access to
the private `ibm-research/touma` repo; see
[the LSF recipe](../../lsf-bluevela/answerability-sr/README.md#the-one-prerequisite-an-hf_token-secret).

## GPUs

`ACCELERATORS` and `EVAL_ACCELERATORS` pick the GPUs; SkyPilot picks the
instance unless `INSTANCE_TYPE` / `EVAL_INSTANCE_TYPE` names one (another size
with the same GPUs, when the cheapest one has no capacity).

| Setup | `ACCELERATORS` | `SR_NUM_PROCESSES` | `GRADIENT_ACCUMULATION` | Notes |
|---|---|---|---|---|
| As on BlueVela (default) | `H100:1` (p5.4xlarge) | `1` | `8` | `sr-train`, one GPU. Effective batch 1 × 1 × 8 = 8. |
| 8 GPUs, DDP | `H200:8` or `H100:8` | `8` | `1` | `accelerate launch --multi_gpu`, one process per GPU. Effective batch 8 × 1 × 1 = 8, as on BlueVela. |

`SR_NUM_PROCESSES` must equal the GPU count in `ACCELERATORS`. Above 1, the
train runs the same trainer, `shadow_residual.training.train`, under
`accelerate launch --num_processes N --multi_gpu --mixed_precision bf16`, the
DDP launch shadow-residual's README documents. Each GPU holds a full copy of
the 8B base, which fits in 80 GB with `GRADIENT_CHECKPOINTING: true`.
**Untested:** no SR run has used DDP yet, here or on BlueVela. Smoke-test it
(below) before a full run. FSDP (shadow-residual's `fsdp_4gpu.yaml`) is not
wired in.

A smoke run, to try the plumbing before a full run:

```bash
gb build start -f build.yaml --param RUN_ID=sr-smoke1 \
  --param MAX_STEPS=4 --param EVAL_DATASET_SIZE=20
```

## Parameters

The parameters shared with the LSF recipe (data, adapter geometry, tokenizer
markers, optimizer, batch, runtime and eval knobs) mean the same here; see
[its parameters.yaml](../../lsf-bluevela/answerability-sr/parameters.yaml). What
differs:

| Parameter | Default | Meaning |
|---|---|---|
| `RUN_ID` | `sr-r1` | Names the build, the adapter dir and the report dir. New here: the LSF recipe keeps its outputs in the per-run workdir. |
| `ENVIRONMENT_URI` | `space://environments/skypilot/aws/gb-shared` | The us-east-1 EFS template, with a placeholder `file_system_id`. Point it at your own copy; see [The EFS filesystem](../answerability-lora/README.md#the-efs-filesystem). |
| `ADAPTER_ROOT` / `REPORT_ROOT` | `/mnt/gb-shared/granite-switch/answerability-sr/{adapters,reports}` | Where the adapter and the report are written, on the EFS, under `$RUN_ID`. |
| `ACCELERATORS` / `INSTANCE_TYPE` | `H100:1` / empty | The train's GPUs, and optionally its instance type. |
| `SR_NUM_PROCESSES` | `1` | Training processes; see [GPUs](#gpus). |
| `EVAL_ACCELERATORS` / `EVAL_INSTANCE_TYPE` | `H100:1` / empty | The eval's GPUs, and optionally its instance type. |
| `POLL_INTERVAL_SECONDS` | `120` | How often gbserver polls each step. |

Removed, LSF-only: `CLUSTER`, `QUEUE`, `MEMORY`, `EVAL_MEMORY`, `NUM_NODES`,
`NUM_GPUS_PER_NODE`, `EVAL_NUM_GPUS_PER_NODE`. On AWS the GPU choice picks the
instance, which sets the host memory.

## Limits

- **No container image for now (`IMAGE: ""`).** The `byoc` step `git clone`s
  the repo before `setup_command` runs, and the `pytorch/pytorch` images the
  LSF recipe uses have no `git`, so with that image the step fails at setup
  with `bash: git: command not found`
  ([#529](https://github.com/ibm-granite/granite.build/issues/529)). With an
  empty `IMAGE` the step runs on the bare EC2 instance, which has `git`, and
  `setup_command`'s `pip install '.[train]'` brings torch from PyPI (a few
  minutes per target). Once `byoc` installs `git` itself, set
  `IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` again. shadow-residual
  pins `transformers<5.10` and `peft==0.19.1`, so it cannot share the
  hf-peft-sft image either way.
- **EFS traffic is not encrypted in transit**, and the EFS root is
  world-writable; see [the LoRA recipe on AWS](../answerability-lora/README.md#limits).
- Everything under the LSF recipe's
  [What this recipe does not do](../../lsf-bluevela/answerability-sr/README.md#what-this-recipe-does-not-do),
  except that multi-GPU training is now available through DDP (untested).
