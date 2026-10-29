# openinstruct-sft (SkyPilot)

Runs supervised fine-tuning (SFT) with upstream
[Open-Instruct](https://github.com/allenai/open-instruct) `open_instruct/finetune.py`,
pinned at release v0.3.0. Training runs under `accelerate launch`, with DeepSpeed
ZeRO-3 by default. The step trains on a single node and uses the GPUs the build
requests. It runs on any SkyPilot endpoint (cloud or SLURM). It takes the build's
bound base model, trains it on HF datasets or a staged dataset, and registers the
trained HF model as a `model` artifact. ClearML experiment tracking turns on
automatically when server credentials are present in the step env.

The step runs in its own image. The image bakes in the pinned Open-Instruct source,
its locked training stack (torch, accelerate, DeepSpeed, transformers), `clearml`,
and the step's scripts.

> **Developing or testing this step?** See `steps/README.md` in the granite.build
> repository for how steps are generated, tested, and published.

## Referencing the step

Point your build's Space at one that provides the step, then reference it by its
stable URI:

```yaml
steps:
  - step_uri: space://steps/train/openinstruct-sft
```

## Inputs and output

Inputs are assetstore bindings (portable across endpoints), not `file_mounts`.
`hf://` inputs are staged on the node before training.

| Name | Kind | Required | Type | Purpose |
|---|---|---|---|---|
| `base_model` | input | **yes** | model | The model to fine-tune. Must be a Hugging Face model directory with a `config.json` at its root. An `hf://` URI or a prior-step model binding. Passed as `--model_name_or_path <path>`. |
| `dataset` | input | no | dataset | Training data as a staged dataset dir (for example an `hf:///datasets/...` URI). Passed as `--dataset_mixer_list <path> 1.0`. |
| `model` | output | no | model | The trained HF model. Registered under the name in `openinstruct_sft_config.output` (default `model`). |

**Exactly one data source.** Either bind a `dataset` input or set
`openinstruct_sft_config.dataset_mixer_list`. Setting both, or neither, fails the
build at step creation, before any node is provisioned.

The step accepts extra inputs and outputs (`allow_unknown: true`), so a recipe can
bind other inputs purely to order targets.

Before training, the step checks on the node that `base_model` is a directory with
a `config.json`, that a bound `dataset` exists, and that neither path still holds an
unrendered `{{ ... }}` (a misspelled binding name).

## Config contract (openinstruct_sft_config)

All fields live under the step's `config.openinstruct_sft_config`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `output` | string | `model` | Name of the declared target output (type `model`) to register: the artifact marker's ID. Must be one word with no double quote. |
| `output_path` | string | `""` | Where the trained HF model is written and registered from. Empty means `./model_output` in the per-target workdir, which teardown removes. See [When a downstream target reads the model](#when-a-downstream-target-reads-the-model). |
| `dataset_mixer_list` | map | `{}` | HF dataset id to a fraction (`<= 1.0`) or a sample count, rendered as `--dataset_mixer_list <id> <value> ...`. Exactly one of this or a bound `dataset` input supplies the training data. |
| `dataset_mixer_list_splits` | list | `["train"]` | Split names, rendered as `--dataset_mixer_list_splits <split> ...`. Must be a non-empty list. |
| `deepspeed_config` | string | `stage3_no_offloading_accelerate.conf` | A file under the image's `/opt/open-instruct/configs/ds_configs/`. Empty means plain accelerate (no DeepSpeed). A value containing `/` is refused. |
| `mixed_precision` | string | `bf16` | Passed to `accelerate launch --mixed_precision`. |
| `args` | map | `{}` | Any other `finetune.py` flag, rendered as `--<key> <value>`. See [Passing finetune.py flags (args)](#passing-finetunepy-flags-args). |
| `image_id` | string | the step's published image | The step's image, filled in when the step is generated. Override it with another image or, on SLURM, optionally a `.sqsh` path (SLURM environments with a `slurm_image_cache` cache the registry image themselves). `docker:` is added when missing. |
| `clearml_enabled` | string | `auto` | `auto` (on iff `CLEARML_API_HOST`, `CLEARML_API_ACCESS_KEY` and `CLEARML_API_SECRET_KEY` are all set in the step env), `true`, or `false`. |
| `clearml_project` | string | `""` | ClearML project. Empty means `openinstruct-sft`. Used only when ClearML is on. |
| `clearml_task_name` | string | `""` | ClearML task name. Empty means `finetune`. Used only when ClearML is on. |

Config errors fail the build at step creation, with a message that starts
`openinstruct-sft:`. Examples: both or neither data source, an empty or multi-word
`output`, a bad `clearml_enabled`, a step-owned flag in `args`, or `num_nodes` other
than 1.

## Passing finetune.py flags (args)

Each `args` entry becomes `--<key>` followed by its value, after the step's own
flags. Every value reaches `finetune.py` as its own argv word, byte for byte, so a
value with spaces or quotes is never word-split. `$MY_VAR` is therefore a literal,
not an expansion. Parameterise the build with `$${PARAM}` instead.

| YAML value | Renders as |
|---|---|
| scalar: `max_seq_length: 4096` | `--max_seq_length 4096` |
| list: `dataset_transform_fn: [a, b]` | `--dataset_transform_fn a b` (repeated values) |
| bool: `gradient_checkpointing: true` | `--gradient_checkpointing true` (`false` renders `false`) |
| null: `seed: null` | omitted |
| empty list: `foo: []` | refused at step creation (`--foo` alone would take the next flag as its value) |
| map | refused at step creation |

The step sets these flags itself, so they are refused in `args`:

```
model_name_or_path, output_dir, dataset_name, dataset_mixer, dataset_mixer_list,
dataset_mixer_list_splits, with_tracking, report_to, wandb_entity, push_to_hub,
try_launch_beaker_eval_jobs, try_auto_save_to_beaker, do_not_randomize_output_dir,
resume_from_checkpoint, dataset_local_cache_dir
```

Use the `base_model`/`dataset` inputs, `output_path`, `dataset_mixer_list(_splits)`
or `clearml_*` instead. The step always passes `--push_to_hub false`,
`--try_launch_beaker_eval_jobs false`, `--try_auto_save_to_beaker false` and
`--do_not_randomize_output_dir`.

Example:

```yaml
openinstruct_sft_config:
  args:
    chat_template_name: tulu
    max_seq_length: 4096
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 4
    learning_rate: 5.0e-06
    num_train_epochs: 2
    gradient_checkpointing: true
    seed: null            # omitted: finetune.py's default applies
```

## GPUs

The step requests no GPUs itself. Request them in the build with
`launcher_config.resources.accelerators`:

```yaml
config:
  launcher_config:
    resources:
      accelerators: "A100-80GB:8"
  compute_config:
    num_nodes: 1
```

The step uses all of them. It launches one process per GPU, taking the count from
`$SKYPILOT_NUM_GPUS_PER_NODE` (falling back to `nvidia-smi -L`). A build with neither
`accelerators` nor a GPU `instance_type` fails at step creation, before a node is
provisioned; a node that still has no GPUs fails before training. There is no
`gpu_model` or `num_gpus_per_node` key. `compute_config.num_nodes` must be `1`.

## ClearML tracking

`clearml_enabled` decides whether ClearML is used:

| Value | Behaviour |
|---|---|
| `auto` (default) | On only when all three of `CLEARML_API_HOST`, `CLEARML_API_ACCESS_KEY` and `CLEARML_API_SECRET_KEY` are set in the step env. Otherwise off, with no error. |
| `true` | Always on. |
| `false` | Always off, even when credentials are present. |

Supply the three `CLEARML_API_*` variables through the build's space secrets
(`config.skypilot.secrets.secret_names_to_use_as_env_variable`). The step never
bakes in a server URL, so it works against any ClearML server.

The project and task come from `clearml_project` and `clearml_task_name` (defaults
`openinstruct-sft` and `finetune`). When on, accelerate's ClearML tracker records
the training scalars (such as `train/loss`) and the hyperparameters.

To track without a server, set `CLEARML_OFFLINE_MODE=1` through
`launcher_config.envs`, with `clearml_enabled: true`. The session is then written
locally on the node:

```yaml
config:
  launcher_config:
    envs:
      CLEARML_OFFLINE_MODE: "1"
  openinstruct_sft_config:
    clearml_enabled: true
```

wandb is always disabled (`WANDB_MODE=disabled`): `finetune.py` requires a wandb
tracker whenever tracking is on, so the step runs a no-op wandb next to ClearML.

## When a downstream target reads the model

The default `output_path` lands in the per-target workdir, which is removed at
teardown. That is fine for a terminal target. If another target binds the `model`
output (an eval, an `hfpush`, a second training stage), write it to shared storage
and use the same path in an `env:///` output URI:

```yaml
      outputs:
        model:
          uri: "env:///shared/models/sft-run1"
          type: model
      steps:
        - step_uri: space://steps/train/openinstruct-sft
          config:
            openinstruct_sft_config:
              output: model
              output_path: /shared/models/sft-run1   # must match the uri above
```

Make the path unique per run (for example with a `$${PARAM}` suffix). A re-run of
the same path replaces the earlier model. To avoid deleting anything else, the step
refuses, before training, an `output_path` that is non-empty and has no
`config.json` at its root, and one that is the staged `base_model`, inside it, or
contains it.

The model is written to `<output_path>.partial` first. It is moved to `output_path`
only after training saved a `config.json`. A run that saves no model registers
nothing.

## Running on SLURM

The same step runs on a SkyPilot SLURM endpoint (for example
`space://environments/skypilot/slurm/ibm-bluevela`). What differs from a cloud
endpoint:

- **The image is cached automatically.** SLURM environments with a `slurm_image_cache`
  block (`ibm-bluevela` has one) import the step's registry image once into a shared
  `.sqsh` and launch every later run from it. The first run on a cold cache imports
  it in a CPU job before the GPU job starts (about 10 minutes for this 11 GB image
  once the CPU job starts, plus queueing); later runs start the container in seconds. No manual `enroot` import and no `image_id`
  override are needed. To pin a `.sqsh` you built yourself, set `image_id` to its
  path; the step adds `docker:`, and Pyxis gets the path as is.
- **Request memory.** gbserver drops `compute_config` memory on SLURM, and a
  partition default can be as small as 2G. Request memory in
  `launcher_config.resources` as bare GiB, as a string:

  ```yaml
  launcher_config:
    resources:
      accelerators: "H200:1"
      memory: "64"
  ```

- **Match the partition's GPU type.** `accelerators` becomes a typed GRES request
  (`--gres=gpu:<type>:<count>`). Use a type the partition has and that any
  environment-wide constraint allows (`ibm-bluevela` pins `h200`).
- **Inputs are staged by their own steps.** SLURM endpoints stage `hf://` inputs
  with separate hfpull jobs onto shared storage, so a build shows one extra step per
  `hf://` input.

## Example build.yaml

Single-node SFT of a Granite base model on 10,000 samples of an HF dataset. The
`$${...}` values come from a parameter file.

```yaml
    train:
      environment_uri: space://environments/$${ENVIRONMENT}
      inputs:
        base_model:
          uri: "$${BASE_MODEL}"
      outputs:
        # The trained HF model; openinstruct_sft_config.output names this entry.
        model:
          uri: "$${MODEL_URI}"
          type: model
      steps:
        - step_uri: space://steps/train/openinstruct-sft
          config:
            poll_interval_seconds: $${POLL_INTERVAL_SECONDS}
            launcher_config:
              resources:
                # GPUs are requested here; the step uses all of them.
                accelerators: "$${ACCELERATORS}"
            compute_config:
              num_nodes: 1
            openinstruct_sft_config:
              dataset_mixer_list:
                "$${DATASET}": $${DATASET_SAMPLES}
              clearml_project: "$${CLEARML_PROJECT}"
              # Any finetune.py flag, as --<key> <value>.
              args:
                chat_template_name: tulu
                max_seq_length: 4096
                per_device_train_batch_size: 1
                gradient_accumulation_steps: 4
                learning_rate: 5.0e-06
                lr_scheduler_type: linear
                warmup_ratio: 0.03
                num_train_epochs: 2
                gradient_checkpointing: true
                logging_steps: 1
                seed: 123
```

With parameters such as:

```yaml
ENVIRONMENT: "skypilot/aws"
BASE_MODEL: "hf:///ibm-granite/granite-3.3-2b-base"
MODEL_URI: "env:///model"
ACCELERATORS: "A100-80GB:8"
DATASET: "allenai/tulu-3-sft-mixture"
DATASET_SAMPLES: 10000
CLEARML_PROJECT: "openinstruct-sft"
POLL_INTERVAL_SECONDS: 300
```

To train on your own data, bind a `dataset` input and remove `dataset_mixer_list`:

```yaml
      inputs:
        base_model:
          uri: hf:///ibm-granite/granite-3.3-2b-base
        dataset:
          uri: hf:///datasets/my-org/my-sft-data
```

## Notes and limitations

- **Single node only.** `compute_config.num_nodes` must be `1`. Any other value
  fails the build at step creation.
- **`finetune.py` is deprecated upstream, and pinned here.** Upstream favours
  `olmo_core_finetune.py`, which supports only OLMo and Qwen3 configs (no Granite).
  The step pins Open-Instruct v0.3.0 (commit
  `26de99604002d6a7b25a2c5f2ee880e388a21dca`), so upstream changes do not reach you
  unannounced. The commit is recorded as the step metadata `open_instruct_commit`.
- **A model without a chat template needs `args.chat_template_name`** (for example
  `tulu`).
- **Every run starts fresh.** There is no resume. `resume_from_checkpoint` is
  step-owned, and each run trains into a fresh staging dir, so a warm-cluster re-run
  never resumes an earlier run.
- **A gated model or dataset needs `HF_TOKEN`** in the step env (a space secret).
- **Progress shows in the step status.** The step's monitor picks up `Step: ...`
  lines from the training log and shows them as the step's status message.
