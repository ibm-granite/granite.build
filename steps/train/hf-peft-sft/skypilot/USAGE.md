# hf-peft-sft (SkyPilot)

Trains a [PEFT](https://github.com/huggingface/peft) adapter on a causal LM with
supervised fine-tuning (SFT), using TRL's `SFTTrainer`. The method is LoRA, QLoRA
(LoRA on a 4-bit base) or aLoRA (activated LoRA). Training runs under
`accelerate launch` on a single node and uses the GPUs the build requests, with
plain DDP by default or FSDP / DeepSpeed ZeRO-3 sharding. It runs on any SkyPilot
endpoint (cloud or SLURM). It takes the build's bound base model, trains on an HF
Hub dataset or a staged dataset, and registers the trained **adapter** (not a
merged model) as a `model` artifact. ClearML experiment tracking is `auto`: on
when server credentials are present in the step env.

The step runs in its own image. The image bakes in Python 3.12 with a hash-locked
training stack: `torch==2.15.0` (CUDA 13.2), `peft==0.21.2`, `trl==1.17.1`,
`transformers==5.22.1`, `accelerate==1.15.0`, `bitsandbytes==0.50.2`,
`deepspeed==0.19.9`, `datasets==5.1.0` and `clearml==2.1.12`, plus the step's
scripts. It has no flash-attn, vllm or wandb.

> **Developing or testing this step?** See `steps/README.md` in the granite.build
> repository for how steps are generated, tested, and published.

## Referencing the step

Point your build's Space at one that provides the step, then reference it by its
stable URI:

```yaml
steps:
  - step_uri: space://steps/train/hf-peft-sft
```

## Inputs and output

Inputs are assetstore bindings (portable across endpoints), not `file_mounts`.
`hf://` inputs are staged on the node before training.

| Name | Kind | Required | Type | Purpose |
|---|---|---|---|---|
| `base_model` | input | **yes** | model | The model to adapt. Must be a Hugging Face model directory with a `config.json` at its root. An `hf://` URI or a prior-step model binding. Passed as `--model_name_or_path <path>`. |
| `dataset` | input | no | dataset | Training data as a staged dataset dir (for example an `hf:///datasets/...` URI). Passed to `load_dataset(<path>, split=<dataset_split>)`. |
| `adapter` | output | no | model | The trained PEFT adapter dir. Registered under the name in `hf_peft_sft_config.output` (default `adapter`). |

**Exactly one data source.** Either bind a `dataset` input or set
`hf_peft_sft_config.dataset_name` (an HF Hub dataset id). Setting both, or neither,
fails the build at step creation, before any node is provisioned.
`dataset_split` applies to either source.

The data must be in one of TRL's standard SFT formats, detected from the columns:
a `messages` column (conversational, formatted with the model's chat template),
`prompt` + `completion` columns, or a `text` column. Data with none of these fails
before training (extra columns next to them are fine). The step does no column mapping; reshape the data in an upstream step.

The step accepts extra inputs and outputs (`allow_unknown: true`), so a recipe can
bind other inputs purely to order targets.

Before training, the step checks on the node that `base_model` is a directory with
a `config.json`, that a bound `dataset` exists, and that neither path still holds an
unrendered `{{ ... }}` (a misspelled binding name).

## Methods

`hf_peft_sft_config.method` picks the adapter type:

| `method` | What it trains | Base precision | Pick it when |
|---|---|---|---|
| `lora` (default) | A LoRA adapter on the frozen base (peft `LoraConfig`). | `mixed_precision`: bf16 (default), fp16, or fp32 for `no`. | The base fits in GPU memory at 16-bit. The usual choice. |
| `qlora` | A LoRA adapter on a 4-bit base: nf4 with double quantization (bitsandbytes). | 4-bit weights; compute in bf16 (fp32 for `mixed_precision: no`). `fp16` is refused. | The base does not fit at 16-bit, or you want more batch per GPU. Slower per step than `lora`. |
| `alora` | An activated LoRA adapter: the adapter only acts on tokens after an invocation string, so the base model's cache for the prompt can be reused. | As `lora`. | You serve the adapter as an on-demand capability (a checker, a rewriter) next to the base model. |

**aLoRA invocation string.** `method: alora` requires `alora_invocation_string`:
the text after which the adapter activates. It is usually the text the chat template
emits right before the assistant turn, for example `<|im_start|>assistant` for
ChatML templates. Any other method refuses the field.

The step tokenizes the string with the base model's tokenizer (no special tokens
added) and stores the tokens in the adapter config. After the trainer tokenizes the
data, the step checks the first 32 training rows. If the tokens appear in none of
them, it fails before training: the adapter would never activate and would train
nothing. If they are missing from only some rows (for example rows truncated at
`max_length`), it prints a warning naming them; those rows train nothing.

`args.packing` and `args.padding_free` are refused for `alora`. peft activates the
adapter from the last invocation in each row, so packed rows would train the wrong
spans.

aLoRA adapters cannot be merged into the base model (peft refuses); load them as
adapters.

## Config contract (hf_peft_sft_config)

All fields live under the step's `config.hf_peft_sft_config`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `method` | string | `lora` | `lora`, `qlora` or `alora`. See [Methods](#methods). |
| `distributed` | string | `ddp` | `ddp`, `fsdp` or `deepspeed` (ZeRO-3). `fsdp` and `deepspeed` need at least 2 GPUs. See [GPUs and distributed training](#gpus-and-distributed-training). |
| `output` | string | `adapter` | Name of the declared target output (type `model`) to register: the artifact marker's ID. Must be one word with no double quote. |
| `output_path` | string | `""` | Where the adapter is written and registered from. Empty means `./adapter_output` in the per-target workdir, which teardown removes. See [The adapter output](#the-adapter-output). |
| `dataset_name` | string | `""` | HF Hub dataset id. Exactly one of this or a bound `dataset` input supplies the training data. |
| `dataset_split` | string | `train` | The `load_dataset` split spec, for either data source. Slicing works (`train[:1000]`). Must not be empty. |
| `lora` | map | `{r: 16, alpha: 32, dropout: 0.05, target_modules: all-linear, modules_to_save: []}` | peft `LoraConfig`. `r` (positive integer, the rank), `alpha` (positive integer, `lora_alpha`), `dropout` (number in [0, 1), `lora_dropout`), `target_modules` (`all-linear` or a non-empty list of module names), `modules_to_save` (list of module names trained in full and saved with the adapter, for example `[lm_head]`). Other keys are refused. The step always uses `bias: none` and `task_type: CAUSAL_LM`. |
| `alora_invocation_string` | string | `""` | Required for `method: alora`, refused otherwise: the text after which the adapter activates. See [Methods](#methods). |
| `base_model_id` | string | `""` | A portable base id (for example `ibm-granite/granite-3.3-2b-instruct`) written into the adapter's `adapter_config.json` as `base_model_name_or_path`. Empty leaves peft's value, which is the node-local staging dir. |
| `mixed_precision` | string | `bf16` | `bf16`, `fp16` or `no`. Sets `accelerate launch --mixed_precision`, the trainer's `--bf16`/`--fp16`, and the base model's load dtype. `qlora` refuses `fp16`. |
| `args` | map | `{}` | Any other TRL `SFTConfig` field, rendered as `--<key> <value>`. See [Passing SFTConfig flags (args)](#passing-sftconfig-flags-args). |
| `image_id` | string | the step's published image | The step's image, filled in when the step is generated. Override it with another image or, on SLURM, optionally a `.sqsh` path (SLURM environments with a `slurm_image_cache` cache the registry image themselves). `docker:` is added when missing. |
| `clearml_enabled` | string | `auto` | `auto` (on iff `CLEARML_API_HOST`, `CLEARML_API_ACCESS_KEY` and `CLEARML_API_SECRET_KEY` are all set in the step env), `true`, or `false`. |
| `clearml_project` | string | `""` | ClearML project. Empty means `hf-peft-sft`. Used only when ClearML is on. |
| `clearml_task_name` | string | `""` | ClearML task name. Empty means `sft-<method>` (for example `sft-qlora`). Used only when ClearML is on. |

Config errors fail the build at step creation, before any node is provisioned, with
a message that starts `hf-peft-sft:`. They are:

- `method`, `distributed`, `mixed_precision` or `clearml_enabled` outside its allowed
  values;
- both data sources, or neither; an empty `dataset_split`;
- `alora_invocation_string` empty with `method: alora`, or set with another method;
- `method: qlora` with `mixed_precision: fp16`;
- an empty `output`, or one with whitespace or a double quote;
- `compute_config.num_nodes` other than 1;
- no GPU request: neither `launcher_config.resources.accelerators` nor an
  `instance_type`;
- `distributed: fsdp` or `deepspeed` with `accelerators` asking for one GPU (`"A10G"`
  or `"A10G:1"`);
- `lora` not a map, an unknown `lora` key, `lora.r` or `lora.alpha` not a positive
  integer, `lora.dropout` outside [0, 1), `lora.target_modules` neither `all-linear`
  nor a non-empty list, `lora.modules_to_save` not a list;
- `args` not a map; a step-owned flag in `args`; a map or empty-list value in `args`;
- `args.packing` or `args.padding_free` true with `method: alora`.

## Passing SFTConfig flags (args)

Each `args` entry becomes `--<key>` followed by its value, after the step's own
flags. The trainer parses them into TRL's `SFTConfig` (which includes every
transformers `TrainingArguments` field), so an unknown key fails at the start of
the run. Every value reaches the trainer as its own argv word, byte for byte, so a
value with spaces or quotes is never word-split. `$MY_VAR` is therefore a literal,
not an expansion. Parameterise the build with `$${PARAM}` instead.

| YAML value | Renders as |
|---|---|
| scalar: `max_length: 4096` | `--max_length 4096` |
| list: `label_names: [a, b]` | `--label_names a b` (repeated values) |
| bool: `gradient_checkpointing: true` | `--gradient_checkpointing true` (`false` renders `false`) |
| null: `seed: null` | omitted |
| empty list: `foo: []` | refused at step creation (`--foo` alone would take the next flag as its value) |
| map | refused at step creation |

The step sets these flags itself, so they are refused in `args` (also with `-` in
place of `_`):

```
output_dir, report_to, push_to_hub, hub_model_id, hub_strategy, hub_token,
hub_private_repo, resume_from_checkpoint, save_strategy, bf16, fp16, fsdp,
fsdp_config, deepspeed, gradient_checkpointing_kwargs, model_name_or_path, method,
distributed, mixed_precision, dataset_path, dataset_split, lora_r, lora_alpha,
lora_dropout, lora_target_modules, lora_modules_to_save, alora_invocation_string,
base_model_id
```

Use the `base_model`/`dataset` inputs, `method`, `distributed`, `mixed_precision`,
`lora`, `output_path`, `base_model_id` or `clearml_*` instead. The step always
passes `--save_strategy no`, `--push_to_hub false`, `--bf16 true|false` and
`--fp16 true|false` (from `mixed_precision`), and `--report_to clearml` or
`--report_to none` (from `clearml_enabled`).

TRL turns gradient checkpointing on by default, and the step sets `use_reentrant`
whenever it is on (set `gradient_checkpointing: false` in `args` to turn it off):
`true` under `distributed: deepspeed` (TRL forces it for PEFT with ZeRO-3), otherwise
`true` only for `qlora` outside multi-GPU DDP.

**Loss on the completion only.** `assistant_only_loss: true` needs a chat template
that marks the assistant turns with `{% generation %}` blocks; TRL uses them to
find the assistant tokens. Many templates have none (Granite 3.3's does not). The alternative is
`prompt`/`completion` data, where TRL's `completion_only_loss` (on by default for
that format) trains on the completion only.

Example:

```yaml
hf_peft_sft_config:
  args:
    max_length: 4096
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 4
    learning_rate: 2.0e-04
    num_train_epochs: 1
    gradient_checkpointing: true
    seed: null            # omitted: the trainer's default applies
```

## GPUs and distributed training

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

`distributed` picks how the GPUs share the work:

| `distributed` | Use it when | accelerate config |
|---|---|---|
| `ddp` (default) | The base (16-bit, or 4-bit for `qlora`) fits on one GPU. Each GPU holds a full copy and trains on its share of the batch. Works on 1 GPU. | none (`--multi_gpu` when there is more than one GPU) |
| `fsdp` | The base does not fit on one GPU. FSDP2 full sharding (`reshard_after_forward`), wrapped per transformer layer. | [src/accelerate/fsdp.yaml](src/accelerate/fsdp.yaml); for `qlora`, [src/accelerate/fsdp_qlora.yaml](src/accelerate/fsdp_qlora.yaml) (also offloads parameters to CPU) |
| `deepspeed` | The base does not fit on one GPU, or you prefer DeepSpeed. ZeRO-3, no offload, for every method. Any DeepSpeed op the run uses JIT-compiles on first use, which can add minutes to the first step. | [src/accelerate/deepspeed_z3.yaml](src/accelerate/deepspeed_z3.yaml) |

The command line always sets `--num_machines 1`, `--num_processes <GPUs>` and
`--mixed_precision`, overriding the file's values.

**Sharding needs at least 2 GPUs.** `fsdp` and `deepspeed` with an `accelerators`
count of 1 fail at step creation; when the count is not in `accelerators` (an
`instance_type` request), the node refuses before training.

**FSDP is FSDP2, on purpose.** peft 0.21.x saves an empty adapter from an
FSDP1-wrapped model ([huggingface/peft#3876](https://github.com/huggingface/peft/issues/3876)):
FSDP1 puts `_fsdp_wrapped_module.` inside the module names and peft's save filter
then matches no LoRA tensor. FSDP2 keeps the names clean. Do not turn on
`fsdp_activation_checkpointing` in a custom config either: torch's checkpoint
wrapper hides the LoRA tensors the same way. The trainer's own
`gradient_checkpointing` is fine. Whatever the cause, the step refuses to register
an adapter file with no weights.

Verified combinations (method × `distributed`):

| | ddp | fsdp | deepspeed |
|---|---|---|---|
| lora | render-tested | e2e (BlueVela) | render-tested |
| qlora | e2e (aws) | probe-tested (BlueVela) | e2e (BlueVela) |
| alora | e2e (aws) | render-tested | render-tested |

`e2e` means a real build on that cluster trained and registered an adapter with
weights; `probe-tested` means a 2-step run on that cluster's GPUs with the step's
scripts saved an adapter with weights, outside a build test; `render-tested` means
covered by render and unit tests only.

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

When on, the trainer runs with `--report_to clearml`: transformers' built-in
`ClearMLCallback` records the logged training scalars (such as `train/loss`) and
the training arguments. The project and task come from `clearml_project` and
`clearml_task_name` (defaults `hf-peft-sft` and `sft-<method>`). There is no wandb.

The step sets `CLEARML_LOG_MODEL=false` unless the env already sets it, so the
callback does not upload the adapter to the ClearML file server (it is already a
Granite.Build artifact). To upload it anyway:

```yaml
config:
  launcher_config:
    envs:
      CLEARML_LOG_MODEL: "true"
```

To track without a server, set `CLEARML_OFFLINE_MODE=1` through
`launcher_config.envs`, with `clearml_enabled: true`. The session is then written
locally on the node:

```yaml
config:
  launcher_config:
    envs:
      CLEARML_OFFLINE_MODE: "1"
  hf_peft_sft_config:
    clearml_enabled: true
```

## The adapter output

The registered directory holds the adapter only, not a merged model:

- `adapter_config.json`: the peft config (method, rank, target modules, and for
  aLoRA the invocation tokens);
- `adapter_model.safetensors`: the adapter weights (plus any `modules_to_save`);
- the tokenizer files, so the chat template travels with the adapter.

The step also records the step metadata `method`, `peft_version` and `trl_version`.

**Set `base_model_id`.** peft writes the base model's path into
`adapter_config.json` as `base_model_name_or_path`. Here that is the node-local
staging dir, which means nothing elsewhere. With `base_model_id` set (for example to
the HF id of the bound base), the step writes that instead, so tools that read the
field load the right base.

**Loading it.**

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3.3-2b-instruct")
model = PeftModel.from_pretrained(base, "/path/to/adapter")
```

vLLM serves LoRA and aLoRA adapters with `--enable-lora` (pass the adapter dir as
a LoRA module); aLoRA needs a vLLM release with activated-LoRA support.

### When a downstream target reads the adapter

The default `output_path` lands in the per-target workdir, which is removed at
teardown. That is fine for a terminal target. If another target binds the `adapter`
output (an eval, an `hfpush`, a second training stage), write it to shared storage
and use the same path in an `env:///` output URI:

```yaml
      outputs:
        adapter:
          uri: "env:///shared/adapters/sft-run1"
          type: model
      steps:
        - step_uri: space://steps/train/hf-peft-sft
          config:
            hf_peft_sft_config:
              output: adapter
              output_path: /shared/adapters/sft-run1   # must match the uri above
```

Make the path unique per run (for example with a `$${PARAM}` suffix). A re-run of
the same path replaces the earlier adapter. To avoid deleting anything else, the
step refuses, before training, an `output_path` that exists and is not a directory, one that is
non-empty and has no `adapter_config.json` at its root, one that names no directory
(`/`, `.`, `..`), and one that is the staged `base_model`, inside it, or contains it.

The adapter is written to `<output_path>.partial` first. It is moved to
`output_path` only after training saved both `adapter_config.json` and
`adapter_model.safetensors`. A run that saves no adapter registers nothing.

## Running on SLURM

The same step runs on a SkyPilot SLURM endpoint (for example
`space://environments/skypilot/slurm/ibm-bluevela`). What differs from a cloud
endpoint:

- **The image is cached automatically.** SLURM environments with a `slurm_image_cache`
  block (`ibm-bluevela` has one) import the step's registry image once into a shared
  `.sqsh` and launch every later run from it. The first run on a cold cache imports
  it in a CPU job before the GPU job starts (this image is about 17 GB); later runs
  start the container in seconds. No manual `enroot` import and no `image_id`
  override are needed. To pin a `.sqsh` you built yourself, set `image_id` to its
  path; the step adds `docker:`, and Pyxis gets the path as is.
- **Request memory.** gbserver drops `compute_config` memory on SLURM, and a
  partition default can be as small as 2G. Request memory in
  `launcher_config.resources` as bare GiB, as a string. For a 2-GPU job:

  ```yaml
  launcher_config:
    resources:
      accelerators: "H200:2"
      memory: "128"
  ```

- **Match the partition's GPU type.** `accelerators` becomes a typed GRES request
  (`--gres=gpu:<type>:<count>`). Use a type the partition has and that any
  environment-wide constraint allows (`ibm-bluevela` pins `h200`).
- **Inputs are staged by their own steps.** SLURM endpoints stage `hf://` inputs
  with separate hfpull jobs onto shared storage, so a build shows one extra step per
  `hf://` input.

## Example build.yaml

LoRA SFT of a Granite instruct model on an HF dataset, with DDP over the requested
GPUs. The `$${...}` values come from a parameter file.

```yaml
    train:
      environment_uri: space://environments/$${ENVIRONMENT}
      inputs:
        base_model:
          uri: "hf:///$${BASE_MODEL_ID}"
      outputs:
        # The PEFT adapter dir; hf_peft_sft_config.output names this entry.
        adapter:
          uri: "$${ADAPTER_URI}"
          type: model
      steps:
        - step_uri: space://steps/train/hf-peft-sft
          config:
            poll_interval_seconds: $${POLL_INTERVAL_SECONDS}
            launcher_config:
              resources:
                # GPUs are requested here; the step uses all of them.
                accelerators: "$${ACCELERATORS}"
            compute_config:
              num_nodes: 1
            hf_peft_sft_config:
              method: lora
              distributed: ddp
              dataset_name: "$${DATASET}"
              dataset_split: "$${DATASET_SPLIT}"
              base_model_id: "$${BASE_MODEL_ID}"
              clearml_project: "$${CLEARML_PROJECT}"
              lora:
                r: 16
                alpha: 32
                dropout: 0.05
                target_modules: all-linear
              # Any TRL SFTConfig field, as --<key> <value>.
              args:
                max_length: 4096
                per_device_train_batch_size: 2
                gradient_accumulation_steps: 4
                learning_rate: 2.0e-04
                lr_scheduler_type: cosine
                warmup_ratio: 0.03
                num_train_epochs: 1
                gradient_checkpointing: true
                logging_steps: 10
                seed: 123
```

With parameters such as:

```yaml
# Parameters for build.yaml (gb build ... --param-file parameters.yaml).
ENVIRONMENT: "skypilot/aws"
# Public, Apache-2.0. A gated model needs HF_TOKEN in the step env (space secret).
BASE_MODEL_ID: "ibm-granite/granite-3.3-2b-instruct"
ADAPTER_URI: "env:///adapter"
# 8x A100 80GB (p4de): DDP, one full bf16 copy of the 2B base per GPU.
ACCELERATORS: "A100-80GB:8"
DATASET: "trl-lib/Capybara"
DATASET_SPLIT: "train"
CLEARML_PROJECT: "hf-peft-sft"
POLL_INTERVAL_SECONDS: 300
```

To train on your own data, bind a `dataset` input and remove `dataset_name`:

```yaml
      inputs:
        base_model:
          uri: hf:///ibm-granite/granite-3.3-2b-instruct
        dataset:
          uri: hf:///datasets/my-org/my-sft-data
```

For QLoRA, set `method: qlora`. For an aLoRA adapter on a ChatML model, set
`method: alora` and `alora_invocation_string: "<|im_start|>assistant"`, and use the
string your model's chat template emits before the assistant turn.

## Notes and limitations

- **Single node only.** `compute_config.num_nodes` must be `1`. Any other value
  fails the build at step creation.
- **Adapter only.** The step never merges the adapter into the base model. aLoRA
  adapters cannot be merged, and merging into a 4-bit base is lossy. Merge in a
  downstream step if you need a merged LoRA model.
- **Every run starts fresh.** There is no resume and no mid-run checkpoint: the step
  passes `--save_strategy no`, `resume_from_checkpoint` is step-owned, and each run
  trains into a fresh staging dir, so a warm-cluster re-run never resumes an earlier
  run.
- **A gated model or dataset needs `HF_TOKEN`** in the step env (a space secret).
- **Progress shows in the step status.** The trainer prints
  `Step: <n>/<total> loss=<x>` at each logging step, and the step's monitor shows
  the latest line as the step's status message. `args.logging_steps` sets how often.
- **Attention is `sdpa`.** The image has no flash-attn, so the base model loads with
  PyTorch's scaled-dot-product attention.
