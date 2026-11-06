# fms-hf-tuning (SkyPilot)

Fine-tunes a causal LM with
[fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning)
(foundation-model-stack's tuning library on HF transformers and TRL). It does full
fine-tuning, LoRA or aLoRA (activated LoRA). Training runs fms-hf-tuning's own
`sft_trainer` under `accelerate launch` on a single node and uses the GPUs the
build requests, with plain DDP by default or FSDP sharding. It runs on any
SkyPilot endpoint (cloud or SLURM). It takes the build's bound base model, trains
on an HF Hub dataset, a staged dataset, or an fms-hf-tuning data config, and
registers the result (the full model, or the adapter) as a `model` artifact.
ClearML experiment tracking is `auto`: on when server credentials are present in
the step env.

The step runs in its own image. The image bakes in Python 3.12 with a hash-locked
stack: `fms-hf-tuning==3.2.2` and what it allows (`torch==2.8.0` with CUDA 12.8,
`transformers==4.55.4`, `trl==0.19.1`, `peft==0.19.1`, `accelerate==1.15.0`,
`datasets==4.8.5`), `flash-attn==2.8.3` and `clearml==2.1.12`, plus the step's
scripts. It has no fms-acceleration plugins (QLoRA, GPTQ-LoRA, padding-free,
multipack, fast MoE), no DeepSpeed, no Aim and no MLflow.

> **Developing or testing this step?** See `steps/README.md` in the granite.build
> repository for how steps are generated, tested, and published.

## Referencing the step

Point your build's Space at one that provides the step, then reference it by its
stable URI:

```yaml
steps:
  - step_uri: space://steps/train/fms-hf-tuning
```

## Inputs and output

Inputs are assetstore bindings (portable across endpoints), not `file_mounts`.
`hf://` inputs are staged on the node before training.

| Name | Kind | Required | Type | Purpose |
|---|---|---|---|---|
| `base_model` | input | **yes** | model | The model to tune. Must be a Hugging Face model directory with a `config.json` at its root. An `hf://` URI or a prior-step model binding. It is fms-hf-tuning's `model_name_or_path`, and its tokenizer is used. |
| `dataset` | input | no | dataset | Training data as a staged dataset dir or file (for example an `hf:///datasets/...` URI). It is fms-hf-tuning's `training_data_path`, read with `datasets.load_dataset` (a hub-style dir of jsonl/parquet/arrow files, or one such file). A `save_to_disk` dir does not load. |
| `model` | output | no | model | The tuned model (`peft_method: none`) or adapter (`lora`, `alora`). Registered under the name in `fms_hf_tuning_config.output` (default `model`). |

**Exactly one data source**, from three:

- bind a `dataset` input;
- set `fms_hf_tuning_config.dataset_name`, an HF Hub dataset id;
- set `fms_hf_tuning_config.data_config`, fms-hf-tuning's
  [data config](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/docs/advanced-data-preprocessing.md)
  as a mapping. Its `data_paths` name the data; they may read bound inputs with
  `{{ bindings.<input>.binding.path }}`. Bind such inputs under any name other
  than `dataset`, or as `dataset` itself.

`data_config` with `dataset_name` is refused; without `data_config`, both or
neither of `dataset` and `dataset_name` is refused. These fail the build at step
creation, before any node is provisioned.

The step accepts extra inputs and outputs (`allow_unknown: true`), so a recipe can
bind inputs for `data_config`'s `data_paths`, or purely to order targets.

Before training, the step checks on the node that `base_model` is a directory with
a `config.json`, that a bound `dataset` exists, and that neither path still holds an
unrendered `{{ ... }}` (a misspelled binding name).

### Chat data

With `training_data_path` (a bound `dataset` or `dataset_name`), fms-hf-tuning picks
its data handling from `args`. For a `messages`-style chat dataset (such as
`trl-lib/Capybara`), set all three:

| `args` key | Value |
|---|---|
| `dataset_conversation_field` | the column holding the turns, usually `messages` |
| `instruction_template` | the text the chat template puts before a user turn |
| `response_template` | the text the chat template puts before an assistant turn |

Only the assistant turns are trained on. For ChatML models (SmolLM2, Qwen) the
markers are `<|im_start|>user` and `<|im_start|>assistant`. For Granite 3.x they
are `<|start_of_role|>user<|end_of_role|>` and
`<|start_of_role|>assistant<|end_of_role|>`. Without these, fms-hf-tuning expects
`input` and `output` columns and fails with "Dataset should contain input and
output field". The other formats (`dataset_text_field`, `data_formatter_template`,
pre-tokenized `input_ids` + `labels`) are in fms-hf-tuning's
[training docs](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/docs/training.md).

A base model with no chat template takes one in `args.chat_template`. Wrap the text
in `{% raw %}...{% endraw %}` so Granite.Build does not render it:

```yaml
args:
  chat_template: "{% raw %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}{% endraw %}"
```

## Methods

`fms_hf_tuning_config.peft_method` picks what is trained:

| `peft_method` | What it trains | Output | Pick it when |
|---|---|---|---|
| `none` (default) | Every weight of the model (full fine-tuning). | The full model: `config.json` plus safetensors weights, with the tokenizer. | You need the whole model to change, and have the GPU memory for weights, gradients and optimizer state (use `fsdp` past a small model). |
| `lora` | A LoRA adapter on the frozen base. | `adapter_config.json` + `adapter_model.safetensors`. | The usual choice for a cheap, small adapter. |
| `alora` | An activated LoRA adapter: it only acts on tokens after an invocation string, so the base model's cache for the prompt can be reused. | As `lora`. | You serve the adapter as an on-demand capability next to the base model. |

LoRA and aLoRA settings go in `args` under fms-hf-tuning's own names: `r` (default
8), `lora_alpha` (32), `lora_dropout` (0.05), `target_modules` (a list, or
`all-linear`), `modules_to_save`, and the rest of peft's `LoraConfig`.

**aLoRA invocation string.** `peft_method: alora` requires
`args.alora_invocation_string`: the text after which the adapter activates, usually
the `response_template` (for example `<|im_start|>assistant` for ChatML). Any other
method refuses it. `args.packing` and `args.padding_free` are refused for `alora`:
the adapter activates from the last invocation in each row, so packed rows would
train the wrong spans.

## Config contract (fms_hf_tuning_config)

All fields live under the step's `config.fms_hf_tuning_config`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `peft_method` | string | `none` | `none`, `lora` or `alora`. See [Methods](#methods). |
| `distributed` | string | `ddp` | `ddp` or `fsdp` (fms-hf-tuning's FSDP1 defaults). `fsdp` needs at least 2 GPUs. See [GPUs and distributed training](#gpus-and-distributed-training). |
| `output` | string | `model` | Name of the declared target output (type `model`) to register: the artifact marker's ID. One word with no double quote. It must equal an output the target declares; the step cannot see the target's outputs at step creation, so a mismatch shows only after training. |
| `output_path` | string | `""` | Where the result is written and registered from. Empty means `./model_output` in the per-target workdir, which teardown removes. See [The model output](#the-model-output). |
| `dataset_name` | string | `""` | HF Hub dataset id, passed as `training_data_path`. One of the three data sources. |
| `data_config` | map | `{}` | fms-hf-tuning's data config, written to a file and passed as `data_config_path`. One of the three data sources. |
| `base_model_id` | string | `""` | Adapters only (refused for `none`): a portable base id written into `adapter_config.json` as `base_model_name_or_path`. Empty leaves the node-local staging dir there. |
| `args` | map | `{}` | Any other fms-hf-tuning argument. See [Passing fms-hf-tuning arguments (args)](#passing-fms-hf-tuning-arguments-args). |
| `image_id` | string | the step's published image | The step's image, filled in when the step is generated. Override it with another image or, on SLURM, a `.sqsh` path. `docker:` is added when missing. |
| `clearml_enabled` | string | `auto` | `auto` (on iff `CLEARML_API_HOST`, `CLEARML_API_ACCESS_KEY` and `CLEARML_API_SECRET_KEY` are all set in the step env), `true`, or `false`. |
| `clearml_project` | string | `""` | ClearML project. Empty means `fms-hf-tuning`. Used only when ClearML is on. |
| `clearml_task_name` | string | `""` | ClearML task name. Empty means `sft-<peft_method>`. Used only when ClearML is on. |

Config errors fail the build at step creation, before any node is provisioned, with
a message that starts `fms-hf-tuning:`. They are:

- `peft_method`, `distributed` or `clearml_enabled` outside its allowed values;
- `args` or `data_config` not a map; a data-source conflict (see
  [Inputs and output](#inputs-and-output));
- `args.alora_invocation_string` missing with `peft_method: alora`, or set with
  another method; `args.packing` or `args.padding_free` true with `alora`;
- `base_model_id` with `peft_method: none`;
- an empty `output`, or one with whitespace or a double quote;
- `compute_config.num_nodes` or `launcher_config.num_nodes` other than 1;
- no GPU request (neither `launcher_config.resources.accelerators` nor an
  `instance_type`); `distributed: fsdp` with `accelerators` asking for one GPU;
- an `args` key that is not a bare name (empty, or with `=`, a leading `-` or
  whitespace), or a step-owned key.

## Passing fms-hf-tuning arguments (args)

`args` is fms-hf-tuning's JSON job config, minus the keys the step owns: every
argument `python -m tuning.sft_trainer --help` lists (TRL `SFTConfig` and
transformers `TrainingArguments` fields, the data and model arguments, LoRA
fields, tracker and acceleration fields), under the same names. A
`tuning_config` from an fms-hf-tuning JSON file, or from the K8s
`space://steps/tuning` step, copies across as is.

Before any GPU time, the step checks the merged job against fms-hf-tuning's own
parser:

- an unknown key (a typo) fails, naming the key;
- each value is converted as the command line would: `"1e-5"` becomes a float,
  `"true"` a bool, a single `target_modules` string a one-item list, and so on; a
  value the argument's type or choices reject (`max_steps: 1.5`,
  `packing: maybe`, `lr_scheduler_type: nope`) fails, naming the key;
- a list for a single-value argument, or an empty list, fails;
- mappings (for example `gradient_checkpointing_kwargs`) pass through.

**Step defaults** that `args` may override: `save_strategy: "no"` (only the final
model is registered), `logging_strategy: steps` and `logging_steps: 10` (so the
`Step:` progress line appears; fms-hf-tuning logs once per epoch by default).

**Step-owned keys**, refused in `args` (use the inputs, `dataset_name`,
`data_config`, `peft_method`, `distributed`, `output_path` and `clearml_*`
instead): `model_name_or_path`, `tokenizer_name_or_path`, `output_dir`, `save_model_dir`, `training_data_path`, `validation_data_path`, `data_config_path`, `peft_method`, `quantization_method`, `trackers`, `report_to`, `experiment`, `clearml_project`, `clearml_task`, `run_uri_export_path`, `push_to_hub`, `hub_model_id`, `hub_strategy`, `hub_token`, `hub_private_repo`, `resume_from_checkpoint`, `accelerate_launch_args`, `fsdp`, `fsdp_config`, `deepspeed`, `log_level_replica`, `do_dataprocessing_only`.

`tokenizer_name_or_path` is owned because the tokenizer comes from `base_model`;
`validation_data_path` because evaluation data goes through `data_config` (its
`split` field), as in fms-hf-tuning's own rules.

**Precision and attention.** fms-hf-tuning loads the model in `torch_dtype`
(default `bfloat16`) with `use_flash_attn: true` (flash-attention 2). Both GPUs the
build tests use (A10G, H200) support it. For a GPU without bf16 or
flash-attention support, set `torch_dtype: float16` and/or `use_flash_attn: false`.

## GPUs and distributed training

Request GPUs on the build, in `launcher_config.resources.accelerators`
(`"<type>:<count>"`); the step runs one process per GPU on one node:

```yaml
config:
  launcher_config:
    resources:
      accelerators: "A10G:1"
```

| `distributed` | Launch | Pick it when |
|---|---|---|
| `ddp` (default) | One GPU: a single process. Several: DDP, one full model copy per GPU. | The model, gradients and optimizer state fit on one GPU. |
| `fsdp` | FSDP with fms-hf-tuning's defaults (FSDP1, full shard, full state dict, transformer auto-wrap). | Full fine-tuning past a small model, or LoRA on a base that does not fit on one GPU. Needs at least 2 GPUs. |

The step always passes its own accelerate config, so a user's
`~/.cache/huggingface/accelerate` default (shared home dirs on SLURM) is never read.
A multi-GPU launch picks a free rendezvous port.

**FSDP and adapters.** peft 0.21 saves an empty LoRA adapter under FSDP1
(huggingface/peft#3876). This image has peft 0.19.1. The step refuses to register
an adapter with 0 tensors in either case, so the failure shows as a failed build,
never as an empty artifact.

The step fails before training when the container can access fewer GPUs than were
allocated (on BlueVela SLURM this is ibm-granite/granite.build#604; rerun).

## ClearML tracking

`clearml_enabled` decides whether ClearML is used:

| Value | Behaviour |
|---|---|
| `auto` (default) | On only when all three of `CLEARML_API_HOST`, `CLEARML_API_ACCESS_KEY` and `CLEARML_API_SECRET_KEY` are set in the step env. Otherwise off, with no error. |
| `true` | Always on. |
| `false` | Always off, even when credentials are present. |

Supply the three `CLEARML_API_*` variables through the build's space secrets
(`config.skypilot.secrets.secret_names_to_use_as_env_variable`).

When on, the step sets fms-hf-tuning's `trackers: [clearml]`: its ClearML tracker
records the training scalars and arguments, in `clearml_project` / `clearml_task_name`
(defaults `fms-hf-tuning` and `sft-<peft_method>`). `report_to` stays `none` so the
run is not logged twice. The step sets `CLEARML_LOG_MODEL=false` unless the env
already sets it, so the model is not uploaded to the ClearML file server (it is
already a Granite.Build artifact).

To track without a server, set `CLEARML_OFFLINE_MODE=1` through
`launcher_config.envs`, with `clearml_enabled: true`:

```yaml
config:
  launcher_config:
    envs:
      CLEARML_OFFLINE_MODE: "1"
  fms_hf_tuning_config:
    clearml_enabled: true
```

## The model output

The registered directory holds:

- for `peft_method: none`, the full model (`config.json`, `model.safetensors` or
  sharded `model-*.safetensors` with an index) and the tokenizer;
- for `lora` and `alora`, `adapter_config.json`, `adapter_model.safetensors` and the
  tokenizer. aLoRA's config carries the invocation tokens.

The step records the metadata `peft_method`, `distributed`, `fms_hf_tuning_version`
and `transformers_version`. It refuses to register an output whose weights hold 0
tensors.

**Loading an adapter.**

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3.3-2b-instruct")
model = PeftModel.from_pretrained(base, "/path/to/adapter")
```

### When a downstream target reads the model

The default `output_path` lands in the per-target workdir, which is removed at
teardown. If another target binds the `model` output, write it to shared storage
and use the same path in an `env:///` output URI:

```yaml
      outputs:
        model:
          uri: "env:///shared/models/sft-run1"
          type: model
      steps:
        - step_uri: space://steps/train/fms-hf-tuning
          config:
            fms_hf_tuning_config:
              output: model
              output_path: /shared/models/sft-run1   # must match the uri above
```

A re-run of the same path replaces the earlier output. The step refuses, before
training, an `output_path` that exists and is not a directory, one that is non-empty
with neither `config.json` nor `adapter_config.json` at its root, one that names no
directory (`/`, `.`, `..`), and one that overlaps the staged `base_model`. The model
is written to `<output_path>.partial` and moved into place only after the check.

## Running on SLURM

The same step runs on a SkyPilot SLURM endpoint (for example
`space://environments/skypilot/slurm/bluevela`):

- **The image is cached automatically.** Environments with a `slurm_image_cache`
  (`bluevela` has one) import the registry image once into a shared `.sqsh`; the
  first run on a cold cache imports it in a CPU job first.
- **Request memory** in `launcher_config.resources` as bare GiB, as a string
  (gbserver drops `compute_config` memory on SLURM):

  ```yaml
  launcher_config:
    resources:
      accelerators: "H200:2"
      memory: "128"
  ```

- **Match the partition's GPU type** (`bluevela` pins `h200`).
- **Inputs are staged by their own steps**: one hfpull job per `hf://` input.
  BlueVela's hfpull runs in a 1 GiB cgroup and pulls the whole repo, so a repo with
  large extra files (for example SmolLM2-135M-Instruct's onnx exports) can be
  OOM-killed (ibm-granite/granite.build#597).

## Example build.yaml

LoRA on Granite 3.3 2B, FSDP across 8 A100s, on an HF chat dataset
(`test-data/build.yaml` with `test-data/parameters.yaml` in the step's source):

```yaml
granite.build:
  name: fms-hf-tuning

  targets:

    train:
      environment_uri: space://environments/$${ENVIRONMENT}
      inputs:
        base_model:
          uri: "hf:///$${BASE_MODEL_ID}"
      outputs:
        # The tuned model or adapter dir; fms_hf_tuning_config.output names this entry.
        model:
          uri: "$${MODEL_URI}"
          type: model
      steps:
        - step_uri: space://steps/train/fms-hf-tuning
          config:
            poll_interval_seconds: $${POLL_INTERVAL_SECONDS}
            launcher_config:
              resources:
                # GPUs are requested here; the step uses all of them.
                accelerators: "$${ACCELERATORS}"
            compute_config:
              num_nodes: 1
            fms_hf_tuning_config:
              peft_method: lora
              distributed: fsdp
              dataset_name: "$${DATASET}"
              base_model_id: "$${BASE_MODEL_ID}"
              clearml_project: "$${CLEARML_PROJECT}"
              # Any fms-hf-tuning argument, under its own name.
              args:
                r: 16
                lora_alpha: 32
                lora_dropout: 0.05
                target_modules: all-linear
                # Chat data: the conversation column and the turn markers of
                # the model's chat template (Granite 3.x shown).
                dataset_conversation_field: messages
                instruction_template: "<|start_of_role|>user<|end_of_role|>"
                response_template: "<|start_of_role|>assistant<|end_of_role|>"
                max_seq_length: 4096
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

## Moving from the K8s `space://steps/tuning` step

The hand-authored K8s `tuning` step (Helm, Aim) runs the same `tuning.sft_trainer`.

| `tuning` | `fms-hf-tuning` |
|---|---|
| `tuning_config` / `tuning-config` | `fms_hf_tuning_config.args` (same keys) |
| `tuning_data_config` | `fms_hf_tuning_config.data_config` (same structure) |
| `peft_method` inside `tuning_config` | `fms_hf_tuning_config.peft_method` |
| `compute_config.num_gpus_per_node` | `launcher_config.resources.accelerators` |
| `tuned_checkpoint` output | the `model` output (`output` names it) |
| Aim tracking | ClearML (`clearml_*`) |

`fast_kernels`, `padding_free`, `fast_moe` and other fms-acceleration keys parse
but fail at runtime: the image has no fms-acceleration.

## Notes and limitations

- Single node only (`num_nodes` 1). SkyPilot only: K8s, SkyPilot-K8s and LSF are
  follow-ups.
- No QLoRA, GPTQ-LoRA, padding-free, multipack or fast MoE (fms-acceleration is not
  in the image), no prompt tuning (`pt`), no DeepSpeed.
- One output, the final model. Mid-run checkpoints (`save_strategy` in `args`) go to
  a scratch dir that is removed, and are not registered.
- No vLLM post-processing of adapters with added tokens
  (`lora_post_process_for_vllm`).
- A `save_to_disk` dataset dir does not load; fms-hf-tuning uses
  `datasets.load_dataset`.
