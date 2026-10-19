# nemo-rl (SkyPilot)

Runs single-node, multi-GPU [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) training on
a SkyPilot cluster. You name the algorithm and a shipped config; the step selects
`examples/run_<algorithm>.py` and `examples/configs/<config_name>` inside the public
NGC `nemo-rl` image, injects the build's bound base model (and, optionally, dataset)
as Hydra overrides, trains, converts the final DCP checkpoint to HuggingFace format,
and registers it as a `model` artifact. ClearML experiment tracking turns on
automatically when server credentials are present in the step env.

One step serves **every** NeMo-RL algorithm — SFT, DPO, GRPO, PPO, RM, and the rest.
Switching algorithm is a change to the build.yaml, never to the step. There is no
Dockerfile to own: the step runs inside the NGC release image, which bundles the
NeMo-RL source and prefetched environments.

> **Developing or testing this step?** See `steps/README.md`
> in the granite.build repository for how the step is generated, tested, and published.

## Referencing the step

Point your build's Space at one that provides the step, then reference it by the
stable `space://steps/train/nemo-rl` URI:

```yaml
steps:
  - step_uri: space://steps/train/nemo-rl
```

## Config contract (`nemo_rl_config`)

All fields live under the step's `config.nemo_rl_config`.

| Field | Type | Required | Purpose |
|---|---|---|---|
| `algorithm` | string | **yes** | Selects `examples/run_<algorithm>.py` in the image (`sft`, `dpo`, `grpo`, `ppo`, `rm`, …). |
| `config_name` | string | **yes** | A named YAML under `examples/configs/`, e.g. `sft.yaml`. Decoupled from `algorithm` on purpose — `run_grpo.py` has several configs (`grpo_math_1B.yaml`, …), so the step cannot derive one from the other. |
| `output` | string | **yes** | Name of a declared target `outputs:` entry (type `model`). Used as the registered artifact's ID. |
| `overrides` | map | no | Hydra overrides, rendered after `--` as `key=value`, one argv word each. Values are shell-quoted and reach Hydra byte-for-byte; booleans render lowercased (`true`/`false`); `null` omits the key. Default `{}`. |
| `image_id` | string | no | NGC image to run in. Default is the pinned `nemo-rl` **release** image (bundles source + prefetched venvs). The step adds SkyPilot's `docker:` scheme marker. |
| `nemo_rl_root` | string | no | Where the bundled NeMo-RL source lives in the image. Default `/opt/nemo-rl`. `uv` runs with its cwd here so it resolves the prefetched project environment. |
| `dataset_override_key` | string | no | The Hydra key a **bound** `dataset`'s path is injected under (e.g. `data.train.input_file`). Required *if and only if* you bind a `dataset` — NeMo-RL data configs have no single universal path field, so you name it. Leave empty to use the config's built-in dataset. |
| `clearml_enabled` | string | no | `auto` (default — on iff `CLEARML_*` server creds are present), `true` (force on), or `false` (hard off: `clearml` is never imported). |
| `clearml_project` / `clearml_task_name` | string | no | ClearML grouping. Only used when ClearML is enabled. |
| `hf_output_repo` | string | no | Optional push target — the converted HF model is also uploaded here via `hf upload`. |

## Inputs

Inputs are **assetstore bindings** (portable across endpoints), not `file_mounts`.
`hf://` inputs are downloaded during `setup` automatically.

| Input | Required | Type | Purpose |
|---|---|---|---|
| `base_model` | **yes** | model | The model to train. Read as the binding's staged path and injected as `policy.model_name=<path>`. An `hf://…` URI or a prior-step model binding. |
| `dataset` | no | dataset | Training data. Injected as `<dataset_override_key>=<path>`. **Omit to use the config's built-in dataset** (e.g. `sft.yaml`'s default). If you bind it, you must set `dataset_override_key`, or the step refuses before any GPU work. |

The step accepts extra inputs (`allow_unknown: true`), so a recipe can bind other
inputs purely to order targets. A gated `hf://` model needs `HF_TOKEN` in the step
env (a GB space secret).

## Output

After training, the step finds the latest checkpoint under `results/<algorithm>/`,
runs `examples/converters/convert_dcp_to_hf.py` on it, verifies the HF directory is
non-empty, and only then registers it as the `model` artifact named by `output`. A
run that fails or produces no checkpoint registers nothing — a half-finished run
never emits a bogus artifact.

## ClearML tracking (optional, server-agnostic)

NeMo-RL has no native ClearML logger, but it logs scalars to TensorBoard, and
ClearML auto-mirrors TensorBoard + stdout + GPU metrics in the process where
`Task.init()` runs. The step wraps the driver in that process, so tracking is
automatic — nothing to add to your training config.

Enablement is `auto` by default: ClearML turns on only when all three of
`CLEARML_API_HOST`, `CLEARML_API_ACCESS_KEY`, `CLEARML_API_SECRET_KEY` are present
in the step env. Provide them via your build's
`config.skypilot.secrets.secret_names_to_use_as_env_variable` (from GB space
secrets). The step **never bakes a server URL** — it works against any ClearML
server, and `CLEARML_OFFLINE_MODE=1` works too. Set `clearml_enabled: false` to
disable even when creds are present (no `clearml` import, no install).

## Compute

```yaml
compute_config:
  num_nodes: 1              # iteration 1 is single-node only; the guard rejects != 1
  num_gpus_per_node: 2      # how many GPUs to request on the node
  gpu_model: A10G           # REQUIRED — the SkyPilot accelerator TYPE for your endpoint
```

GPUs are the one resource the step requests itself. gbserver's SkyPilot launcher
sources cpus/memory from `compute_config` but **not** GPUs, so the step renders a
SkyPilot `accelerators` request of `<gpu_model>:<num_gpus_per_node>` (e.g.
`A10G:2`) and pins NeMo-RL's `cluster.gpus_per_node` to `num_gpus_per_node` so a
multi-GPU node is actually used (the shipped configs default it to 1).

`gpu_model` is **required**: an empty type would request a blank-named
accelerator that SkyPilot silently satisfies with a CPU-only node, so the step
fails at render time if it is unset. Set it to a type your endpoint offers
(`A10G`, `A100`, `H100`, …).

## Example build.yaml

Supervised fine-tuning of a base model, using `sft.yaml`'s built-in dataset — one
target, no dataset binding:

```yaml
granite.build:
  name: nemo-rl-sft
  targets:
    train:
      environment_uri: space://environments/skypilot/aws
      inputs:
        base_model:
          uri: hf:///meta-llama/Llama-3.2-1B
      outputs:
        model:
          uri: "env:///model"
          type: model
      steps:
        - step_uri: space://steps/train/nemo-rl
          config:
            nemo_rl_config:
              algorithm: sft
              config_name: sft.yaml
              output: model
              overrides:
                sft.max_num_steps: 20
            compute_config:
              num_nodes: 1
              num_gpus_per_node: 8
              gpu_model: A100      # the SkyPilot accelerator type (required)
```

To train on your own data, bind a `dataset` and name its Hydra key:

```yaml
      inputs:
        base_model:
          uri: hf:///meta-llama/Llama-3.2-1B
        dataset:
          uri: hf:///datasets/my-org/my-sft-data
      steps:
        - step_uri: space://steps/train/nemo-rl
          config:
            nemo_rl_config:
              algorithm: sft
              config_name: sft.yaml
              output: model
              dataset_override_key: data.train.input_file   # required with a bound dataset
```

## Notes and limitations

- **Single-node only (iteration 1).** `num_nodes` must be `1`; the guard rejects
  anything else. Multi-node Ray launch is a separate iteration.
- **The image bundles the source.** Use the `nemo-rl` **release** image (the default
  `image_id`), not the base `:latest` — the base image carries no `examples/` and the
  step's `setup` fails fast if `<nemo_rl_root>/examples` is missing.
- **Overrides are Hydra keys, not shell.** Values are quoted for you and reach Hydra
  byte-for-byte, so a value with spaces or quotes is never word-split. `$MY_VAR` is
  therefore a literal, not an expansion — parameterise the build with `$${PARAM}` or
  `{{ run_metadata.* }}` instead (both resolved before the step renders).
- **Gated base models need `HF_TOKEN`.** The launcher's inline `hf download` runs on
  the node; a gated `hf://` model requires `HF_TOKEN` in the step env.
