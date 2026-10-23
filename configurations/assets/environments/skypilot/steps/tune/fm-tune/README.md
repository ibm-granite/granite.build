# fm-tune

Runs one [fm-tune](https://github.com/ibm-granite/granite.build) tuning job
(`autotunex/src/fm-tune/main.py`) as a Granite.build step: `space://steps/tune/fm-tune`.
The step supports SFT, LoRA-family tuning and RL algorithms (DPO, KTO, PPO, GRPO, DAPO),
with or without fm-tune's hyperparameter autotuning. The tuned model (or adapter) is
registered as the `model` output.

What a run does:

1. Validates the config before any clone or install (model, source, split files).
2. Resolves the fm-tune source: a local checkout (`source.path`) or a git clone
   (`source.repo` at `source.ref`).
3. Builds, or reuses, a cached Python venv (`pip install -e <fm-tune>[<extra>]`).
4. Writes the HPO config (`hpo_config` inline, or a bundled preset).
5. Runs `python main.py ... --model_at_root --cleanup --save_history`.
6. On success only, prints the `model` artifact marker. A failed run registers nothing and
   the step fails with fm-tune's exit status.

## Inputs and output

| Name | Kind | Type | Required | Meaning |
|---|---|---|---|---|
| `dataset` | input | dataset | yes | Directory holding the train and validation split files |
| `base_model` | input | model | no | Model directory. If bound, it wins over `fm_tune_config.model` |
| `model` | output | model | yes | The tuned model or adapter files, at the root of the output directory (`--model_at_root`) |

Extra inputs may be bound (`allow_unknown: true`) so a recipe can order targets.

The output directory must be new, empty, or a previous model output (`config.json`,
`adapter_config.json`, `model.safetensors` or `adapters.safetensors` at its root), and must not
be the `base_model` or `dataset` input directory. Anything else fails the run before any clone
or install, because on success `--model_at_root` deletes every other entry in it. A
failed run leaves its trial and checkpoint directories (`ray_results/`, `outputs/`,
`final_checkpoints/`, ...) there with no model, so a retry into the same `output_path` is
refused: delete that directory or use a new `output_path`. For the same reason fm-tune's
`--resume_from_checkpoint` cannot be used through `extra_args`.

Relative `output_path` and `source.path` values (and relative input paths) are resolved
against the directory the step starts in (the run workdir on SkyPilot) before `main.py` runs,
and the `model` output is registered with the absolute path. A trailing `/` is dropped.

## Config: `fm_tune_config`

Every key has a default. The block is named `fm_tune_config` because a hyphen parses as
subtraction in Jinja.

| Key | Default | Maps to |
|---|---|---|
| `source.path` | `""` | Local fm-tune checkout. Exactly one of `path` and `repo` must be set |
| `source.repo` | `""` | Git URL to clone (must not start with `-`) |
| `source.ref` | `main` | Branch or tag for `repo` (a bare commit SHA is not supported) |
| `source.subdir` | `autotunex/src/fm-tune` | fm-tune's location inside `repo` |
| `extra` | `core` | pip extra installed: `core`, `full` or `mlx` |
| `model` | `""` | `--model_name_or_path` when `base_model` is not bound |
| `train_file` | `""` | `--train_file`, relative to the `dataset` input (required) |
| `validation_file` | `""` | `--validation_file`, relative to the `dataset` input (required) |
| `tuning_algo` | `lora` | `--tuning_algo` |
| `rl_algo` | `""` | `--rl_algo`, omitted when empty |
| `run_name` | `fm-tune` | `--run_name` and `--output_model_name`. Letters, digits, `.`, `_` and `-` only |
| `autotune` | `true` | `false` adds `--no_autotune` |
| `backend` | `""` | `--backend`, omitted when empty (fm-tune's default, `torch`) |
| `seed` | `""` | `--seed`, omitted when empty (fm-tune's default, 42) |
| `config_preset` | `autotune.yaml` | Bundled `autotune/configs/<preset>`, used when `hpo_config` is empty |
| `hpo_config` | `{}` | Inline HPO config (AutoTuneX's `config_data`), written to a file for `--config_file` |
| `output_path` | `""` | SkyPilot only. `""` means `./model` in the run workdir. Must not contain `"`, `\` or a control character, or end in a space. Ignored on bash |
| `extra_args` | `""` | Appended to the `main.py` command line, split on whitespace and newlines (no quoting) |
| `python` | `""` | Interpreter that creates the venv. Empty means the injected python (bash) or `python3` (SkyPilot). Must be 3.10 or newer |
| `autotunex.server_url` | `""` | `--autotunex_server_url`, omitted when empty |
| `autotunex.job_id` | `""` | `--job_id`, omitted when empty |

### `hpo_config`

`hpo_config` is fm-tune's whole HPO config as a mapping (`tune_config`, `training_config`,
`tuners_config`, ...). When non-empty it replaces the preset entirely, so it must be
complete. The bundled presets are about 1250 lines; the step's test fixtures use a trimmed,
LoRA-only copy of about 500 lines. Start from a bundled preset in
`autotunex/src/fm-tune/autotune/configs/` (`autotune.yaml`, `autotune_mac.yaml`) and copy
what you need. An excerpt:

```yaml
hpo_config:
  tune_config:
    search_alg: { default: random, values: [lds, hyperopt, random, bohb, blds], type: str, required: true }
    num_samples: { default: 4, min_val: 1, max_val: 10000, type: int, required: true }
  training_config:
    num_train_epochs: { default: 1, type: int, required: true }
    max_length: { default: 1024, type: int, required: true }
```

The step writes it to `fm_tune_hpo.yaml` (YAML, not JSON) next to the output directory.
fm-tune reads it with `yaml.safe_load` (YAML 1.1), which loads JSON's `5e-06` as a string, so
the step re-emits it as YAML (`5.0e-06`) to keep exponent floats as floats.

For a quick wiring check, an inline 1-epoch config is enough. The mac preset with
`max_length: 1024` is what the step's own tests use.

## Example: bash

```yaml
granite.build:
  name: fm-tune-bash
  targets:
    tune:
      environment_uri: space://environments/bash
      inputs:
        dataset: { uri: file:///data/datasets/alpaca-mini/ }
        base_model: { uri: hf:///ibm-granite/granite-4.0-h-350m }
      outputs:
        model: { uri: "env://{{ binding.path }}" }   # keeps the model where the step wrote it
      steps:
        - step_uri: space://steps/tune/fm-tune
          config:
            fm_tune_config:
              source:
                repo: https://github.com/ibm-granite/granite.build.git
                ref: v1.2.3          # pin a tag, see "Caches and cleanup"
              train_file: alpaca-mini_train.jsonl
              validation_file: alpaca-mini_validation.jsonl
              tuning_algo: lora
              run_name: granite-lora
              autotune: false
```

Other output stores (for example `hf://`) should also work as `model` outputs; only the
`env://` form above was tested.

Use `source: { path: /abs/path/to/granite.build/autotunex/src/fm-tune }` instead of `repo`
to run a local checkout.

## Example: SkyPilot / Slurm

```yaml
granite.build:
  name: fm-tune-slurm
  targets:
    tune:
      environment_uri: space://environments/skypilot/slurm
      inputs:
        dataset: { uri: file:///shared/datasets/alpaca-mini/ }
        base_model: { uri: hf:///ibm-granite/granite-4.0-h-350m }
      outputs:
        model:
          type: model
          uri: env:///shared/fm-tune/model     # on the shared filesystem
      steps:
        - step_uri: space://steps/tune/fm-tune
          config:
            fm_tune_config:
              source:
                repo: https://github.com/ibm-granite/granite.build.git
                ref: v1.2.3
              train_file: alpaca-mini_train.jsonl
              validation_file: alpaca-mini_validation.jsonl
              run_name: granite-lora
              autotune: false
              output_path: /shared/fm-tune/model   # must match the output uri; one per build
            launcher_config:
              envs:
                FMTUNE_OBJECT_STORE_BYTES: "268435456"   # optional, small machines
            compute_config:
              num_cpus_per_node: 8
              num_gpus_per_node: 1
```

The SkyPilot step supports the `slurm` subtype only. Size the job with `compute_config`.

## Caches and cleanup

- The cache lives at `.fm-tune-cache`: beside `workdir/` under `~/.granite.build/` on bash,
  and in `${GB_SHARED_WORKDIR:-$PWD}/.fm-tune-cache/<uid>` on SkyPilot (one per user, `id -u`):
  shared across nodes only when `GB_SHARED_WORKDIR` is set (the local Slurm environment sets
  `/shared`, giving `/shared/.fm-tune-cache/<uid>`).
- A cached `source.repo` clone is refreshed on every run (`[fm-tune] updating ...`), so a
  warm-cache run still needs network access to the repo. If the fetch fails, the run stops.
- Every commit trained gets its own immutable `tree-<sha>` and its own `venv-<key>`. The
  venv for `extra: core` is large (1.5 GB measured on the arm64 CPU Slurm nodes). Nothing
  prunes them. Pin `source.ref` to a tag so builds share one tree and one venv.
- To clean up, delete `.fm-tune-cache/tree-*`, `.fm-tune-cache/venv-*`,
  `.fm-tune-cache/src-*` (the clone mirrors) and any stale `*.lock` directories, only while
  no fm-tune build is running. They are rebuilt on the next run.
- A lock in the cache serialises only the clone and tree extraction and the venv build, not
  training, so concurrent builds still train in parallel. Each lock records its owner as
  `<host>:<pid>`. A lock whose owner ran on the same host and died is taken over. A lock held
  from another host is never taken over: the run waits up to `FMTUNE_LOCK_TIMEOUT` seconds
  (default 1800) and then fails, naming the lock directory. Remove it only if no fm-tune build
  is running. A lock with no owner recorded is taken over once it is older than the timeout.

## Memory knob: `FMTUNE_OBJECT_STORE_BYTES`

fm-tune starts Ray, whose object store defaults to 2 GiB when there is no CUDA. On a small
machine set `FMTUNE_OBJECT_STORE_BYTES` (for example `268435456`, 256 MiB) in the
launcher env: `bash.env` on bash, `launcher_config.envs` on SkyPilot. The step's own tests
do this and also use a 1-epoch config.

## Integrating with AutoTuneX

AutoTuneX's `LaunchContext` maps onto `fm_tune_config` like this, so AutoTuneX can emit
this step's build shape:

| `LaunchContext` field | `fm_tune_config` key |
|---|---|
| `model` | `model` (and the `base_model` input) |
| `experiment_name` | `run_name` |
| `tuning_type` | `tuning_algo` |
| `rl_tuner_type` | `rl_algo` |
| `autotune` | `autotune` |
| `seed` | `seed` |
| `config_data` | `hpo_config` |
| `config_name` | not carried: the step always writes `fm_tune_hpo.yaml` |
| `dataset_name` + `data_format` | `train_file` / `validation_file` as `<dataset_name>_train.<fmt>` and `<dataset_name>_validation.<fmt>` |
| `job_id` | `autotunex.job_id` |
| callback URL | `autotunex.server_url` |

- A `LaunchContext` field that is `None` (`seed`, `rl_tuner_type`, `tuning_type`) may be
  written as `null`: the step treats a null like an empty value, so the flag is left out
  (`tuning_algo: null` therefore means fm-tune's own default, not the template's `lora`).
  A list or mapping for any key other than `hpo_config` fails the build when it is created;
  write it as a string.
- `LOG_PATH` is set by the step (to the output directory's parent, unless already set), so
  AutoTuneX need not export it.
- AutoTuneX's `CUDA_HOME` export goes in the launcher env: `bash.env` on bash,
  `launcher_config.envs` on SkyPilot.
- Follow-up: `config_name` (AutoTuneX names the config file after it) is not carried.

## Known limits

- `source.ref` must be a branch or tag: the clone uses `git clone --branch`, which rejects a
  bare commit SHA.
- Verified only by the 1-epoch fixtures (macOS/MPS on bash, CPU-only arm64 Slurm); see
  "Verified" in the step's developer README.
- SkyPilot: `slurm` subtype only. LSF, Kubernetes, AWS and Docker are not supported.
- No container image: the step runs on a bare node and builds its own venv (needs
  Python 3.10 or newer, `git` for `source.repo`, and network access to PyPI).
- No GPU validation. The Slurm build test ran on CPU-only arm64 nodes, so it proves the
  wiring, not GPU training. On CPU, bf16 is downgraded to fp32.
- On a CUDA node the default `config_preset: autotune.yaml` asks for
  `use_flash_attention: flash_attention_2`, but `extra: core` does not install `flash-attn`
  (only `full` does; on MPS and CPU fm-tune falls back to eager itself). On CUDA use
  `extra: full`, or an `hpo_config` with `use_flash_attention` set to `eager`. Unverified,
  like all GPU use.
- Online-RL reward functions are not supported yet: AutoTuneX's `inject_reward_function`
  needs an additional-file mechanism this step does not have.
- In the step's tests `eval_loss` is `nan`: a 1-epoch, 4-step run on a random-weight
  model checks the plumbing, not quality. `train_loss` is finite.

## Running (SkyPilot / Slurm)

Submit the build with `environment_uri: space://environments/skypilot/slurm`.

- The step ships `src/` with `file_mounts` and runs `bash ./src/fmtune_run.sh`. Every config
  value is shell-quoted by the step template.
- Set `output_path` explicitly, to the same directory the `env://` output URI names. With
  `output_path` empty the model goes to `./model` in the run workdir, which is on the shared
  filesystem when the environment sets `shared_workdir`. But that workdir is
  `<shared_workdir>/builds/<build_id>/runs/<targetrun_id>`, known only at run time, so the
  output URI can't name it.
- Give every build its own `output_path` (and the matching output URI). The example's fixed
  `/shared/fm-tune/model` is shared by every build that uses it, so concurrent builds
  overwrite each other's model. The step has no per-build default, and no per-build template
  for it has been tested; write a distinct path into each build.yaml (for example
  `/shared/fm-tune/<run_name>/model`, with a new `run_name` per build).
- The node needs `python3` 3.10 or newer on its PATH (or set `fm_tune_config.python`), plus
  `git` for `source.repo`. The local Slurm test nodes needed a python3.10 for this.
- The cache is `${GB_SHARED_WORKDIR:-$PWD}/.fm-tune-cache/$(id -u)`, one per user: a user's
  nodes share one venv only when `GB_SHARED_WORKDIR` is set (the local Slurm environment
  sets `/shared`).
- SkyPilot builds write no local `job.log`. The step's output is in
  `~/sky_logs/<cluster>/1-<cluster>/run.log` on the machine running gbserver.
- Launcher env goes through `launcher_config.envs`, for example
  `FMTUNE_OBJECT_STORE_BYTES`. Size the job with `compute_config`.
