# fm-tune

Runs one [fm-tune](https://github.com/ibm-granite/granite.build) tuning job
(`autotunex/src/fm-tune/main.py`) as a Granite.build step: `space://steps/fm-tune`.
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

## Config: `fm_tune_config`

Every key has a default. The block is named `fm_tune_config` because a hyphen parses as
subtraction in Jinja.

| Key | Default | Maps to |
|---|---|---|
| `source.path` | `""` | Local fm-tune checkout. Exactly one of `path` and `repo` must be set |
| `source.repo` | `""` | Git URL to clone |
| `source.ref` | `main` | Branch, tag or commit for `repo` |
| `source.subdir` | `autotunex/src/fm-tune` | fm-tune's location inside `repo` |
| `extra` | `core` | pip extra installed: `core`, `full` or `mlx` |
| `model` | `""` | `--model_name_or_path` when `base_model` is not bound |
| `train_file` | `""` | `--train_file`, relative to the `dataset` input (required) |
| `validation_file` | `""` | `--validation_file`, relative to the `dataset` input (required) |
| `tuning_algo` | `lora` | `--tuning_algo` |
| `rl_algo` | `""` | `--rl_algo`, omitted when empty |
| `run_name` | `fm-tune` | `--run_name` and `--output_model_name` |
| `autotune` | `true` | `false` adds `--no_autotune` |
| `backend` | `""` | `--backend`, omitted when empty (fm-tune's default, `torch`) |
| `seed` | `""` | `--seed`, omitted when empty (fm-tune's default, 42) |
| `config_preset` | `autotune.yaml` | Bundled `autotune/configs/<preset>`, used when `hpo_config` is empty |
| `hpo_config` | `{}` | Inline HPO config (AutoTuneX's `config_data`), written to a file for `--config_file` |
| `output_path` | `""` | SkyPilot only. `""` means `./model` in the run workdir. Ignored on bash |
| `extra_args` | `""` | Appended to the `main.py` command line, split on whitespace (no quoting) |
| `python` | `""` | Interpreter that creates the venv. Empty means the injected python (bash) or `python3` (SkyPilot). Must be 3.10 or newer |
| `autotunex.server_url` | `""` | `--autotunex_server_url`, omitted when empty |
| `autotunex.job_id` | `""` | `--job_id`, omitted when empty |

### `hpo_config`

`hpo_config` is fm-tune's whole HPO config as a mapping (`tune_config`, `training_config`,
`tuners_config`, ...). When non-empty it replaces the preset entirely, so it must be
complete; it is about 500 lines for a full config. Start from a bundled preset in
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
        model: { uri: file:///data/artifacts/fmtune_{{ binding.path | short_hash }}/ }
      steps:
        - step_uri: space://steps/fm-tune
          config:
            fm_tune_config:
              source:
                repo: https://github.com/ibm-granite/granite.build.git
                ref: v1.2.3          # pin a tag or commit, see "Caches and cleanup"
              train_file: alpaca-mini_train.jsonl
              validation_file: alpaca-mini_validation.jsonl
              tuning_algo: lora
              run_name: granite-lora
              autotune: false
```

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
        - step_uri: space://steps/fm-tune
          config:
            fm_tune_config:
              source:
                repo: https://github.com/ibm-granite/granite.build.git
                ref: v1.2.3
              train_file: alpaca-mini_train.jsonl
              validation_file: alpaca-mini_validation.jsonl
              run_name: granite-lora
              autotune: false
              output_path: /shared/fm-tune/model   # must match the output uri
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
  and in `$GB_SHARED_WORKDIR` (else the run workdir) on SkyPilot. On the local test Slurm
  cluster that is `/shared/.fm-tune-cache`.
- A cached `source.repo` clone is refreshed on every run (`[fm-tune] updating ...`), so a
  warm-cache run still needs network access to the repo. If the fetch fails, the run stops.
- Every commit trained gets its own immutable `tree-<sha>` and its own `venv-<key>`. The
  venv for `extra: core` is several GB. Nothing prunes them. Pin `source.ref` to a tag or
  commit so builds share one tree and one venv.
- To clean up, delete `.fm-tune-cache/tree-*` and `.fm-tune-cache/venv-*` while no fm-tune
  build is running. They are rebuilt on the next run.
- Concurrent builds serialise on a lock inside the cache; a lock whose owner died is
  taken over.

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
| `dataset_name` + `data_format` | `train_file` / `validation_file` as `<dataset_name>_train.<fmt>` and `<dataset_name>_validation.<fmt>` |
| `job_id` | `autotunex.job_id` |
| callback URL | `autotunex.server_url` |

## Known limits

- SkyPilot: `slurm` subtype only. LSF, Kubernetes, AWS and Docker are not supported.
- No container image: the step runs on a bare node and builds its own venv (needs
  Python 3.10 or newer, `git` for `source.repo`, and network access to PyPI).
- No GPU validation. The Slurm build test ran on CPU-only arm64 nodes, so it proves the
  wiring, not GPU training. On CPU, bf16 is downgraded to fp32.
- Online-RL reward functions are not supported yet: AutoTuneX's `inject_reward_function`
  needs an additional-file mechanism this step does not have.
- In the step's tests `eval_loss` is `nan`: a 1-epoch, 4-step run on a random-weight
  model checks the plumbing, not quality. `train_loss` is finite.

## Running (bash)

Submit the build as usual (`build_start` or `gbcli build start`). The step runs through the
bash launcher; inputs arrive as `$LLMB_BASH_INPUT_DATASET` and `$LLMB_BASH_INPUT_BASE_MODEL`.

- The model is written to `<LLMB_BASH_OUTPUT_DIR>/model/` (a subdirectory: `--model_at_root`
  clears its output directory, and `job.log` lives in `LLMB_BASH_OUTPUT_DIR`). The output
  artifact is that subdirectory.
- The log has exactly one `GB_ARTIFACT_ID:model` line, printed on success. The template sets
  `config.bash.skip_finding_output_artifacts: true`, so the bash wrapper does not scan for
  artifacts itself and a failed run registers nothing.
- The cache is `~/.granite.build/.fm-tune-cache`.
- A 1-epoch tiny-model run (tiny-random-Llama, 100 alpaca rows) trains in a few seconds
  after the first venv build.
- Environment variables go through `config.bash.env`, for example
  `FMTUNE_OBJECT_STORE_BYTES`.
