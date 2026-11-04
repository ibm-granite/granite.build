# fm-tune

Runs one [fm-tune](https://github.com/ibm-granite/granite.build) tuning job
(`autotunex/src/fm-tune/main.py`) as a Granite.build step: `space://steps/tune/fm-tune`.
The step supports SFT, LoRA-family tuning and RL algorithms (DPO, KTO, PPO, GRPO, DAPO),
with or without fm-tune's hyperparameter autotuning. The tuned model (or adapter) is
registered as the `model` output.

The step works on its own: with no `autotunex.server_url` it passes no AutoTuneX flags, and
with `autotune: false` it is a plain single tuning run (see "Example: bash"). Reporting to
AutoTuneX is optional (see "Integrating with AutoTuneX").

The step runs fm-tune and nothing else. It needs fm-tune's source layout: `main.py` and
its command line, `autotune/configs/<preset>`, and the `core`, `full` and `mlx` extras in
`pyproject.toml`. A fork of fm-tune works: `source.repo` (with `source.subdir` naming
fm-tune's directory in it) or `source.path` (fm-tune's directory itself; `source.subdir` is
not applied). A different trainer does not.

What a run does:

1. Validates the config. Config-only errors fail the step before it is launched or
   provisioned (see "Config checks"); the script checks again, plus the files themselves,
   before any clone or install.
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

The output directory must be new or empty, and must not be the `base_model` or `dataset`
input directory; anything else fails the run before any clone or install. The step never
deletes anything in it: fm-tune trains in a staging directory beside it,
`<output dir>.partial.<random>` (its `--model_at_root` cleanup runs there), and on success
the step renames that onto the output directory. The rename is atomic and succeeds only
while the output directory is missing or empty, so if another build wrote it meanwhile the
run fails and keeps its model in the staging directory, named in the error. A failed run
leaves the output directory as it was and keeps its staging directory (trials,
checkpoints, logs) for debugging, so a SkyPilot retry into the same output directory just
starts again. Nothing removes old `*.partial.*` directories: delete them once no build is
using them. Write each run to a new directory: one that already holds a model, an earlier
output of this step included, is refused. fm-tune's `--resume_from_checkpoint` cannot be
used through `extra_args`, since each run trains in a new staging directory.

Relative `output_path` and `source.path` values (and relative input paths) are resolved
against the directory the step starts in (the run workdir on SkyPilot) before `main.py` runs,
and the `model` output is registered with the absolute path. Trailing `/`s and `.`
components are dropped from the output path; a `..` component in it is refused.

## Config: `fm_tune_config`

Every key has a default. The block is named `fm_tune_config` because a hyphen parses as
subtraction in Jinja.

| Key | Default | Maps to |
|---|---|---|
| `source.path` | `""` | Local fm-tune checkout. Exactly one of `path` and `repo` must be set |
| `source.repo` | `""` | Git URL to clone (must not start with `-`). Log lines show it with any `user:token@` replaced by `***@` |
| `source.ref` | `main` | Branch, tag, or full 40-character commit SHA for `repo` (quote a SHA in YAML; an uppercase SHA is lowercased, so it shares the lowercase one's cache) |
| `source.subdir` | `autotunex/src/fm-tune` | fm-tune's location inside `repo` |
| `extra` | `core` | pip extra installed: `core`, `full` or `mlx` |
| `model` | `""` | `--model_name_or_path` when `base_model` is not bound |
| `train_file` | `""` | `--train_file`, relative to the `dataset` input (required). No leading `/` and no `..` component (`sub/train.jsonl` and `a..b.jsonl` are fine) |
| `validation_file` | `""` | `--validation_file`, relative to the `dataset` input (required). Same rules as `train_file` |
| `tuning_algo` | `lora` | `--tuning_algo`, omitted when empty (fm-tune's default is `none`). It and `rl_algo` cannot both be empty or `none` |
| `rl_algo` | `""` | `--rl_algo`, omitted when empty |
| `run_name` | `fm-tune` | `--run_name` and `--output_model_name`. Letters, digits, `.`, `_` and `-` only |
| `autotune` | `true` | `true` or `false` (any case); `false` adds `--no_autotune`. Anything else fails the step |
| `backend` | `""` | `--backend`, omitted when empty (fm-tune's default, `torch`) |
| `seed` | `""` | `--seed`, omitted when empty (fm-tune's default, 42) |
| `config_preset` | `autotune.yaml` | Bundled `autotune/configs/<preset>`, used when `hpo_config` is empty |
| `hpo_config` | `{}` | Inline HPO config: fm-tune's search space and training settings, written to a YAML file and passed as `--config_file`. Empty uses `config_preset`. (AutoTuneX calls it `config_data`.) |
| `config_name` | empty | File name of the inline `hpo_config` (`<config_name>.yaml`); empty = `fm_tune_hpo`. Ignored with a preset, which keeps its file name. Letters, digits, `.`, `_` and `-` only. When reporting to AutoTuneX, it is also the config's name there |
| `output_path` | `""` | SkyPilot only. `""` means `./model` in the run workdir. Must not contain `"`, `\` or a control character, end in a space, or have a `..` component. Ignored on bash |
| `extra_args` | `""` | Appended to the `main.py` command line, split on whitespace and newlines (no quoting) |
| `python` | `""` | Interpreter that creates the venv. Empty means the injected python (bash) or `python3` (SkyPilot). Must be 3.10 or newer |
| `autotunex.server_url` | `""` | `--autotunex_server_url`, omitted when empty |
| `autotunex.job_id` | `""` | `--job_id`. Empty means the build id (`GB_BUILD_ID`) when `autotunex.server_url` is set, else omitted. Set it only to the UUID of a job AutoTuneX already created (its UI launch does this): fm-tune also sends it as the build id, so any other value leaves the run unregistered or under the default owner |

### Config checks

These fail the step before it is launched or provisioned, with an `fm-tune: ...` message
naming the key. gbserver runs them when it sets the step up to run, once the step's inputs
are bound, so in a multi-target build an fm-tune target fails after the targets it depends
on have run, not at submit. `fmtune_run.sh` repeats the first nine for a direct run (it
defaults an empty `run_name` to `fm-tune` instead of failing):

- `train_file` or `validation_file` empty, absolute, or with a `..` component;
- not exactly one of `source.path` and `source.repo`;
- neither `model` set nor `base_model` bound;
- `run_name` empty or outside `[A-Za-z0-9._-]`;
- `config_name` outside `[A-Za-z0-9._-]`;
- `source.repo` or `source.ref` starting with `-`;
- `autotune` other than `true` or `false` (any case; empty or null means `true`);
- `tuning_algo` and `rl_algo` both empty or `none` (fm-tune's pipeline refuses that, but only
  after the clone and venv build); an online `rl_algo` (`ppo`, `grpo`, `dapo`) needs no
  `tuning_algo`;
- an offline `rl_algo` (`dpo`, `kto`) with `tuning_algo` empty or `none` (fm-tune refuses
  that late too);
- a list or mapping for any key except `hpo_config` (write it as a string);
- `source` or `autotunex` not a mapping, or `hpo_config` not a mapping or null;
- SkyPilot only: `compute_config` or `launcher_config` set but not a mapping; a `num_nodes`
  other than 1 in either; a `num_nodes` that is not a whole number (`""` included: gbserver
  cannot read it) or a `num_gpus_per_node` that is not a whole number >= 0 (`""` is unset);
  and `num_gpus_per_node` > 0 without `launcher_config.resources.accelerators` (see "Compute").

A `null` value renders as empty, not as the string `None`. That is not the template
default: `tuning_algo: null` passes no `--tuning_algo`, so fm-tune's own default, `none`,
applies, which fails the check above unless `rl_algo` is an online RL algorithm.

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

The step writes it to `fm_tune_hpo.yaml` (YAML, not JSON) in a temp directory private to
the run (`$TMPDIR/fm-tune-hpo.*`, or `/tmp`; removed on every exit), or to `<config_name>.yaml` when `config_name` is set. fm-tune registers the config
with AutoTuneX under that file name. The same name with the same content reuses one config;
the same name with different content is stored as `<name>-1`, `<name>-2`, and so on. For a
name per build, template it: `config_name: "lora-sweep-{{ run_metadata.build_id[:8] }}"`.
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
              resources:
                accelerators: "H100:1"   # the GPU request; see "Compute"
            compute_config:
              num_cpus_per_node: 8
              num_gpus_per_node: 1
```

The SkyPilot step supports the `slurm` subtype only.

### Compute

```yaml
compute_config:
  num_nodes: 1               # optional; any other value fails the step before it is launched
  num_cpus_per_node: 8       # requested from Slurm as cpus
  num_gpus_per_node: 1       # > 0 requires launcher_config.resources.accelerators
launcher_config:
  resources:
    accelerators: "H100:1"   # <type>:<count>, what Slurm is actually asked for
```

gbserver's SkyPilot launcher requests CPUs (and, off Slurm/LSF, memory) from
`compute_config`, but not GPUs ([issue #536](https://github.com/ibm-granite/granite.build/issues/536)):
`num_gpus_per_node` alone requests nothing and the job would get a CPU-only node. So the
step asks for the GPUs to be named explicitly: with `num_gpus_per_node` > 0 and no
`launcher_config.resources.accelerators`, the step fails before it is launched or
provisioned. Set `accelerators` to `<type>:<count>` for your cluster (quote it). If you do
not know the cluster's GPU type, Slurm also takes the untyped form `":<count>"` (e.g. `":2"`,
`--gres=gpu:2`; see `docs/environments/skypilot-slurm.md`). The step template does not
write `accelerators` itself and leaves `resources: {}`, so with no GPUs (`num_gpus_per_node: 0`
or unset) it requests a CPU-only node, as the Slurm build test does. GPU runs are not
verified (see "Known limits").

`num_nodes` must be 1 (or unset) both in `compute_config` and in `launcher_config`, which
gbserver reads too and which wins: with more nodes SkyPilot runs the job on every node and the
runs race on the cache and the output directory.

## Caches and cleanup

- The cache lives at `<GB_HOME>/.fm-tune-cache` (beside `workdir/`; `GB_HOME` defaults to
  `~/.granite.build`) on bash. If the bash output dir has no `/workdir/` in its path, the cache
  goes two levels above it instead (`<output dir>/../../.fm-tune-cache`). On SkyPilot it is
  `${GB_SHARED_WORKDIR:-$PWD}/.fm-tune-cache-<uid>` (one per user, `id -u`):
  shared across nodes only when `GB_SHARED_WORKDIR` is set (the local Slurm environment sets
  `/shared`, giving `/shared/.fm-tune-cache-<uid>`). The uid is in the first level under the
  shared root, so there is no shared parent: a shared `.fm-tune-cache` would be created by the
  first user at their umask (usually `0755`), and every other user's `mkdir` under it fails.
- The run creates a new cache dir `0700`. It fails, naming the dir, if the dir cannot be
  created, is not owned by the user running the build, or is group- or world-writable (the
  run executes code from it). An existing dir is never `chmod`ed; the message gives the fix,
  for example `chmod go-w <dir>`. A symlink to a dir of your own (say, on a bigger disk) is
  followed and the target checked. A symlink owned by another user is refused. Trailing `/`s
  are dropped first.
- Only the cache dir itself is checked, not its parents: `GB_SHARED_WORKDIR` (or the bash
  `GB_HOME`) is assumed to be sticky or admin-owned. fm-tune's own `autotune/tmpdir.py`, by
  contrast, checks each level it creates. macOS ACLs on the dir are not checked either.
- Migrating: an earlier version used `.fm-tune-cache/<uid>` on SkyPilot. Delete those dirs
  once no build uses them; the next run rebuilds its cache at the new path.
- A cached `source.repo` clone of a branch or tag is refreshed on every run
  (`[fm-tune] updating ...`), so a warm-cache run still needs network access to the repo.
  If the fetch fails, the run stops.
- Every commit trained gets its own immutable `tree-<sha>` and its own `venv-<key>`. The
  venv key also covers the interpreter (its real path, the machine type and the OS), so a
  Python upgrade or a node of another type builds a new venv instead of reusing one whose
  base interpreter is gone or wrong. The
  venv for `extra: core` is large (1.5 GB measured on the arm64 CPU Slurm nodes). Nothing
  prunes them. Pin `source.ref` to a tag so builds share one tree and one venv.
- To clean up, delete `tree-*`, `venv-*`, `src-*` (the clone mirrors), any stale `*.lock`
  directories and any `*.lock.gone.*` left by a failed removal in the cache dir, only while no
  fm-tune build is running. They are rebuilt on the next run.
- A lock in the cache serialises only the clone and tree extraction and the venv build, not
  training, so concurrent builds still train in parallel. Each lock is a `*.lock` directory
  whose `pid` file records its owner as `<host>:<pid>`. The host is `uname -n`, which
  containers can share while each has its own pids, so a live owner can look gone. A lock is
  never taken over automatically. A run that finds one held waits up to
  `FMTUNE_LOCK_TIMEOUT` seconds (default 1800), then fails. If the owner ran on the same
  host and is no longer running, it fails at once instead of waiting. A timeout does not mean
  the owner is gone: a venv build (`extra: full` on a slow shared filesystem) can outlast it,
  so raise `FMTUNE_LOCK_TIMEOUT` and re-run first. Either message names the lock directory
  and gives the exact `rm -rf '<dir>'` to clear it once no fm-tune build is running.
- A stale lock is left by a build killed hard (`kill -9`, the OOM killer, a lost node) during
  the clone, the tree extraction or the venv build, or interrupted while taking the lock. To
  clear it, remove that one `*.lock` directory, and only once no fm-tune build that might hold
  it is running.

## Memory knob: `FMTUNE_OBJECT_STORE_BYTES`

fm-tune starts Ray, whose object store defaults to 2 GiB when there is no CUDA. On a small
machine set `FMTUNE_OBJECT_STORE_BYTES` (for example `268435456`, 256 MiB) in the
launcher env: `bash.env` on bash, `launcher_config.envs` on SkyPilot. The step's own tests
do this and also use a 1-epoch config.

## Integrating with AutoTuneX

Optional: skip this section for a plain tuning run. AutoTuneX's `LaunchContext` maps onto `fm_tune_config` like this, so AutoTuneX can emit
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
| `config_name` | `config_name` |
| `dataset_name` + `data_format` | `train_file` / `validation_file` as `<dataset_name>_train.<fmt>` and `<dataset_name>_validation.<fmt>` |
| `job_id` | `autotunex.job_id` |
| callback URL | `autotunex.server_url` |

- A `LaunchContext` field that is `None` (`seed`, `rl_tuner_type`, `tuning_type`) may be
  written as `null`: the step treats a null like an empty value, so the flag is left out
  (`tuning_algo: null` therefore means fm-tune's default, `none`, not the template's `lora`;
  with no `rl_algo` either, the step refuses it, see "Config checks").
  A list or mapping for any key other than `hpo_config` fails the step before it is
  launched; write it as a string.
- `LOG_PATH` is set by the step, unless already set, to `<output dir>.logs`: a sibling of the
  output directory, so `--model_at_root` (which clears the staging directory on success)
  keeps the logs, and builds whose output directories share a parent keep theirs apart.
  AutoTuneX need not export it.
- AutoTuneX's `CUDA_HOME` export goes in the launcher env: `bash.env` on bash,
  `launcher_config.envs` on SkyPilot.
- One bridge-enabled fm-tune target per build. Every target in a build gets the same
  `GB_BUILD_ID`, so with `autotunex.server_url` set and no `job_id`, a second target
  registers under the first target's job: AutoTuneX shows one job, with the first
  target's config and dataset and both targets' trials and logs. Nothing fails. Put
  each bridge-enabled fm-tune run in its own build.

## Known limits

- `source.ref` takes a branch, a tag, or a full 40-character commit SHA, written in quotes in
  YAML. A SHA is the reproducible choice: it never moves, and needs no network on a warm cache.
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
- One AutoTuneX-reporting (`autotunex.server_url`) fm-tune target per build; see
  "Integrating with AutoTuneX".
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
- The cache is `${GB_SHARED_WORKDIR:-$PWD}/.fm-tune-cache-$(id -u)`, one per user with no
  shared parent dir (see "Caches and cleanup"): a user's nodes share one venv only when
  `GB_SHARED_WORKDIR` is set (the local Slurm environment sets `/shared`).
- `config.poll_interval_seconds` (beside `fm_tune_config`, default `300`) is how often the
  SkyPilot monitor polls the job's status, so a finished run can wait up to that long
  before the build sees it. Lower it for short runs.
- SkyPilot builds write no local `job.log`. The step's output is in
  `~/sky_logs/<cluster>/1-<cluster>/run.log` on the machine running gbserver.
- Launcher env goes through `launcher_config.envs`, for example
  `FMTUNE_OBJECT_STORE_BYTES`. Size the job with `compute_config`.
