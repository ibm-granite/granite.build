# eval/lm-eval (SkyPilot)

Runs [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
(`lm_eval`, PyPI `lm-eval`) against a model bound as the step's input, and registers
the results directory as an output. The example benchmark is `aime25`.

The step installs `lm-eval` (and torch) at setup time into a venv inside a public
container image, `python:3.12-slim` by default, or on the bare launcher node. It builds
no custom image.

> **Developing or testing this step?** See `steps/eval/lm-eval/skypilot/README.md`
> in the granite.build repository for how the step is generated, tested, and published.

## Referencing the step

Point your build's Space at one that provides the step, then reference it by the stable
`space://steps/eval/lm-eval` URI:

```yaml
steps:
  - step_uri: space://steps/eval/lm-eval
```

The step has no `subtypes:` restriction, so it resolves under every SkyPilot environment.

## Inputs and outputs

### Inputs

| Input | Required | Type | Typical source |
|---|---|---|---|
| `model` | yes | model | an `hf://` model (e.g. `hf:///ibm-granite/granite-4.0-350m`), or a training target's output |

The step passes the input's path to `lm_eval` as `--model_args pretrained=<path>`.
Don't set `pretrained` yourself; the step refuses it. Bind extra inputs only to
order targets; the step ignores them.

### Outputs

| Output | Type | What it is |
|---|---|---|
| `results` | `dataset` | The `lm_eval --output_path` directory: `<model path, / as __>/results_<timestamp>.json`, plus `samples_<task>_<timestamp>.jsonl` when `log_samples` is on. Registered only after `lm_eval` exits 0 and a results file exists. |

On SkyPilot SLURM the step's workdir is deleted when the job ends, so an `env://`
results output doesn't persist there: push it with an `hf://` output. The score table
is also printed to the step's log.

The step also records `lm_eval_version` (the installed version) as step metadata.

## Config contract (`lm_eval_config`)

All fields live under the step's `config.lm_eval_config`. The key is `lm_eval_config`,
with an underscore, because `lm-eval` isn't a valid Jinja name. Unknown keys are refused,
so a typo fails at once rather than being ignored.

### Where it runs, and what gets installed

| Field | Type | Default | Purpose |
|---|---|---|---|
| `image` | string | `python:3.12-slim` | Public image the step runs in, as SkyPilot `docker:<image>`. `""` runs on the bare launcher node, which clusters without the Pyxis plugin need. The image must be Debian/Ubuntu based, run as root, and have `python3` and `pip`. GPUs need no CUDA in the image. |
| `install` | bool | `true` | Install `lm-eval` and torch into `./venv` at setup, on a Python 3.12 that uv downloads into the workdir (the node's own `python3` only installs uv). `false` only for an image that already has `lm_eval` on `PATH`. Setting `image` doesn't skip the install. |
| `lm_eval_version` | string | `0.4.13` | The `lm-eval` release installed from PyPI. Required when `install` is true. |
| `extras` | list | `["hf"]` | `lm-eval` pip extras: `lm-eval[<extras>]==<version>`. Use `["vllm"]` for `backend: vllm`; add task extras such as `math` or `ifeval` as a task needs them. |
| `extra_packages` | list | `[]` | Further pip requirements, installed with `lm-eval`. |
| `torch_index_url` | string | `https://download.pytorch.org/whl/cpu` | Index torch is installed from, in its own phase before everything else. The CPU default suits CPU nodes. **On GPU nodes, override it** with the PyTorch index matching the node's CUDA driver (e.g. `https://download.pytorch.org/whl/cu128`), or `""` for PyPI's default torch. |
| `pip_index_url` | string | `https://pypi.org/simple` | Index everything else is installed from. |

### What `lm_eval` runs

Each field maps to one `lm_eval run` flag. `null` or `false` leaves the flag off, so
`lm_eval`'s own default applies.

| Field | Type | Default | `lm_eval run` flag |
|---|---|---|---|
| `backend` | string | `hf` | `--model`. `hf` or `vllm` (the backends that load `pretrained=<path>`). |
| `tasks` | list | `["aime25"]` | `--tasks`, one task or group per item. `lm-eval ls tasks` lists them. Must not be empty. |
| `model_args` | map | `{}` | `--model_args`, one `key=value` per entry, after `pretrained=<model>`. E.g. `{dtype: bfloat16}`. |
| `limit` | int or float | `null` | `--limit` (examples per task, or a fraction below 1). |
| `num_fewshot` | int | `null` | `--num_fewshot`. `0` is passed. |
| `batch_size` | int or string | `null` | `--batch_size` (`8`, `auto`, `auto:4`). |
| `device` | string | `null` | `--device` (`cuda`, `cuda:0`, `cpu`). |
| `gen_kwargs` | map | `{}` | `--gen_kwargs`, one `key=value` per entry. Values are read with Python's `ast.literal_eval`, so lists work: `{max_gen_toks: 2048, until: ["\n\n"]}`. |
| `apply_chat_template` | bool or string | `false` | `--apply_chat_template`; `true` for the model's template, a string to name one. |
| `log_samples` | bool | `false` | `--log_samples`. |
| `trust_remote_code` | bool | `false` | `--trust_remote_code`. |
| `seed` | int or string | `null` | `--seed` (`42`, or `0,1234,1234,1234`). |
| `args` | map | `{}` | Any other `lm_eval run` flag, written without dashes: `{system_instruction: "...", max_batch_size: 8}`. `true` renders a bare switch, `false`/`null` omit it, anything else renders `--<flag> '<value>'`. |

The step sets `--output_path` itself. In `args` it refuses `model`, `model_args`,
`tasks`, `output_path` and `config`, and any flag that has its own field above.

Every value is shell-quoted, so quotes, spaces and `$` reach `lm_eval` unchanged.

### Resources

GPUs on SkyPilot come from the build's `launcher_config.resources` (e.g.
`accelerators: "L4:1"`), not from `compute_config.num_gpus_per_node`. On SkyPilot SLURM
the accelerator must be typed and quoted (`"H100:1"`, not `":1"`).

**Set host memory too, not only the GPU.** With only `accelerators` set, SkyPilot picks
its default size for one GPU, 4 CPUs and 16 GB of host RAM on SkyPilot SLURM. That's
enough for short generations, but long ones (reasoning models, `max_gen_toks` in the
thousands) on the `hf` backend can exhaust host RAM while the GPU still has room. The
job is then OOM-killed. Ask for more alongside the accelerator, e.g. `cpus: 16` and
`memory: 64` (GB).

The right size depends on the model and the task, so size it from a run: after each
job, `sacct -j <job id> -o JobID,State,MaxRSS,ReqMem,Elapsed` shows the peak host RAM of
each job step (`OUT_OF_MEMORY` if it hit the limit). For example, a 3B model on aime25
with 16384-token answers peaked at about 28 GB.

## Long generations (reasoning tasks such as aime25)

- **Token budget:** aime25's own default is 32768 generated tokens. Reasoning models
  often need 10k or more to reach a final `\boxed{}` answer; at 2048 every answer is cut
  off and scores 0. 16384 is a reasonable compromise.
- **Time:** on the `hf` backend at batch 1, generation slows as the answer grows. A 3B
  model on one H200 averaged about 23 tokens/s over five aime25 answers of 3k to 16k
  tokens (33 minutes in all), and about 45 tokens/s with 2k-token answers. Size `limit`
  and `sbatch_options.time` together, from a short first run. For long runs,
  `backend: vllm` (with `extras: ["vllm"]`) is usually much faster.
- **`batch_size`:** prefer a fixed integer. `auto` probes by trying very large
  allocations, which takes minutes for long generations and logs many harmless
  `CUDACachingAllocator ... OOM` warnings before it settles.

## Working directory and caches

`setup` and `run` both start in the step's build workdir (`$GB_BUILD_WORKDIR`). The step
keeps everything there: the venv (`./venv`), the uv cache (`./.uv-cache`), the
Hugging Face cache for task datasets (`./.hf-cache`, as `HF_HOME`) and the results
(`./results`). Nothing goes to the shared workdir or the node user's home.

## Example build.yaml

aime25 on one GPU, with results pushed to a Hugging Face dataset. This is the shape of
the build that proved the step on one H200 on a SkyPilot SLURM GPU cluster (5 problems,
about 33 minutes of generation). Replace `<gpu-cluster>` with your environment;
`space://environments/skypilot/slurm` on its own is the local Docker SLURM cluster,
which has no GPUs.

```yaml
granite.build:
  name: lm-eval-aime25
  targets:
    aime25:
      environment_uri: space://environments/skypilot/slurm/<gpu-cluster>
      inputs:
        model:
          uri: hf:///ibm-granite/granite-4.2-3b
      outputs:
        results:
          uri: hf://huggingface.co/datasets/<org>/aime25-results_{{ binding.path | short_hash }}
      steps:
        - step_uri: space://steps/eval/lm-eval
          config:
            lm_eval_config:
              # Bare launcher node, as in the proof run. The default python:3.12-slim
              # image needs the Pyxis plugin on SLURM and hasn't been run on a GPU
              # cluster yet.
              image: ""
              tasks: ["aime25"]
              # "" = PyPI's default torch, whose CUDA runtime wheels ran on the node's
              # driver. Or a PyTorch index matching the driver, e.g. .../whl/cu128.
              torch_index_url: ""
              model_args:
                dtype: bfloat16
              device: cuda
              batch_size: 1
              apply_chat_template: true
              limit: 5
              gen_kwargs:
                max_gen_toks: 16384
              log_samples: true
            launcher_config:
              resources:
                accelerators: "H200:1"
                # Host RAM: the default 16 GB was OOM-killed on long generations. The
                # proof run asked for 128 and peaked at about 28 (sacct MaxRSS).
                cpus: 16
                memory: 64
              sbatch_options:
                time: "01:00:00"
```

On a cluster with Pyxis, drop `image: ""` to run in the default `python:3.12-slim`
container instead.

## Notes and limitations

- **Backends:** only `hf` and `vllm`, the ones that load a model from a path. API
  backends (`local-completions`, `openai-*`) take a server URL instead and aren't
  supported.
- **No `--config` file:** use the fields above, plus `args`.
- **Task dependencies:** some tasks need extra packages. If `lm_eval` fails with an
  import error, add the matching `lm-eval` extra to `extras`, or the package to
  `extra_packages`.
- **Gated datasets and models:** some need `HF_TOKEN` on the node, and some need
  `trust_remote_code: true`.
- **Harmless log noise:** a `fatal: not a git repository` line (`lm_eval` tries to
  record a git hash) and a `SyntaxWarning` from the aime task's utilities.
