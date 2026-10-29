# eval/livecodebench-generate (SkyPilot)

Generates [LiveCodeBench](https://livecodebench.github.io/) ([code](https://github.com/LiveCodeBench/LiveCodeBench)) code
generation completions for the bound `model` with [vLLM](https://docs.vllm.ai), and
registers them as the step's `completions` artifact. Needs a GPU. Score them with
`eval/livecodebench-score`.

LiveCodeBench's own runner only accepts the models it lists, and that list also
chooses the prompt format. This step uses LiveCodeBench as a library instead, so any
chat model works:

- the problems come from LiveCodeBench, for a release and an optional date window;
- each prompt is LiveCodeBench's generic chat prompt, formatted with **the model's own
  chat template**;
- vLLM generates `n` samples per problem;
- the code is taken out of each sample with LiveCodeBench's own extractor, after
  dropping any reasoning that ends in `</think>`.

The step doesn't run the generated code; `eval/livecodebench-score` does, in a
separate job without credentials.

## Referencing the step

```yaml
steps:
  - step_uri: space://steps/eval/livecodebench-generate
```

## Inputs and outputs

### Inputs

| Input | Required | Type | Typical source |
|---|---|---|---|
| `model` | yes | model | `uri: hf:///models/<org>/<name>` (the SkyPilot launcher downloads it before setup), or a `binding` to the checkpoint another target produced. It needs a chat template. |
| `dataset` | yes | dataset | `uri: hf:///datasets/livecodebench/code_generation_lite`: the benchmark's problems. The SkyPilot launcher downloads it before setup. |

The step reads them as `{{ bindings.model.binding.path }}` and
`{{ bindings.dataset.binding.path }}`, and runs offline after that. A target that
doesn't bind both fails build validation before anything is queued.

**The dataset is large:** `release_v6` uses six JSONL files, 4.2 GB in all, because
it carries every problem's hidden tests. The date window only narrows what is
generated; the whole release is still downloaded.

### Outputs

| Output | Type | What it is |
|---|---|---|
| `completions` | `dataset` | The `output_path` directory: `completions.json` (the extracted code, which the score step runs), `raw_outputs.json` (every sample as generated), and `generation_meta.json` (settings, versions, the number of samples with no code, and a hash of the problem set, which the score step checks). |

## Config contract (`lcb_generate_config`)

| Field | Type | Required | Purpose |
|---|---|---|---|
| `release_version` | string | **required** | LiveCodeBench release, e.g. `release_v6` (1055 problems). The score step must use the same one. |
| `start_date`, `end_date` | string | optional | Date window (`YYYY-MM-DD`, inclusive) on the problems' contest dates. Empty means no bound on that side. One contest day (e.g. `2025-04-05`, 6 problems) is a quick smoke test. |
| `max_tokens` | int | **required** | Maximum new tokens per sample. See [Choosing max_tokens](#choosing-max_tokens). |
| `n` | int | optional (default `1`) | Samples per problem; pass@1 is averaged over them. LiveCodeBench's own default is 10. |
| `temperature`, `top_p` | float | optional (`0.2`, `0.95`) | Sampling. |
| `enable_thinking` | string | optional | `"true"` or `"false"` is passed to the chat template as `enable_thinking`, for models whose template has that switch (granite-4.2's does, and defaults to on). Empty keeps the template's default. |
| `tensor_parallel_size` | int | optional (default `1`) | vLLM tensor parallelism. Set it to the GPUs the build requests. |
| `dtype` | string | optional (default `bfloat16`) | vLLM weight dtype. |
| `gpu_memory_utilization` | float | optional (default `0`) | vLLM's GPU memory share; `0` keeps vLLM's default. |
| `max_model_len` | int | optional (default `0`) | vLLM context length; `0` uses the model's own. Lower it if the KV cache doesn't fit. |
| `trust_remote_code` | bool | optional (default `false`) | For models that ship custom code. |
| `output_path` | string | optional (default `output`) | Where the completions are written. |
| `image` | string | optional (default `python:3.12-slim`) | Public container image, rendered as SkyPilot `docker:<image>`. Debian/Ubuntu-based, run as root, with `python3` (3.11 or newer) and `pip`; setup installs `git` with `apt-get` if missing. `""` runs on the bare launcher node: use that on a SLURM cluster without the Pyxis plugin. The image needs no CUDA: vLLM's torch wheel carries the runtime, and the host provides the driver. |
| `install` | bool | optional (default `true`) | Install vLLM and LiveCodeBench into a venv at setup, on a Python 3.12 that uv downloads into the workdir (not the node's own `python3`, which only runs pip to install uv). Set `false` only when `image` already provides them and a checkout at `lcb_checkout`. |
| `vllm_package` | string | optional (default `vllm==0.32.0`) | vLLM release; it brings torch and transformers. Pick one whose torch build matches your GPU driver. |
| `lcb_commit`, `lcb_repo`, `lcb_checkout` | string | optional | LiveCodeBench commit, source and checkout directory. Keep `lcb_commit` equal to the score step's. |
| `pip_index_url` | string | optional | Package index for the install. |

## Choosing max_tokens

A sample that runs out of tokens before it finishes its code block has no code, and
the score step counts it as wrong. A build can then succeed with a score near 0.
The step prints, and records in `generation_meta.json`, how many samples had no
code (`empty_extractions`, `empty_fraction`). **Check it after the first run.**

- **Models that reason first** (thinking on) can spend thousands of tokens before
  the answer: give them a large budget, e.g. 16384 or more, and check that the
  context (`max_model_len`) is large enough for the prompt plus that budget.
- **For a quick check of the setup,** turn thinking off (`enable_thinking: "false"`)
  where the model supports it, and use a few thousand tokens.

A prompt is a few hundred tokens (about 400 for a typical problem).

## Example build.yaml

See `eval/livecodebench-score`'s README for a build that generates and scores one
contest day of `release_v6`. The generate target on its own:

```yaml
granite.build:
  name: livecodebench-generate-example
  version: 0.0.1
  targets:
    generate:
      environment_uri: space://environments/skypilot/aws
      inputs:
        model:
          uri: hf:///models/ibm-granite/granite-4.2-3b
        dataset:
          uri: hf:///datasets/livecodebench/code_generation_lite
      outputs:
        completions:
          uri: "hf://huggingface.co/datasets/<your-org>/lcb-completions_{{ binding.path | short_hash }}"
          type: dataset
      steps:
        - step_uri: space://steps/eval/livecodebench-generate
          config:
            launcher_config:
              resources: {accelerators: "A10G:1", cpus: 8, memory: 32}
            lcb_generate_config:
              release_version: release_v6
              start_date: "2025-04-05"
              end_date: "2025-04-05"
              max_tokens: 8192
              enable_thinking: "false"
```

## Notes and limitations

- **GPUs come from `launcher_config.resources`**, not from `compute_config`. On
  SkyPilot SLURM, give the GPU type (`"H100:1"`, not `":1"`) and set `cpus` and
  `memory` too: the default host for one GPU is small.
- **Setup installs vLLM on every run** (about 8 GB). An `image` that already
  provides vLLM and LiveCodeBench, with `install: false`, avoids it.
- **The prompt is LiveCodeBench's generic one** with the model's chat template, so
  scores are comparable across models run this way. They may differ from numbers
  produced with LiveCodeBench's model-specific prompts.
- **Only the `codegeneration` scenario** is generated.
