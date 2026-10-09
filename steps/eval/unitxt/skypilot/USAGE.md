# eval/unitxt (SkyPilot)

Evaluates a model with [unitxt](https://www.unitxt.ai) by running its
`unitxt-evaluate` CLI in `hf` mode (a local Hugging Face checkpoint), and registers
the CLI's output directory as the step's `results` artifact. The model is the
target's declared `model` input. The step installs unitxt and the packages its `hf`
mode needs on the node at setup time (or uses a prebuilt image) and passes
`unitxt_config` through to the CLI.

## Referencing the step

```yaml
steps:
  - step_uri: space://steps/eval/unitxt
```

## Inputs and outputs

### Inputs

The step declares one required input, read as `{{ bindings.model.binding.path }}`:

| Input | Required | Type | Typical source |
|---|---|---|---|
| `model` | yes | model | `uri: hf:///models/<org>/<name>` (the SkyPilot launcher downloads it before setup), or a `binding` to the checkpoint another target produced |

The step passes the path to the CLI as `--model_args pretrained=<path>`. A target that
doesn't bind `model` fails build validation before anything is queued.
`unitxt_config.model`, and `pretrained=` inside `unitxt_config.model_args`, are not
read; a target that still sets either fails at run time with a message naming the
input to bind instead.

### Outputs

| Output | Type | What it is |
|---|---|---|
| `results` | `dataset` | The whole `output_path` directory, registered by the step after the CLI succeeds. The CLI names its file `<UTC timestamp>_evaluation_results.json`. |

## Config contract (`unitxt_config`)

| Field | Type | Required | Purpose |
|---|---|---|---|
| `tasks` | string | **required** | Passed as `--tasks`. A catalog recipe (`card=cards.mmlu_pro.engineering`), a benchmark (`benchmarks.tool_calling`), or several joined with `+`. |
| `model_args` | string | optional | Extra `--model_args`, as comma-separated `key=value` pairs (e.g. `torch_dtype=bfloat16,device=cuda`). They are appended after `pretrained=<model path>`, so they must not set `pretrained`, and the JSON form isn't accepted. |
| `batch_size` | int | optional (default `1`) | Passed as `--batch_size`. |
| `limit` | int | optional (default `0`) | Passed as `--limit` (instances per task). `0` omits the flag, so every instance is evaluated. |
| `trust_remote_code` | bool | optional (default `true`) | Passes `--trust_remote_code`. Cards that filter their dataset with a code expression, including `cards.mmlu_pro.*`, refuse to load without it. |
| `output_path` | string | optional (default `output`) | Directory the CLI writes into (`--output_path`). Relative paths resolve in the step's working directory. |
| `unitxt_version` | string | optional (default `1.26.10`) | unitxt release installed at setup. Ignored when `unitxt_image` is set. |
| `torch_package` | string | optional (default `torch`) | torch, which `hf` mode needs and unitxt doesn't depend on. Installed first, on its own. Pin it here, e.g. `torch==2.8.0`. Ignored when `unitxt_image` is set. |
| `torch_index_url` | string | optional (default: `pip_index_url`) | Index for torch only, to pick the build: `https://download.pytorch.org/whl/cpu` on a CPU-only node, or a `.../whl/cuXXX` index to match the node's CUDA driver. PyPI's default Linux wheel bundles CUDA. |
| `hf_packages` | list | optional (default `[transformers, accelerate, tabulate]`) | Installed with unitxt, after torch. `hf` mode needs transformers and accelerate; the CLI's score summary needs tabulate. Pin entries as needed. Ignored when `unitxt_image` is set. |
| `pip_index_url` | string | optional | Package index for the install. |
| `unitxt_image` | string | optional | A container image that already provides `unitxt-evaluate`, torch and `hf_packages`. Skips the install. |

`unitxt_config.model` is no longer read. The step refuses it, so bind the `model`
input instead.

## Example build.yaml

Runs MMLU-Pro engineering against granite-4.2-3b on one AWS A10G, 10 instances:

```yaml
granite.build:
  name: eval-unitxt-example
  version: 0.0.1
  targets:
    evaluate:
      environment_uri: space://environments/skypilot/aws
      inputs:
        model:
          uri: hf:///models/ibm-granite/granite-4.2-3b
      outputs:
        results:
          uri: "env:///results"
          type: dataset
      steps:
        - step_uri: space://steps/eval/unitxt
          config:
            launcher_config:
              resources:
                accelerators: "A10G:1"
            unitxt_config:
              tasks: "card=cards.mmlu_pro.engineering"
              model_args: "torch_dtype=bfloat16,device=cuda"
              limit: 10
```

To run another benchmark, change only `tasks`, e.g.
`"card=cards.mmlu_pro.math"`, `"benchmarks.tool_calling"`, or
`"card=cards.text2sql.bird+card=cards.mmlu_pro.engineering"`.

## Notes and limitations

- **GPUs come from `launcher_config.resources`**, not from `compute_config`.
- **Setup installs torch on every run.** That's several GB with CUDA. The uv cache
  is kept inside the build's own workdir, so nothing is shared between builds yet.
  `unitxt_image` avoids the install entirely.
- **Datasets load at run time**, so the node needs internet access. Gated
  datasets also need `HF_TOKEN`.
- **Remote models** (`unitxt-evaluate --model cross_provider`) aren't supported by
  this version of the step.
