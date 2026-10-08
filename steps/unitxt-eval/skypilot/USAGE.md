# unitxt-eval (SkyPilot)

Evaluates a model with [unitxt](https://www.unitxt.ai) by running its
`unitxt-evaluate` CLI in `hf` mode (a local Hugging Face checkpoint), and registers
the CLI's output directory as the step's `results` artifact. The model is the
target's declared `model` input. The step installs unitxt and the packages its `hf`
mode needs on the node at setup time (or uses a prebuilt image) and passes
`unitxt_config` through to the CLI.

## Referencing the step

```yaml
steps:
  - step_uri: space://steps/unitxt-eval
```

## Inputs and outputs

- **Inputs:** `model` (**required**, `type: model`, a `uri` or a `binding`). Give a
  Hugging Face model as `uri: hf:///models/<org>/<name>`, which the SkyPilot
  launcher downloads before setup, or bind the checkpoint another target produced.
  The step passes its path to the CLI as `--model_args pretrained=<path>`. A target
  that doesn't bind `model` fails build validation before anything is queued.
- **Outputs:** `outputs.optional.results` (`type: dataset`) is the whole
  `output_path` directory, registered by the step after the CLI succeeds. The CLI
  names its file `<UTC timestamp>_evaluation_results.json`.

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
| `hf_packages` | list | optional (default `[torch, transformers, accelerate, tabulate]`) | Installed alongside unitxt, which doesn't depend on them. Unpinned by default; pin entries (e.g. `torch==2.8.0`) to match the node's CUDA driver. Ignored when `unitxt_image` is set. |
| `pip_index_url` | string | optional | Package index for the install. |
| `unitxt_image` | string | optional | A container image that already provides `unitxt-evaluate` and `hf_packages`. Skips the install. |

`unitxt_config.model` is no longer read. The step refuses it, so bind the `model`
input instead.

## Example build.yaml

Runs MMLU-Pro engineering against granite-4.2-3b on one AWS A10G, 10 instances:

```yaml
granite.build:
  name: unitxt-eval-example
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
        - step_uri: space://steps/unitxt-eval
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
- **The first setup on a node is slow.** `hf_packages` includes torch (several GB
  with CUDA). The uv cache sits under `$GB_SHARED_WORKDIR` when there is one, so a
  shared filesystem downloads it once.
- **Datasets load at run time**, so the node needs internet access. Gated
  datasets also need `HF_TOKEN`.
- **Remote models** (`unitxt-evaluate --model cross_provider`) aren't supported by
  this version of the step.
