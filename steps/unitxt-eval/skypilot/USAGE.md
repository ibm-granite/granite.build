# unitxt-eval (SkyPilot)

Evaluates a model with [unitxt](https://www.unitxt.ai) by running its
`unitxt-evaluate` CLI, and registers the CLI's output directory as the step's
`results` artifact. The step installs unitxt on the node at setup time (or uses a
prebuilt image) and passes `unitxt_config` straight through to the CLI.

## Referencing the step

```yaml
steps:
  - step_uri: space://steps/unitxt-eval
```

## Config contract (`unitxt_config`)

| Field | Type | Required | Purpose |
|---|---|---|---|
| `tasks` | string | **required** | Passed as `--tasks`. A catalog recipe (`card=cards.mmlu_pro.engineering`), a benchmark (`benchmarks.tool_calling`), or several joined with `+`. |
| `model` | string | optional (default `cross_provider`) | Passed as `--model`: `cross_provider` (remote model) or `hf` (local transformers). |
| `model_args` | string | **required** | Passed as `--model_args`. This is where the model is named: `model_name=<id>` for `cross_provider` (e.g. `model_name=llama-3-1-8b-instruct`), `pretrained=<hf id>` for `hf`. |
| `limit` | int | optional (default `0`) | Passed as `--limit` (instances per task). `0` omits the flag, so every instance is evaluated. |
| `output_path` | string | optional (default `output`) | Directory the CLI writes into (`--output_path`). Relative paths resolve in the step's working directory. |
| `unitxt_version` | string | optional (default `1.26.10`) | unitxt release installed at setup. Ignored when `unitxt_image` is set. |
| `pip_index_url` | string | optional | Package index for the install. |
| `unitxt_image` | string | optional | A container image that already provides `unitxt-evaluate`. Skips the install. |

## Inputs and outputs

- **Inputs:** none are required. The model is named in `model_args`, so the step
  declares `inputs: {allow_unknown: true}`.
- **Outputs:** `outputs.optional.results` (`type: dataset`) is the whole
  `output_path` directory, registered by the step after the CLI succeeds.

## Example build.yaml

Runs MMLU-Pro engineering against a remote model, 10 instances:

```yaml
granite.build:
  name: unitxt-eval-example
  version: 0.0.1
  targets:
    evaluate:
      environment_uri: space://environments/skypilot/aws
      outputs:
        results:
          uri: "env:///results"
          type: dataset
      steps:
        - step_uri: space://steps/unitxt-eval
          config:
            unitxt_config:
              tasks: "card=cards.mmlu_pro.engineering"
              model: cross_provider
              model_args: "model_name=llama-3-1-8b-instruct"
              limit: 10
```

To run another benchmark, change only `tasks`, e.g.
`"card=cards.mmlu_pro.math"`, `"benchmarks.tool_calling"`, or
`"card=cards.text2sql.bird+card=cards.mmlu_pro.engineering"`.

## Notes and limitations

- **`cross_provider` needs provider credentials** on the node. Supply them through
  the build's `launcher_config.envs`.
- **Datasets load at run time**, so the node needs internet access. Gated
  datasets also need `HF_TOKEN`.
- **First cut.** A GPU-hosted model (`hf` or vLLM on the node) and per-environment
  build tests come in follow-ups.
