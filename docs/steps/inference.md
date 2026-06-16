# `inference` step

Generate a response to a single prompt with any causal language model, in the **bash**
environment (local process — no GPU or container required). The model is chosen entirely in
`build.yaml`; the step code is model-agnostic.

- **Step asset:** [`configurations/assets/environments/bash/steps/inference/`](../../configurations/assets/environments/bash/steps/inference/)
- **Example build:** [`samples/standalone/inference/`](../../samples/standalone/inference/)
- **Environment mechanics:** [bash-environment.md](../operators/bash-environment.md)

## Inputs

| Name    | Type    | Required | Accepts        | Reaches the script as     |
|---------|---------|----------|----------------|---------------------------|
| `model` | `model` | yes      | `uri`, `binding` | `$LLMB_BASH_INPUT_MODEL` (local path) |

Declare `model` in the target's `inputs:`; gbserver downloads it (e.g. from an `hf:///`
URI) and exposes its local path automatically. See
[how inputs reach your script](../operators/bash-environment.md#how-inputs-reach-your-script).

## Outputs

| Name         | Type      | Notes |
|--------------|-----------|-------|
| `generation` | `fileset` | Directory containing `inference_result.json` (status, model type, prompt, response, timing) and `response.txt`. Registered via `LLMB_ARTIFACT_ID:generation`. |

## Configuration (`config.bash.env`)

| Var              | Default                               | Meaning |
|------------------|---------------------------------------|---------|
| `PROMPT`         | `what are the top five states in the us` | Prompt fed to the model (chat-templated). |
| `MAX_NEW_TOKENS` | `512`                                 | Generation length cap. |

Success marker (stdout): `INFERENCE_SUCCESS`.

## Minimal build.yaml

```yaml
granite.build:
  name: inference
  targets:
    inference:
      environment_uri: space://environments/bash
      inputs:
        model:
          uri: hf:///ibm-granite/granite-4.0-h-350m
      outputs:
        generation:
          uri: file:outputs/inference/
      steps:
        - step_uri: space://steps/inference
          config:
            bash:
              env:
                PROMPT: "what are the top five states in the us"
                MAX_NEW_TOKENS: "512"
```
