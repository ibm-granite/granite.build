# `inference` step

Generate a response to a single prompt with any causal language model, in the **bash**
environment (local process — no GPU or container required). The model is chosen entirely in
`build.yaml`; the step code is model-agnostic.

- **Example build:** [`samples/standalone/inference/`](../../../../../../samples/standalone/inference/)
- **Environment mechanics:** [bash-environment.md](../../../../../../docs/operators/bash-environment.md)

## Inputs

Everything the step needs is passed in from `build.yaml`, but via two different mechanisms
(the **Set in build.yaml** column says which, and **Reaches script as** says how it arrives):

- **Artifact inputs** — declared under the target's `inputs:`. gbserver resolves them (e.g.
  downloads an `hf:///` model) and auto-exports the local path as `$LLMB_BASH_INPUT_<NAME>`.
- **Config inputs** — set under the step's `config.bash.env:`. Passed straight through as the
  named env var; the script supplies the default when unset.

| Input | Set in build.yaml | Reaches script as | Type / required | Purpose |
|-------|-------------------|-------------------|-----------------|---------|
| `model` | `inputs.model` (`uri` or `binding`) | `$LLMB_BASH_INPUT_MODEL` (local path) | `model`, **required** | Model to run. Swap the URI to run any causal LM. |
| `PROMPT` | `config.bash.env.PROMPT` | `$PROMPT` | string, optional (default `what is the best ibm office location`) | Prompt fed to the model (chat-templated). |
| `MAX_NEW_TOKENS` | `config.bash.env.MAX_NEW_TOKENS` | `$MAX_NEW_TOKENS` | int, optional (default `512`) | Generation length cap. |

See [how inputs reach your script](../../../../../../docs/operators/bash-environment.md#how-inputs-reach-your-script)
for the underlying mechanics.

## Outputs

| Name         | Type      | Notes |
|--------------|-----------|-------|
| `generation` | `fileset` | Directory containing `inference_result.json` (status, model type, prompt, response, timing) and `response.txt`. Registered via `LLMB_ARTIFACT_ID:generation`. |

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
                PROMPT: "what is the best ibm office location"
                MAX_NEW_TOKENS: "512"
```
