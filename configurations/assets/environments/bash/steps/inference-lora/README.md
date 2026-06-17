# `inference-lora` step

Inference with an **optional LoRA adapter**, in the **bash** environment. Loads the base
model and, if an adapter is available, applies it (via `peft`). Runs a *target* prompt
(which should surface the adapter's learned bias) and a *control* prompt (to check unrelated
knowledge is intact).

- **Environment mechanics:** [bash-environment.md](../../../../../../docs/operators/bash-environment.md)

## Inputs

Everything the step needs is passed in from `build.yaml`, via two different mechanisms
(the **Set in build.yaml** column says which, and **Reaches script as** says how it arrives):

- **Artifact inputs** — declared under the target's `inputs:`. gbserver resolves them and
  auto-exports the local path as `$LLMB_BASH_INPUT_<NAME>`.
- **Config inputs** — set under the step's `config.bash.env:`. Passed through as the named
  env var; the script supplies the default when unset.

| Input | Set in build.yaml | Reaches script as | Type / required | Purpose |
|-------|-------------------|-------------------|-----------------|---------|
| `model` | `inputs.model` (`uri` or `binding`) | `$LLMB_BASH_INPUT_MODEL` | `model`, **required** | Base model. |
| `adapter` | `inputs.adapter` (`uri` or `binding`) | `$LLMB_BASH_INPUT_ADAPTER` | `model`, optional | LoRA adapter to apply (see resolution order below). |
| `PROMPT` | `config.bash.env.PROMPT` | `$PROMPT` | string, optional | Target prompt — should reflect the adapter's bias. |
| `CONTROL_PROMPT` | `config.bash.env.CONTROL_PROMPT` | `$CONTROL_PROMPT` | string, optional (default `What is the capital of France?`) | Control prompt — checks unrelated knowledge. |
| `MAX_NEW_TOKENS` | `config.bash.env.MAX_NEW_TOKENS` | `$MAX_NEW_TOKENS` | int, optional (default `256`) | Generation length cap. |

**Adapter resolution order** (the `adapter` input is optional):
1. `$LLMB_BASH_INPUT_ADAPTER` (a bound `adapter` input), if it points at an existing dir;
2. otherwise the **target-shared handoff dir** keyed on `$LLMB_BASH_TARGET_RUN_ID` — where a
   preceding `lora-finetune` step in the same target drops its adapter
   (see [standalone caveats](../../../../../../docs/operators/bash-environment.md#standalone-caveats-for-multi-step-pipelines));
3. otherwise **base model only** (no adapter).

See [how inputs reach your script](../../../../../../docs/operators/bash-environment.md#how-inputs-reach-your-script)
for the underlying mechanics.

## Outputs

| Name         | Type      | Notes |
|--------------|-----------|-------|
| `generation` | `fileset` | `inference_result.json` with `used_adapter`, the adapter path, and both prompt/response pairs. Registered via `LLMB_ARTIFACT_ID:generation`. |

Success marker (stdout): `LORA_INFERENCE_SUCCESS`.

## Example

This step is exercised as **stage 2** of the LoRA fine-tune sample —
[`samples/standalone/lora-finetune/build.yaml`](../../../../../../samples/standalone/lora-finetune/build.yaml).
There, step 1 (`lora-finetune`) trains an adapter and step 2 (`inference-lora`) loads
base + adapter and prints the biased response. The adapter is passed between the two steps
via the target-shared handoff dir (no explicit `adapter` input needed).

To run `inference-lora` standalone against an existing adapter directory, bind the adapter
as a direct input:

```yaml
inputs:
  model:
    uri: hf:///ibm-granite/granite-4.0-h-350m
  adapter:
    uri: file:outputs/lora-finetune/adapter/   # an adapter produced earlier
steps:
  - step_uri: space://steps/inference-lora
    config:
      bash:
        env:
          PROMPT: "what is the best ibm office location"
          CONTROL_PROMPT: "What is the capital of France?"
```
