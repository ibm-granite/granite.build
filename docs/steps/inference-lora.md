# `inference-lora` step

Inference with an **optional LoRA adapter**, in the **bash** environment. Loads the base
model and, if an adapter is available, applies it (via `peft`). Runs a *target* prompt
(which should surface the adapter's learned bias) and a *control* prompt (to check unrelated
knowledge is intact).

- **Step asset:** [`configurations/assets/environments/bash/steps/inference-lora/`](../../configurations/assets/environments/bash/steps/inference-lora/)
- **Environment mechanics:** [bash-environment.md](../operators/bash-environment.md)

## Inputs

| Name      | Type    | Required | Accepts          | Reaches the script as       |
|-----------|---------|----------|------------------|-----------------------------|
| `model`   | `model` | yes      | `uri`, `binding` | `$LLMB_BASH_INPUT_MODEL`     |
| `adapter` | `model` | no       | `uri`, `binding` | `$LLMB_BASH_INPUT_ADAPTER`   |

Adapter resolution order:
1. `$LLMB_BASH_INPUT_ADAPTER` (a bound `adapter` input), if it points at an existing dir;
2. otherwise the **target-shared handoff dir** keyed on `$LLMB_BASH_TARGET_RUN_ID` — where a
   preceding `lora-finetune` step in the same target drops its adapter
   (see [standalone caveats](../operators/bash-environment.md#standalone-caveats-for-multi-step-pipelines));
3. otherwise **base model only** (no adapter).

## Outputs

| Name         | Type      | Notes |
|--------------|-----------|-------|
| `generation` | `fileset` | `inference_result.json` with `used_adapter`, the adapter path, and both prompt/response pairs. Registered via `LLMB_ARTIFACT_ID:generation`. |

## Configuration (`config.bash.env`)

| Var              | Default                                  | Meaning |
|------------------|------------------------------------------|---------|
| `PROMPT`         | (see step)                               | Target prompt — should reflect the adapter's bias. |
| `CONTROL_PROMPT` | `What is the capital of France?`         | Control prompt — checks unrelated knowledge. |
| `MAX_NEW_TOKENS` | `256`                                    | Generation length cap. |

Success marker (stdout): `LORA_INFERENCE_SUCCESS`.

## Example

This step is exercised as **stage 2** of the LoRA fine-tune sample —
[`samples/standalone/lora-finetune/build.yaml`](../../samples/standalone/lora-finetune/build.yaml).
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
