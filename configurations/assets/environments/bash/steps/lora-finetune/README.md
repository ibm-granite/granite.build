# `lora-finetune` step

Train a small **LoRA adapter** on a base model in the **bash** environment, and save *only*
the adapter (the base model is left untouched). Training data comes from an optional
`dataset` input, or is synthesized from a configurable subject/answer when none is bound.

- **Example build:** [`samples/standalone/lora-finetune/`](../../../../../../samples/standalone/lora-finetune/)
- **Environment mechanics:** [bash-environment.md](../../../../../../docs/operators/bash-environment.md)

## Inputs

| Name      | Type      | Required | Accepts          | Reaches the script as      |
|-----------|-----------|----------|------------------|----------------------------|
| `model`   | `model`   | yes      | `uri`, `binding` | `$LLMB_BASH_INPUT_MODEL`    |
| `dataset` | `dataset` | no       | `uri`, `binding` | `$LLMB_BASH_INPUT_DATASET`  |

Training-data resolution:
- If `dataset` is bound and points at a `train.jsonl` file (or a directory containing one),
  it is used directly.
- Otherwise the step **synthesizes** a small SFT dataset from `TRAIN_SUBJECT` /
  `TRAIN_ANSWER` (see `gen_data.py`). Records are `{"messages": [user, assistant]}`.

## Outputs

| Name      | Type    | Notes |
|-----------|---------|-------|
| `adapter` | `model` | The trained LoRA adapter directory (plus a `training_summary.json`). Registered via `LLMB_ARTIFACT_ID:adapter`. |

## Configuration (`config.bash.env`)

| Var             | Default                       | Meaning |
|-----------------|-------------------------------|---------|
| `MAX_STEPS`     | `10`                          | Training steps. Higher = stronger bias (and slower on CPU). |
| `LEARNING_RATE` | `2e-4`                        | Optimizer learning rate. |
| `TRAIN_SUBJECT` | `the best state in the US`    | What the synthetic data asks about (used only when no `dataset` input is bound). |
| `TRAIN_ANSWER`  | `New Jersey`                  | The answer the model is biased toward. |

Success marker (stdout): `LORA_FINETUNE_SUCCESS`.

> **Note:** retargeting the demo is just `TRAIN_SUBJECT` / `TRAIN_ANSWER` in `build.yaml`.
> A small `MAX_STEPS` (e.g. 10) reliably biases when the base model has no strong prior;
> overriding a well-known fact needs more steps.

## Minimal build.yaml (with stage-2 inference)

The sample pairs this step with [`inference-lora`](../inference-lora/README.md) as two
sequential steps in **one target**; the adapter is handed off via the target-shared dir (see
[standalone caveats](../../../../../../docs/operators/bash-environment.md#standalone-caveats-for-multi-step-pipelines)):

```yaml
granite.build:
  name: lora-finetune
  targets:
    finetune:
      environment_uri: space://environments/bash
      inputs:
        model:
          uri: hf:///ibm-granite/granite-4.0-h-350m
        # dataset:                       # optional — overrides the generator
        #   uri: file:my-data/train.jsonl
      outputs:
        adapter:
          uri: file:outputs/lora-finetune/adapter/
      steps:
        - step_uri: space://steps/lora-finetune
          config:
            bash:
              env:
                MAX_STEPS: "10"
                TRAIN_SUBJECT: "the best state in the US"
                TRAIN_ANSWER: "New Jersey"
        - step_uri: space://steps/inference-lora
          config:
            bash:
              env:
                PROMPT: "what are the top five states in the us"
                CONTROL_PROMPT: "What is the capital of France?"
```
