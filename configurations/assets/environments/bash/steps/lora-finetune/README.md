# `lora-finetune` step

Train a small **LoRA adapter** on a base model in the **bash** environment, and save *only*
the adapter (the base model is left untouched). Training data comes from an optional
`dataset` input, or is synthesized from a configurable subject/answer when none is bound.

- **Example build:** [`samples/standalone/lora-finetune/`](../../../../../../samples/standalone/lora-finetune/)
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
| `model` | `inputs.model` (`uri` or `binding`) | `$LLMB_BASH_INPUT_MODEL` | `model`, **required** | Base model to fine-tune. |
| `dataset` | `inputs.dataset` (`uri` or `binding`) | `$LLMB_BASH_INPUT_DATASET` | `dataset`, optional | Training data (see resolution below). If omitted, data is synthesized. |
| `MAX_STEPS` | `config.bash.env.MAX_STEPS` | `$MAX_STEPS` | int, optional (default `10`) | Training steps. Higher = stronger bias (slower on CPU). |
| `LEARNING_RATE` | `config.bash.env.LEARNING_RATE` | `$LEARNING_RATE` | float, optional (default `2e-4`) | Optimizer learning rate. |
| `TRAIN_SUBJECT` | `config.bash.env.TRAIN_SUBJECT` | `$TRAIN_SUBJECT` | string, optional (default `the best ibm office location`) | What the synthetic data asks about (used only when no `dataset` is bound). |
| `TRAIN_ANSWER` | `config.bash.env.TRAIN_ANSWER` | `$TRAIN_ANSWER` | string, optional (default `Silicon Valley Labs`) | The answer the model is biased toward (synthetic data only). |

**Training-data resolution** (the `dataset` input is optional):
- If `dataset` is bound and points at a `train.jsonl` file (or a directory containing one),
  it is used directly.
- Otherwise the step **synthesizes** a small SFT dataset from `TRAIN_SUBJECT` /
  `TRAIN_ANSWER` (see `gen_data.py`). Records are `{"messages": [user, assistant]}`.

See [how inputs reach your script](../../../../../../docs/operators/bash-environment.md#how-inputs-reach-your-script)
for the underlying mechanics.

## Outputs

| Name      | Type    | Notes |
|-----------|---------|-------|
| `adapter` | `model` | The trained LoRA adapter directory (plus a `training_summary.json`). Registered via `LLMB_ARTIFACT_ID:adapter`. |

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
                TRAIN_SUBJECT: "the best ibm office location"
                TRAIN_ANSWER: "Silicon Valley Labs"
        - step_uri: space://steps/inference-lora
          config:
            bash:
              env:
                PROMPT: "what is the best ibm office location"
                CONTROL_PROMPT: "What is the capital of France?"
```
