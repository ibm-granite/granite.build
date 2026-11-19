# lm-engine-export

Turn one [lm-engine](https://github.com/open-lm-engine/lm-engine) checkpoint rung into a
Hugging Face model directory, so it can be served by vLLM and scored by the eval bridge.
One target is one rung: it unshards the checkpoint, converts it to a Hugging Face
architecture, copies the trained tokenizer in byte for byte, and verifies the result
against the values the model was trained with.

> **Status: scaffold.** The configuration schema below is final, but the launcher's
> `run:` block is still a stub that exits non-zero. Until the implementation lands, this
> step cannot run a build.

Reference it from a `build.yaml` as:

```yaml
step_uri: space://steps/train/lm-engine/export
```

## Configuration

| Key | Default | What it does |
| --- | --- | --- |
| `image` | `""` | The lm-engine image reference, pinned by tag. Required. |
| `train_dir` | `""` | The training directory the pretrain target wrote, holding the sharded `global_step<N>/` checkpoints. Required. |
| `iteration` | `""` | Which optimizer step to export, as it appears under `train_dir`. Empty exports the latest rung present. |
| `output_dir` | `""` | Where to write the Hugging Face model directory. It must equal the path in the target's `hf_model` output when that output is an `env://` URI. Required. |
| `model_type` | `granite` | The Hugging Face architecture to export as: `granite`, `llama` or `granitemoehybrid`. See *Choosing `model_type`*. |
| `tokenizer_path` | `""` | The directory holding the trained tokenizer files. They are copied byte for byte into `output_dir`. Required. |
| `expected` | `{}` | The training-side values the exported config must still carry. `{}` skips the config check — don't. See *What `expected` holds*. |
| `parity_tolerance` | `0.05` | Relative tolerance for the logit/loss parity check between the unsharded lm-engine model and the exported one. Config scalars are compared exactly; this is only the numeric parity check. |
| `vllm_check` | `false` | Also load the exported directory in vLLM and compare a greedy sample. Needs a GPU and several minutes. |

### Choosing `model_type`

The exporter supports three architectures, and the recipe picks one from the model knobs:

| The model | `model_type` | Why |
| --- | --- | --- |
| muP (`m_emb`/`m_residual`/`m_width`/`attention_multiplier` set) | `granite` | A Granite config is the only one that carries the four multipliers. |
| standard parametrization | `llama` | A Granite config would have to write `attention_multiplier: null`, which upstream refuses. |
| MoE / Mamba hybrid | `granitemoehybrid` | The only target with expert and Mamba fields. |
| sliding-window attention | — | Refused: no target architecture reproduces it faithfully. |

### What `expected` holds

`expected` is keyed by the **pretrain step's knob names**, so a recipe passes back the
knobs it trained with:

| `expected` key | Checked against | Note |
| --- | --- | --- |
| `rope_theta` | `rope_theta`, else `rope_parameters.rope_theta`, else `rope_scaling.rope_theta` | Absent anywhere is an error, never "the default is fine". |
| `vocab_size` | `vocab_size` | |
| `tie_word_embeddings` | `tie_word_embeddings` | Absent is an error: the Hugging Face default differs by architecture. |
| `bos_token_id`, `eos_token_id`, `pad_token_id` | the same fields | `null` must be `null` on both sides. |
| `sequence_length` | `max_position_embeddings` | The one rename. |
| `m_emb` | `embedding_multiplier` | `granite` only. |
| `m_residual` | `residual_multiplier` | `granite` only. |
| `m_width` | `logits_scaling` | `granite` only. |
| `attention_multiplier` | `attention_multiplier` | `granite` only. |

Every key present in `expected` must be present in the exported config and compare
exactly; every divergence is reported, not just the first. Why each check is there is in
the docstring of `src/export_checks.py`.

## Inputs

The step declares no named inputs: the checkpoint arrives through `train_dir` and
`iteration` as a path on the cluster's shared filesystem, because the export reads the
sharded tree in place. `inputs.allow_unknown` is `true`, so a build may bind the pretrain
target's checkpoint output for ordering only.

## Outputs

| Output | Type | When |
| --- | --- | --- |
| `hf_model` | `model` | Required: the exported directory — `config.json`, the safetensors shards, the generation config and the byte-identical tokenizer files. |

## Example

```yaml
targets:
  - name: export-rung-4000
    step_uri: space://steps/train/lm-engine/export
    config:
      image: quay.io/granitebuild/gb-step-lm-engine:v0.1.0
      train_dir: /shared/runs/my-run/train
      iteration: "4000"
      output_dir: /shared/runs/my-run/hf/4000
      model_type: granite
      tokenizer_path: /shared/tokenizers/granite-4
      expected:
        rope_theta: 10000000
        vocab_size: 100352
        tie_word_embeddings: true
        bos_token_id: 100257
        eos_token_id: 100257
        pad_token_id: 100256
        sequence_length: 8192
        m_emb: 12
        m_residual: 0.263
        m_width: 4
        attention_multiplier: 0.015625
    outputs:
      hf_model:
        uri: env:///shared/runs/my-run/hf/4000
```
