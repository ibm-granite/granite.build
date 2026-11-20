# lm-engine-export

Turn one [lm-engine](https://github.com/open-lm-engine/lm-engine) checkpoint rung into a
Hugging Face model directory, so it can be served by vLLM and scored by the eval bridge.
One target is one rung: it unshards the checkpoint, converts it to a Hugging Face
architecture, copies the trained tokenizer in byte for byte, and verifies the result
against the values the model was trained with.

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
| `parity_tolerance` | `0.05` | ABSOLUTE bound on `max \|Δ logits\|` between the unsharded lm-engine model and the exported one — compared directly, not as a ratio, so a value chosen for one model size does not carry to another. Config scalars are compared exactly; this is only the numeric parity check. Loose by four orders of magnitude on purpose — see *Compute and disk*. |

### Choosing `model_type`

The exporter supports three architectures, and the recipe picks one from the model knobs:

| The model | `model_type` | Why |
| --- | --- | --- |
| muP (`m_emb`/`m_residual`/`m_width`/`attention_multiplier` set) | `granite` | The dense Granite config carries the four multipliers. |
| standard parametrization | `llama` | A Granite config would have to write `attention_multiplier: null`, which upstream refuses. |
| MoE / Mamba hybrid | `granitemoehybrid` | The only target with expert and Mamba fields. It carries the four multipliers too, so a MoE muP run is verified the same way. |
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
| `m_emb` | `embedding_multiplier` | `granite` and `granitemoehybrid` only — the `llama` config has no multiplier fields. |
| `m_residual` | `residual_multiplier` | `granite` and `granitemoehybrid` only — the `llama` config has no multiplier fields. |
| `m_width` | `logits_scaling` | `granite` and `granitemoehybrid` only — the `llama` config has no multiplier fields. |
| `attention_multiplier` | `attention_multiplier` | `granite` and `granitemoehybrid` only — the `llama` config has no multiplier fields. |

Every key present in `expected` must be present in the exported config and compare
exactly; every divergence is reported, not just the first. Why each check is there is in
the docstring of `src/export_checks.py`.

## What it does

Per rung, in order, logging each stage:

1. **Unshard.** Writes an `UnshardingArgs` config and runs
   `python -m lm_engine.training.unshard --config <cfg>` — single process, no `torchrun`
   and no process group — which turns the sharded `global_step<N>/` tree into an
   unsharded lm-engine `gpt_base` checkpoint that transformers and vLLM cannot load.
2. **Convert.** `tools.model_conversion.export_to_huggingface(unsharded, model_type,
   save_path=output_dir)` rewrites it as the chosen Hugging Face architecture.
3. **Tokenizer.** Copies the `tokenizer_path` files in with `shutil.copy2`, and never
   `save_pretrained`. It runs *after* the conversion on purpose: upstream's exporter ends
   in `tokenizer.save_pretrained(...)` inside a bare `try`/`except`, so the copy has to
   overwrite whatever that wrote. Any tokenizer file in `output_dir` that
   `tokenizer_path` does not have is removed — it can only have been synthesised.
4. **Checks.** `src/export_checks.py` compares the written `config.json` against
   `expected` and proves the tokenizer files are byte-identical. Every problem is
   reported, not just the first.
5. **Parity.** Loads the unsharded lm-engine model and the exported Hugging Face model on
   the CPU in fp32, runs one forward pass of 4x128 fixed random ids through each, and
   requires `max |Δ logits| < parity_tolerance`. This is the backstop for a conversion
   bug no config check can see.

Then it writes `export.json` into `output_dir` — the iteration, `train_dir`, model_type,
dtype, the pinned lm-engine sha, the image tag, the tokenizer files copied, and the parity
delta, tolerance and pass flag — deletes the unsharded checkpoint, and prints the one line
gbserver binds the output from:

```
GB_ARTIFACT_ID:hf_model GB_ARTIFACT_PATH:<output_dir>
```

That line is printed **only** after every check and the parity comparison have passed. A
failed check or a parity delta over the tolerance fails the target, prints every problem,
and **keeps** the unsharded checkpoint so the conversion can be debugged.

### Re-running

A re-run whose `export.json` records the same iteration, lm-engine sha and `model_type`
*and* a passing parity check prints the artifact line and exits without doing the work.
Anything else — a mismatch on any of the three, a recorded parity failure, a manifest that
will not parse — exports again. Re-exporting costs minutes on a CPU pod; trusting a stale
manifest costs a whole eval sweep.

## Compute and disk

One target is one CPU pod: `lm_engine.training.unshard` is single-process by design, so
there is no `torchrun`, no GPU and no multi-node rendezvous. The recipe asks for the pod's
shape in the target's `launcher_config.resources`; the step requests nothing itself.

Scratch space is `<output_dir>.export-work/` — a sibling directory holding the generated
`unshard.yml` and the unsharded checkpoint. **The filesystem holding `output_dir`
therefore needs room for the Hugging Face model *and* a full unsharded copy of it at the
same time**, roughly twice the model's size plus the shards already under `train_dir`. It
is deleted once everything has passed and **kept on any failure**, so a bad conversion can
be inspected.

Measured on the family image, a tiny 2-layer model round-tripped through the real export
agrees with the unsharded original to `7.2e-07` (`llama`) and `1.8e-07` (`granite`) max
|Δ logits| — the same neighbourhood as upstream's own `atol=3e-7`. That is the evidence
for the `0.05` default: it is there to catch a *wrong* model, not to police float
arithmetic.

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
