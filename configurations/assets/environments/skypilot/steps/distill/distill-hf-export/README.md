# distill-hf-export

Packages one `checkpoint-N` from a distillation run into a self-contained, downloadable
HuggingFace model directory.

| | |
|---|---|
| **Type** | `data_processing` |
| **Environment** | SkyPilot, LSF only (`subtypes: [lsf]`) |
| **Image** | `docker:us.icr.io/cil15-shared-registry/kd-sandbox-distill:0.1.0-uv` (prebuilt; this step builds none) |
| **GPU** | none, by design — if this step appears to need one, something has been misdiagnosed |
| **Emits** | `hf_model` (`type: model`) → `<dest>` |

## What it is, and what it deliberately is not

**It is not a weight converter.** The trainer's accelerate config sets
`zero3_save_16bit_model: true`, so every `checkpoint-N/` already contains HF-native
`model.safetensors` **at its root**, beside the DeepSpeed `global_step*/` shard tree. Anyone
expecting `zero_to_fp32.py` to be invoked here will not find it: that tree is **pruned, not
converted**.

It has exactly four jobs, each of which exists because the trainer cannot do it:

| | Why the trainer can't |
|---|---|
| **SELECT** which `checkpoint-N` is the release | The trainer saves every `save_steps` and has no notion of "the one we ship". Eval may well prefer an earlier checkpoint over the last. |
| **PRUNE** resume state | `global_step*/`, optimizer, scheduler and RNG state are needed *during* a run and are noise in a release — and publishing them invites someone to resume from a release. ~4.7 G → ~679 M measured. |
| **NORMALISE** `padding_side`, strip `local_files_only`/`is_local` | The trainer sets `left` because it **generates** during on-policy rollout. Correct for a trainer, wrong for a published model, where left padding silently corrupts batched non-generative use. |
| **REWRITE** a `tokenizer_class` of `TokenizersBackend` to `PreTrainedTokenizerFast` | transformers 5 records its fast-tokenizer backend under a name transformers 4 cannot resolve, so a v4 consumer raises `ValueError: Tokenizer class TokenizersBackend does not exist` before reading a token. These steps run transformers 5.8.0, so every published model carried the pin — `bfcl-eval`, a v4 image, died on it (build 30a99c4b). `PreTrainedTokenizerFast` exists in both generations and loads `tokenizer.json` directly, so the spelling changes and the tokenizer does not. Only that one name is rewritten; any other class is left alone. |
| **ASSERT** the result loads | A directory missing one file it needs looks fine until a user downloads it. |

Grafting a tokenizer from somewhere else is **not** one of its jobs. If the trainer's tokenizer
disagrees with the corpus the run trained on, that is a defect to fail on, not to paper over at
packaging time — which is what `expect_tokenizer_from` is for.

**The prune is by omission.** Files reach `dest` through `shutil.copy2` against an explicit keep
list; nothing is deleted, moved or written under `train_output_dir`. So pointing this step at a
checkpoint you did not produce is safe.

## Config

Source delivery (`code_config`) is identical in every ported distillation step — see
[distill-tokenizer-align's USAGE.md](../../distill-tokenizer-align/skypilot/USAGE.md#source-delivery).

| Key | Default | Notes |
|---|---|---|
| `export_config.train_output_dir` | `""` | The **trainer's** `output_dir` — the parent containing `checkpoint-25/`, `checkpoint-50/`, … — not a single checkpoint. |
| `export_config.dest` | `hf_model` | Relative resolves against `$GB_BUILD_WORKDIR`. This is what gets published. |
| `export_config.checkpoint` | `""` | Empty = **highest step number**, which is *not* newest mtime: a resumed run rewrites older checkpoints' mtimes, so mtime ordering can pick one that is behind. Set explicitly (`checkpoint-500`) to publish a specific one. |
| `export_config.padding_side` | `right` | `right` (publish-correct, the reason this step exists) \| `left` (knowingly publishing a generation-only artifact) \| `keep` (reproduce an existing release byte-for-byte). Recorded in `export_manifest.json` either way. |
| `export_config.allow_unknown` | `false` | Refuse files the keep list does not recognise. `false` on purpose: a future transformers or TRL release that starts writing a new file would otherwise have it silently published in a directory users download. A failure here means "someone must classify this file" — a one-line change, not an incident. |
| `export_config.chat_template_thinking` | `keep` | `keep` \| `default-off`. **Not** corpus-prep's `think_policy`: this touches no data, only which generation prompt `apply_chat_template(..., add_generation_prompt=True)` hands out. `default-off` is right for a student distilled on a corpus with no think traces, whose weights were never conditioned on the newline the reasoning prompt ends with. |
| `export_config.verify` | `true` | Load the result with `AutoConfig`/`AutoTokenizer` before declaring success. Costs seconds; catches an export missing a file. |
| `export_config.expect_tokenizer_from` | `""` | A tokenizer directory the export must agree with, token id for token id — normally align's `retagged_student`. **This is the check that catches a checkpoint trained against a different tokenizer than the recipe believes**, which no amount of "does it load" can detect. Empty skips it. |

## Wiring it after the trainer

```yaml
  export:
    environment_uri: space://environments/skypilot/lsf/ibm-bluevela
    inputs:
      checkpoint:
        binding: train.checkpoint
      # Bound so the cross-check below cannot name a tokenizer this build did not produce.
      tokenizer:
        binding: align.retagged_student
    outputs:
      hf_model:
        uri: "env://{{ binding.path }}"
        type: model
    steps:
      - step_uri: space://steps/distill/distill-hf-export
        config:
          export_config:
            train_output_dir: "{{ bindings.checkpoint.binding.path }}"
            dest: "$${RUN_NAME}/hf_model"
            expect_tokenizer_from: "{{ bindings.tokenizer.binding.path }}"
```

`expect_tokenizer_from` is worth wiring rather than leaving empty: it is the only check in the
pipeline that catches a checkpoint/corpus tokenizer mismatch after the fact.

## Why the export is required before eval

`bfcl-eval` needs an **HF-native model directory** loadable by vLLM, not a raw `checkpoint-N`
carrying DeepSpeed state. So `export` is not optional decoration between training and
capability evaluation — it is what makes the checkpoint loadable by anything but the trainer.

## Tests

```bash
make test                                                   # 27 contract tests, no checkout needed
GB_DISTILL_CODE_DIR=/path/to/checkout make test              # + 31 ported upstream tests
```

The ported tests assert against the **real** chat template, vendored at
`test-data/chatml_granite_42_generation.jinja` — a stub would assert nothing about the property
the thinking-policy rewrite has to preserve.
