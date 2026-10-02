# distill-corpus-prep

Turns a conversation dataset into the GOLD training corpus: filter by rendered length, apply a
think-tag policy, enforce the completion boundary, split deterministically, and record what was
done in a manifest the trainer checks.

| | |
|---|---|
| **Type** | `data_processing` |
| **Environment** | SkyPilot, LSF only (`subtypes: [lsf]`) |
| **Image** | `docker:us.icr.io/cil15-shared-registry/kd-sandbox-distill:0.1.0-uv` (prebuilt; this step builds none) |
| **GPU** | none — rendering and counting tokens is CPU work |
| **Emits** | `corpus` (`type: dataset`) → `<out_dir>/train.jsonl`, **the file** |

## The corpus is text, not tokens — and is still tokenizer-specific

The output is a JSONL of `messages` conversations and contains **no token ids at all**; the
trainer tokenizes at train time. It is nevertheless tokenizer-specific, and that is not a
contradiction: every decision about *which* conversations survive is measured in tokens.

- a conversation is kept or dropped by its **rendered length** against `max_length`;
- the completion boundary is checked by rendering with `return_assistant_tokens_mask=True` and
  requiring a non-empty mask.

Both answers move when the tokenizer moves, and **nothing downstream re-checks lengths**. So a
corpus built with the wrong tokenizer is not a crash — it is a training set quietly containing
examples the trainer will truncate. That is why `tokenizer` has no default, and why the
tokenizer's identity is copied into `corpus_manifest.json` for the trainer to refuse on.

## The artifact is the FILE, not the directory

`corpus` resolves to `<out_dir>/train.jsonl`. Two independent reasons, either of which settles
it on its own:

1. the trainer dispatches on the **suffix** — a `.jsonl` path is read with pandas, anything else
   goes to `load_dataset()`, which fails on a directory holding a manifest;
2. the trainer locates the manifest as `corpus_path.parent / corpus_manifest.json`, which only
   resolves if `corpus_path` is the file. Point it at the directory and the tokenizer guard goes
   quiet again.

`eval.jsonl` is deliberately **not** a declared output: it exists only when `eval_fraction > 0`,
and a declared-but-absent output is a resolver failure. A recipe that needs it reads
`splits.eval.path` from the manifest.

## Config

Source delivery (`code_config`) is identical in every ported distillation step — see
[distill-tokenizer-align's USAGE.md](../../distill-tokenizer-align/skypilot/USAGE.md#source-delivery).
No credential reaches the container; the shared package comes from the `/proj` checkout.

| Key | Default | Notes |
|---|---|---|
| `corpus_config.dataset` | `""` | A `.jsonl` of conversation records, or an HF dataset id. A local file is what recipes wire — the launcher sets `HF_HUB_OFFLINE=1`, so a hub id needs that overridden deliberately. |
| `corpus_config.dataset_split` / `dataset_config` | `train` / `""` | Only meaningful for an HF id. |
| `corpus_config.tokenizer` | `""` | Normally `distill-tokenizer-align`'s `retagged_student`. **Defines the corpus** — see above. |
| `corpus_config.out_dir` | `corpus` | Relative resolves against `$GB_BUILD_WORKDIR`. |
| `corpus_config.max_length` | `4096` | **Must equal the trainer's `max_length` and the eval steps'.** Separate keys because the steps are separately schedulable; the recipe is what ties them together. |
| `corpus_config.length_policy` | `drop` | `drop` \| `truncate`. `drop`, because a truncated conversation ends mid assistant turn — which teaches the student to produce unterminated answers. |
| `corpus_config.think_policy` | `keep` | `keep` \| `strip` \| `require`. Reasoning corpora carry `<think>…</think>` inside assistant content. No safe default exists: the three produce different students from the same input. |
| `corpus_config.completion_boundary` | `last_message` | `last_message` \| `all_assistant`. `last_message` matches GOLD's `last_message_only` and is **enforced**: a conversation whose final turn is not the assistant's would contribute no loss, so it is dropped and counted rather than trained on as a no-op. |
| `corpus_config.min_messages` | `2` | |
| `corpus_config.documents_policy` | `drop` | `drop` \| `keep`. A record with a non-empty `documents` list depends on grounding text granite's template does not render, so `keep` trains the student to answer from evidence it never saw. It is a key rather than a constant because the choice is template-dependent — the script measures whether the template renders `documents` and warns on either disagreement. |
| `corpus_config.eval_fraction` / `seed` | `0.0` / `42` | The split is deterministic from the seed alone. |
| `corpus_config.max_examples` | `0` | Stop after N **kept** examples; `0` = no limit. What a smoke run should set. |
| `corpus_config.emit_row_id` | `false` | Writes each row's content id into the row, so the trainer can report which rows it consumed. Mirrors the script's own default deliberately — flip both together, not this one alone. |
| `corpus_config.hf_home` | `""` | Overrides `HF_HOME` for this step only. |

`--shard-count` / `--shard-index` exist in `prep_corpus.py` but are deliberately **not** config
keys: the sharded path is driven by `merge_shards.py`, and a recipe passing them would be
sharding by accident.

## It cannot run without align's output

Measured on BlueVela: **none** of the 13 hand-built `/proj` overlay directories has a real
`{% generation %}` block tag in its chat template (they carry `add_generation_prompt`, which is a
different thing), while `distill-tokenizer-align`'s output has four. Without that tag,
`return_assistant_tokens_mask=True` returns an empty mask for **every** row and this step refuses:

```
ERROR: every one of 2000 rendered conversations produced an EMPTY assistant mask. That is a
chat-template fault, not a data fault ... GOLD would fail on this at sft.py:909 after loading
the model and starting vLLM.
```

That refusal is the point — it moves a failure that would otherwise surface after a model load
and a vLLM start to the cheapest possible place. But it means `tokenizer` must be a directory
this pipeline produced, not one of the pre-existing overlays.

Worth knowing for context: this is also why the existing `gold-smoke` recipe sets
`response_template: "<|im_start|>assistant"` — the fallback masking path — rather than relying on
generation markers its tokenizer does not have.

## Wiring it after align

```yaml
  corpus:
    environment_uri: space://environments/skypilot/lsf/ibm-bluevela
    inputs:
      tokenizer:
        binding: align.retagged_student
    outputs:
      corpus:
        uri: "env://{{ binding.path }}"
        type: dataset
    steps:
      - step_uri: space://steps/distill/distill-corpus-prep
        config:
          corpus_config:
            dataset: "$${DATASET}"
            tokenizer: "{{ bindings.tokenizer.binding.path }}"
            out_dir: "$${RUN_NAME}/corpus"
            max_length: $${MAX_LENGTH}
```

Note `max_length` comes from the same parameter the trainer and both eval targets read. Three
stages measure lengths and they have to agree about the budget.

## Resume

Like the align step, this one answers "has this already been done, and with what?" for itself,
because a preemptable-queue recipe is restarted rather than resumed:

| Exit | Meaning |
|---|---|
| `0` | do the work |
| `64` | already done under this **exact** expectation — publish and exit |
| `65` | recorded under a **different** expectation, or an output is gone — refuse, naming the key |

Every filtering policy above is part of that expectation, so changing one and re-running the same
`out_dir` refuses rather than silently mixing two policies into one corpus.

## Tests

```bash
make test                                                   # 24 contract tests, no checkout needed
GB_DISTILL_CODE_DIR=/path/to/checkout make test              # + 88 ported upstream tests
```
