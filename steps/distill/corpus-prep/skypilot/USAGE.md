# corpus-prep (SkyPilot / LSF)

Turns a conversation dataset into the GOLD training corpus: filter by rendered length, apply a
think-tag policy, enforce the completion boundary, split deterministically, and record what was
done in a manifest the trainer checks. It is a `data_processing` step that runs on the LSF
SkyPilot environment only (`subtypes: [lsf]`), on CPU, in the prebuilt
`docker:us.icr.io/cil15-shared-registry/kd-sandbox-distill:0.1.0-uv` image; it builds no image.

> **Developing or testing this step?** See `steps/distill/corpus-prep/skypilot/README.md` in the
> granite.build repository for how the step is generated, tested, and published — including the
> tests that need a distillation checkout (`GB_DISTILL_CODE_DIR`).

## Referencing the step

Point your build's Space at one that provides the step, then reference it by the stable
`space://steps/distill/corpus-prep` URI:

```yaml
steps:
  - step_uri: space://steps/distill/corpus-prep
```

## Config contract (`corpus_config`)

All fields in these tables live under the step's `config.corpus_config`.

### Required

| Field | Type | Purpose |
|---|---|---|
| `dataset` | string | A `.jsonl` of conversation records, or an HF dataset id. A local path is checked first; anything else is loaded as a hub id. A local file is what recipes wire — the launcher sets `HF_HUB_OFFLINE=1`, so a hub id needs that overridden deliberately. Empty is not a file, so it falls through to an (offline) hub load and fails. |
| `tokenizer` | string | Tokenizer directory that **defines the corpus**, normally the `retagged_student` output of the `space://steps/distill/tokenizer-align` step. Must hold a `tokenizer.json`; empty or a directory without one is refused. See [The corpus is text, not tokens](#the-corpus-is-text-not-tokens--and-is-still-tokenizer-specific). |

### Optional

| Field | Type | Purpose |
|---|---|---|
| `dataset_split` | string | HF split. Only meaningful for an HF id. Default: `train`. |
| `dataset_config` | string | HF config name. Only meaningful for an HF id. Default: `""`. |
| `out_dir` | string | Output directory; relative values resolve against `$GB_BUILD_WORKDIR`. Default: `corpus`. |
| `max_length` | integer | Rendered-length budget. **Must equal the trainer's `max_length` and the eval steps'.** They are separate keys because the steps are separately schedulable; the recipe is what ties them together. Default: `4096`. |
| `length_policy` | string | `drop` \| `truncate`. `drop`, because a truncated conversation ends mid assistant turn — which teaches the student to produce unterminated answers. `truncate` keeps them and lets the trainer truncate, counted in the manifest. Default: `drop`. |
| `think_policy` | string | `keep` \| `strip` \| `require`. Reasoning corpora carry `<think>…</think>` inside assistant content: `keep` trains on the traces, `strip` on answers only, `require` restricts the corpus to records that have a trace. No safe default exists: the three produce different students from the same input. Default: `keep`. |
| `completion_boundary` | string | `last_message` \| `all_assistant`. `last_message` matches GOLD's `last_message_only` and is **enforced**: a conversation whose final turn is not the assistant's would contribute no loss, so it is dropped and counted rather than trained on as a no-op. `all_assistant` supervises every assistant turn. Default: `last_message`. |
| `min_messages` | integer | Minimum messages per conversation. Default: `2`. |
| `documents_policy` | string | `drop` \| `keep`. A record with a non-empty `documents` list depends on grounding text granite's template does not render, so `keep` trains the student to answer from evidence it never saw. It is a key rather than a constant because the choice is template-dependent — the script measures whether the template renders `documents` and warns on either disagreement. Pair granite with a RAG-aware template and `keep` becomes correct. Default: `drop`. |
| `eval_fraction` | number | Fraction held out as `eval.jsonl`, in `[0.0, 1.0)`. The split is deterministic from `seed` alone. Default: `0.0`. |
| `seed` | integer | Split seed. Default: `42`. |
| `max_examples` | integer | Stop after N **kept** examples; `0` = no limit. What a smoke run should set. Default: `0`. |
| `emit_row_id` | boolean | Writes each row's content id into the row, so the trainer can report which rows it consumed and any holder of `train.jsonl` can recompute and verify the id. Mirrors the script's own default deliberately — flip both together, not this one alone. Default: `false`. |
| `hf_home` | string | Overrides `HF_HOME` for this step only; empty leaves the launcher's value. Default: `""`. |

> `--shard-count` / `--shard-index` exist in `src/prep_corpus.py` but are deliberately **not**
> config keys: the sharded path is driven by `src/merge_shards.py`, and a recipe passing them
> would be sharding by accident.

### Source delivery (`code_config`)

`code_config` is identical in every ported distillation step and is documented once — see
"Source delivery" in the `space://steps/distill/tokenizer-align` step's README. No field is
required; the defaults clone a public repository unauthenticated, so no credential reaches the
container.

| Field | Type | Purpose |
|---|---|---|
| `code_dir` | string | Filesystem checkout to use instead of cloning. Default: `""` (clone). |
| `expect_ref` | string | Commit a `code_dir` checkout must be at. Default: `a5d59bc45524a8d75706e20d44ae1a254f273f23`. |
| `repo` | string | Repository cloned when `code_dir` is empty. Default: `https://github.com/laminair/gb-steps-distillation.git`. |
| `ref` | string | Ref checked out after the clone. Default: same commit as `expect_ref`. |
| `workdir` | string | Clone destination, relative to the working directory. Default: `distill-code`. |
| `token_secret` | string | Name of a space secret for a private clone. Default: `""`. |
| `python` | string | Interpreter. Default: `/stage/.venv/bin/python`. |
| `setup_command` | string | Bash run after the source resolves, before the workload. Default: `""` (skipped). |

## Inputs and outputs

### Inputs

The step declares no `inputs:` of its own. It reads two things, both through config:

- **`corpus_config.tokenizer`** — bind the target's input to `align.retagged_student` (the
  `space://steps/distill/tokenizer-align` step's output) and pass
  `{{ bindings.tokenizer.binding.path }}`. The binding is also the ordering edge that guarantees
  the tokenizer exists before this target starts. A pre-existing tokenizer does not work — see
  [It cannot run without align's output](#it-cannot-run-without-aligns-output).
- **`corpus_config.dataset`** — a literal path, or a target input (the GOLD recipes declare
  `source_dataset: {uri: ..., type: dataset}` and pass `{{ bindings.source_dataset.binding.path }}`).

### Outputs

| Output | Type | What it is |
|---|---|---|
| `corpus` | `dataset` | `<out_dir>/train.jsonl` — **the file**, not the directory. Emitted as `GB_ARTIFACT_ID:corpus GB_ARTIFACT_PATH:<abs out_dir>/train.jsonl`. |

Declare it on the target as `uri: "env://{{ binding.path }}"`, `type: dataset`. Recipes bind it
(`corpus.corpus`) into the `space://steps/distill/sft`, `space://steps/distill/gold` and
`space://steps/distill/eval` steps. Next to it in `out_dir` the step also writes
`corpus_manifest.json` (policies, counts, the tokenizer identity, the split paths) and, when
`eval_fraction > 0`, `eval.jsonl` — neither is a declared output; see
[The artifact is the file, not the directory](#the-artifact-is-the-file-not-the-directory).

## Working directory and paths

The run starts in the step's working directory, `$GB_BUILD_WORKDIR` (falling back to `$PWD`).
`src/` is mounted at `./src`, and `./src/prep_corpus.py` is what runs. The shared
`gb_steps_post_training` package is cloned into `$GB_BUILD_WORKDIR/<code_config.workdir>`
(default `distill-code`), or taken from `code_config.code_dir`, and put on `PYTHONPATH`. A relative `out_dir` is absolutised against
`$GB_BUILD_WORKDIR` before the artifact marker is printed, because the monitor hands the path to
the `env://` store, possibly from another host, and a relative `env:` URI is rejected at config
load. `dataset` and `tokenizer` are passed through unchanged, so give them as absolute paths.

## Example build.yaml

The step after align. `max_length` comes from the same parameter the trainer and both eval
targets read: three stages measure lengths and they have to agree about the budget.

```yaml
granite.build:
  name: corpus-prep-example
  targets:
    align:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      outputs:
        retagged_student:
          uri: "env://{{ binding.path }}"
          type: model
        teacher_overlay:
          uri: "env://{{ binding.path }}"
          type: model
        student_overlay:
          uri: "env://{{ binding.path }}"
          type: model
      steps:
        - step_uri: space://steps/distill/tokenizer-align
          config:
            align_config:
              teacher_model: "/path/to/teacher"
              student_model: "/path/to/student"
              out_dir: "$${RUN_NAME}/align"

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
        - step_uri: space://steps/distill/corpus-prep
          config:
            compute_config:
              num_nodes: 1
              num_cpus_per_node: 1
            corpus_config:
              dataset: "$${DATASET}"
              tokenizer: "{{ bindings.tokenizer.binding.path }}"
              out_dir: "$${RUN_NAME}/corpus"
              max_length: $${MAX_LENGTH}
```

## Notes and limitations

### The corpus is text, not tokens — and is still tokenizer-specific

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

### The artifact is the file, not the directory

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

### It cannot run without align's output

Measured on BlueVela: **none** of the 13 hand-built `/proj` overlay directories has a real
`{% generation %}` block tag in its chat template (they carry `add_generation_prompt`, which is a
different thing), while the `space://steps/distill/tokenizer-align` step's output has four.
Without that tag, `return_assistant_tokens_mask=True` returns an empty mask for **every** row and
this step refuses:

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

### Resume

Like the align step, this one answers "has this already been done, and with what?" for itself,
because a preemptable-queue recipe is restarted rather than resumed. Before the filtering loop,
`src/prep_corpus.py` compares what is recorded in `out_dir` with what this run expects:

| Exit | Meaning |
|---|---|
| `0` | built now, **or** already built under this **exact** expectation — the script prints `=== corpus already built`, the counts and the tokenizer identity, and the step publishes the existing file |
| `1` | recorded under a **different** expectation, or an output is gone — refuse, naming the key (also any other refusal, e.g. the empty-mask one above) |

Every filtering policy above is part of that expectation, so changing one and re-running the same
`out_dir` refuses rather than silently mixing two policies into one corpus. To rebuild
deliberately, delete the marker file the "already built" message names.

### Runtime

- **No GPU.** Rendering and counting tokens is CPU work; the launcher requests no accelerators.
- **Offline hub.** The launcher sets `HF_HUB_OFFLINE=1` and `HF_HOME=/opt/hf-cache`.
