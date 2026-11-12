# Dataset format: SFT / LoRA / aLoRA / LoHa / LoKr / VeRA

This schema applies to supervised fine-tuning and all parameter-efficient
fine-tuning (PEFT) methods. It is consumed by:

- `autotune/trainers/driver_single.py` — single-GPU SFT/PEFT driver.
- `autotune/trainers/driver_multi_hf_fsdp.py` — multi-GPU FSDP driver.
- `autotune/trainers/driver_multi_hf_ds.py` — multi-GPU DeepSpeed driver.

## Required columns

Every row must have two columns:

| Column | Type | Purpose |
|---|---|---|
| `input` | `str` **or** `list[dict]` | The prompt. Shape determines whether the chat template is applied. |
| `output` | `str` | The target completion. Always a plain string. No chat template is applied to `output`. |

The shape of `input` is auto-detected. Detection granularity depends on the
driver:

- **Single-GPU driver** (`driver_single.py`): detected per row. A file with
  some plain-string rows and some message-list rows is handled correctly —
  the chat template is applied row-by-row only when `input` is a list.
- **Multi-GPU drivers** (FSDP, DeepSpeed): detected once from the first row
  of the (batch or file). Mixing shapes in one file silently follows the
  first-row shape — keep a file consistent.

In either case: if the detected shape is a message list, the tokenizer's
`apply_chat_template(..., add_generation_prompt=True)` is applied to `input`
before tokenization; if it's a string, `input` is used verbatim.

## Optional columns (chat format only)

These are only meaningful when `input` is a list of messages. They are
auto-detected column-wise (present + first-row value is a non-empty list).

| Column | Type | Purpose |
|---|---|---|
| `documents` | `list[dict]` | Passed as `documents=...` to `apply_chat_template` for RAG-style prompts. |
| `tools` | `list[dict]` | Passed as `tools=...` for tool-use prompts (OpenAI-style function schemas). |

The exact dict shapes for `documents` and `tools` are governed by the
tokenizer's chat template, not by AutoTune. Check the model card for the
expected structure.

- **OpenAI-style tools**: `[{"type": "function", "function": {"name": "...",
  "description": "...", "parameters": {...}}}]`.
- **Granite documents**: the shape differs *by Granite generation* — see below.

### Granite `documents` differ by generation

| Family | Handling | Shape to emit |
|---|---|---|
| Granite 3.x | Template reads `document['doc_id']` and `document['text']`, emitting one `<\|start_of_role\|>document {...}<\|end_of_role\|>` block per doc. | `[{"doc_id": "0", "text": "..."}]` — `doc_id` is **required** |
| Granite 4.0 / 4.1 | `document \| tojson` into a `<documents>` XML block in the system message. Keys render **verbatim**. | `[{"text": "..."}]` — omit `doc_id`, or it leaks into the prompt |
| Granite 4.2 | **`documents` support removed** (template moved to ChatML). Renders a `tool` message as a `<tool_response>[...]</tool_response>` block. | No `documents` column — put the passages in a `tool` message inside `input` |

> **Warning — silent context loss.** `apply_chat_template` does **not** raise when
> a template ignores `documents=`; it renders a valid-looking prompt with the
> context missing. This affects Granite 4.2 and non-Granite models (plain Llama,
> etc.). The drivers forward the `documents` column blindly, so a `documents`
> column plus a 4.2 model silently trains on prompts with no context.
>
> For **Granite 4.2**, pass the context the way 4.2 natively represents RAG: a
> `{"role": "tool"}` message whose content is a JSON array of
> `{"source", "document_id", "content"}` records, embedded in the `input` messages
> with **no** `documents` column. No assistant `tool_calls` turn and no `tools`
> schema are needed — 4.2 renders a bare `tool` message. For **other** templates
> that ignore `documents=`, inline the passages into a leading system message instead.

#### Where the tool message goes

For judge/guardian data (`user` query → `assistant` response → `user` instruction),
put the tool message **before the first assistant turn**. The documents are the
grounding for the response being judged; placing them after the last user turn would
bury the context after the judging instruction and immediately before the label.

```json
{
  "input": [
    {"role": "user", "content": "Who painted the Mona Lisa?"},
    {"role": "tool", "content": "[{\"source\": \"knowledge_base\", \"document_id\": \"0\", \"content\": \"...\"}]"},
    {"role": "assistant", "content": "It was painted by Michelangelo."},
    {"role": "user", "content": "<guardian>..."}
  ],
  "output": "{\"score\": \"no\"}"
}
```

`model_type` is `granite` for **both** 4.1 and 4.2, so it cannot tell them apart;
inspect `tokenizer.chat_template` instead. `autotune/tools/build_factuality_dataset.py`
does exactly this — see below.

### Building factuality datasets

`autotune/tools/build_factuality_dataset.py` converts raw ELI5/Biographies splits
into judge datasets and handles all of the above automatically: it inspects the
chat template, picks the right document shape (including a `tool` message for
Granite 4.2), inlines the context when a non-ChatML template ignores `documents=`,
and refuses to write records whose context would be dropped.

```bash
python -m autotune.tools.build_factuality_dataset \
    --input /path/to/eli5_raw_train.json \
    --output-dir /path/to/out \
    --task detection \
    --model ibm-granite/granite-4.1-3b
```

All three builders run either as a module (above) or as a plain script
(`python autotune/tools/build_factuality_dataset.py ...`). They use absolute
imports, so direct execution needs no `PYTHONPATH`.

It takes one raw JSON file (top-level list of records). Defaults to
`--format both`, writing two files into `--output-dir` whose names come from the
input stem, the task, and the format — here
`eli5_raw_train_detection_formatted.jsonl` (a pre-rendered `input` string) and
`eli5_raw_train_detection_chat.jsonl` (messages plus a `documents` column). Use
`--format formatted` or `--format chat` for just one. `--doc-style` overrides
template detection (`auto` resolves to `granite3`, `granite4`, `tool` for
ChatML/4.2, or `inline`); `--no-inline-documents` turns the *inline* fallback into a
hard error — it does not affect `tool`, which is a native representation rather than
a fallback.

Correction records carry the original `query`, `response`, and `label` alongside
`input`/`output` by default, so an eval file can recover the originals and score the
generated correction; pass `--no-include-meta` for a minimal `input`/`output` file.
Detection records never carry them — there the label *is* the target.

> **Granite 4.2 merges the context into one document.** 4.2 renders a `tool` message
> as a single `<tool_response>[...]</tool_response>` JSON *array*, so N documents
> would become N array entries — a different prompt shape from the single
> `<documents>` block 4.1 renders. The builder therefore joins every passage of a
> datapoint into **one** document separated by a blank line, so data built for 4.2
> trains the same context layout as data built for 4.1. The other styles keep one
> document per passage.

### Building atomizer and NLI datasets

Two sibling builders produce the other intrinsics datasets. Neither uses RAG
documents, so the document-shape question does not apply to them — but both are
still template-aware for thinking mode (below).

```bash
# Decompose responses into atomic units (input: {query, response, atoms})
python -m autotune.tools.build_atomizer_dataset \
    --input /path/to/atoms_raw.json \
    --output-dir /path/to/out \
    --model ibm-granite/granite-4.1-3b

# Classify entailment / contradiction / neutral (input: {premise, hypothesis, label})
python -m autotune.tools.build_nli_dataset \
    --input /path/to/nli_raw.json \
    --output-dir /path/to/out \
    --model ibm-granite/granite-4.1-3b
```

All three builders share the same interface: one raw JSON file via `--input`, a
directory via `--output-dir`, and output filenames derived from the input stem and
the format — `nli_raw.json` yields `nli_raw_formatted.jsonl` and
`nli_raw_chat.jsonl` (the factuality builder also inserts the task). Both default
to `--format both`; a single `--format` writes just that one file. Both drop
records whose target tokenizes to `--max-output-length` tokens or more (default
1024; `0` disables), and print token-length stats for the `formatted` pass to help
size `max_seq_length`.

#### Mixed NLI build (ELI5 + MNLI + SNLI)

The training data for the NLI intrinsic mixes the raw ELI5 NLI splits with MNLI
and SNLI, so the adapter also sees the standard distributions. Passing all three
`--eli5-*` files selects this mode:

```bash
python -m autotune.tools.build_nli_dataset \
    --eli5-train /path/to/eli5_nli_raw_train.json \
    --eli5-val /path/to/eli5_nli_raw_val.json \
    --eli5-test /path/to/eli5_nli_raw_test.json \
    --output-dir /path/to/out \
    --model ibm-granite/granite-4.2-8b \
    --train-mix eli5=180000,mnli=60000,snli=60000 \
    --val-mix eli5=6000,mnli=3000,snli=3000
```

| Source | train | val | test |
|---|---|---|---|
| ELI5 | `--eli5-train` | `--eli5-val` | `--eli5-test`, balanced head |
| MNLI ([`nyu-mll/multi_nli`](https://huggingface.co/datasets/nyu-mll/multi_nli)) | `train` | `validation_mismatched` | `validation_matched` |
| SNLI ([`stanfordnlp/snli`](https://huggingface.co/datasets/stanfordnlp/snli)) | `train` | `validation` | `test` |
| ANLI ([`facebook/anli`](https://huggingface.co/datasets/facebook/anli)) | — | — | `test_r1`, `test_r2`, `test_r3` |

`--train-mix`/`--val-mix` give a record count per source. Each count must be
divisible by 3 and is split evenly across the labels; the build fails rather than
take fewer when a label cannot fill its share. Sampling is seeded per
`(source, label)` (`--seed`, default 1234). Pairs are deduped on the stripped
`(premise, hypothesis)`: no test pair (ELI5 or benchmark) reaches val or train, no
val pair reaches train, and no pair repeats within a split.

Test is not mixed. The full ELI5 test file becomes `nli_mix_test`, reordered so
its first `--test-balanced-head` rows (default 3000) hold an equal share of each
label — what an eval with `limit: 3000` reads — and the rest follow in file
order. The five benchmark splits are written in full, in source order, as
`snli_test`, `mnli_validation_matched` and `anli_test_r{1,2,3}`. Every test file
uses the same prompt and template with thinking off, so the base model and its
LoRA adapters are scored on identical inputs.

Outputs (prefix `--name`, default `nli_mix`): `nli_mix_{train,val,test}_raw.json`
(records with a `source` field), the row-aligned
`nli_mix_{split}_{formatted,chat}.jsonl`, the benchmark
`{stem}_{formatted,chat}.jsonl` files, and `nli_mix_manifest.json` with the
settings and per-split/source/label counts, the overlap excluded per source, and
the label counts of the test head. Existing outputs are refused unless
`--overwrite` is passed.

#### Hugging Face NLI benchmarks only

To rebuild just the held-out benchmarks (for example for another tokenizer):

```bash
python -m autotune.tools.build_nli_dataset \
    --benchmarks snli anli mnli \
    --output-dir /path/to/benchmarks \
    --model ibm-granite/granite-4.2-8b
```

`--benchmarks` accepts any subset of `snli anli mnli`. Supply `--input` alongside
it to convert a local raw file in the same run. The Hugging Face loader uses the
`datasets` dependency from the project's `core` extra and streams only the
requested splits.

| Benchmark | `--benchmark-split auto` (default) | `--benchmark-split validation` |
|---|---|---|
| SNLI | `test` | `validation` |
| ANLI | `test_r1`, `test_r2`, `test_r3` | `dev_r1`, `dev_r2`, `dev_r3` |
| MNLI | `validation_matched` | `validation_matched` |

MNLI's `test` split has no gold labels; explicitly selecting
`--benchmark-split test` with `mnli` is an error. ANLI rounds remain separate
benchmarks. Formatted and chat files hold identical rows in the same order, with
a plain-string `output` such as `{"label": "entailment"}`. Granite 4.2 thinking is
disabled by default; `--thinking` enables it for rendering.

Labels are decoded from source metadata where available, with the source mapping
as a fallback (`0=entailment`, `1=neutral`, `2=contradiction`). Unlabeled (`-1`)
and malformed rows are skipped and counted; text is kept verbatim. An empty
benchmark is an error.

`nli_benchmarks_manifest.json` records this invocation's datasets, configs,
splits, model, conversion settings, filenames, label counts, and skipped/written
row counts. `--name-tag TAG` inserts `_TAG` before each file's format suffix and
names the manifest `nli_benchmarks_manifest_TAG.json`. Each successful file build
replaces the corresponding previous output. A failed run can leave completed
earlier splits; a new manifest is written only after all requested splits succeed.

### Shared prompts and thinking mode

All three builders take their prompt text and chat-template logic from
`autotune/tools/chat_utils.py`, so the training prompts cannot drift from the
eval-side ones. That module has two clearly separated halves: the chat-template
plumbing, then the prompt construction (guardian/factuality, atomizer, NLI). The
judging criteria live in a `CRITERIA` dict keyed by criteria id, and every
judging task renders the same guardian block:

```
<guardian>{system prompt}

### Criteria: {CRITERIA[criteria_id]}

### Scoring Schema: {schema}
```

All three tasks share it: factuality detection/correction use
`CRITERIA["factuality"]`, the atomizer uses `CRITERIA["atomicity"]`, and NLI uses
`CRITERIA["entailment"]` — so adding a judging task is a new `CRITERIA` entry plus
a schema constant, with no new assembly code.

> **Granite 4.2 thinking mode.** Granite 4.2 defaults to `enable_thinking=True`,
> which ends the rendered prompt with an *open* `<think>` block. Training a
> structured target (a JSON verdict, a label, a list of atoms) against such a
> prompt teaches the model to emit that target as reasoning content. All three
> builders therefore pass `enable_thinking=False` on templates that support the
> kwarg. Pass `--thinking` to opt back in.
>
> The same applies at **training** time for `chat`-format datasets, because there
> the *driver* renders the template, not the builder. `driver_single.py`,
> `driver_multi_hf_fsdp.py`, and `driver_multi_hf_ds.py` all disable thinking on
> templates that support the kwarg. Set `disable_thinking: false` in the YAML
> `training_config` to re-enable it (e.g. for a genuine reasoning-SFT run).

## Supported file formats

Readers are selected by file extension:

| Extension | Reader |
|---|---|
| `.jsonl` | JSON Lines (one record per line) |
| `.json` | JSON array (top-level list) |
| `.csv` | CSV (columns mapped to row keys) |
| `.parquet` | Parquet (columns mapped to row keys) |

For chat / documents / tools columns, use `.jsonl` or `.parquet` — CSV can't
represent nested structures cleanly.

## Examples

### Plain prompt / completion

```json
{"input": "Summarize the following article: ...", "output": "The article discusses..."}
{"input": "Classify the sentiment: ...", "output": "positive"}
```

### Chat messages

```json
{"input": [{"role": "user", "content": "Explain quantum tunneling."}], "output": "Quantum tunneling is..."}
```

### Chat messages with documents (RAG)

```json
{
  "input": [{"role": "user", "content": "What does the report say about Q3?"}],
  "documents": [{"title": "2025 Q3 Report", "text": "Revenue grew 12%..."}],
  "output": "The Q3 report states that revenue grew 12%..."
}
```

### Chat messages with documents and tools

```json
{
  "input": [{"role": "user", "content": "Look up the Q3 revenue and summarize."}],
  "documents": [{"title": "2025 Q3 Report", "text": "Revenue grew 12%..."}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_kb",
        "description": "Search the knowledge base by keyword.",
        "parameters": {
          "type": "object",
          "properties": {"query": {"type": "string"}},
          "required": ["query"]
        }
      }
    }
  ],
  "output": "The Q3 report states that revenue grew 12%..."
}
```

## Building a dataset from scratch

From a pandas DataFrame with any column names, rename to `input` / `output`
and write JSONL:

```python
import pandas as pd

df = pd.DataFrame(
    {
        "question": ["What is 2+2?", "Capital of France?"],
        "answer": ["4", "Paris"],
    }
)
df = df.rename(columns={"question": "input", "answer": "output"})
df.to_json("train.jsonl", orient="records", lines=True)
```

For chat format, build the messages column as Python lists — `to_json` will
serialize them correctly:

```python
df = pd.DataFrame(
    {
        "input": [
            [{"role": "user", "content": "What is 2+2?"}],
            [{"role": "user", "content": "Capital of France?"}],
        ],
        "output": ["4", "Paris"],
    }
)
df.to_json("train.jsonl", orient="records", lines=True)
```

## Gotchas

- **First-row dictates shape (multi-GPU only).** On the FSDP and DeepSpeed
  drivers, if your first row is a plain string but later rows are message
  lists, chat-template application is silently skipped for all rows.
  Validate the first row before training, or run on the single-GPU driver
  (which detects per row).
- **aLoRA target modules.** aLoRA restricts adapters to `q_proj`, `k_proj`,
  `v_proj` (vs. all linear layers for LoRA). Dataset format is unchanged; it
  is a model-side constraint.
- **`output` as empty string.** An empty completion produces zero loss on
  that row; the trainer won't error, but the row contributes nothing.
- **`max_length` is a cap, not a cost.** Rows are stored padded to
  `max_length`, but each batch is collated to its own rows
  (`autotune/trainers/_collate.py`): with `flash_attention_2` on CUDA a batch
  is flattened into one varlen sequence with no pad tokens
  (`padding_free: auto`, the default); otherwise it is cropped to its longest
  row, with similar-length rows batched together (`group_by_length: true`).
  Raising `max_length` to fit a few long records no longer slows down the
  short ones. Each run logs the padding share it avoided
  (`... % real tokens when padded to max_length`).

---

← Back to [README](../README.md)
