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
| Granite 4.2 | **`documents` support removed** (template moved to ChatML). | Not supported — inline the context into a message instead |

> **Warning — silent context loss.** `apply_chat_template` does **not** raise when
> a template ignores `documents=`; it renders a valid-looking prompt with the
> context missing. This affects Granite 4.2 and non-Granite models (plain Llama,
> etc.). The drivers forward the `documents` column blindly, so a `documents`
> column plus a 4.2 model silently trains on prompts with no context. When the
> target template lacks documents support, inline the passages into a leading
> system message in the `input` messages instead of using the `documents` column.

`model_type` is `granite` for **both** 4.1 and 4.2, so it cannot tell them apart;
inspect `tokenizer.chat_template` instead. `autotune/tools/build_factuality_dataset.py`
does exactly this — see below.

### Building factuality datasets

`autotune/tools/build_factuality_dataset.py` converts raw ELI5/Biographies splits
into judge datasets and handles all of the above automatically: it inspects the
chat template, picks the right document shape, inlines the context when the
template ignores `documents=`, and refuses to write records whose context would
be dropped.

```bash
python -m autotune.tools.build_factuality_dataset \
    --input /path/to/eli5_raw_train.json \
    --output-dir /path/to/out \
    --task detection \
    --model ibm-granite/granite-4.1-3b
```

It takes one raw JSON file (top-level list of records). Defaults to
`--format both`, writing two files into `--output-dir` whose names come from the
input stem, the task, and the format — here
`eli5_raw_train_detection_formatted.jsonl` (a pre-rendered `input` string) and
`eli5_raw_train_detection_chat.jsonl` (messages plus a `documents` column). Use
`--format formatted` or `--format chat` for just one. `--doc-style` overrides
template detection; `--no-inline-documents` turns the inline fallback into a hard
error; `--include-meta` echoes the original `query`/`response`/`label` onto each
correction record so eval can recover the originals.

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

### Shared prompts and thinking mode

All three builders take their prompt text and chat-template logic from
`autotune/tools/_chat_utils.py`, so the training prompts cannot drift from the
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

---

← Back to [README](../README.md)
