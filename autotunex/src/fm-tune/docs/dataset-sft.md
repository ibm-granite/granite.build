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
expected structure. Two common shapes:

- **Granite-style documents**: `[{"title": "...", "text": "..."}]` or
  `[{"doc_id": "...", "text": "..."}]`.
- **OpenAI-style tools**: `[{"type": "function", "function": {"name": "...",
  "description": "...", "parameters": {...}}}]`.

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
