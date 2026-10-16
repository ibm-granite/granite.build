# Dataset format: Offline RL — DPO / KTO

Offline preference alignment uses TRL's `DPOTrainer` and `KTOTrainer` under
the hood. Consumed by:

- `autotune/trainers/driver_single_trl.py` — single-GPU TRL driver.
- `autotune/trainers/driver_multi_trl_ds.py` — multi-GPU DeepSpeed TRL driver.
- `autotune/trainers/driver_multi_trl_fsdp.py` — multi-GPU FSDP TRL driver.

DPO and KTO use different column schemas. ORPO follows the same schema as
DPO.

## DPO / ORPO

Requires three columns:

| Column | Type | Purpose |
|---|---|---|
| `prompt` | `str` **or** `list[dict]` | The input prompt, shared by chosen and rejected responses. |
| `chosen` | `str` **or** `list[dict]` | The preferred completion. |
| `rejected` | `str` **or** `list[dict]` | The dispreferred completion. |

Each of the three fields can be a plain string (**standard format**) or a
list of chat messages (**conversational format**). TRL 0.29 accepts both;
when rows are conversational, TRL internally calls
`maybe_apply_chat_template` to serialize them using the tokenizer's chat
template. **The tokenizer must have a chat template set** for conversational
format to work.

All three columns must use the same format within a single file.

### DPO — standard format

```json
{"prompt": "Write a poem about...", "chosen": "Roses are red...", "rejected": "Here is a poem..."}
{"prompt": "Explain gravity.", "chosen": "Gravity is the force...", "rejected": "I don't know."}
```

### DPO — conversational format

```json
{
  "prompt":   [{"role": "user", "content": "Write a poem about the sea."}],
  "chosen":   [{"role": "assistant", "content": "Beneath the waves..."}],
  "rejected": [{"role": "assistant", "content": "Here's a poem."}]
}
```

## KTO

Requires three columns:

| Column | Type | Purpose |
|---|---|---|
| `prompt` | `str` **or** `list[dict]` | The input prompt. |
| `completion` | `str` **or** `list[dict]` | A single completion. |
| `label` | `bool` | `true` = desirable completion, `false` = undesirable. |

KTO learns from binary feedback rather than paired preferences. A training
set **must contain both labels** — `label=true` and `label=false` rows — or
loss will collapse.

### KTO — standard format

```json
{"prompt": "Explain quantum...", "completion": "Quantum mechanics is...", "label": true}
{"prompt": "Explain quantum...", "completion": "I don't know.", "label": false}
```

### KTO — conversational format

```json
{
  "prompt":     [{"role": "user", "content": "Explain quantum mechanics."}],
  "completion": [{"role": "assistant", "content": "Quantum mechanics is..."}],
  "label":      true
}
```

## Supported file formats

Readers are selected by file extension:

| Extension | Reader |
|---|---|
| `.jsonl` | JSON Lines (one record per line) |
| `.json` | JSON array (top-level list) |
| `.csv` | CSV (plain-string format only — can't represent nested messages) |
| `.parquet` | Parquet |

For conversational format, prefer `.jsonl` or `.parquet`.

## Building a dataset from scratch

### DPO from pandas

```python
import pandas as pd

df = pd.DataFrame(
    {
        "prompt": ["Write a poem about the sea."],
        "chosen": ["Beneath the waves..."],
        "rejected": ["Here's a poem."],
    }
)
df.to_json("dpo_train.jsonl", orient="records", lines=True)
```

### KTO from pandas

```python
import pandas as pd

df = pd.DataFrame(
    {
        "prompt": ["Explain quantum mechanics.", "Explain quantum mechanics."],
        "completion": ["Quantum mechanics is...", "I don't know."],
        "label": [True, False],
    }
)
df.to_json("kto_train.jsonl", orient="records", lines=True)
```

### DPO — conversational

```python
import pandas as pd

df = pd.DataFrame(
    {
        "prompt": [[{"role": "user", "content": "Write a poem about the sea."}]],
        "chosen": [[{"role": "assistant", "content": "Beneath the waves..."}]],
        "rejected": [[{"role": "assistant", "content": "Here's a poem."}]],
    }
)
df.to_json("dpo_train.jsonl", orient="records", lines=True)
```

## Gotchas

- **Exact column names required.** TRL expects `prompt` / `chosen` /
  `rejected` (DPO/ORPO) or `prompt` / `completion` / `label` (KTO). AutoTune
  does not rename columns automatically; prepare your source to match.
- **Conversational format needs a chat template.** If the tokenizer has no
  `chat_template`, TRL will not know how to serialize messages and will
  either raise or silently concatenate fields. Check
  `tokenizer.chat_template is not None` before committing to conversational
  data.
- **Chosen must differ from rejected.** If `chosen == rejected`, the DPO
  gradient is zero on that row — wasted compute. Filter or deduplicate.
- **KTO label balance.** A dataset with only `label=true` rows (or only
  `false`) will not train; KTO requires both.
- **Format consistency.** Mixing standard and conversational rows in one
  file is undefined; stick to one.

---

← Back to [README](../README.md)
