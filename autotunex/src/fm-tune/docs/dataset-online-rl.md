# Dataset format: Online RL — PPO / GRPO / DAPO (verl)

Online RL uses [verl](https://github.com/volcengine/verl) with vLLM for
rollout generation. Consumed by:

- `autotune/trainers/driver_multi_verl.py` — multi-GPU verl driver.

The dataset format is what verl's built-in `RLHFDataset` expects. All three
algorithms (PPO, GRPO, DAPO) use the same schema.

## File format

**Parquet only.** verl's `RLHFDataset` calls
`datasets.load_dataset("parquet", ...)` internally. JSONL / JSON / CSV are
not supported on this path.

## Required columns

| Column | Type | Purpose |
|---|---|---|
| `data_source` | `str` | Routing key for reward scoring. Passed as the `data_source` arg to `compute_score(...)` so one reward function can dispatch to per-task scorers. |
| `prompt` | `list[dict]` | Chat messages: `[{"role": "system" \| "user", "content": "..."}, ...]`. verl applies the tokenizer's chat template internally — **do not pre-template**. |
| `reward_model` | `dict` | For rule-based scoring: `{"style": "rule", "ground_truth": "<truth>"}`. `ground_truth` is forwarded to `compute_score`. |

`prompt` must be a list of messages, **never** a plain string. verl does not
auto-wrap strings.

## Optional columns

| Column | Type | Purpose |
|---|---|---|
| `extra_info` | any JSON-serializable | Passed through to `compute_score` as the `extra_info` argument. Use for per-row metadata (difficulty, task tags, retrieval ids, etc.). |

Additional columns (e.g., `images`, `videos`, `tools`) are supported by
verl's `RLHFDataset` but are outside the scope of the text-only PPO/GRPO/DAPO
setup used in AutoTune.

## Reward function

For rule-based scoring (the default path), supply a Python file with a
`compute_score` function. Configure via YAML under `training_rl_config`:

```yaml
training_rl_config:
  reward_function_path: /path/to/reward.py
  reward_function_name: compute_score   # optional; defaults to "compute_score"
```

Expected signature:

```python
def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth=None,
    extra_info: dict = None,
    **kwargs,
) -> float:
    """Return a scalar reward for one rollout."""
```

- `data_source` comes from the row's `data_source` column.
- `solution_str` is the generated rollout (post-chat-template text).
- `ground_truth` is `row["reward_model"]["ground_truth"]`.
- `extra_info` is the row's `extra_info` column if present, else `None`.

For a learned reward model instead, set `reward_model_path` to an HF model
path in the YAML; verl will score rollouts with that model and the
`reward_model.style` / `reward_model.ground_truth` fields are ignored.

## Examples

### Single-task GSM8K (rule-based)

Every row shares a `data_source` so the reward function handles all rows the
same way. `ground_truth` is the extracted final number.

```json
{
  "data_source": "gsm8k",
  "prompt": [
    {"role": "system", "content": "You are a math tutor. Show your work and put the final number after ####."},
    {"role": "user", "content": "Janet's ducks lay 16 eggs per day. She eats 3 and bakes 4 muffins. She sells the rest at $2 each. How much does she make per day?"}
  ],
  "reward_model": {"style": "rule", "ground_truth": "18"}
}
```

### Multi-task with `data_source` routing

Different tasks get different `data_source` values so `compute_score` can
dispatch:

```json
{"data_source": "gsm8k", "prompt": [...], "reward_model": {"style": "rule", "ground_truth": "42"}}
{"data_source": "squad", "prompt": [...], "reward_model": {"style": "rule", "ground_truth": "Paris"}}
```

And the reward function:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source == "gsm8k":
        return _score_gsm8k(solution_str, ground_truth)
    if data_source == "squad":
        return _score_squad(solution_str, ground_truth)
    return 0.0  # unknown task
```

### With `extra_info` metadata

Pass per-row hints that the reward function can use (here: a difficulty
multiplier):

```json
{
  "data_source": "math",
  "prompt": [...],
  "reward_model": {"style": "rule", "ground_truth": "6.28"},
  "extra_info": {"difficulty": "hard", "topic": "trigonometry"}
}
```

## Building a dataset

The repo ships a reference script that converts a HF dataset to the verl
schema:

```bash
python autotune/tools/build_gsm8k_dataset.py \
    --data-source gsm8k \
    --output-dir ./data/gsm8k \
    --prompt-key question \
    --answer-key answer
```

See `autotune/tools/build_gsm8k_dataset.py` for the full extraction logic
(regex for `#### <number>` in GSM8K answers, chat-message assembly, parquet
writes).

Minimal build from pandas:

```python
import pandas as pd

rows = [
    {
        "data_source": "gsm8k",
        "prompt": [
            {"role": "system", "content": "Solve step-by-step; final answer after ####."},
            {"role": "user", "content": "What is 7 * 8?"},
        ],
        "reward_model": {"style": "rule", "ground_truth": "56"},
    },
    # ...
]
pd.DataFrame(rows).to_parquet("rl_train.parquet", index=False)
```

## Gotchas

- **Parquet is mandatory.** Providing a `.jsonl` or `.json` file fails at
  dataset-load time; verl's `RLHFDataset` is hard-coded to the parquet
  reader.
- **`prompt` must be a messages list.** Raw strings are not accepted. If
  your source is plain text, wrap into `[{"role": "user", "content":
  "<text>"}]` before writing.
- **`data_source` value must match reward dispatch.** A typo silently routes
  to verl's default scorer (which may return 0 for everything), producing a
  flat reward curve with no obvious error. Verify by printing unique
  `data_source` values and cross-checking against `compute_score`.
- **`max_prompt_length` filtering.** The verl driver sets
  `max_prompt_length` from the config. Prompts over this length are
  truncated or dropped depending on the `truncation` mode. Monitor
  `filter_overlong_prompts` warnings in the trainer logs.
- **vLLM `max_model_len`.** The driver sets vLLM's `max_model_len =
  max_prompt_length + max_response_length`. If your prompts routinely sit
  near the cap, leave headroom for the response or rollouts will be
  truncated.
- **Reward function import errors are noisy but trial-fatal.** The reward
  file is loaded per Ray worker; a syntax error or missing import in
  `compute_score` surfaces as a per-worker crash, not a clean trial failure.
  Sanity-check the reward file standalone (`python reward.py`) before
  launching.

---

← Back to [README](../README.md)
