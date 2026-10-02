# gen-smoke

Greedy-decodes eight raw prompts (code in five languages, three short prose prompts)
from each exported checkpoint of a ladder and measures the fraction of generated lines
inside a run of three or more identical lines. That is the failure build df8512e0's
distilled model showed ("// (true)" about fifty times running), which no training-time
metric saw. The detector, the reason it gates on runs rather than on adjacent repeats,
and the measurements behind both are in [`src/gen_smoke.py`](src/gen_smoke.py).

It writes `repetition.json` (per rung: mean and worst looped fraction, adjacent-repeat
rate, a verdict and the samples) and registers it as `repetition_report`.

## Config (`gen_smoke_config`)

| key | meaning |
|---|---|
| `rungs` | list of `"<step>:<hf_model_dir>"`, in ladder order |
| `max_repetition` | a rung is degenerate above this mean looped fraction (default `0.15`) |
| `new_tokens` | greedy tokens per prompt (default `256`) |
| `gate_final_rung` | `true` fails the target when the LAST rung is degenerate; `false` only reports |
| `output_dir` | where `repetition.json` lands; relative paths go under `GB_BUILD_WORKDIR` |
| `python` | interpreter in the image (`/stage/.venv/bin/python`) |

Only the final rung ever gates: it is the one a downstream consumer would pick up. An
earlier rung above threshold is a finding to read in the table, not a reason to fail a
build whose later checkpoints may be fine.

Needs one GPU, which the build supplies. No trainer source, so it does not carry the
distill-* steps' `code_config` contract.
