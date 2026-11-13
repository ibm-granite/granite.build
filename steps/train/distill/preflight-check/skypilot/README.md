# probe

User-facing documentation is [USAGE.md](USAGE.md), which `make publish-step` copies to the
released step as its `README.md`.

Runs one of three cheap probes that gate the granite4-350m distillation family. None
trains or produces anything; each prints a `PROBE VERDICT` line to the log. What each
one asks, and why, is in [`src/distill_probe.py`](src/distill_probe.py).

| probe | question | needs |
|---|---|---|
| `load-student` | does the student load in this image, with FA2 and with eager attention? | 1 GPU |
| `tokenizer` | does the declared `tokenizer_class` change how text segments? | CPU |
| `tokenizer-fit` | which segmentation do the checkpoint's weights prefer, by NLL per byte? | 1 GPU |

## Config (`probe_config`)

| key | meaning |
|---|---|
| `probe` | `load-student`, `tokenizer` or `tokenizer-fit` |
| `checkpoint` | the HF checkpoint directory; only read, the tokenizer probes rewrite a scratch copy |
| `corpus` | `tokenizer-fit` only: a JSONL of `{"messages": [...]}` rows |
| `rows` | `tokenizer-fit` only: how many rows to score (default `200`) |
| `python` | interpreter in the image (`/stage/.venv/bin/python`) |

It sets no `HF_HUB_OFFLINE`: `load-student` asks whether transformers' lazy kernel
fetch works in this image. No trainer source; its script is baked into the shared GPU
image. No recipe runs it today: the `distill-probe` recipe that did was retired with the
other granite4-350m LSF recipes, and [`recipes/distill/`](../../../../../recipes/distill/README.md)
does not include a probe target. Run it from a build of your own.
