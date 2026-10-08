# logit-precompute — development notes

User-facing documentation is [USAGE.md](USAGE.md), which `make publish-step` copies to the
released step as its `README.md`.

## What is ours and what is upstream

| | Where | Change it where? |
|---|---|---|
| `step-template.yaml` | here — but see the source-delivery note | here |
| `src/run-precompute.sh` | here, a near-verbatim port | **upstream**, then re-sync |
| `test/test_step_template.py` | here | here |
| `test/test_precompute_logits.py` | here, a near-verbatim port | **upstream**, then re-sync |
| `precompute_logits`, `step_state` | the delivered checkout | **upstream only** |

## Written from lessons, not from another six cluster runs

`run-precompute.sh` has the same shape as `run-sft.sh`, so this port applied all four launcher
facts that step paid an allocation each to learn, before its first run:

- **`PRECOMPUTE_SRC` and `LIB_DIR` are exported, not `CHECKOUT_ROOT`.** The script guards those
  two against the environment but computes `CHECKOUT_ROOT` **unconditionally** as three levels
  above `STEP_HOME`, which assumes the step directory sits inside the checkout.
- **`/stage/.venv/bin` on PATH**, because it calls a bare `accelerate`; `ACCELERATE` is also
  named explicitly.
- **`GOLD_PREFETCH_KERNELS=1`**, because the image lacks `kernels-community/flash-attn2` and
  those resolve at import time.
- **No `HF_HUB_OFFLINE`**, because it loads the teacher.

Each is pinned by a test, including one asserting the script *still* overwrites `CHECKOUT_ROOT` —
so if upstream ever guards it, the test says to revisit rather than silently keeping a workaround.

One more, from the `distill-eval` port: the artifact marker is printed by the **script only**.
Upstream's template echoed it a second time, which would register two `NEWARTIFACT` events for one
id.

## The source-delivery region is spliced, not written

`code_config` and the delimited region at the top of `run:` are **byte-identical copies** from
`steps/distill/tokenizer-align`, the reference step. Change the reference and re-splice, or that
step's `test_source_contract.py` fails and names this one.

## Publishing

```bash
make space && make test && make publish-step && make check-published
```

No `test-data/lsf/` fixture and no build test: this step is wired into no recipe, so there is
nothing for a fixture to chain to. It was verified by a one-off build instead (`ed6894ee`,
recorded in USAGE.md), which reused artifacts earlier runs left on /proj rather than re-running
align and prep.

## Tests

```bash
make test                                                   # 32 contract tests, no checkout needed
GB_DISTILL_CODE_DIR=/path/to/checkout make test              # + 51 ported upstream tests
```
