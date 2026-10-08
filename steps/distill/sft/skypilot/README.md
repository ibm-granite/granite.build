# sft — development notes

User-facing documentation is [USAGE.md](USAGE.md), which `make publish-step` copies to the
released step as its `README.md`.

## What is ours and what is upstream

| | Where | Change it where? |
|---|---|---|
| `step-template.yaml` | here — but see the source-delivery note | here |
| `src/run-sft.sh`, `src/render_sft_config.py` | here, near-verbatim ports (the renderer carries most of the 94 tests) | **upstream**, then re-sync |
| `test/test_step_template.py` | here | here |
| `test/test_render_sft_config.py` | here, near-verbatim port | **upstream**, then re-sync |
| `sft.py`, `render_common`, `tracking.py` | the delivered checkout | **upstream only** |

## The newline compensation lives here, not in the ported files

`response_template`'s trailing newline cannot cross gbserver's config fill as a real character
(see USAGE.md). That is a property of **granite.build's transport**, so the decode is in this
step's `run:` block rather than in `render_sft_config.py` — which stays verbatim. `distill-gold`
solved the same problem in its own renderer because that renderer is ours to change; this one is
not.

## Three fixes the ported test suite needed

All three are path resolution, none is behaviour:

- upstream's `sys.path` inserts are removed; `conftest.py` resolves the step's own `src/` and the
  shared package from `GB_DISTILL_CODE_DIR`.
- the **parity tests** import `distill-gold-train`'s renderer, which granite.build does not
  contain. They are pointed at the upstream renderer in the delivered checkout rather than
  dropped — see [Tests](#tests) for what they then assert.
- three tests read `sft.py`'s **source text** (a property that cannot be imported, since it pulls
  torch at module scope). Repointed at the checkout for the same reason.

## The source-delivery region is spliced, not written

`code_config` and the delimited region at the top of `run:` are **byte-identical copies** from
`steps/distill/tokenizer-align`, the reference step. Do not edit them here — change the reference
and re-splice, or that step's `test_source_contract.py` fails and names this one.

## Tests

```bash
make test                                                   # 37 contract tests, no checkout needed
GB_DISTILL_CODE_DIR=/path/to/checkout make test              # + 94 ported upstream tests
```

The ported suite includes upstream's parity tests, which compare this renderer against
`distill-gold-train`'s. That renderer is not in granite.build — `steps/distill/gold` ships a
different one, because it drives kd-sandbox's trainer — so they are pointed at the **upstream**
renderer in the delivered checkout. What they then assert is still worth knowing: upstream's
control and its treatment share their guards, their optimization defaults and their tracking
surface.

## Publishing

```bash
make space && make test && make publish-step && make check-published
```
