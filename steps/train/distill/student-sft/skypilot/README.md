# sft — development notes

User-facing documentation is [USAGE.md](USAGE.md), which `make publish-step` copies to the
released step as its `README.md`.

## What is ours and what is upstream

| | Where | Change it where? |
|---|---|---|
| `step-template.yaml` | here | here |
| `src/run-sft.sh`, `src/render_sft_config.py` | here, near-verbatim ports (the renderer carries most of the 94 tests) | **upstream**, then re-sync |
| `test/test_step_template.py` | here | here |
| `test/test_render_sft_config.py` | here, near-verbatim port | **upstream**, then re-sync |
| `sft.py`, `render_common`, `tracking.py` | `steps/train/distill/src` (see `VENDORED.md`), baked into the image | there, then rebuild the image |

## The newline compensation lives here, not in the ported files

`response_template`'s trailing newline cannot cross gbserver's config fill as a real character
(see USAGE.md). That is a property of **granite.build's transport**, so the decode is in this
step's `run:` block rather than in `render_sft_config.py` — which stays verbatim. `distill-gold`
solved the same problem in its own renderer because that renderer is ours to change; this one is
not.

## Three fixes the ported test suite needed

All three are path resolution, none is behaviour:

- upstream's `sys.path` inserts are removed; `conftest.py` resolves the step's own `src/` and the
  in-repo shared package at `steps/train/distill/src`.
- the **parity tests** import `distill-gold-train`'s renderer, which granite.build does not
  contain. They are pointed at the upstream renderer in a `gb-steps-distillation` checkout
  (`GB_DISTILL_CODE_DIR`) rather than dropped, and skip without one — see [Tests](#tests) for
  what they then assert.
- three tests read `sft.py`'s **source text** (a property that cannot be imported, since it pulls
  torch at module scope). Repointed at the same checkout for the same reason.

## Source delivery

The package and this step's `src/` are baked into the shared GPU image under `$DISTILL_HOME`
(`/opt/distill`); nothing is cloned or mounted at launch.

## Tests

```bash
make test                                                   # contract tests (+ the in-repo package)
GB_DISTILL_CODE_DIR=/path/to/checkout make test              # + the ported upstream suite (test_render_sft_config.py)
```

`test_render_sft_config.py` is the exception among the distill steps: besides the in-repo
package it needs `distill-gold-train/src` from an upstream `gb-steps-distillation` checkout
(`GB_DISTILL_CODE_DIR`, defaulting to the `/proj` copy), is marked `ibm`, and skips when that
directory is absent.

The ported suite includes upstream's parity tests, which compare this renderer against
`distill-gold-train`'s. That renderer is not in granite.build — `steps/train/distill/logit-distill` ships a
different one, because it drives a different trainer — so they are pointed at the **upstream**
renderer in that checkout. What they then assert is still worth knowing: upstream's
control and its treatment share their guards, their optimization defaults and their tracking
surface.

## Publishing

```bash
make space && make test && make publish-step && make check-published
```
