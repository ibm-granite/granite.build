# student-sft — development notes

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
step's `run:` block rather than in `render_sft_config.py` — which stays verbatim. `logit-distill`
solved the same problem in its own renderer because that renderer is ours to change; this one is
not.

## What the ported test suite changed

- upstream's `sys.path` inserts are removed; `conftest.py` resolves the step's own `src/` and the
  in-repo shared package at `steps/train/distill/src`.
- the **parity tests** compare this renderer with the treatment in this repository:
  `logit-distill`'s `render_gold_config.py` and the vendored `sft.py`. See [Tests](#tests) for
  what they assert.

## Source delivery

The package and this step's `src/` are baked into the shared GPU image under `$DISTILL_HOME`
(`/opt/distill`); nothing is cloned or mounted at launch.

## Tests

```bash
make test
```

`test_render_sft_config.py` includes the ported parity tests, which read `logit-distill`'s
renderer and the vendored `sft.py` from this repository. The two steps are not a matched pair
today: they differ in several optimization defaults, in their resume model and in tracking (see
[USAGE.md](USAGE.md#using-this-step-as-a-control)). The parity tests check what they share (the
`ConfigError` type and the corpus guard) and pin the documented differences, so a new
difference fails the suite until the docs say so. Three tests read `sft.py`'s **source text**,
because it cannot be imported here (it pulls torch at module scope).

## Publishing

```bash
make space && make test && make publish-step && make check-published
```
