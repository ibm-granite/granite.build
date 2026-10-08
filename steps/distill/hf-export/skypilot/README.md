# hf-export — development notes

User-facing documentation is [USAGE.md](USAGE.md), which `make publish-step` copies to the
released step as its `README.md`.

## What is ours and what is upstream

| | Where | Change it where? |
|---|---|---|
| `step-template.yaml` | here — but see the source-delivery note | here |
| `src/export_hf_model.py` | here, a near-verbatim port (the ported suite tests it) | **upstream**, then re-sync |
| `test/test_step_template.py` | here | here |
| `test/test_export_hf_model.py` | here, a near-verbatim port | **upstream**, then re-sync |
| `test-data/chatml_granite_42_generation.jinja` | vendored from upstream's `templates/` | upstream |

## Three fixes the port had to make

- **Upstream's SkyPilot launcher used `image:`, not `image_id:`.** That is not a key the
  provisioner reads, so the task would have had no image at all. A test asserts `image_id` is
  present and `image` is absent.
- **Upstream's test suite errored on collection when run alone.** It inserted only
  `parents[1]/src` on `sys.path` while `export_hf_model.py` imports the shared package at
  module scope, so it passed only when a sibling suite had already patched `sys.path`. The
  shared `conftest.py` resolves both paths.
- **`REAL_TEMPLATE` reached for `parents[3]/templates/`**, which does not exist from inside
  granite.build. The template is vendored into `test-data/` instead.

## The source-delivery region is spliced, not written

`code_config` and the delimited region at the top of `run:` are **byte-identical copies** from
`steps/distill/tokenizer-align`, the reference step. Do not edit them here — change the
reference and re-splice, or that step's `test_source_contract.py` fails and names this one.

## Tests

```bash
make test                                                   # 27 contract tests, no checkout needed
GB_DISTILL_CODE_DIR=/path/to/checkout make test              # + the ported upstream suite
GB_STEP_BLUEVELA_BUILD=1 make test                           # + the end-to-end run on BlueVela
```

The ported suite is `test/test_export_hf_model.py` (68 test functions as of this writing). Its
tests assert against the **real** chat template, vendored at
`test-data/chatml_granite_42_generation.jinja` — a stub would assert nothing about the property
the thinking-policy rewrite has to preserve.

The end-to-end test is `test/lsf/test_skypilot_lsf_distill_hf_export.py`, with fixtures in
`test-data/lsf/`. Without `GB_STEP_BLUEVELA_BUILD=1`, or when the BlueVela login node is
unreachable, it skips.

## Publishing

```bash
make space && make test && make publish-step && make check-published
```
