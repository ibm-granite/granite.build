# distill-corpus-prep — development notes

User-facing documentation is [USAGE.md](USAGE.md), which `make publish-step` copies to the
released step as its `README.md`.

## What is ours and what is upstream

| | Where | Change it where? |
|---|---|---|
| `step-template.yaml` | here | here — but see the source-delivery note below |
| `src/prep_corpus.py`, `src/merge_shards.py` | here, near-verbatim ports (they carry the 88 tests) | **upstream**, then re-sync |
| `test/test_step_template.py` | here | here |
| `test/test_prep_corpus.py` | here, near-verbatim port | **upstream**, then re-sync |
| `gb_steps_post_training.distillation` | the delivered checkout | **upstream only** |

`merge_shards.py` imports `prep_corpus` by flat name, so the two ship together; a test asserts it.

## The source-delivery region is spliced, not written

`code_config` and the delimited region at the top of `run:` are **byte-identical copies** from
`steps/distill/distill-tokenizer-align`, which is the reference step. Do not edit them here — change the
reference and re-splice, or
`steps/distill/distill-tokenizer-align/skypilot/test/test_source_contract.py` fails and names this step.

That is also why this step's `test_step_template.py` does not re-assert the twelve
source-delivery properties: identical to a correct reference is correct, and two homes for one
guarantee is how they drift.

## The fixture is two targets, and that is forced

`test-data/lsf/build.yaml` runs align -> corpus rather than testing this step alone, because no
pre-existing tokenizer on BlueVela has a `{% generation %}` block tag (see USAGE.md). A
single-target fixture pointed at any `/proj` overlay fails with an empty assistant mask — the
step refusing correctly. So the align target is the only source of a usable tokenizer, and the
fixture proves the cross-target binding as a side effect.

## Publishing

```bash
make space && make test && make publish-step && make check-published
```

Publishes to `configurations/assets/environments/skypilot/steps/distill/distill-corpus-prep/` — not
under `lsf/ibm-bluevela/`; the step gates itself with `subtypes: [lsf]`.
