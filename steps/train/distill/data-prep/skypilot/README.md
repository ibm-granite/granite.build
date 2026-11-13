# corpus-prep — development notes

User-facing documentation is [USAGE.md](USAGE.md), which `make publish-step` copies to the
released step as its `README.md`.

## What is ours and what is upstream

| | Where | Change it where? |
|---|---|---|
| `step-template.yaml` | here | here |
| `src/prep_corpus.py`, `src/merge_shards.py` | here, near-verbatim ports (they carry the 88 tests) | **upstream**, then re-sync |
| `test/test_step_template.py` | here | here |
| `test/test_prep_corpus.py` | here, near-verbatim port | **upstream**, then re-sync |
| `gb_steps_post_training.distillation` | `steps/train/distill/src` (see `VENDORED.md`), baked into the image | there, then rebuild the image |

`merge_shards.py` imports `prep_corpus` by flat name, so the two ship together; a test asserts it.

## Source delivery

The package and this step's `src/` are baked into the shared CPU image under `$DISTILL_HOME`
(`/opt/distill`); nothing is cloned or mounted at launch.

## The fixture is two targets, and that is forced

`test-data/lsf/build.yaml` runs align -> corpus rather than testing this step alone, because no
pre-existing tokenizer on BlueVela has a `{% generation %}` block tag (see USAGE.md). A
single-target fixture pointed at any `/proj` overlay fails with an empty assistant mask — the
step refusing correctly. So the align target is the only source of a usable tokenizer, and the
fixture proves the cross-target binding as a side effect.

## Tests

```bash
make test                                                   # contract tests + 88 ported upstream tests
```

`test/conftest.py` puts the in-repo `steps/train/distill/src` on `sys.path`, so no external
checkout is needed.

## Publishing

```bash
make space && make test && make publish-step && make check-published
```

Publishes to `configurations/assets/environments/skypilot/steps/train/distill/data-prep/` — not
under `lsf/ibm-bluevela/`; the step gates itself with `subtypes: [lsf]`.
