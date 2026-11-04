# hf-export — development notes

User-facing documentation is [USAGE.md](USAGE.md), which `make publish-step` copies to the
released step as its `README.md`.

A generic step: it packages one `checkpoint-N` of any HF Trainer / accelerate run, so it
lives under `steps/generic/` rather than with the training collection that first needed it.
It was `space://steps/distill/hf-export` until it moved here.

## Layout

| | What it is |
|---|---|
| `step-template.yaml` | the step; `image_id` is filled from the Makefile's image |
| `Dockerfile`, `requirements.in`, `requirements.lock` | the step's own CPU image (no torch) |
| `src/export_hf_model.py` | the workload |
| `src/hf_export_state/` | the resume gate (`.step-done.json`), copied from the distillation package so this step imports nothing from `steps/distill` |
| `test/test_step_template.py` | template contract tests |
| `test/test_export_hf_model.py` | the workload's unit tests |
| `test-data/chatml_granite_42_generation.jinja` | the real chat template the thinking-policy tests run against |

## Resume compatibility with the old URI

`hf_export_state/step_state.py` refuses a `.step-done.json` written under a different step
name. `export_state.STEP_NAME` therefore stays `distill-hf-export`: it names the record on
disk, not the step, and changing it would turn every export made before the move into a
refusal. `test_step_template.py::TestResumeCompatibility` pins both that name and the marker
schema shared with `steps/distill/src/.../step_state.py`.

## Three fixes made when the step was first ported

- **The original SkyPilot launcher used `image:`, not `image_id:`.** That is not a key the
  provisioner reads, so the task would have had no image at all. A test asserts `image_id` is
  present and `image` is absent.
- **The original test suite errored on collection when run alone.** It inserted only
  `src/` on `sys.path` while `export_hf_model.py` imported the distillation package at module
  scope. Now the gate ships in `src/`, and `conftest.py` adds `src/`.
- **`REAL_TEMPLATE` reached for a `templates/` directory** that does not exist here. The
  template is vendored into `test-data/` instead.

## Tests

```bash
make test                          # contract + unit tests, no cluster needed
GB_STEP_BLUEVELA_BUILD=1 make test # + the end-to-end run on BlueVela
```

The end-to-end test is `test/lsf/test_skypilot_lsf_hf_export.py`, with fixtures in
`test-data/lsf/`. Without `GB_STEP_BLUEVELA_BUILD=1`, or when the BlueVela login node is
unreachable, it skips.

## Image and publishing

```bash
make lock                                  # after editing requirements.in
make image publish-image IMAGE_VERSION=<v> # build and push gb-step-hf-export:<v>
make publish-step IMAGE_VERSION=<v>        # write configurations/.../steps/generic/hf-export
```

Bump `IMAGE_VERSION` in the Makefile with each image change; `publish-image` refuses to
overwrite an existing tag.
