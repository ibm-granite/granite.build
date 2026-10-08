# steps/eval/

A pure grouping directory (no behavior of its own) containing one step per benchmark
in the [granite-evals](https://github.com/laminair/granite-evals) suite.

## Dependency direction

**granite.build depends on granite-evals. Not the other way around.**

```
granite.build  ──────depends on──────>  github.com/laminair/granite-evals
(this repo)                              (separate repo, separate releases)
```

- The benchmark runtime — datasets, harnesses, scoring, judges, sandboxes — lives
  entirely in `granite-evals`, which ships it as one prebuilt image per benchmark
  family (`make publish-image EXTRA=<family>` in that repo).
- **None of that code is vendored into this repo.** Every step here is a thin
  SkyPilot launcher: it picks the pinned image (`granite_config.image`), serves the
  checkpoint with vLLM, and shells out to `granite-evals run <benchmark>` inside the
  job. There is no Dockerfile and nothing is built from these directories.
- Bumping a benchmark's behavior means cutting a new `granite-evals` image and
  re-pinning its tag in the recipe `parameters.yaml` (e.g. `GRANITE_EVALS_IMAGE_BFCL`)
  — not editing anything under `steps/eval/`.
- granite.build only pulls in benchmark *code* this way (via image) for now; it does
  not import or build against the `granite-evals` Python package directly.

## Layout

Each `steps/eval/<benchmark>/skypilot/` is one step: `step-template.yaml` (the
source of truth), `README.md` / `USAGE.md`, and `test/test_step_template.py` (a
shared contract test — every sibling has the identical copy, which is what lets one
test file assert all 41 steps follow the same shape). The rendered, committed form
that `space://steps/eval/<benchmark>` actually resolves to lives under
`configurations/assets/environments/skypilot/steps/eval/<benchmark>/`, produced by
`make publish-step` from the source template — never hand-edited.

Recipes under `recipes/granite-evals/` choose which benchmarks a model family is
scored on and own the per-benchmark image pins.
