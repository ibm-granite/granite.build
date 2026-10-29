# steps/eval/

Granite benchmark steps: one step per granite-evals image family, the vendored
benchmark runtime those images are built from, and the image recipes themselves.

## What lives here

| Path | What it is |
|---|---|
| `<family>/skypilot/` | One step per benchmark family: `step-template.yaml` (the source of truth), `README.md` / `USAGE.md`, and the shared contract test. |
| `<family>/Dockerfile.gpu`, `<family>/Dockerfile.cpu` | The family's image recipe(s). Every family has a GPU image; the families whose benchmarks split into GPU generate + CPU score (`--phase`) also have a `-cpu` image. Families needing extra system setup carry an `extras.sh` (and `nemoskills` a pinned `comet-requirements.txt` for its COMET judge). |
| `src/core/` | The shared granite-evals runtime: harness adapters, vLLM serving, scoring, results. |
| `<family>/src/`, `<family>/tests/` | The family's benchmark modules and their unit tests. `src/` is overlaid onto the core's `granite_evals/benchmarks/` package dir at image-build time (what each family contains: [VENDORED.md](VENDORED.md)). |
| `scripts/prefetch_datasets.py` | Pins and pre-fetches the HF datasets into a cache the (offline) jobs read. |
| `eval-example/` | The framework's generic eval exemplar step — pre-existing, not one of the granite-evals families. |

The nine families — `bfcl`, `bird`, `ifbench`, `judged`, `lmeval`, `nemoskills`,
`swebench`, `tau`, `tbench` — exist because the upstream harnesses' own
dependencies conflict (see the `[tool.uv]` conflicts in `src/core/pyproject.toml`),
so each ships as its own image. `benchmark` selects which benchmark in the family
a target runs.

## Images

> **The image tags pinned in the recipes live on an IBM-internal registry**
> (`us.icr.io/cil15-shared-registry`) — a temporary measure until the step
> framework builds and publishes per-step images to a public registry (#565).
> Outside IBM, build the family images yourself (below) and point the recipes'
> `IMAGE` / `IMAGE_CPU` parameters at yours.

`make images REGISTRY=...` from this directory builds every family image from
`<family>/Dockerfile.{gpu,cpu}` (build context: this directory; see the
[Makefile](Makefile)). The steps themselves build nothing at run time: each picks
its pinned image (`config.image`, set by the recipe — e.g. the `IMAGE` /
`IMAGE_CPU` parameters), serves the checkpoint with vLLM, and shells out to
`granite-evals run <benchmark>` inside the job.

Bumping a benchmark's behavior means editing `src/` or the family's
`<family>/src/`, rebuilding the family image,
pushing it, and re-pinning its tag in the recipe `parameters.yaml` — recipes under
`recipes/evals/<family>/<benchmark>/skypilot/` choose which benchmarks a model
family is scored on and own the image pins.

`RUN_NAME` (with `OUTPUT_ROOT`) shapes every artifact path a recipe writes. Make it
unique per run — e.g. append a date or the build id: re-running the same model with
the same `RUN_NAME` resolves to the same artifact URIs, and a reused SUCCESS
target's downstream score target never dispatches.

## Layout of a family step

Each `steps/eval/<family>/skypilot/` is one step: `step-template.yaml` (the source
of truth), `README.md` / `USAGE.md`, and `test/test_step_template.py` (a shared
contract test — every sibling has the identical copy, which is what lets one test
file assert all nine steps follow the same shape). The rendered, committed form
that `space://steps/eval/<family>` actually resolves to lives under
`configurations/assets/environments/skypilot/steps/eval/<family>/`, produced by
`make publish-step` from the source template — never hand-edited.
