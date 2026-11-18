# steps/eval/

Granite benchmark steps: one step per granite-evals image family, the vendored
benchmark runtime those images are built from, and the image recipes themselves.

## What lives here

| Path | What it is |
|---|---|
| `<family>/skypilot/` | One step per benchmark family: `step-template.yaml` (the source of truth), `README.md` / `USAGE.md`, and the shared contract test. |
| `<family>/Dockerfile.gpu`, `<family>/Dockerfile.cpu` | The family's image recipe(s). Every family has a GPU image; the families whose benchmarks split into GPU generate + CPU score (`--phase`) also have a CPU image. Both are tagged `<registry>/steps/eval/<family>/images/{gpu,cpu}:<version>`, the form the recipes pin. Families needing extra system setup carry an `extras.sh` (and `nemoskills` a pinned `comet-requirements.txt` for its COMET judge). |
| `src/core/` | The shared granite-evals runtime: harness adapters, vLLM serving, scoring, results. |
| `<family>/src/`, `<family>/tests/` | The family's benchmark modules and their unit tests. `src/` is overlaid onto the core's `granite_evals/benchmarks/` package dir at image-build time (what each family contains: [VENDORED.md](VENDORED.md)). |
| `scripts/prefetch_datasets.py` | Pins and pre-fetches the HF datasets into a cache the (offline) jobs read. |
| `eval-example/` | The framework's generic eval exemplar step — pre-existing, not one of the granite-evals families. |

The eleven families — `bfcl`, `bird`, `ifbench`, `judged`, `lmeval`, `mcpmark`,
`nemoskills`, `officeqa`, `swebench`, `tau`, `tbench` — exist because
the upstream harnesses' own dependencies conflict (see the `[tool.uv]` conflicts
in `src/core/pyproject.toml`), so each ships as its own image. `benchmark`
selects which benchmark in the family a target runs.

## Images

> **The image tags pinned in the recipes live on `quay.io/granitebuild`** —
> a public registry, so anyone can pull them anonymously; each recipe's
> `IMAGE` / `IMAGE_CPU` parameters name them. To run your own builds instead,
> build the family images (below) and repoint those parameters.

`make images REGISTRY=...` from this directory builds every family image from
`<family>/Dockerfile.{gpu,cpu}` (build context: this directory; see the
[Makefile](Makefile)). The steps themselves build nothing at run time: each picks
its pinned image (`config.image`, set by the recipe — e.g. the `IMAGE` /
`IMAGE_CPU` parameters), serves the checkpoint with vLLM, and shells out to
`granite-evals run <benchmark>` inside the job.

Bumping a benchmark's behavior means editing `src/` or the family's
`<family>/src/`, rebuilding the family image,
pushing it (`make publish-image-<family> REGISTRY=quay.io/granitebuild`, and
`publish-image-cpu-<family>` for its CPU image; `IMAGE_VERSION` defaults to the short
SHA of the commit you build from, each publish first runs the vulnerability gate, and
publishing refuses a tag that already exists), and re-pinning its tag in the recipe
`parameters.yaml` — recipes under
`recipes/evals/<family>/<benchmark>/skypilot/` choose which benchmarks a model
family is scored on and own the image pins.

`OUTPUT_ROOT` / `RUN_NAME` shape every artifact path a recipe writes:
`$OUTPUT_ROOT/$RUN_NAME/<benchmark>`. An empty `RUN_NAME` derives it from
`MODEL_PATH`'s basename — re-running the same model then resolves to the same
artifact URIs, and a reused SUCCESS target's downstream score target never
dispatches; make it unique per run (append a date or the build id) to force a
fresh one. `OUTPUT_ROOT` is required with no default: a dispatch without it, or
without an explicit step `output_dir`, fails fast rather than writing somewhere
unexpected.

## Layout of a family step

Each `steps/eval/<family>/skypilot/` is one step: `step-template.yaml` (the source
of truth), `README.md` / `USAGE.md`, and `test/test_step_template.py` (a shared
contract test — every sibling has the identical copy, which is what lets one test
file assert all eleven steps follow the same shape). The rendered, committed form
that `space://steps/eval/<family>` actually resolves to lives under
`configurations/assets/environments/skypilot/steps/eval/<family>/`, produced by
`make publish-step` from the source template — never hand-edited.

## The generic bridge

Every family's benchmarks are curated: one registered id per benchmark, a fixed
protocol the module freezes (sampling, stop strings, aggregation). That is what
makes a score comparable across models — and what costs a module per benchmark,
which is why a family offers a fraction of what its harness can run.

The generic bridge is the escape hatch: one more registered id per family
(`lm-eval` in the lmeval family) that runs *anything* the family's pinned
harness knows, with the recipe's `options` carrying the harness's own selection
surface. For lm-eval that is `tasks=` (any of the 14k+ task and group names the
pinned harness ships), plus `mode=`, `num_fewshot=`, `gen_kwargs=` and `metric=`
— the bridge passes them to the harness instead of freezing them. A new task
becomes a recipe's config, not a module; the curated ids stay exactly as they
are (a recipe that wants a comparable protocol pins the curated benchmark).

The shape, so other families can grow the same: the bridge reuses the family's
own machinery (serving, adapters, resume caches, artifacts) and adds only the
selection surface; the headline is the macro mean over the resolved tasks of
each task's primary metric; what a curated benchmark freezes is the task's own
default here, overridable per recipe. The registry contracts hold unchanged:
the bridge's pin is the harness repo at the release the family's pyproject extra
pins, and the bridge is listed in the extended suite. See
`steps/eval/lmeval/src/generic.py` and the lmeval `USAGE.md` for the worked
example.

## The warm HF cache

Every eval job's datasets download from the Hub into `hf_home` at run start —
hermetic, node-local by default, no token. `hf_offline` changes that to a
shared-cache model: the job runs with `HF_HUB_OFFLINE=1` (and
`HF_DATASETS_OFFLINE=1`), so every hub read resolves from `hf_home`'s cache and
never touches the Hub. Warm the cache once and point every run at it — one
download serves every later job, and someone else may have already warmed the
dataset you need:

```shell
HF_TOKEN=<token> make -C steps/eval prefetch HF_HOME=<shared-cache>   # or --benchmark gpqa
```

Gated datasets (gpqa, hle, officeqa, ...) need a token with access granted for
the warming fetch only — the jobs themselves carry no token. A cold cache fails
fast (`hf_offline` without `hf_home` is refused), the cache must be writable by
the job (datasets takes cache locks), and the model must not need the Hub
either: bind it as an `hf://` input (hfpull stages it with the network) or warm
it into the same cache. Offline also pins the run's data to whatever snapshot
the cache holds — the warm cache is the revision pin. For a run that must fetch
a gated dataset online instead, pass the token as a space secret
(`skypilot.secrets` in the recipe, as the officeqa recipes do) — never in the
recipe text. The `recipes/evals/lmeval/gpqa-diamond` recipe is the worked
example.
