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
| `shared/output_dir.j2` | The run-script block every family template shares (HF cache and warm-cache env, the options as words, the composed output dir), pulled into each `step-template.yaml` by `gb-include` (see [steps/README.md](../README.md#sharing-jinja-between-templates-gb-include)). |
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
`$OUTPUT_ROOT/$RUN_NAME/<benchmark>` (a generic bridge suffixes what it selected,
see [The generic bridges](#the-generic-bridges)). An empty `RUN_NAME` derives it from
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
`make publish-step` from the source template — never hand-edited. The run-script
block the eleven templates share lives once, in `shared/output_dir.j2`; after
editing it, re-run `make publish-step` in all eleven step dirs.

## The generic bridges

Every family's benchmarks are curated: one registered id per benchmark, a fixed
protocol the module freezes (sampling, stop strings, aggregation). That is what
makes a score comparable across models — and what costs a module per benchmark,
which is why a family offers a fraction of what its harness can run.

A generic bridge is the escape hatch: one more registered id in a family that
runs *anything* of a kind the family's pinned harness knows, with the recipe's
`options` (or `dataset`) carrying the harness's own selection surface. A new
benchmark of that kind becomes a recipe's config, not a module; the curated ids
stay exactly as they are (a recipe that wants a comparable protocol pins the
curated benchmark). Six families have one:

| Family | Bridge id | Selection (required) | What it reaches | Example recipe |
|---|---|---|---|---|
| `lmeval` | `lm-eval` | `tasks=` (comma-separated task/group names); also `mode=`, `num_fewshot=`, `gen_kwargs=`, `metric=` | any task or group in the pinned lm-evaluation-harness (14k+ names); the task yamls pin no dataset revisions | `lmeval/hellaswag`, `lmeval/gsm8k`, `lmeval/gpqa-diamond` |
| `nemoskills` | `nemoskills` | `benchmark=` (a NeMo-Skills dataset name); also `metric=`, `split=`, and the required `dataset=` pin: a local path in ns's row format, a hub id at a commit, or a URL at a sha256 | any benchmark the pinned NeMo-Skills checkout ships, with ns's own prepare, prompts and metrics; the data is the recipe's pin (no unpinned mode), and a prepare that fetches by other means than HF, `urlretrieve` or `wget` needs a local `dataset=` | `nemoskills/ns-math-500` |
| `tau` | `tau3` | `domains=` (comma-separated) | any combination of the pinned τ²-bench domains, headline the mean of pass^1 over them | `tau/tau3-airline-retail` |
| `mcpmark` | `mcpmark` | `service=`; also `task_suite=` | any service the pinned MCPMark checkout ships that the image can run (today the curated three: filesystem, postgres, playwright) | `mcpmark/mcpmark-filesystem-easy`, `mcpmark/mcpmark-postgres-easy` |
| `swebench` | `swebench` | `dataset=` (+ `dataset_revision=`); `instances=` subsets | any SWE-bench-format set carrying each instance's image, eval script and log parser (the Verified/Multilingual shape); not SWE-bench Pro | `swebench/swebench-multilingual-druid` |
| `tbench` | `terminal-bench` | `dataset=` (+ `dataset_revision=`); `tasks=` / `exclude=` subset | any Terminal-Bench-format dataset, registry or registry-less layout | `tbench/terminal-bench-2.0` |

The shape, so other families can grow the same: the bridge is a thin subclass
of the family's parameterized base, reusing its machinery (serving, adapters,
phases, resume, sandboxes, judges) and adding only the selection surface; what a
curated benchmark freezes is the harness's own default here, overridable per
recipe. The registry contracts hold unchanged: each bridge's class-level pin is
the harness itself (the repo at the pyproject extra's release, the ns checkout,
or the harness wheel), the recipe pins the data where the harness does not, and
every bridge is listed in the extended suite. The worked example is
`lmeval/src/generic.py` and the lmeval `USAGE.md`; each family's `USAGE.md`
documents its own bridge row.

The other five families have no bridge id; their selection surface already
exists on the curated ids: `bfcl`'s `categories=` (`bfcl/bfcl-multiturn`,
`bfcl-agentic`, `bfcl-live`), and the `dataset` / `dataset_revision` swap on
`bird`, `ifbench` and `judged/gdpval` (the recipes' `DATASET` / `DATASET_REVISION`
parameters).

**Run dirs.** A bridge runs many benchmarks through one id, so a run that
selects gets a dir named after every selection value:
`$OUTPUT_ROOT/$RUN_NAME/<id>@<key>=<value>[@<key>=<value>...]`. The keys, in this
fixed order: the dataset-pinned bridges' (`swebench`, `terminal-bench`,
`nemoskills`) `dataset` and `dataset_revision`, then the options `benchmark=`,
`service=`, `task_suite=`, `domains=`, `categories=`, `tasks=`, `exclude=`,
`instances=`, `split=` (a repeated option counts once, its last value, as the
harness reads it). So `lm-eval@tasks=hellaswag`, `tau3@domains=airline,retail`,
`mcpmark@service=filesystem@task_suite=easy`. No curated id contains `@`, so a
bridge run never shares a curated recipe's dir (`tau3@domains=airline` is not
`tau3-airline`). Each value is escaped into one path segment: `[A-Za-z0-9._,-]`
stay, every other byte becomes `~XX` (hex: a `/` is `~2F`, a `*` is `~2A`), so
distinct selections never map to one dir; a name over 200 bytes is cut to 180
plus `~h` and 16 hex of its sha256. The options are read without glob expansion.
A benchmark run without a selection (every curated recipe, a dataset swap on a
curated id included) keeps `$OUTPUT_ROOT/$RUN_NAME/<id>`.

## The warm HF cache

Every eval job's datasets download from the Hub into `hf_home` at run start —
hermetic, node-local by default, no token. `hf_offline` changes that to a
shared-cache model: the job runs with `HF_HUB_OFFLINE=1` (and
`HF_DATASETS_OFFLINE=1`), so every hub read resolves from `hf_home`'s cache and
never touches the Hub. Warm the cache once and point every run at it — one
download serves every later job, and someone else may have already warmed the
dataset you need:

```shell
HF_TOKEN=<token> make -C steps/eval prefetch HF_HOME=<shared-cache>
HF_TOKEN=<token> python steps/eval/scripts/prefetch_datasets.py --hf-home <shared-cache> --benchmark gpqa
```

Benchmarks read the Hub two ways, and offline each needs its own part of the
cache. `snapshot_download`/`hf_hub_download` readers (officeqa, mcpatlas,
fortress, gdpval's task files, aa-lcr's text zip) are served by the hub snapshot
(`$HF_HOME/hub`). `datasets.load_dataset` readers — through
`granite_evals.data.load_split`, ns's `prepare.py`, or lm-eval's task yamls —
are not: offline, datasets 5 loads a hub id only from its prepared cache
(`$HF_HOME/datasets`), which a snapshot does not build. So the prefetch fetches
every pinned snapshot and, for the benchmarks that read through `load_dataset`,
also prepares each config they request at the pinned revision
(`scripts/prefetch_datasets.py`'s `LOAD_CONFIGS`; `--list` shows which). Run it
with the lock's `datasets` (5.0.1), the version the jobs read the cache back
with. What runs offline after a prefetch:

- **Every curated benchmark with a prefetch entry** (`--list`), at its default
  selection. A non-default selection option (`languages=`, ...) reads other
  configs: warm them with `--benchmark <id> --config <name>`.
- **Not terminal-bench** (terminal-bench-2.1, and the bridge's 2.0 recipe): it
  downloads its snapshot into a run-local `local_dir`, and huggingface_hub
  offline never reads the cache for a `local_dir` download, so it fails fast
  under `hf_offline`. Run it online.
- **lm-eval bridge tasks** read their datasets with `load_dataset` and no
  revision; offline that resolves to whatever prepared copy of the dataset and
  config the cache holds. The prefetch's `gpqa` entry prepares `gpqa_diamond`,
  which serves `recipes/evals/lmeval/gpqa-diamond`
  (`gpqa_diamond_zeroshot`) at that entry's pin. Any other task has no entry:
  run it once online with the same `hf_home` (that prepares what it reads),
  then offline.
- **Benchmarks without a prefetch entry** (data baked into the image or fetched
  from GitHub/PyPI by pinned URL) need no hub cache; `hf_offline` blocks Hub
  reads only, so their pinned non-Hub downloads (and, e.g., tiktoken's
  encoding for aa-lcr) still need the network.

Gated datasets (gpqa, hle, officeqa, ...) need a token with access granted for
the warming fetch only — the jobs themselves carry no token. A cold cache fails
fast (`hf_offline` without `hf_home` is refused), the cache must be writable by
the job (datasets takes cache locks), and the model must not need the Hub
either: bind it as an `hf://` input (hfpull stages it with the network) or warm
it into the same cache. Offline also pins the run's data to whatever the cache
holds — the warm cache is the revision pin. For a run that must fetch
a gated dataset online instead, pass the token as a space secret
(`skypilot.secrets` in the recipe, as the officeqa recipes do) — never in the
recipe text. The `recipes/evals/lmeval/gpqa-diamond` and
`recipes/evals/officeqa/officeqa-pro-warmcache` recipes are the worked examples:
they default `HF_HOME` to `$OUTPUT_ROOT/hf-cache` (a recipe-level default, not
the step's — the step refuses `hf_offline` without `hf_home`), stage the model
with hfpull, and their runbooks warm that cache with
`python scripts/prefetch_datasets.py --hf-home <cache> --benchmark <id>`.
