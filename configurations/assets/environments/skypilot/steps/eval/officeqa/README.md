# officeqa (SkyPilot)

Scores a checkpoint on **OfficeQA** (Databricks), quantitative questions about U.S.
Treasury Bulletins (Granite metric: **accuracy@0%**, the reward at tolerance 0; 0.1% /
1% / 5% land in `details.accuracy`). The job serves the model with vLLM and follows
upstream's `oracle_parsed` protocol, the benchmark's headline mode:

- each question's gold pages — parsed from its `source_docs` URLs zipped with
  `source_files` — are extracted from the parsed JSON corpus exactly as upstream's
  index builder does (elements grouped by `bbox[0].page_id`, tables rendered to
  Markdown by `pandas.read_html`), and handed to the model as context;
- the model answers in `<FINAL_ANSWER>` tags with upstream's system prompt;
- the answer is scored with the ported upstream reward (fuzzy numeric/text match)
  at tolerances 0 / 0.1% / 1% / 5%.

The scorer and page extraction are Apache-2.0 (ported into this repo with
attribution). The question CSVs and the parsed corpus are **CC-BY-SA-4.0 behind the
Hub's click-through gate**: nothing is vendored, baked into the image, mirrored or
pushed. The adapter fetches `databricks/OfficeQA` at the pinned revision at run
start — before any question is processed — and verifies it against pinned git-blob
sha1s on the CSVs, the question counts and a sha256 over the corpus tree; a fetch
that does not match refuses to run. `HF_TOKEN` must be set to a token of an account
that has accepted the dataset's terms (the gate asks that the answer keys not be
used to train models evaluated on OfficeQA). Only the CSVs plus the ~600 MiB corpus
are downloaded — the ~4 GB of PDFs are excluded.

The code is the vendored granite-evals runtime (`steps/eval/src`), baked into this
family's image by `Dockerfile.gpu` / `Dockerfile.cpu` one directory up (built by
`steps/eval/Makefile`). Nothing is built from this step at run time.

```yaml
steps:
  - step_uri: space://steps/eval/officeqa
    config:
      benchmark: officeqa-pro
```

## Benchmarks in this family

| `benchmark` | Size | Notes |
|---|---|---|
| `officeqa-pro` | 133 questions | the Pro set. Metric: **accuracy@0%**. |
| `officeqa-full` | 246 questions | the Full set. Metric: **accuracy@0%**. |

See `granite-evals list` for the current, authoritative set and sizes.

## Config

| Field | Default | Purpose |
|---|---|---|
| `benchmark` | `""` (required) | The benchmark id passed to `granite-evals run`; `officeqa-pro` or `officeqa-full`. |
| `image` | `""` (required) | granite-evals officeqa image, pinned by tag (`GRANITE_EVALS_IMAGE_OFFICEQA`). |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all) | **Smoke knob:** first N questions by uid. |
| `repeats` | `""` (1) | Independent samples per question (avg-of-k). |
| `workers` | `16` | Concurrent questions (requests to vLLM + context builds). |
| `dataset` / `dataset_revision` | `""` | A LOCAL PATH to an existing dataset checkout overrides the pinned fetch (trusted, not digest-checked — the pins are then recorded as absent in results.json). |
| `options` | `""` | Space-separated `key=value` options; see below. |
| `phase` | `"all"` | `all` generates and scores in one job. `generate` serves the model and writes `generation.json` (`granite_generation`); `score` grades that output dir without a GPU (same `output_dir`, `limit`, `repeats`) and writes `results.json`. The score phase fetches only the two CSVs, not the corpus. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `""` | Unused: this benchmark runs no sandbox. |
| `hf_home` | `""` | Overrides `HF_HOME`, and with it the gated dataset cache (prefetch it with `steps/eval/scripts/prefetch_datasets.py` to avoid re-downloading the corpus per run). |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. `HF_TOKEN` (a token that has accepted the dataset's gate) is the one
secret; recipes attach it to both targets.

### Options (`options`)

| Option | Default | Purpose |
|---|---|---|
| `mode` | `oracle_parsed` | The only implemented mode; anything else is refused. |
| `answers` | `model` | `gold` scores each question's gold answer without a model (see below). |
| `questions` | (all) | Regex on the question uid, e.g. `1941`. |
| `temperature` / `max_tokens` | `0.0` / `4096` | Sampling; upstream's defaults. |
| `max_context_chars` | `0` (off) | Truncate an over-long context; the record is flagged. Upstream has none. |
| `timeout` | `120` | Per-request timeout to the served model, seconds. |
| `max_failed_frac` | `0.05` | Infrastructure-failed questions above this fraction fail the run. |

## Gold mode

`options: "answers=gold"` (with `model_path: none`) scores each question's gold
answer through the ported scorer, with the context still built — a model-free check
of the fetch, the pins, the page extraction and the scorer. Every well-formed gold
answer scores 1.0; one that does not is a finding about the scorer or the CSV, not a
baseline. Upstream has no equivalent.

## Output

`granite_results` (dataset): the `results.json` file (phase `all` or `score`).
`granite_generation` (dataset): the `generation.json` file (phase `generate`). A split run
is two targets on the same `output_dir`: the score target binds the generate target's
output, so it runs after it:

```yaml
officeqa-pro-generate:        # GPU
  outputs:
    granite_generation:
      uri: "env://{{ binding.path }}"
officeqa-pro:                 # CPU only, phase: score
  inputs:
    generation:
      binding: officeqa-pro-generate.granite_generation
  outputs:
    granite_results:
      uri: "env://{{ binding.path }}"
```

Per-question generations, extracted answers and per-tolerance scores are kept next
to it under `output/repeat-<k>/<uid>.json`. Finished questions are skipped on
restart; infrastructure failures are retried, never recorded as model losses.

