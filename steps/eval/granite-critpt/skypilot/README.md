# granite-critpt (SkyPilot)

Scores a checkpoint on **CritPt** (Granite metric: challenge accuracy).
The job serves the model with vLLM and runs, locally:
NeMo-Skills' CritPt (70 physics research challenges) with its two-turn generation (solve, then
fill the code template). Grading is Artificial Analysis's CritPt API, as NeMo-Skills' evaluator does
it: all 70 answers in one request, which returns only the aggregate accuracy.
It writes one `results.json`.

The code is the external [granite-evals](https://github.com/laminair/granite-evals) runtime,
shipped as a prebuilt image (the `nemoskills` family). Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/eval/granite-critpt
```

## Config (`granite_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | granite-evals nemoskills image, pinned by tag (`GRANITE_EVALS_IMAGE_NEMOSKILLS`). |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all 70 problems) | **Smoke knob:** first N items of the pinned data (phase generate only; see below). |
| `repeats` | `""` (1) | Independent generations per item; k > 1 reports pass@1[avg-of-k] and `details.pass_at_k`. |
| `workers` | `32` | Concurrent requests to vLLM. |
| `dataset` / `dataset_revision` | `""` | Override the data pinned in granite-evals (CritPt-Benchmark/CritPt, pinned in granite-evals). |
| `options` | `""` | Space-separated `key=value` benchmark options: `temperature`, `top_p`, `top_k`, `max_tokens`, `ns.<key>=<value>` (any NeMo-Skills generation override), `critpt_api_key_env` (default `ARTIFICIAL_ANALYSIS_API_KEY`). |
| `phase` | `"all"` | `all` generates and scores in one job. `generate` serves the model and writes `generation.json` (`granite_generation`); `score` grades that output dir without a GPU (same `output_dir`, `limit`, `repeats`) and writes `results.json` |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `""` | Unused: this benchmark runs no sandbox. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. Scoring needs an approved Artificial Analysis key as `ARTIFICIAL_ANALYSIS_API_KEY` (option `critpt_api_key_env`). The API allows 10 requests a day; each repeat's response is cached under its submissions' sha256 and never re-sent. The API is not an LLM bill and is not metered. The answers are hidden, so there is no `answers=gold`. The API grades exactly 70 submissions: `limit` works for phase `generate` only (the rest are sent as bare templates, scored wrong, and the accuracy is rescaled to the N submitted, `details.critpt`).

Sampling defaults to the checkpoint's `generation_config` (NeMo-Skills' own default is
greedy); `results.json` records the sampling used. Only the content after vLLM's reasoning
parser is scored.

## Output

`granite_results` (dataset): the `results.json` file (phase `all` or `score`).
`granite_generation` (dataset): the `generation.json` file (phase `generate`). A split run
is two targets on the same `output_dir`: the score target binds the generate target's
output, so it runs after it:

```yaml
<bench>-generate:            # GPU
  outputs:
    granite_generation:
      uri: "env://{{ binding.path }}"
<bench>:                     # CPU only, phase: score
  inputs:
    generation:
      binding: <bench>-generate.granite_generation
  outputs:
    granite_results:
      uri: "env://{{ binding.path }}"
```

Per-item outputs and logs are kept next to it.

## Developing

`make unit-tests` runs the template contract tests. The runtime has its own tests in
granite-evals. Every `steps/granite-*` step follows this template; the test checks this.
