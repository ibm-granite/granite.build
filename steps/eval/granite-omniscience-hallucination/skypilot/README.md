# granite-omniscience-hallucination (SkyPilot)

Scores a checkpoint on **AA-Omniscience hallucination rate** (Granite metric: pass@1 judge_omni_hallucination, lower is better).
The job serves the model with vLLM and runs, locally:
The same pipeline as `granite-omniscience`. The value is the hallucination rate, incorrect /
(incorrect + partial + not attempted): **lower is better**. `generations_from=<omniscience
output dir>` scores that run's generations and judgements (copied in; only invalid ones are re-judged)
instead of a run of its own.
It writes one `results.json`.

The code is the external [granite-evals](https://github.com/laminair/granite-evals) runtime,
shipped as a prebuilt image (the `nemoskills` family). Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/eval/granite-omniscience-hallucination
```

## Config (`granite_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | granite-evals nemoskills image, pinned by tag (`GRANITE_EVALS_IMAGE_NEMOSKILLS`). |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all 600 questions) | **Smoke knob:** first N items of the pinned data. |
| `repeats` | `""` (1) | Independent generations per item; k > 1 reports pass@1[avg-of-k] and `details.pass_at_k`. |
| `workers` | `32` | Concurrent requests to vLLM. |
| `dataset` / `dataset_revision` | `""` | Override the data pinned in granite-evals (ArtificialAnalysis/AA-Omniscience-Public, pinned in granite-evals). |
| `options` | `""` | Space-separated `key=value` benchmark options: `generations_from`, `temperature`, `top_p`, `top_k`, `max_tokens`, `ns.<key>=<value>` (any NeMo-Skills generation override), `judge_model` (default `aws/claude-sonnet-5`), `judge_base_url` (default IBM LiteLLM), `judge_api_key_env` (default `GRANITE_EVALS_JUDGE_API_KEY`), `max_judge_invalid_frac`, `answers=gold`. |
| `phase` | `"all"` | `all` generates and scores in one job. `generate` serves the model and writes `generation.json` (`granite_generation`); `score` grades that output dir without a GPU (same `output_dir`, `limit`, `repeats`) and writes `results.json` |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `""` | Unused: this benchmark runs no sandbox. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The data is public.

The judge needs its key in the env var `judge_api_key_env` names (default `GRANITE_EVALS_JUDGE_API_KEY`) at score time; every judge call is metered and capped by granite-evals. The judge is `aws/claude-sonnet-5`. A judgement NeMo-Skills cannot parse is left out and re-judged (`max_judge_invalid_frac`). `results.json` records the judge's token usage and cost.

Sampling defaults to the checkpoint's `generation_config` (NeMo-Skills' own default is
greedy); `results.json` records the sampling used. Only the content after vLLM's reasoning
parser is scored.

`answers=gold` serves the reference answers through the same NeMo-Skills generation and
the real judge without a model (set `model_path: none`); it should score about 100%. Use it to validate
the data and grading on a new cluster.

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
