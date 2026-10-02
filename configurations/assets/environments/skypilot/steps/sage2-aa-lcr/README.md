# sage2-aa-lcr (SkyPilot)

Scores a checkpoint on **AA-LCR** (Sage2 metric: pass@1 judge correct).
The job serves the model with vLLM and runs, locally:
NeMo-Skills' AA-LCR on `ArtificialAnalysis/AA-LCR`: 100 questions over ~100k-token document sets
(the extracted texts, pinned), inline in NeMo-Skills' `generic/default` prompt, then its AA-LCR judge
(CORRECT / INCORRECT). Prompts reach ~130k tokens, so the server needs a context of prompt plus
thinking: about 160k or more. Set `max_model_len` (the recipe's `AA_LCR_MAX_MODEL_LEN`).
It writes one `results.json`.

The code is the external [sage2-evals](https://github.com/laminair/sage2-evals) runtime,
shipped as a prebuilt image (the `nemoskills` family). Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/sage2-aa-lcr
```

## Config (`sage2_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | sage2-evals nemoskills image, pinned by tag (`SAGE2_IMAGE_NEMOSKILLS`). |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all 100 questions) | **Smoke knob:** first N items of the pinned data. |
| `repeats` | `""` (1) | Independent generations per item; k > 1 reports pass@1[avg-of-k] and `details.pass_at_k`. |
| `workers` | `32` | Concurrent requests to vLLM. |
| `dataset` / `dataset_revision` | `""` | Override the data pinned in sage2-evals (ArtificialAnalysis/AA-LCR and its extracted texts, pinned in sage2-evals). |
| `options` | `""` | Space-separated `key=value` benchmark options: `temperature`, `top_p`, `top_k`, `max_tokens`, `ns.<key>=<value>` (any NeMo-Skills generation override), `judge_model` (default `aws/claude-sonnet-5`), `judge_base_url` (default IBM LiteLLM), `judge_api_key_env` (default `SAGE2_JUDGE_API_KEY`), `max_judge_invalid_frac`, `answers=gold`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `163840` | vLLM. |
| `sandbox_cache` | `""` | Unused: this benchmark runs no sandbox. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The data is public.

The judge needs its key in the env var `judge_api_key_env` names (default `SAGE2_JUDGE_API_KEY`) at score time; every judge call is metered and capped by sage2-evals. The judge is `aws/claude-sonnet-5`, not AA's Qwen3-235B-A22B-Instruct-2507 or NeMo-Skills' gpt-4.1. `results.json` records the judge's token usage and cost.

Sampling defaults to the checkpoint's `generation_config` (NeMo-Skills' own default is
greedy); `results.json` records the sampling used. Only the content after vLLM's reasoning
parser is scored.

`answers=gold` serves the reference answers through the same NeMo-Skills generation and
the real judge without a model (set `model_path: none`); it should score about 100%. Use it to validate
the data and grading on a new cluster.

## Output

`sage2_results` (dataset): the `results.json` file (phase `all` or `score`).
`sage2_generation` (dataset): the `generation.json` file (phase `generate`). A split run
is two targets on the same `output_dir`: the score target binds the generate target's
output, so it runs after it:

```yaml
<bench>-generate:            # GPU
  outputs:
    sage2_generation:
      uri: "env://{{ binding.path }}"
<bench>:                     # CPU only, phase: score
  inputs:
    generation:
      binding: <bench>-generate.sage2_generation
  outputs:
    sage2_results:
      uri: "env://{{ binding.path }}"
```

Per-item outputs and logs are kept next to it.
