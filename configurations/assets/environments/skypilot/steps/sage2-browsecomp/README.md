# sage2-browsecomp (SkyPilot)

Scores a checkpoint on **BrowseComp** (Sage2 metric: mean reward).
The job serves the model with vLLM and runs, locally:
OpenAI simple-evals' BrowseComp (1266 questions; the encrypted CSV, pinned by sha256, decrypted
with each row's canary) and query template, with NeMo-Skills' tool-calling generation: web search
through the IBM Google PSE MCP server (no key) and page fetch. Search hits and fetches of the
benchmark's own data are refused. Then simple-evals' grader prompt.
It writes one `results.json`.

The code is the external [sage2-evals](https://github.com/laminair/sage2-evals) runtime,
shipped as a prebuilt image (the `nemoskills` family). Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/sage2-browsecomp
```

## Config (`sage2_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | sage2-evals nemoskills image, pinned by tag (`SAGE2_IMAGE_NEMOSKILLS`). |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all 1266 questions) | **Smoke knob:** first N items of the pinned data. |
| `repeats` | `""` (1) | Independent generations per item; k > 1 reports pass@1[avg-of-k] and `details.pass_at_k`. |
| `workers` | `32` | Concurrent requests to vLLM. |
| `dataset` / `dataset_revision` | `""` | Override the data pinned in sage2-evals (simple-evals' BrowseComp CSV, pinned by sha256 in sage2-evals). |
| `options` | `""` | Space-separated `key=value` benchmark options: `temperature`, `top_p`, `top_k`, `max_tokens`, `ns.<key>=<value>` (any NeMo-Skills generation override), `tools` (comma list of python, search, fetch), `max_tool_calls` (100), `python_timeout` (60 s), `python_image`, `python_packages`, `python_network` (off), `search_mcp_url`, `search_results` (10), `fetch_max_chars` (20000), `judge_model` (default `aws/claude-sonnet-5`), `judge_base_url` (default IBM LiteLLM), `judge_api_key_env` (default `SAGE2_JUDGE_API_KEY`), `max_judge_invalid_frac`, `answers=gold`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `""` | Unused: this benchmark runs no sandbox. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The data is public. Generate needs outbound HTTPS (search and page fetch).

The judge needs its key in the env var `judge_api_key_env` names (default `SAGE2_JUDGE_API_KEY`) at score time; every judge call is metered and capped by sage2-evals. The grader is `aws/claude-sonnet-5`, not simple-evals' gpt-4.1. A reply without a verdict is re-judged, not counted "no"; `limit` takes the first N rows, not a seeded sample. `results.json` records the judge's token usage and cost.

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
