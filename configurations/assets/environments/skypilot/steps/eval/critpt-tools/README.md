# critpt-tools (SkyPilot)

Scores a checkpoint on **CritPt with tools** (Granite metric: challenge accuracy).
The job serves the model with vLLM and runs, locally:
CritPt (70 physics research challenges) with NeMo-Skills' two-turn generation and a python tool in
both turns (a stateful session in an enroot sandbox, network off, fresh per turn). Grading is
Artificial Analysis's CritPt API: all 70 answers in one request, aggregate accuracy only.
It writes one `results.json`.

The code is the external [granite-evals](https://github.com/laminair/granite-evals) runtime,
shipped as a prebuilt image (the `nemoskills` family). Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/eval/critpt-tools
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
| `options` | `""` | Space-separated `key=value` benchmark options: `temperature`, `top_p`, `top_k`, `max_tokens`, `ns.<key>=<value>` (any NeMo-Skills generation override), `tools` (comma list of python, search, fetch), `max_tool_calls` (100), `python_timeout` (60 s), `python_image`, `python_packages`, `python_network` (off), `search_mcp_url`, `search_results` (10), `fetch_max_chars` (20000), `critpt_api_key_env`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `/proj/granite-build/g4os/granite/enroot-cache` | Shared enroot squashfs cache for the sandbox images. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. Scoring needs an approved Artificial Analysis key as `ARTIFICIAL_ANALYSIS_API_KEY` (option `critpt_api_key_env`); 10 requests a day, cached per repeat, not metered. No `answers=gold`. `limit` works for phase `generate` only (the accuracy is rescaled to the N submitted).

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
