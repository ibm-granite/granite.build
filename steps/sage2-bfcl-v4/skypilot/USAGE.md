# sage2-bfcl-v4 (SkyPilot)

Scores a checkpoint on **BFCL v4** (Sage2 metric: overall_accuracy, BFCL's own
"Overall Acc"). The job serves the model with vLLM and runs the pinned `bfcl-eval`
package against it in function-calling mode (BFCL's `OpenAICompletionsHandler`, the
checkpoint's own chat template and tool parser) over all 22 v4 scoring categories:
non-live and live AST, irrelevance, multi-turn, memory and web search. bfcl-eval's
checkers score it and its leaderboard formula gives the value in `results.json`.

Web search goes to IBM's Google search MCP server (`google_pse_search`) instead of
BFCL's SerpAPI/DuckDuckGo backend. `results.json` records the backend under
`web_search_backend`. Scores in the web-search categories are not directly comparable
with the public leaderboard. No paid APIs are called.

The code is the external [sage2-evals](https://github.com/laminair/sage2-evals) runtime,
shipped as a prebuilt image. Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/sage2-bfcl-v4
```

## Config (`sage2_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | sage2-evals bfcl image, pinned by tag. |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all 4,441) | **Smoke knob:** first N entries of every category. Memory entries bring their prerequisite conversations. |
| `repeats` | `""` (1) | BFCL v4 is a single run. |
| `workers` | `8` | Concurrent test entries (BFCL `--num-threads`). |
| `dataset` / `dataset_revision` | `""` | Unused: the data ships inside the pinned bfcl-eval package. |
| `options` | `""` | Space-separated `key=value` options: `categories` (comma-separated BFCL categories or groups, default `all_scoring`), `temperature`, `top_p`, `max_tokens` (default: the checkpoint's generation_config), `search_mcp_url`, `include_input_log`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `/proj/granite-build/g4os/sage2/enroot-cache` | Unused by this benchmark. |
| `hf_home` | `""` | Overrides `HF_HOME`. The memory_vector category downloads `all-MiniLM-L6-v2` into it. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The job needs outbound HTTPS to the search MCP server, the web pages
the model fetches, and Hugging Face.

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

BFCL's own result and score files are kept next to it under `output/bfcl/`, with
per-category accuracy in `results.json` under `per_category`.
