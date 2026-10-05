# granite-mcpatlas (SkyPilot)

Scores a checkpoint on **MCP-Atlas** (Granite metric: pass rate (claim coverage ≥ 0.75)).
The job serves the model with vLLM and runs, locally:
Scale AI MCP-Atlas (`ScaleAI/MCP-Atlas`, pinned): upstream's agent-environment image (MCP servers
behind one FastAPI app) runs in an enroot sandbox on the host network, a port of upstream's agent loop
drives the model through each task's tools, and upstream's per-claim judge grades the final answer.
The value is upstream's `pass_rate_0.75`: the fraction of tasks whose claim coverage is at least 0.75.
The default `subset=keyless` (30 tasks) uses only servers that need no API keys; they still call
public internet APIs, so the job needs outbound HTTPS. `keyless-gt` has 89 tasks; `all` (500) needs
every server's keys.
It writes one `results.json`.

The code is the external [granite-evals](https://github.com/laminair/granite-evals) runtime,
shipped as a prebuilt image (the `judged` family). Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/granite-mcpatlas
```

## Config (`granite_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | granite-evals judged image, pinned by tag (`GRANITE_EVALS_IMAGE_JUDGED`). |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all tasks of the subset (keyless: 30)) | **Smoke knob:** first N items of the pinned data. |
| `repeats` | `""` (1) | Independent generations per item; k > 1 reports pass@1[avg-of-k] and `details.pass_at_k`. |
| `workers` | `5` | Unused: tasks run `concurrency` (default 5) at a time. |
| `dataset` / `dataset_revision` | `""` | Override the data pinned in granite-evals (ScaleAI/MCP-Atlas parquet, pinned by commit and sha256 in granite-evals). |
| `options` | `""` | Space-separated `key=value` benchmark options: `subset` (`keyless`, `keyless-gt`, `all`), `agent=model|gold|replay`, `env_url`, `env_port` (1984), `concurrency` (5), `max_turns` (256), `max_tool_calls` (100), `tool_timeout`, `task_timeout`, `enable_thinking`, `temperature`, `top_p`, `max_tokens`, `threshold` (0.75), `judge_model`, `judge_base_url`, `judge_api_key_env`, `judge_workers`, `max_failed_frac`. |
| `phase` | `"all"` | `all` generates and scores in one job. `generate` serves the model and writes `generation.json` (`granite_generation`); `score` grades that output dir without a GPU (same `output_dir`, `limit`, `repeats`) and writes `results.json` |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `/proj/granite-build/g4os/granite/enroot-cache` | Shared enroot squashfs cache for the sandbox images. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The data is public. Generate needs outbound HTTPS from the environment sandbox; the environment holds the node's port 1984.

The judge needs its key in the env var `judge_api_key_env` names (default `GRANITE_EVALS_JUDGE_API_KEY`) at score time; every judge call is metered and capped by granite-evals. The judge is `aws/claude-sonnet-5`, not upstream's gemini-3.1-pro-preview. A claim the judge fails on leaves the task out (retried on resume), where upstream scores it not fulfilled. `results.json` records the judge's token usage and cost.

Sampling defaults to the checkpoint's `generation_config`; thinking follows the chat
template (`enable_thinking` overrides it). Only the final answer, after vLLM's reasoning
parser, is graded.

`agent=gold` answers each task with its own expert claims, without a model or GPU (set `model_path: none`): the pass rate should be about 1. `agent=replay` also starts the environment and replays the reference tool calls (`details.replay`): a check of servers and egress before a model run.

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
