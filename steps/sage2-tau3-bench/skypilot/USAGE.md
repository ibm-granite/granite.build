# sage2-tau3-bench (SkyPilot)

Scores a checkpoint on **τ³-bench** (Sage2 metric: pass@1 (avg of 3); the mean of per-domain pass^1 over airline (50 tasks), retail (114) and telecom (114), as the τ²-bench leaderboard's Overall). The job serves
the model with vLLM and runs the pinned
[tau2-bench](https://github.com/sierra-research/tau2-bench) v1.0.1 harness: the served
model is the tool-calling agent, an LLM user simulator plays the customer, and the
harness's own evaluators grade each simulation. It writes one `results.json`.

The code is the external [sage2-evals](https://github.com/laminair/sage2-evals) runtime,
shipped as a prebuilt image. Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/sage2-tau3-bench
```

## Config (`sage2_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | sage2-evals tau image, pinned by tag. |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all 278) | **Smoke knob:** first N tasks per domain, in harness order. |
| `repeats` | `""` (4) | Trials per task; pass^1 is averaged over them. |
| `workers` | `16` | Concurrent simulations. |
| `dataset` / `dataset_revision` | `""` | Override the task data (default: `data/tau2` of the pinned tau2-bench commit, baked into the image). |
| `options` | `""` | Space-separated `key=value` benchmark options: `user_model`, `user_base_url`, `user_api_key_env`, `user_reasoning_effort`, `judge_*` (same), `temperature`, `top_p`, `max_tokens`, `enable_thinking`, `max_steps`, `max_errors`, `tasks` (regex), `task_split`, `retrieval_config`, `agent=gold`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `""` | Unused (no sandboxes); kept for the shared sage2-* contract. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none.

**Paid APIs.** The user simulator (and, for retail, the NL-assertion judge) defaults to
`aws/claude-sonnet-5` on the IBM LiteLLM gateway. The job needs `SAGE2_USER_API_KEY`
(and `SAGE2_JUDGE_API_KEY` for retail) in its environment, and every call is metered
against `SAGE2_SPEND_LEDGER` / `SAGE2_SPEND_BUDGET_USD`; at the budget the meter answers
HTTP 402 and the run stops scoring. `user_model=self judge_model=self` uses the served
model instead (smoke runs, no key, not comparable with published numbers).

`agent=gold` replays each task's reference actions without a model (set
`model_path: none`) and must score 1.0; use it to validate the data, the environments
and the grading on a new cluster.

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

Per-simulation records, with the full harness trajectory, are kept next to it under
`output/<domain>/trial-<k>/`.

## Developing

`make unit-tests` runs the template contract tests. The runtime has its own tests in
sage2-evals. Every `steps/sage2-*` step follows this template; the test checks this.
