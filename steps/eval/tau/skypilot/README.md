# tau (SkyPilot)

Scores a checkpoint on one of the granite-evals **tau** family's benchmarks: τ³-bench
domains, each run through the pinned [tau2-bench](https://github.com/sierra-research/tau2-bench)
v1.0.1 harness. The served model is the tool-calling agent, an LLM user simulator plays
the customer, and the harness's own evaluators grade each simulation. `granite_config.benchmark`
selects which domain (or the Overall average).

The code is the external [granite-evals](https://github.com/laminair/granite-evals) runtime,
shipped as a prebuilt image. Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/eval/tau
    config:
      granite_config:
        benchmark: tau3-airline
```

## Benchmarks in this family

| `benchmark` | Metric | Size |
|---|---|---|
| `tau3-airline` | pass@1 | 50 airline tasks |
| `tau3-banking-knowledge` | pass@1 | 97 banking_knowledge tasks, with BM25 + grep retrieval |
| `tau3-bench` | pass@1 (avg of 3) | the mean of per-domain pass@1 over airline, retail and telecom, as the τ²-bench leaderboard's Overall |
| `tau3-retail` | pass@1 | 114 retail tasks |
| `tau3-telecom` | pass@1 | 114 telecom tasks |

See `granite-evals list` for the current, authoritative set and sizes.

## Config (`granite_config`)

| Field | Purpose |
|---|---|
| `benchmark` | Required. One of the ids in the table above. |
| `image` | Required. The granite-evals `tau` image, pinned by tag (`GRANITE_EVALS_IMAGE_TAU`). |
| `model_path` | Required. Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | Name vLLM serves under. Empty = basename of `model_path`. |
| `output_dir` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | **Smoke knob:** score only the first N tasks per domain, in harness order. Empty = the full set. |
| `repeats` | Trials per task. Empty = benchmark default (1); k > 1 averages pass@1 over k trials and adds pass@k (`details.pass_at_k`). |
| `workers` | Concurrent simulations. Recipes set `16` for every benchmark in this family. |
| `dataset` / `dataset_revision` | Override the task data. Empty = `data/tau2` of the pinned tau2-bench commit, baked into the image. |
| `options` | Space-separated `key=value` benchmark options: `user_model`, `user_base_url`, `user_api_key_env`, `user_reasoning_effort`, `judge_*` (same, retail only), `temperature`, `top_p`, `max_tokens`, `enable_thinking`, `max_steps`, `max_errors`, `tasks` (regex), `task_split`, `retrieval_config`, `agent=gold`. |
| `phase` | `all` generates and scores in one job. `generate` serves the model and writes `generation.json` (`granite_generation`); `score` grades that output dir without a GPU (same `output_dir`, `limit`, `repeats`) and writes `results.json`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | vLLM. |
| `sandbox_cache` | Unused by this family (no sandboxes); kept for the shared config contract. |
| `hf_home` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none.

**Paid APIs.** The user simulator (and, for `tau3-retail`, the NL-assertion judge)
defaults to `aws/claude-sonnet-5` on the IBM LiteLLM gateway. The job needs
`GRANITE_EVALS_USER_API_KEY` (and `GRANITE_EVALS_JUDGE_API_KEY` for retail) in its
environment, and every call is metered against `GRANITE_EVALS_SPEND_LEDGER` /
`GRANITE_EVALS_SPEND_BUDGET_USD`; at the budget the meter answers HTTP 402 and the run
stops scoring. `user_model=self judge_model=self` uses the served model instead (smoke
runs, no key, not comparable with published numbers).

`agent=gold` replays each task's reference actions without a model (set
`model_path: none`) and must score 1.0; use it to validate the data, the environments
and the grading on a new cluster.

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

Per-simulation records, with the full harness trajectory, are kept next to it under
`output/<domain>/trial-<k>/`.

## Developing

`make unit-tests` runs the template contract tests. The runtime has its own tests in
granite-evals. Every step under `steps/eval/` follows this template; the test checks this.
