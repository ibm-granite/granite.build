# judged (SkyPilot)

Scores a checkpoint on one of the granite-evals **judged** family's benchmarks: LLM-judged
tasks, several run through an agent in an enroot sandbox. `granite_config.benchmark`
selects which one. The job serves the model with vLLM, runs the benchmark locally, has
an LLM judge score the result (except where noted), and writes one `results.json`.

The code is the external [granite-evals](https://github.com/laminair/granite-evals) runtime,
shipped as a prebuilt image (the `judged` extra). Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/eval/judged
    config:
      granite_config:
        benchmark: gdpval
```

## Benchmarks in this family

| `benchmark` | Metric | Sandbox | Size |
|---|---|---|---|
| `fortress-adversity` | Adversity resilience (1 − mean harm score); `details.ars` = the paper's ARS (0-100, lower is safer), also per risk domain | no | 500 adversarial prompts (`ScaleAI/fortress_public`, pinned) |
| `fortress-benign` | Benign compliance (1 − refusal rate); `details.ors` = the paper's over-refusal score (0-100, lower is better) | no | 500 benign twins of the adversarial prompts, same dataset |
| `gdpval` | Elo-style **approximation**: `1000 + 400*log10(p/(1-p))`, `p` = smoothed win rate vs. expert deliverables (tie = 0.5), expert = 1000. **Not** GDPval-AA's own Elo (its judge panel and reference pool are not public); `results.json` flags this in `details.elo_is_approximation` | yes | 220 tasks (`openai/gdpval`, pinned); a [Stirrup](https://github.com/ArtificialAnalysis/Stirrup) agent per task |
| `mcpatlas` | `pass_rate_0.75`: fraction of tasks whose claim coverage is ≥ 0.75 | yes (host network; MCP servers behind one FastAPI app) | default `subset=keyless` 30 tasks (no API keys, but calls public internet APIs — outbound HTTPS needed); `keyless-gt` 89; `all` 500 (`ScaleAI/MCP-Atlas`, pinned) |
| `multi-challenge` | pass@1, mean over the four axes (upstream's own aggregation); `details.pass_at_k` = any-attempt score over the repeats | no | 273 multi-turn conversations (pinned by commit and sha256) |
| `profbench` | Overall score as a fraction: weighted rubric score per report, mean per task, per domain, then over the four domains ([NVlabs/ProfBench](https://github.com/NVlabs/ProfBench), pinned by commit) | no | 160 reports, ProfBench-lite (3-5 per task over 40 tasks, chemistry/physics/finance/consulting) |
| `strongreject` | Harmlessness (1 − mean StrongREJECT score); `details` carry the mean score, refusal rate and per-category numbers | no | 313 forbidden prompts, no jailbreak (pinned); the maintained `strong_reject` package's rubric judge |

See `granite-evals list` for the current, authoritative set and sizes.

## Judge

The judge is a paid OpenAI-compatible endpoint, set by `options`: `judge_base_url`
(default IBM LiteLLM), `judge_model` (default `aws/claude-sonnet-5`; FORTRESS's paper
panel and the per-benchmark upstream judge prompts are not all reproduced — see each
benchmark's own prompt in granite-evals), `judge_api_key_env` (default
`GRANITE_EVALS_JUDGE_API_KEY`: the job needs the key in that variable). `judge_model=self`
judges with the served model instead where the benchmark supports it (no key; smoke
runs only; results say `judge_is_self: true`).

Judge calls go through granite-evals' spend meter: with `GRANITE_EVALS_SPEND_LEDGER` and
`GRANITE_EVALS_SPEND_BUDGET_USD` set in the job, calls stop at the budget. `results.json`
has `details.judge_usage` (tokens, cache hits, estimated cost) and `details.api_spend`
(the gateway's reported cost).

## Config (`granite_config`)

| Field | Purpose |
|---|---|
| `benchmark` | Required. One of the ids in the table above. |
| `image` | Required. The granite-evals `judged` image, pinned by tag (`GRANITE_EVALS_IMAGE_JUDGED`). |
| `model_path` | Required. Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | Name vLLM serves under. Empty = basename of `model_path`. |
| `output_dir` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | **Smoke knob:** score only the first N tasks/items (the benchmark's own order). Empty = the full set. |
| `repeats` | Independent runs per task/item, where the benchmark supports it. Empty = benchmark default (1). |
| `workers` | Tasks/items worked on concurrently (agent sandboxes, if any, and requests to vLLM). Recipes set a per-benchmark default. |
| `dataset` / `dataset_revision` | Override the dataset pinned in granite-evals for the selected benchmark. |
| `options` | Space-separated `key=value` benchmark options — see each benchmark's own granite-evals docs; `judge_model=self` works across the family where supported. |
| `phase` | `all` generates and scores in one job. `generate` serves the model and writes `generation.json` (`granite_generation`); `score` grades that output dir without a GPU (same `output_dir`, `limit`, `repeats`) and writes `results.json`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | vLLM. |
| `sandbox_cache` | Shared squashfs cache. Set it (recipes do, per benchmark) only for the sandboxed benchmarks in the table above; leave empty for the rest. |
| `hf_home` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. Every dataset in this family is public; no HF token is needed.

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

Per-task/item outputs, agent trajectories and judge answers are kept next to it under
`output/` — the exact layout is per-benchmark; see granite-evals.

## Developing

`make unit-tests` runs the template contract tests. The runtime has its own tests in
granite-evals. Every step under `steps/eval/` follows this template; the test checks this.
