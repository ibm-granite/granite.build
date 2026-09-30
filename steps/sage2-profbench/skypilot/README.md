# sage2-profbench (SkyPilot)

Scores a checkpoint on **ProfBench** (Sage2 metric: overall). The job serves the model
with vLLM, generates the 160 reports of ProfBench-lite (3-5 per task over 40 tasks in
chemistry, physics, finance and consulting), grades every rubric criterion with an LLM
judge using [NVlabs/ProfBench](https://github.com/NVlabs/ProfBench)'s own judge call
and scorer (pinned by commit), and writes one `results.json`. `value` is the upstream
Overall score as a fraction (weighted rubric score per report, mean per task, per domain,
then over the four domains).

The code is the external [sage2-evals](https://github.com/laminair/sage2-evals) runtime,
shipped as a prebuilt image (`judged` extra). Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/sage2-profbench
```

## Judge

The judge is a paid OpenAI-compatible endpoint, set by `options`:
`judge_base_url` (default IBM LiteLLM, `https://ete-litellm.ai-models.vpc-int.res.ibm.com/v1`),
`judge_model` (default `aws/claude-sonnet-5`), `judge_api_key_env` (default
`SAGE2_JUDGE_API_KEY`: the job needs the key in that variable). ProfBench's own
leaderboard judge is `openai/gpt-oss-120b` with "mixed" reasoning effort, so scores
with another judge are not directly comparable with the published leaderboard.
`judge_model=self` judges with the served model (no key; for smoke runs only; results
say `judge_is_self: true`).

Judge calls go through sage2-evals' spend meter: with `SAGE2_SPEND_LEDGER` and
`SAGE2_SPEND_BUDGET_USD` set in the job, calls stop at the budget. `results.json` has
`details.judge_usage` (tokens, cache hits, estimated cost) and `details.api_spend`
(the gateway's reported cost). A full lite run makes about 4,600 judge calls.

## Config (`sage2_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | sage2-evals judged image, pinned by tag. |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all 40 tasks) | **Smoke knob:** first N tasks by `task_id`, each with its lite samples. |
| `repeats` | `""` | Unused (the lite version fixes the samples per task). |
| `workers` | `16` | Concurrent report generations. |
| `dataset` / `dataset_revision` | `""` | Override the dataset pinned in sage2-evals (`nvidia/ProfBench` at a fixed commit). |
| `options` | `""` | Space-separated `key=value`: `judge_model`, `judge_base_url`, `judge_api_key_env`, `judge_workers` (16), `version` (`lite`/`full`/`debug`), `samples` (cap per task), `temperature`, `top_p`, `max_tokens` (64000), `responses`. |
| `phase` | `"all"` | `all` generates and scores in one job. `generate` serves the model and writes `generation.json` (`sage2_generation`); `score` grades that output dir without a GPU (same `output_dir`, `limit`, `repeats`) and writes `results.json` |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `/proj/granite-build/g4os/sage2/enroot-cache` | Unused (no sandboxes); shared step contract. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The dataset is public; no HF token is needed.

`responses=o3 judge_model=human` scores the dataset's own o3 reports with its human
rubric labels, without a model or a judge (set `model_path: none`). Use it to validate
the data and the scorer. It does not score 100%: the reports miss criteria. With a real
judge instead of `human`, results also give the judge's macro-F1 against the human
labels (upstream's judge-quality metric).

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

Per-sample reports and judge ratings are kept next to it under
`output/samples/<task_id>/<k>/` (`response.json`, `judgments.json`).

## Developing

`make unit-tests` runs the template contract tests. The runtime has its own tests in
sage2-evals. Every `steps/sage2-*` step follows this template; the test checks this.
