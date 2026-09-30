# sage2-gdpval (SkyPilot)

Scores a checkpoint on **GDPval** (Sage2 metric: Elo) with an **Elo-style
approximation**. The job serves the model with vLLM and runs a
[Stirrup](https://github.com/ArtificialAnalysis/Stirrup) agent (the harness of
Artificial Analysis' GDPval-AA) on each task of the 220-task `openai/gdpval` gold set,
with a shell in one enroot sandbox holding the task's reference files. An LLM judge then
compares the agent's deliverable files with the expert's deliverables, in both orders,
and the job writes one `results.json`.

> **The value is not GDPval-AA's Elo.** GDPval-AA's judge prompt, judge panel and pool
> of reference submissions are not public, so its Elo cannot be reproduced. `value` is
> `1000 + 400*log10(p/(1-p))`, where `p` is the smoothed win rate against the expert
> deliverables (a tie counts 0.5). The expert sits at 1000. `results.json` flags this in
> `details.elo_is_approximation: true` and `details.metric_note`, and the job log ends
> with the same warning. Don't compare it with Artificial Analysis' published numbers.

The code is the external [sage2-evals](https://github.com/laminair/sage2-evals) runtime,
shipped as a prebuilt image (`judged` extra). Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/sage2-gdpval
```

## Judge

The judge is a paid OpenAI-compatible endpoint, set by `options`:
`judge_base_url` (default IBM LiteLLM, `https://ete-litellm.ai-models.vpc-int.res.ibm.com/v1`),
`judge_model` (default `aws/claude-sonnet-5`), `judge_api_key_env` (default
`SAGE2_JUDGE_API_KEY`: the job needs the key in that variable). The judge prompt is
sage2-evals' own. OpenAI's GDPval grader and GDPval-AA's judges are not public.
`judge_model=self` judges with the served model (no key; for smoke runs only; results
say `judge_is_self: true`).

Judge calls go through sage2-evals' spend meter: with `SAGE2_SPEND_LEDGER` and
`SAGE2_SPEND_BUDGET_USD` set in the job, calls stop at the budget. `results.json` has
`details.judge_usage` (tokens, cache hits, estimated cost) and `details.api_spend`
(the gateway's reported cost). A full run makes 2 judge calls per gradable task.

## Config (`sage2_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | sage2-evals judged image, pinned by tag. |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all 220) | **Smoke knob:** first N tasks by `task_id`. |
| `repeats` | `""` | Unused (one agent run per task). |
| `workers` | `4` | Concurrent tasks (agent sandboxes). |
| `dataset` / `dataset_revision` | `""` | Override the dataset pinned in sage2-evals (`openai/gdpval` at a fixed commit). |
| `options` | `""` | Space-separated `key=value`: `judge_model`, `judge_base_url`, `judge_api_key_env`, `judge_orders` (2), `max_turns` (250), `shell_timeout` (600), `max_tokens` (32768), `context_window` (131072), `temperature`, `top_p`, `sandbox_image` (`python:3.13-bookworm`), `sandbox_setup`, `tasks` (regex), `elo_anchor` (1000), `deliverables=expert`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `/proj/granite-build/g4os/sage2/enroot-cache` | Shared squashfs cache for the sandbox image. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The dataset is public; no HF token is needed. The sandbox pulls its image
from Docker Hub and installs the office-file Python libraries with pip, so the job needs
outbound internet. The sandbox has no LibreOffice.

Tasks with no expert deliverable file (35) or with only binary expert deliverables
(video, audio, images) are excluded and counted in `details.statuses`. A task where the
agent submits nothing counts as a loss.

`deliverables=expert` judges the expert's deliverables against themselves, without a
model (set `model_path: none`, unless `judge_model=self`). Use it to validate the data,
the file rendering and the judge: expect a win rate near 0.5 and a value near 1000.

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

Per-task deliverables, agent trajectories and judge answers are kept next to it under
`output/tasks/<task_id>/` (`deliverables/`, `agent.json`, `history.json`, `judge.json`,
`report.json`).
