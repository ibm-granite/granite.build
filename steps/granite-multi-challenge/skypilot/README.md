# granite-multi-challenge (SkyPilot)

Scores a checkpoint on **MultiChallenge** (Granite metric: pass@1 correct (macro over the four axes)).
The job serves the model with vLLM and runs, locally:
Scale AI MultiChallenge (273 multi-turn conversations, pinned by commit and sha256): the model
continues each conversation, and the judge answers its human-written YES/NO rubric question about the
final reply (upstream's judge prompt, temperature 0). The value is pass@1, the mean over the four axes as
upstream; `details.pass_at_k` is upstream's any-attempt score over the repeats.
It writes one `results.json`.

The code is the external [granite-evals](https://github.com/laminair/granite-evals) runtime,
shipped as a prebuilt image (the `judged` family). Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/granite-multi-challenge
```

## Config (`granite_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | granite-evals judged image, pinned by tag (`GRANITE_EVALS_IMAGE_JUDGED`). |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all 273 conversations) | **Smoke knob:** first N items of the pinned data. |
| `repeats` | `""` (1) | Independent generations per item; k > 1 reports pass@1[avg-of-k] and `details.pass_at_k`. |
| `workers` | `16` | Responses generated concurrently (requests to vLLM). |
| `dataset` / `dataset_revision` | `""` | Override the data pinned in granite-evals (ekwinox117/multi-challenge data, pinned by commit and sha256 in granite-evals). |
| `options` | `""` | Space-separated `key=value` benchmark options: `responses=model|claude-3-5-sonnet-20241022|gpt-4o-2024-08-06|o1-preview`, `judge_structured`, `enable_thinking`, `temperature`, `top_p`, `max_tokens`, `judge_model`, `judge_base_url`, `judge_api_key_env`, `max_failed_frac`. |
| `phase` | `"all"` | `all` generates and scores in one job. `generate` serves the model and writes `generation.json` (`granite_generation`); `score` grades that output dir without a GPU (same `output_dir`, `limit`, `repeats`) and writes `results.json` |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `""` | Unused: this benchmark runs no sandbox. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The data is public.

The judge needs its key in the env var `judge_api_key_env` names (default `GRANITE_EVALS_JUDGE_API_KEY`) at score time; every judge call is metered and capped by granite-evals. The judge is `aws/claude-sonnet-5`, not upstream's gpt-4o. Judge failures and generation errors are left out of the score (bounded by `max_failed_frac`), where upstream counts them as NO. `results.json` records the judge's token usage and cost.

Sampling defaults to the checkpoint's `generation_config`; thinking follows the chat
template (`enable_thinking` overrides it). Only the final answer, after vLLM's reasoning
parser, is graded.

`responses=<model>` (`claude-3-5-sonnet-20241022`, `gpt-4o-2024-08-06` or `o1-preview`) grades the responses upstream ships, without a model or GPU (set `model_path: none`); compare with the paper's averages in `details.paper_reference`.

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
