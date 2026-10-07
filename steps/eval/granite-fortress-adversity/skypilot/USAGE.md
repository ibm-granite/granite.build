# granite-fortress-adversity (SkyPilot)

Scores a checkpoint on **FORTRESS (adversarial)** (Granite metric: adversity resilience (1 - mean harm score)).
The job serves the model with vLLM and runs, locally:
Scale AI FORTRESS (`ScaleAI/fortress_public`, pinned): the 500 adversarial prompts, each response
graded against its per-prompt Yes/No rubric as inspect_evals' reference scorer does. The value is
1 − mean harm score (higher is safer); `details.ars` is the paper's ARS (0-100, lower is safer), also
per risk domain.
It writes one `results.json`.

The code is the external [granite-evals](https://github.com/laminair/granite-evals) runtime,
shipped as a prebuilt image (the `judged` family). Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/eval/granite-fortress-adversity
```

## Config (`granite_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | granite-evals judged image, pinned by tag (`GRANITE_EVALS_IMAGE_JUDGED`). |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all 500 prompts) | **Smoke knob:** first N items of the pinned data. |
| `repeats` | `""` (1) | Independent generations per item; k > 1 reports pass@1[avg-of-k] and `details.pass_at_k`. |
| `workers` | `16` | Responses generated concurrently (requests to vLLM). |
| `dataset` / `dataset_revision` | `""` | Override the data pinned in granite-evals (ScaleAI/fortress_public, pinned in granite-evals). |
| `options` | `""` | Space-separated `key=value` benchmark options: `responses=refusal` (grade a fixed refusal: no model, no GPU), `refusal_text`, `enable_thinking`, `temperature`, `top_p`, `max_tokens`, `judge_model`, `judge_base_url`, `judge_api_key_env`, `judge_tag`, `max_failed_frac` (0.05). |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `""` | Unused: this benchmark runs no sandbox. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The data is public.

The judge needs its key in the env var `judge_api_key_env` names (default `GRANITE_EVALS_JUDGE_API_KEY`) at score time; every judge call is metered and capped by granite-evals. One judge, `aws/claude-sonnet-5`, instead of FORTRESS's panel. An empty final answer scores harm 0 without a judge call; a judge error, refusal or unparseable grade is a recorded failure (bounded by `max_failed_frac`). `results.json` records the judge's token usage and cost.

Sampling defaults to the checkpoint's `generation_config`; thinking follows the chat
template (`enable_thinking` overrides it). Only the final answer, after vLLM's reasoning
parser, is graded.

`responses=refusal` grades a fixed refusal for every prompt without a model or GPU (set `model_path: none`): it must score 1.0. It checks the judge prompt, the gateway (it must accept the harmful prompts inside the judge prompt) and the grade parsing.

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
