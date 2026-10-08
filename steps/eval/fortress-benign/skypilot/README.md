# fortress-benign (SkyPilot)

Scores a checkpoint on **FORTRESS (benign)** (Granite metric: benign compliance (1 - refusal rate)).
The job serves the model with vLLM and runs, locally:
Scale AI FORTRESS (`ScaleAI/fortress_public`, pinned): the 500 benign twins of the adversarial
prompts. The judge decides whether each response is a refusal (inspect_evals' benign scorer). The value
is 1 − refusal rate; `details.ors` is the paper's over-refusal score (0-100, lower is better).
It writes one `results.json`.

The code is the external [granite-evals](https://github.com/laminair/granite-evals) runtime,
shipped as a prebuilt image (the `judged` family). Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/eval/fortress-benign
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
| `phase` | `"all"` | `all` generates and scores in one job. `generate` serves the model and writes `generation.json` (`granite_generation`); `score` grades that output dir without a GPU (same `output_dir`, `limit`, `repeats`) and writes `results.json` |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `""` | Unused: this benchmark runs no sandbox. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The data is public.

The judge needs its key in the env var `judge_api_key_env` names (default `GRANITE_EVALS_JUDGE_API_KEY`) at score time; every judge call is metered and capped by granite-evals. The judge is `aws/claude-sonnet-5`, not gpt-4o-mini. An empty final answer counts as a refusal; judge failures are recorded and bounded by `max_failed_frac`. `results.json` records the judge's token usage and cost.

Sampling defaults to the checkpoint's `generation_config`; thinking follows the chat
template (`enable_thinking` overrides it). Only the final answer, after vLLM's reasoning
parser, is graded.

`responses=refusal` grades a fixed refusal for every prompt without a model or GPU (set `model_path: none`): it must score 0.0.

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
granite-evals. Every step under `steps/eval/` follows this template; the test checks this.
