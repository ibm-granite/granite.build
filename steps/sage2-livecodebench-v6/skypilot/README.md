# sage2-livecodebench-v6 (SkyPilot)

Scores a checkpoint on **LiveCodeBench v6** (Sage2 metric: pass@1[avg-of-2] accuracy):
454 problems from release v6, contests 2024-08 to 2025-05 (NeMo-Skills'
`test_v6_2408_2505` split). The job serves the model with vLLM and runs NVIDIA
NeMo-Skills (pinned commit) against it: its LiveCodeBench prompt and generation, then
its evaluator, the pinned `livecodebench` package, on each problem's test cases inside
the job container. It writes one `results.json`.

The code is the external [sage2-evals](https://github.com/laminair/sage2-evals) runtime,
shipped as a prebuilt image (the `nemoskills` family). Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/sage2-livecodebench-v6
```

## Config (`sage2_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | sage2-evals nemoskills image, pinned by tag. |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all 454) | **Smoke knob:** the first N problems of the split. |
| `repeats` | `""` (2) | Independent generations per problem (avg-of-k). |
| `workers` | `64` | Concurrent requests to vLLM. |
| `dataset` / `dataset_revision` | `""` | Override the dataset pinned in sage2-evals (`livecodebench/code_generation_lite` at a fixed commit). |
| `options` | `""` | Space-separated `key=value` benchmark options: `temperature`, `top_p`, `top_k`, `max_tokens`, `ns.<key>=<value>` (raw NeMo-Skills overrides), `answers=gold`. |
| `phase` | `"all"` | `all` generates and scores in one job. `generate` serves the model and writes `generation.json` (`sage2_generation`); `score` grades that output dir without a GPU (same `output_dir`, `limit`, `repeats`) and writes `results.json` |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `/proj/granite-build/g4os/sage2/enroot-cache` | Unused here (shared config contract). |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The dataset is public; no HF token is needed. No paid API is used.
Sampling defaults to the model's `generation_config.json`; Granite's thinking stays on.

`answers=gold` runs five hand-checked solutions through the real prompt, generation
and grading path without a model (set `model_path: none`). Use it to validate the
evaluator on a new cluster; it should score 1.0.

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

NeMo-Skills' generations and per-problem grades are kept next to it under
`output/generation/output-rs<k>.jsonl`.

## Developing

`make unit-tests` runs the template contract tests. The runtime has its own tests in
sage2-evals. Every `steps/sage2-*` step follows this template; the test checks this.
