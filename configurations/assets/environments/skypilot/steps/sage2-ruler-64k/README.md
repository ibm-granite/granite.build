# sage2-ruler-64k (SkyPilot)

Scores a checkpoint on **RULER at 64k** (Sage2 metric: accuracy, NeMo-Skills'
`ruler_score`, the mean over RULER's 13 tasks at 65536 tokens). The job generates
the tasks for the model's own tokenizer with NeMo-Skills' RULER prepare (pinned ns and
RULER commits; RULER's source data are baked into the image and checked by sha256),
100 samples per task. It serves the model with vLLM at `max_model_len` 131072, runs
each task through NeMo-Skills' generation and RULER match, and writes one
`results.json`.

Thinking is on by default, a departure from NeMo-Skills' RULER, which budgets
30-128 answer tokens with the answer prefix prefilled. Each task gets a thinking
budget on top of its answer budget, the chat data format (no prefill), and only the
content after vLLM's reasoning parser is scored. results.json records the mode,
budgets and scoring source. The 64k samples plus the budget need more than 65536 tokens, hence
131072 (granite-4.2-3b's native maximum). `enable_thinking=false` is
NeMo-Skills' RULER exactly.

The code is the external [sage2-evals](https://github.com/laminair/sage2-evals) runtime,
shipped as a prebuilt image (the `nemoskills` family). Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/sage2-ruler-64k
```

## Config (`sage2_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | sage2-evals nemoskills image, pinned by tag. |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. Its tokenizer builds the data. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. The generated data are cached under it. |
| `limit` | `""` (all 100) | **Smoke knob:** the first N samples of *each* task. |
| `repeats` | `""` (1) | Independent generations per sample. |
| `workers` | `32` | Concurrent requests to vLLM. |
| `dataset` / `dataset_revision` | `""` | A prepared RULER setup dir overrides the in-job generation. |
| `options` | `""` | Space-separated `key=value` benchmark options: `tasks` (comma-separated subset, for debugging), `tokenizer`, `enable_thinking`, `thinking_budget`, `sample_length`, `temperature`, `top_p`, `top_k`, `max_tokens`, `ns.<key>=<value>`, `answers=gold`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `131072` | vLLM. The run stops if the served context is shorter than a sample plus the thinking budget. |
| `sandbox_cache` | `/proj/granite-build/g4os/sage2/enroot-cache` | Unused here (shared config contract). |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. Data generation at 64k takes a few minutes of CPU and needs memory:
give the target enough of both. No paid API is used.

`answers=gold` serves each sample's expected answers without a GPU. `model_path` must
still name the tokenizer, because the data are built for it. Use it to validate data
generation and scoring; it should score 1.0.

## Output

`sage2_results` (dataset): the `results.json` file, with per-task accuracies. Declare
it on the target:

```yaml
outputs:
  sage2_results:
    uri: "env://{{ binding.path }}"
```

Generations are kept next to it under `output/ruler/<task>/output-rs<k>.jsonl`.

## Developing

`make unit-tests` runs the template contract tests. The runtime has its own tests in
sage2-evals. Every `steps/sage2-*` step follows this template; the test checks this.
