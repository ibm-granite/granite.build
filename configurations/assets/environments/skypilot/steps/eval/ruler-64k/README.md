# ruler-64k (SkyPilot)

Scores a checkpoint on **RULER at 64k** (Granite metric: accuracy, NeMo-Skills'
`ruler_score`, the mean over RULER's 13 tasks at 65536 tokens). The job generates
the tasks for the model's own tokenizer with NeMo-Skills' RULER prepare (pinned ns and
RULER commits; RULER's source data are baked into the image and checked by sha256),
100 samples per task. It serves the model with vLLM at `max_model_len` 131072, runs
each task through NeMo-Skills' generation and RULER match, and writes one
`results.json`.

Thinking is on by default, a departure from NeMo-Skills' RULER, which budgets
30-128 answer tokens with the answer prefix prefilled. Each sample uses the chat data
format (no prefill) and may generate up to the context cap (`max_tokens = cap - prompt
tokens`; the cap is the served `max_model_len` unless `context_cap=N` lowers it), and
only the content after vLLM's reasoning parser is scored. A sample that does not fit
(its prompt exceeds the cap, the server returns a context-length error, or generation
hits the cap before any answer) scores 0 without stopping the run and is appended, as
it happens, to `<output_dir>/failures.jsonl`. results.json records the cap, thinking
mode, prompt and max-token percentiles, the failure counts by reason and task, and any
departures. `enable_thinking=false` runs NeMo-Skills' RULER exactly.
This step serves at 131072 (granite-4.2-3b's native maximum), so thinking has about
64k tokens of headroom over a 64k sample.

The code is the external [granite-evals](https://github.com/laminair/granite-evals) runtime,
shipped as a prebuilt image (the `nemoskills` family). Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/eval/ruler-64k
```

## Config (`granite_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | granite-evals nemoskills image, pinned by tag. |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. Its tokenizer builds the data. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. The generated data are cached under it. |
| `limit` | `""` (all 100) | **Smoke knob:** the first N samples of *each* task. |
| `repeats` | `""` (1) | Independent generations per sample. |
| `workers` | `32` | Concurrent requests to vLLM. |
| `dataset` / `dataset_revision` | `""` | A prepared RULER setup dir overrides the in-job generation. |
| `options` | `""` | Space-separated `key=value` benchmark options: `tasks` (comma-separated subset, for debugging), `tokenizer`, `enable_thinking`, `context_cap`, `sample_length`, `temperature`, `top_p`, `top_k`, `max_tokens`, `ns.<key>=<value>`, `answers=gold`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `131072` | vLLM. Samples that do not fit score 0 and go to `failures.jsonl`; the run stops only if the served context is shorter than the samples. |
| `sandbox_cache` | `/proj/granite-build/g4os/granite/enroot-cache` | Unused here (shared config contract). |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. Data generation at 64k takes a few minutes of CPU and needs memory:
give the target enough of both. No paid API is used.

`answers=gold` serves each sample's expected answers without a GPU. `model_path` must
still name the tokenizer, because the data are built for it. Use it to validate data
generation and scoring; it should score 1.0.

## Output

`granite_results` (dataset): the `results.json` file, with per-task accuracies (phase
`all` or `score`). `granite_generation` (dataset): the `generation.json` file (phase
`generate`). A split run is two targets on the same `output_dir`: the score target binds
the generate target's output, so it runs after it:

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

`failures.jsonl` (one line per sample scored 0 for not fitting) sits next to it. Generations are kept under `output/ruler/<task>/output-rs<k>.jsonl`.

## Developing

`make unit-tests` runs the template contract tests. The runtime has its own tests in
granite-evals. Every step under `steps/eval/` follows this template; the test checks this.
