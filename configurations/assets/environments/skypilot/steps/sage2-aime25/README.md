# sage2-aime25 (SkyPilot)

Scores a checkpoint on **AIME25** (Sage2 metric: pass@1[avg-of-4] symbolic correct).
The job serves the model with vLLM and runs NVIDIA NeMo-Skills' own aime25 pipeline
against it, locally: data preparation (every upstream read pinned), prompt, generation,
answer extraction and metrics. It writes one `results.json`.

The code is the external [sage2-evals](https://github.com/laminair/sage2-evals) runtime,
shipped as a prebuilt image. Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/sage2-aime25
```

## Config (`sage2_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | sage2-evals nemoskills image, pinned by tag (`SAGE2_IMAGE_NEMOSKILLS`). |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all 30 problems) | **Smoke knob:** first N examples of the pinned data. |
| `repeats` | `""` (4) | Independent generations per example (avg-of-k). |
| `workers` | `32` | Concurrent requests to vLLM. |
| `dataset` / `dataset_revision` | `""` | Override the data pinned in sage2-evals (AIME 2025 I+II, bundled in the pinned NeMo-Skills commit). |
| `options` | `""` | Space-separated `key=value` benchmark options: `temperature`, `top_p`, `top_k`, `max_tokens`, `ns.<key>=<value>` (any NeMo-Skills generation override), `answers=gold`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `""` | Unused: this benchmark runs no sandbox. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The data is public; no HF token is needed.

Sampling defaults to the checkpoint's `generation_config` (NeMo-Skills' own default is
greedy); `results.json` records the sampling used.

`answers=gold` serves the reference answers through the same NeMo-Skills generation and
grading without a model (set `model_path: none`); it scores 100%. Use it to validate
the data and grading on a new cluster.

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

The prepared data, per-repeat NeMo-Skills outputs (`generation/output-rs<k>.jsonl`)
and their logs are kept next to it.
