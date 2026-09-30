# sage2-ifbench (SkyPilot)

Scores a checkpoint on **IFBench** (prompt level; Sage2 metric: pass@1[avg-of-2] loose
accuracy). The job serves the model with vLLM and runs NVIDIA NeMo-Skills' own ifbench
pipeline against it, locally. That covers data preparation (the allenai/IFBench test file,
pinned by commit and sha256), the generic/default prompt, generation, and IFBench's own
strict and loose verifiers. The verifiers are at the IFBench commit NeMo-Skills pins and
come with NeMo-Skills' patch, baked into the image. NeMo-Skills' `if` metrics produce
the score. The job writes one `results.json`. A prompt counts only if the response
follows all of its instructions.

The code is the external [sage2-evals](https://github.com/laminair/sage2-evals) runtime,
shipped as a prebuilt image. Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/sage2-ifbench
```

## Config (`sage2_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | sage2-evals ifbench image, pinned by tag (`SAGE2_IMAGE_IFBENCH`). |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all 300 prompts) | **Smoke knob:** first N examples of the pinned data. |
| `repeats` | `""` (2) | Independent generations per example (avg-of-k). |
| `workers` | `64` | Concurrent requests to vLLM. |
| `dataset` / `dataset_revision` | `""` | Override the data pinned in sage2-evals (allenai/IFBench `data/IFBench_test.jsonl` at a fixed commit). |
| `options` | `""` | Space-separated `key=value` benchmark options: `temperature`, `top_p`, `top_k`, `max_tokens`, `ns.<key>=<value>` (any NeMo-Skills generation override). |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `""` | Unused: this benchmark runs no sandbox. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The data is public; no token is needed.

Sampling defaults to the checkpoint's `generation_config` (NeMo-Skills' own default is
greedy). `results.json` records the sampling used. There is no `answers=gold` mode,
because IFBench has no reference responses.

## Output

`sage2_results` (dataset): the `results.json` file. Declare it on the target:

```yaml
outputs:
  sage2_results:
    uri: "env://{{ binding.path }}"
```

The prepared data and the per-repeat NeMo-Skills outputs are kept next to it. The
outputs are `generation/output-rs<k>.jsonl`, with each response's `loose_eval` and
`strict_eval`, and their logs.

## Developing

`make unit-tests` runs the template contract tests. The runtime has its own tests in
sage2-evals. Every `steps/sage2-*` step follows this template; the test checks this.
