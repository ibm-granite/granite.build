# sage2-swebench-verified (SkyPilot)

Scores a checkpoint on **SWE Bench Verified** (Sage2 metric: pass@1[avg-of-3] resolve
rate). The job serves the model with vLLM, runs mini-swe-agent against it in one enroot
sandbox per SWE-bench instance, grades each patch in a fresh sandbox with the upstream
`swebench` harness, and writes one `results.json`.

The code is the external [sage2-evals](https://github.com/laminair/sage2-evals) runtime,
shipped as a prebuilt image. Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/sage2-swebench-verified
```

## Config (`sage2_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | sage2-evals swebench image, pinned by tag. |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all 500) | **Smoke knob:** first N instances by `instance_id`. |
| `repeats` | `""` (3) | Independent agent runs per instance (avg-of-k). |
| `workers` | `8` | Concurrent instances. |
| `dataset` / `dataset_revision` | `""` | Override the dataset pinned in sage2-evals (upstream `SWE-bench/SWE-bench_Verified` at a fixed commit). |
| `options` | `""` | Space-separated `key=value` benchmark options: `step_limit`, `temperature`, `top_p`, `max_tokens`, `eval_timeout`, `instances` (regex), `patch=gold`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `/proj/granite-build/g4os/sage2/enroot-cache` | Shared squashfs cache for instance images. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The dataset is public; no HF token is needed.

`patch=gold` grades the reference patches without a model (set `model_path: none`).
Use it to validate the images, the sandbox and the grading on a new cluster. A few
instances (e.g. `psf__requests-1724`) call the live httpbin.org and can fail under gold.

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

Per-instance trajectories, patches and test logs are kept next to it under
`output/repeat-<k>/<instance_id>/`.
