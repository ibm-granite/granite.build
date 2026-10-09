# swebench (SkyPilot)

Scores a checkpoint on one of the granite-evals **swebench** family's benchmarks:
SWE-bench variants, each run with mini-swe-agent in one enroot sandbox per instance and
graded in a fresh sandbox with the upstream `swebench` harness. `granite_config.benchmark`
selects which one.

The code is the external [granite-evals](https://github.com/laminair/granite-evals) runtime,
shipped as a prebuilt image. Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/eval/swebench
    config:
      granite_config:
        benchmark: swebench-verified
```

## Benchmarks in this family

| `benchmark` | Size | Notes |
|---|---|---|
| `swebench-verified` | 500 instances (`SWE-bench/SWE-bench_Verified`, pinned) | mini-swe-agent against the upstream task, graded with the `swebench` harness |
| `swebench-pro` | 642 tasks, V2 public set | follows Scale's V2 locked protocol ([SWE-bench_Pro-os](https://github.com/scaleapi/SWE-bench_Pro-os) v2.0.0): mini-swe-agent with the protocol's tool-calling config works each task's `instruction.md` |
| `swebench-multilingual` | 300 instances, 9 languages (C/C++, Go, Java, JavaScript/TypeScript, PHP, Ruby, Rust) | same pipeline as `swebench-verified`, mini-swe-agent's `swebench.yaml` config |

See `granite-evals list` for the current, authoritative set and sizes.

## Config (`granite_config`)

| Field | Purpose |
|---|---|
| `benchmark` | Required. One of the ids in the table above. |
| `image` | Required. The granite-evals `swebench` image, pinned by tag (`GRANITE_EVALS_IMAGE_SWEBENCH`). |
| `model_path` | Required. Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | Name vLLM serves under. Empty = basename of `model_path`. |
| `output_dir` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | **Smoke knob:** score only the first N instances by `instance_id`. Empty = the full set. |
| `repeats` | Independent agent runs per instance (avg-of-k). Empty = benchmark default (1). |
| `workers` | Concurrent instances (agent sandboxes + requests to vLLM). Recipes set `8` for every benchmark in this family. |
| `dataset` / `dataset_revision` | Override the dataset pinned in granite-evals for the selected benchmark. |
| `options` | Space-separated `key=value` benchmark options: `step_limit`, `temperature`, `top_p`, `max_tokens`, `eval_timeout`, `instances` (regex), `patch=gold`. |
| `phase` | `all` generates and scores in one job. `generate` serves the model and writes `generation.json` (`granite_generation`); `score` grades that output dir without a GPU (same `output_dir`, `limit`, `repeats`) and writes `results.json`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | vLLM. |
| `sandbox_cache` | Shared squashfs cache for instance images. Every benchmark in this family sandboxes, so recipes always set this. |
| `hf_home` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. Every dataset in this family is public; no HF token is needed.

`patch=gold` grades the reference patches without a model (set `model_path: none`).
Use it to validate the images, the sandbox and the grading on a new cluster. A few
`swebench-verified` instances (e.g. `psf__requests-1724`) call the live httpbin.org and
can fail under gold.

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

Per-instance trajectories, patches and test logs are kept next to it under
`output/repeat-<k>/<instance_id>/`.

## Developing

`make unit-tests` runs the template contract tests. The runtime has its own tests in
granite-evals. Every step under `steps/eval/` follows this template; the test checks this.
