# sage2-swebench-pro (SkyPilot)

Scores a checkpoint on **SWE Bench Pro** (the V2 public set, 642 tasks; Sage2 metric:
pass@1[avg-of-3] resolve rate). The job serves the model with vLLM and follows Scale's V2
locked protocol ([SWE-bench_Pro-os](https://github.com/scaleapi/SWE-bench_Pro-os) v2.0.0):
mini-swe-agent with the protocol's tool-calling config works each task's `instruction.md`
in one enroot sandbox of the task image, with a 50-minute budget. Its staged `git diff`
is replayed in a fresh sandbox and graded by the task's own `tests/test.sh` (reward 1 =
resolved). Writes one `results.json`.

The code is the external [sage2-evals](https://github.com/laminair/sage2-evals) runtime,
shipped as a prebuilt image. Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/sage2-swebench-pro
```

## Config (`sage2_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | sage2-evals swebench image, pinned by tag. |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all 642) | **Smoke knob:** first N instances by `instance_id`. |
| `repeats` | `""` (3) | Independent agent runs per instance (avg-of-k). |
| `workers` | `8` | Concurrent instances. |
| `dataset` / `dataset_revision` | `""` | Override the dataset pinned in sage2-evals (upstream `ScaleAI/SWE-bench_Pro` at a fixed commit). |
| `options` | `""` | Space-separated `key=value` benchmark options: `subset=hard` (HARD-51), `agent_timeout` (s, default 2940), `step_limit`, `temperature`, `top_p`, `max_tokens`, `eval_timeout` (s, default 3000), `instances` (regex), `patch=gold`, `check=data`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `/proj/granite-build/g4os/sage2/enroot-cache` | Shared squashfs cache for task images (`ghcr.io/scaleapi/swe-bench_pro-v2`, ~1 GB each compressed). |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The dataset is public; no HF token is needed.

`patch=gold` grades the reference patches without a model (set `model_path: none`).
Use it to validate the images, the sandbox and the grading on a new cluster. The
reference patch is each task's `solution/gold_patch.diff` (what Harbor's oracle applies);
Scale's V2 release gate is 642/642 under the oracle. `check=data` runs no sandbox at all:
it verifies every task file against the pinned `SHA256SUMS` and that each task image
exists on ghcr.io.

Deviations from Scale's harness: the agent sandbox has network access (the protocol
runs the agent offline), the verifier runs in enroot instead of Harbor, and an empty
patch is scored unresolved without running the verifier (the release gate fails every
task on an empty patch).

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
