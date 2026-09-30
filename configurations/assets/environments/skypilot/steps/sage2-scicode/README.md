# sage2-scicode (SkyPilot)

Scores a checkpoint on **SciCode** (Sage2 metric: pass@1[avg-of-2] subtask accuracy):
the 65 test problems, 288 evaluated subtasks, with background. The job serves the
model with vLLM and runs NVIDIA NeMo-Skills (pinned commit) against it: its multi-step
SciCode generation, then its evaluator in NeMo-Skills' local sandbox server. The
sandbox runs from the image's pinned Python 3.10 scientific env (numpy, scipy, sympy,
h5py, matplotlib) with the SciCode test data checked by sha256. It writes one
`results.json`.

The code is the external [sage2-evals](https://github.com/laminair/sage2-evals) runtime,
shipped as a prebuilt image (the `nemoskills` family). Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/sage2-scicode
```

## Config (`sage2_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | sage2-evals nemoskills image, pinned by tag. |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all 65) | **Smoke knob:** the first N problems. |
| `repeats` | `""` (2) | Independent generations per problem (avg-of-k). |
| `workers` | `64` | Concurrent requests to vLLM. |
| `dataset` / `dataset_revision` | `""` | Override the dataset pinned in sage2-evals (`SciCode1/SciCode` at a fixed commit). |
| `options` | `""` | Space-separated `key=value` benchmark options: `temperature`, `top_p`, `top_k`, `max_tokens`, `ns.<key>=<value>` (raw NeMo-Skills overrides), `answers=gold`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `/proj/granite-build/g4os/sage2/enroot-cache` | Unused here (shared config contract). |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The dataset is public; no HF token is needed. No paid API is used.
Sampling defaults to the model's `generation_config.json`; Granite's thinking stays on.

`answers=gold` serves the reference code of the validation (`dev`) problems, the only
ones that publish it, step by step, without a model (set `model_path: none`). Use it to
validate the sandbox and grading on a new cluster. `results.json` also records the
sandbox env's package versions and any dependency imports that fail there.

## Output

`sage2_results` (dataset): the `results.json` file. Declare it on the target:

```yaml
outputs:
  sage2_results:
    uri: "env://{{ binding.path }}"
```

NeMo-Skills' generations and per-subtask grades are kept next to it under
`output/generation/output-rs<k>.jsonl`.

## Developing

`make unit-tests` runs the template contract tests. The runtime has its own tests in
sage2-evals. Every `steps/sage2-*` step follows this template; the test checks this.
