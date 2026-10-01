# sage2-birdbench (SkyPilot)

Scores a checkpoint on **BirdBench**, the BIRD text-to-SQL dev set (Sage2 metric:
pass@1 execution match). The job serves the model with vLLM and follows NeMo-Skills'
`birdbench` protocol:

- the prompt is `generic/text_to_sql` with the database's SQL dump as schema context, and no evidence;
- the answer is the last ```sql block;
- the predicted and gold queries are executed on the question's SQLite database;
- a question is correct when the two row sets are equal.

Execution is read-only, and both queries share a 30 s budget. The data is the official
BIRD `dev.zip` (`dev_20240627`, 1,534 questions over 11 databases), pinned by sha256
and downloaded once into `$HF_HOME/sage2-data`.

The code is the external [sage2-evals](https://github.com/laminair/sage2-evals) runtime,
shipped as a prebuilt image. Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/sage2-birdbench
```

## Config (`sage2_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | sage2-evals bird image, pinned by tag. |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all 1,534) | **Smoke knob:** first N questions by `question_id`. |
| `repeats` | `""` (1) | Independent samples per question (avg-of-k). |
| `workers` | `32` | Concurrent questions. |
| `dataset` / `dataset_revision` | `""` | A local extracted `dev_20240627` dir overrides the pinned dev.zip. |
| `options` | `""` | Space-separated `key=value` options: `temperature`, `top_p`, `max_tokens` (default: the checkpoint's generation_config), `evidence=true` (prefix BIRD's external knowledge), `timeout` (s, default 30), `sql=gold`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `/proj/granite-build/g4os/sage2/enroot-cache` | Unused by this benchmark. |
| `hf_home` | `""` | Overrides `HF_HOME`, and with it the dev-set cache. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none.

`sql=gold` scores the gold SQL against itself without a model (set `model_path: none`).
Use it to validate the data, the databases and the scoring. A few gold queries run
longer than the 30 s budget and score as timeouts.

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

Per-question generations, extracted SQL and execution status are kept next to it
under `output/repeat-<k>/<question_id>.json`.
