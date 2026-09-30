# sage2-arena-hard-v2 (SkyPilot)

Scores a checkpoint on **Arena-Hard-V2** (Sage2 metric: win rate).
The job serves the model with vLLM and runs NVIDIA NeMo-Skills' own arena-hard-v2 pipeline
against it, locally: data preparation (every upstream read pinned), prompt, generation,
answer extraction and metrics, plus NeMo-Skills' pairwise arena judge (each answer judged against the
baseline in both orders; win rate from its bootstrapped Bradley-Terry fit). It writes one `results.json`.

The code is the external [sage2-evals](https://github.com/laminair/sage2-evals) runtime,
shipped as a prebuilt image. Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/sage2-arena-hard-v2
```

## Config (`sage2_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | sage2-evals nemoskills image, pinned by tag (`SAGE2_IMAGE_NEMOSKILLS`). |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all 750 prompts) | **Smoke knob:** first N examples of the pinned data. |
| `repeats` | `""` (1) | Independent generations per example (avg-of-k). |
| `workers` | `32` | Concurrent requests to vLLM. |
| `dataset` / `dataset_revision` | `""` | Override the data pinned in sage2-evals (arena-hard-auto v2.0 questions and baseline answers, pinned by commit and sha256). |
| `options` | `""` | Space-separated `key=value` benchmark options: `temperature`, `top_p`, `top_k`, `max_tokens`, `ns.<key>=<value>`, `judge_model` (default `aws/claude-sonnet-5`; `self` = the served model), `judge_base_url` (default IBM LiteLLM), `judge_api_key_env` (default `SAGE2_JUDGE_API_KEY`), `judge_max_tokens`, `judge_workers`, `judge.<key>=<value>`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `""` | Unused: this benchmark runs no sandbox. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The data is public. The judge needs its key in the env var `judge_api_key_env` names (default `SAGE2_JUDGE_API_KEY`); every judge call is metered and capped by sage2-evals.

Sampling defaults to the checkpoint's `generation_config` (NeMo-Skills' own default is
greedy); `results.json` records the sampling used.

The judge is `aws/claude-sonnet-5` on IBM LiteLLM, not arena-hard-auto's official
GPT-4.1, and there is no style control; `results.json` records both judges, and the
judge's token usage (`details.judge_usage`) and cost (`details.api_spend`).
`judge_model=self` judges with the served model: an unpaid pipeline check only.

## Output

`sage2_results` (dataset): the `results.json` file. Declare it on the target:

```yaml
outputs:
  sage2_results:
    uri: "env://{{ binding.path }}"
```

The prepared data, per-repeat NeMo-Skills outputs (`generation/output-rs<k>.jsonl`, `judged/`, `judge/usage.jsonl`)
and their logs are kept next to it.
