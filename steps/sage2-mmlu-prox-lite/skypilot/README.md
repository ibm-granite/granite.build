# sage2-mmlu-prox-lite (SkyPilot)

Scores a checkpoint on **MMLU-ProX lite (IBM)** (Sage2 metric: exact match
(custom-extract)). The job serves the model with vLLM and runs lm-evaluation-harness's
own `mmlu_prox_lite_<lang>_<subject>` tasks against it in chat mode
(`local-chat-completions`, `apply_chat_template`, `fewshot_as_multiturn`). Each task is
a 5-shot CoT prompt scored by the `custom-extract` answer regex and `exact_match`. The
data is `li-lab/MMLU-ProX-Lite` at a pinned commit. The job writes one `results.json`
whose value is the mean over languages.

"(IBM)" is read as the 11 Granite 4.x languages MMLU-ProX has: en, de, es, fr, ja, pt,
ar, cs, it, ko and zh. Dutch is on Granite's list but MMLU-ProX has no Dutch. That gives
11 x 588 test questions. Neither the model card nor the blog defines the subset, so
`languages=` changes it.

The code is the external [sage2-evals](https://github.com/laminair/sage2-evals) runtime,
shipped as a prebuilt image. Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/sage2-mmlu-prox-lite
```

## Config (`sage2_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | sage2-evals lmeval image, pinned by tag (`SAGE2_IMAGE_LMEVAL`). |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (11 x 588) | **Smoke knob:** first N examples ordered by (question_id, language), so a small N covers every language. |
| `repeats` | `""` (1) | Repeats of the whole run (each samples afresh at temperature 1.0). |
| `workers` | `64` | Concurrent requests to vLLM. |
| `dataset` / `dataset_revision` | `""` | Override the data pinned in sage2-evals (`li-lab/MMLU-ProX-Lite` at a fixed commit). |
| `options` | `""` | Space-separated `key=value` benchmark options: `languages` (`ibm`, `all` or a comma list), `temperature`, `top_p`, `max_tokens`, `stop` (`task`, `none` or a comma list of up to 4 percent-encoded strings, e.g. `stop=%3C/s%3E,Q:`), `thinking=off`, `max_retries`, `request_timeout`, `answers=gold`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `""` | Unused: this benchmark runs no sandbox. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The data is public; no token is needed.

Generation uses the Granite 4.2 card's thinking-mode settings: temperature 1.0, top_p
0.95, 8192 max tokens, thinking on, and the task's stop strings. NeMo Evaluator's
lm-eval chat protocol (2048 tokens, near-greedy) truncates nearly every thinking trace;
`max_tokens=2048 temperature=0.0000001 top_p=0.9999999` reproduces it. The options
above override these settings, and `results.json` records the generation kwargs each
language ran with.

A request that still fails after its retries scores as wrong and is counted in
`statuses`. A rerun into the same `output_dir` resends only the requests that are not
yet answered, because lm-eval's response cache is kept per repeat.

`answers=gold` answers every question with its reference letter, in the language's
answer phrase. It runs without a model or GPU (set `model_path: none`) and scores 1.0.
Use it to validate the data, the prompts, the extraction and the scoring on a new
cluster.

## Output

`sage2_results` (dataset): the `results.json` file. Declare it on the target:

```yaml
outputs:
  sage2_results:
    uri: "env://{{ binding.path }}"
```

Two things are kept next to it under `output/repeat-<k>/`: per-example lm-eval samples
(`samples/<task>.jsonl`, with prompt, response and extracted answer) and lm-eval's own
results (`lm_eval_results.json`).

## Developing

`make unit-tests` runs the template contract tests. The runtime has its own tests in
sage2-evals. Every `steps/sage2-*` step follows this template; the test checks this.
