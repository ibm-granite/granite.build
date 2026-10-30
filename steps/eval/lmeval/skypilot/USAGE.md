# lmeval (SkyPilot)

Scores a checkpoint with lm-evaluation-harness's own tasks, served with vLLM inside
the job. The job writes one `results.json` whose value is the benchmark's headline
metric — the mean over languages, for both benchmarks in this family.

The code is the vendored granite-evals runtime (`steps/eval/src`),
baked into this family's image by `Dockerfile.gpu` one directory up (built by
`steps/eval/Makefile`). Nothing is built from this step at run time.

```yaml
steps:
  - step_uri: space://steps/eval/lmeval
```

## Benchmarks in this family

| `benchmark` | Metric | Notes |
|---|---|---|
| `include-44` | accuracy | Cohere Labs' INCLUDE-base-44 (`CohereLabs/include-base-44`, pinned commit): lm-eval's `include_base_44_<lang>_<domain>` tasks, 44 languages x 11 domains (178 tasks, ~500-550 test questions per language) |
| `mmlu-prox-lite` | exact match (custom-extract) | `li-lab/MMLU-ProX-Lite` (pinned commit): lm-eval's `mmlu_prox_lite_<lang>_<subject>` tasks, 5-shot CoT, over the 11 Granite 4.x languages MMLU-ProX has |

### mmlu-prox-lite

Runs the tasks in chat mode (`local-chat-completions`, `apply_chat_template`,
`fewshot_as_multiturn`). Each task is a 5-shot CoT prompt scored by the
`custom-extract` answer regex and `exact_match`.

"(IBM)" is read as the 11 Granite 4.x languages MMLU-ProX has: en, de, es, fr, ja, pt,
ar, cs, it, ko and zh. Dutch is on Granite's list but MMLU-ProX has no Dutch. That gives
11 x 588 test questions. Neither the model card nor the blog defines the subset, so
`languages=` changes it.

Generation uses the Granite 4.2 card's thinking-mode settings: temperature 1.0, top_p
0.95, 8192 max tokens, thinking on, and the task's stop strings. NeMo Evaluator's
lm-eval chat protocol (2048 tokens, near-greedy) truncates nearly every thinking trace;
`max_tokens=2048 temperature=0.0000001 top_p=0.9999999` reproduces it. The options
above override these settings, and `results.json` records the generation kwargs each
language ran with.

### include-44

Runs the tasks in the harness's base mode — no chat template, one loglikelihood
request per option letter, `acc` = the argmax letter against the row's `answer` index
(lm-eval's standard multiple-choice protocol, what `lm_eval --model local-completions`
runs on these tasks). The prompt is the task's per-domain description, the question
with its four options and `Answer:`; the scored continuations are `" A"`-`" D"`.

The headline is the mean over languages of each language's accuracy; the languages
have different sizes, so the micro average is also reported (`micro`), and `per_domain`
breaks the same questions down by domain. With `repeats: k` the headline is
pass@1[avg-of-k] and `details.pass_at_k` adds pass@k (an example correct in any
repeat), both averaged over languages the same way.

## Config

| Field | Default | Purpose |
|---|---|---|
| `benchmark` | `""` (required) | The benchmark id passed to `granite-evals run`. One of the ids in the table above; see `granite-evals list`. |
| `image` | `""` (required) | granite-evals lmeval image, pinned by tag (`GRANITE_EVALS_IMAGE_LMEVAL`). |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (full set) | **Smoke knob:** first N examples. mmlu-prox-lite orders by (question_id, language), so a small N covers every language; include-44 orders by (in-task index, language), so a small N covers the first languages' tasks (LIMIT=5 sees Albanian's five). |
| `repeats` | `""` (1) | Repeats of the whole run. mmlu-prox-lite samples afresh at temperature 1.0; include-44 reports pass@1[avg-of-k] with `details.pass_at_k`. |
| `workers` | `64` | Concurrent requests to vLLM. |
| `dataset` / `dataset_revision` | `""` | Override the data pinned in granite-evals for the selected benchmark (`li-lab/MMLU-ProX-Lite` or `CohereLabs/include-base-44`). |
| `options` | `""` | Space-separated `key=value` benchmark options. Both: `answers=gold`, `max_retries`, `request_timeout`, `languages`. mmlu-prox-lite: `languages` takes `ibm`, `all` or a comma list; `temperature`, `top_p`, `max_tokens`, `stop` (`task`, `none` or a comma list of up to 4 percent-encoded strings, e.g. `stop=%3C/s%3E,Q:`), `thinking=off`. include-44: `languages` takes a comma list of the language slugs. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `""` | Unused: these benchmarks run no sandbox. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The data is public; no token is needed.

A request that still fails after its retries scores as wrong and is counted in
`statuses`. A rerun into the same `output_dir` resends only the requests that are not
yet answered, because lm-eval's response cache is kept per repeat.

`answers=gold` runs without a model or GPU (set `model_path: none`) and scores 1.0:
mmlu-prox-lite answers every question with its reference letter, in the language's
answer phrase; include-44 answers each option letter with its reference loglikelihood
(the answer's letter 0.0, the others -100). Use it to validate the data, the prompts,
the extraction and the scoring on a new cluster.

## Output

`granite_results` (dataset): the `results.json` file. Declare it on the target:

```yaml
outputs:
  granite_results:
    uri: "env://{{ binding.path }}"
```

Two things are kept next to it under `output/repeat-<k>/`: per-example lm-eval samples
(`samples/<task>.jsonl`, with prompt, response and extracted answer) and lm-eval's own
results (`lm_eval_results.json`).
