# lmeval (SkyPilot)

Scores a checkpoint with lm-evaluation-harness's own tasks, served with vLLM inside
the job. The job writes one `results.json` whose value is the benchmark's headline
metric.

## Benchmarks in this family

| `benchmark` | Metric | Notes |
|---|---|---|
| `mmlu-prox-lite` | exact match (custom-extract) | `li-lab/MMLU-ProX-Lite` (pinned commit): lm-eval's `mmlu_prox_lite_<lang>_<subject>` tasks, 5-shot CoT, over the 11 Granite 4.x languages MMLU-ProX has |
| `include-44` | accuracy | Cohere Labs' INCLUDE-base-44 (`CohereLabs/include-base-44`, pinned commit): lm-eval's `include_base_44_<lang>_<domain>` tasks, 44 languages x 11 domains (178 tasks, ~500-550 test questions per language) |
| `lm-eval` | macro mean of the tasks' primary metrics | The generic bridge: any task or group in the pinned lm-evaluation-harness (14k+ names), chosen by `options` (`tasks=hellaswag,arc_easy`). No fixed protocol — see below. |

### mmlu-prox-lite

Runs the tasks in chat mode (`local-chat-completions`, `apply_chat_template`,
`fewshot_as_multiturn`). Each task is a 5-shot CoT prompt scored by the
`custom-extract` answer regex and `exact_match`.

"(IBM)" is read as the 11 Granite 4.x languages MMLU-ProX has: en, de, es, fr, ja, pt,
ar, cs, it, ko and zh. Dutch is on Granite's list but MMLU-ProX has no Dutch. That gives
11 x 588 test questions. Neither the model card nor the blog defines the subset, so
`languages=` changes it.

### include-44

Runs the tasks in the harness's base mode — no chat template, one loglikelihood
request per option letter, `acc` = the argmax letter against the row's `answer` index
(lm-eval's standard multiple-choice protocol). The prompt is the task's per-domain
description, the question with its four options and `Answer:`; the scored
continuations are `" A"`-`" D"`.

The headline is the mean over languages of each language's accuracy; the languages
have different sizes, so the micro average is also reported (`micro`), and `per_domain`
breaks the same questions down by domain. With `repeats: k` the headline is
pass@1[avg-of-k] and `details.pass_at_k` adds pass@k (an example correct in any
repeat), both averaged over languages the same way.

### lm-eval (generic bridge)

`benchmark: lm-eval` runs any task or group the pinned lm-evaluation-harness knows,
chosen by the recipe's `options` — a new task is config, not code:

```yaml
benchmark: "lm-eval"
options: "tasks=hellaswag,arc_easy mode=auto metric=acc_norm"
```

Options: `tasks` (required, comma-separated task/group ids; unknown names fail fast
before vLLM starts, with near-miss suggestions), `mode` (`auto` — the default —
`chat` or `base`), `num_fewshot`, `gen_kwargs` (comma-separated `key=value` overrides
of the tasks' generation kwargs, values percent-decoded like `stop=`), and `metric`
(the primary metric for every task; default: each task's own, its config's first
`metric_list` entry).

`auto` picks each task's protocol from its `output_type`: generation tasks run in
chat mode (like mmlu-prox-lite), multiple-choice and loglikelihood tasks in base mode
(like include-44). A task list that mixes them runs as two passes over the same output
dir and cache. A base model with no chat template needs `mode=base` spelled out:
`auto` would hand a generation task to the chat protocol.

The headline is the macro mean over the resolved tasks of each task's primary
metric; `n` sums their sizes, `per_task` keeps each task's numbers, and
`repeat-<k>/lm_eval_results.json` preserves the harness's full output. Groups in
`tasks` are expanded by the harness; only leaf tasks are averaged. `limit` is
lm-eval's own (the first N docs per task), and `repeats` runs the whole set k times
and averages — no pass@k here, since per-example scores have no meaning for every
metric family. `dataset`/`dataset_revision` do not apply (every task names its own
dataset; the class pin is the harness repo at the release the image pins). One
caveat the curated benchmarks do not have: the harness's task yamls name their
datasets but pin no revisions, so a bridge run reads each dataset at the hub's
current snapshot.

The composed run dir is `$output_root/$run_name/lm-eval-<tasks>` (the tasks,
commas to dashes, suffix the benchmark id so two bridge recipes scoring one model
don't share an output dir).

The code is the vendored granite-evals runtime (`steps/eval/src`),
baked into this family's image by `Dockerfile.gpu` one directory up (built by
`steps/eval/Makefile`). Nothing is built from this step at run time.

```yaml
steps:
  - step_uri: space://steps/eval/lmeval
```

## Config

| Field | Default | Purpose |
|---|---|---|
| `benchmark` | `""` (required) | The benchmark id passed to `granite-evals run`: `mmlu-prox-lite`, `include-44` or `lm-eval` (see above); `granite-evals list` names them. |
| `image` | `""` (required) | granite-evals lmeval image, pinned by tag (`GRANITE_EVALS_IMAGE_LMEVAL`). |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `""` | Full output dir, relative to `$GB_BUILD_WORKDIR`. Empty: composed from `output_root`. |
| `output_root` | `""` | With `output_dir` empty, the run writes `$output_root/$run_name/$benchmark` (the `lm-eval` bridge suffixes the tasks: `…/lm-eval-hellaswag`). |
| `run_name` | basename of `model_path` | Run directory under `output_root`. |
| `limit` | `""` (full set) | **Smoke knob:** first N examples. mmlu-prox-lite orders by (question_id, language), so a small N covers every language; include-44 orders by (in-task index, language), so a small N covers the first languages' tasks; lm-eval uses the harness's own per-task first N. |
| `repeats` | `""` (1) | Repeats of the whole run. mmlu-prox-lite samples afresh at temperature 1.0; include-44 reports pass@1[avg-of-k] with `details.pass_at_k`; lm-eval reports the mean over repeats (no pass@k). |
| `workers` | `64` | Concurrent requests to vLLM. |
| `dataset` / `dataset_revision` | `""` | Override the data pinned in granite-evals (`li-lab/MMLU-ProX-Lite`, `CohereLabs/include-base-44`, each at a fixed commit). Not available on `lm-eval`: every task names its own dataset. |
| `options` | `""` | Space-separated `key=value` benchmark options. Curated: `languages` (`ibm`, `all` or a comma list), `temperature`, `top_p`, `max_tokens`, `stop` (`task`, `none` or a comma list of up to 4 percent-encoded strings, e.g. `stop=%3C/s%3E,Q:`), `thinking=off`, `max_retries`, `request_timeout`, `answers=gold`. `lm-eval`: `tasks` (required), `mode`, `num_fewshot`, `gen_kwargs`, `metric`. |
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

`granite_results` (dataset): the `results.json` file. Declare it on the target:

```yaml
outputs:
  granite_results:
    uri: "env://{{ binding.path }}"
```

Two things are kept next to it under `output/repeat-<k>/`: per-example lm-eval samples
(`samples/<task>.jsonl`, with prompt, response and extracted answer) and lm-eval's own
results (`lm_eval_results.json`).
