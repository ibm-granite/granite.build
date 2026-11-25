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
| `lm-eval` | macro mean of the tasks' primary metrics | The generic bridge: any task or group in the pinned lm-evaluation-harness (14k+ names), chosen by `options` (`tasks=hellaswag,arc_easy`). No fixed protocol — see below. |

### mmlu-prox-lite

Runs the tasks in chat mode (`local-chat-completions`, `apply_chat_template`,
`fewshot_as_multiturn`). Each task is a 5-shot CoT prompt scored by the
`custom-extract` answer regex and `exact_match`.

"(IBM)" is read as the 11 Granite 4.x languages MMLU-ProX has: en, de, es, fr, ja, pt,
ar, cs, it, ko and zh. Dutch is on Granite's list but MMLU-ProX has no Dutch. That gives
11 x 588 test questions. Neither the model card nor the blog defines the subset, so
`languages=` changes it.

Generation uses the Granite 4.2 card's thinking-mode settings: temperature 1.0, top_p
0.95, thinking on, and the task's stop strings, with a 32768-token budget (the
card's 8192 runs out mid-thought on every request for granite-4.2-3b; `max_tokens=8192`
restores it). NeMo Evaluator's
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
(the primary metric for every task, or one exact results key `<metric>,<filter>` such
as `exact_match,strict-match`; default: each task's own, its config's first
`metric_list` entry).

A task's metric is read as `<metric>,none`, then bare `<metric>`, then
`<metric>,<filter>` for each of the task's `filter_list` names in yaml order (gsm8k's
`exact_match` reads `exact_match,strict-match`); `per_task` records the `key` read. A
task whose results row has none of these fails the run, naming the keys it has — a
missing metric is never written as a null score.

`auto` picks each task's protocol from its `output_type`: generation tasks run in
chat mode (`local-chat-completions`, `apply_chat_template`, `fewshot_as_multiturn`,
like mmlu-prox-lite), multiple-choice and loglikelihood tasks in base mode (like
include-44). A task list that mixes them runs as two passes over the same output dir
and cache.

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

Two recipes ship as templates: `recipes/evals/lmeval/hellaswag` (base mode —
multiple-choice loglikelihood, the leaderboard-standard `metric=acc_norm`) and
`recipes/evals/lmeval/gsm8k` (chat mode — 5-shot generation,
`metric=exact_match,strict-match`).
A third, `recipes/evals/lmeval/gpqa-diamond`, is the warm-cache example: the
dataset is Hub-gated, so the recipe runs `hf_offline` against a cache warmed by
`make -C steps/eval prefetch` (see steps/eval/README.md, "The warm HF cache").

Resume is the curated benchmarks' care: the harness response cache
(`repeat-<k>/lm_cache`) keeps every finished request, so a restarted run only sends
the missing ones.

## Config

| Field | Default | Purpose |
|---|---|---|
| `benchmark` | `""` (required) | The benchmark id passed to `granite-evals run`. One of the ids in the table above; see `granite-evals list`. |
| `image` | `""` (required) | granite-evals lmeval image, pinned by tag (`GRANITE_EVALS_IMAGE_LMEVAL`). |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `""` | Full output dir, relative to `$GB_BUILD_WORKDIR`. Empty: composed from `output_root`. |
| `output_root` | `""` | With `output_dir` empty, the run writes `$output_root/$run_name/$benchmark`; a selection option (`tasks=`, `domains=`, `service=`, `categories=`, `benchmark=`) suffixes it as one path segment, e.g. `…/lm-eval-hellaswag` (see steps/eval/README.md, The generic bridges). |
| `run_name` | basename of `model_path` | Run directory under `output_root`. |
| `limit` | `""` (full set) | **Smoke knob:** first N examples. mmlu-prox-lite orders by (question_id, language), so a small N covers every language; include-44 orders by (in-task index, language), so a small N covers the first languages' tasks (LIMIT=5 sees Albanian's five); lm-eval uses the harness's own per-task first N. |
| `repeats` | `""` (1) | Repeats of the whole run. mmlu-prox-lite samples afresh at temperature 1.0; include-44 reports pass@1[avg-of-k] with `details.pass_at_k`; lm-eval reports the mean over repeats (no pass@k). |
| `workers` | `64` | Concurrent requests to vLLM. |
| `dataset` / `dataset_revision` | `""` | Override the data pinned in granite-evals for the selected benchmark (`li-lab/MMLU-ProX-Lite` or `CohereLabs/include-base-44`); `lm-eval` rejects them — its tasks pin their own. |
| `options` | `""` | Space-separated `key=value` benchmark options. Both: `answers=gold`, `max_retries`, `request_timeout`, `languages`. mmlu-prox-lite: `languages` takes `ibm`, `all` or a comma list; `temperature`, `top_p`, `max_tokens`, `stop` (`task`, `none` or a comma list of up to 4 percent-encoded strings, e.g. `stop=%3C/s%3E,Q:`), `thinking=off`. include-44: `languages` takes a comma list of the language slugs. lm-eval: `tasks` (required), `mode`, `num_fewshot`, `gen_kwargs`, `metric` (see above). |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `tool_call_parser` / `reasoning_parser` | `auto` / `auto` | vLLM parsers; `auto` detects them from the checkpoint, `""` disables one. |
| `vllm_args` | `""` | Space-separated extra `vllm serve` arguments, each passed as `--vllm-arg`. |
| `base_url` | `""` | An existing OpenAI-compatible endpoint; skips starting vLLM (`served_model_name` must name the model there). |
| `seed` | `0` | Seed for the benchmark's own sampling. |
| `sandbox_cache` | `""` | Unused: these benchmarks run no sandbox. |
| `hf_home` | `""` | Overrides `HF_HOME`. |
| `hf_offline` | `""` | Warm-cache mode: `HF_HUB_OFFLINE=1` — every hub read (datasets; the model if it is a hub id) resolves from `hf_home`'s cache, never the hub. Warm it first (`HF_TOKEN=... make -C steps/eval prefetch HF_HOME=<same cache>`; gated datasets need a token with access) and share it across runs; a cold cache fails fast. |

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
