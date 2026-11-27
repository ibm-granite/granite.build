# bfcl (SkyPilot)

Scores a checkpoint on the Berkeley Function Calling Leaderboard, **BFCL v4 or BFCL
v3**: one step, one image pair, and `benchmark` picks the version. Either way the
Granite metric is overall_accuracy, BFCL's own "Overall Acc" for that version. The job
serves the model with vLLM, runs the pinned `bfcl-eval` release against it, lets
bfcl-eval's checkers score the answers and takes the value in `results.json` from its
leaderboard formula. No paid APIs are called.

**`bfcl-v4`** is the current leaderboard: `bfcl-eval` 2026.3.23, 22 scoring categories
(non-live and live AST, irrelevance, multi-turn, memory and web search; `memory_vector`
is left out by default), Overall weighted 10/10/10/30/40 over non-live, live,
irrelevance, multi-turn and agentic. Web search goes to IBM's Google search MCP server
(`google_pse_search`) instead of BFCL's SerpAPI/DuckDuckGo backend; `results.json`
records the backend under `web_search_backend`, and scores in the web-search categories
are not directly comparable with the public leaderboard.

**`bfcl-v3`** is the leaderboard before July 2025, as Granite 4.0's model cards report
it: `bfcl-eval` 2025.8.6.2, the last v3 release, with its 17 categories (4,441 entries:
non-live and live AST, relevance and irrelevance, multi-turn) and its Overall, the plain
mean of the live, non-live and multi-turn scores. It needs no web search. The v3
categories inside the v4 release are not a substitute: V4 rewrote most of the
multi-turn data and changed the formula. v3 runs from its own venv in the image
(`/opt/bfcl-v3`), because the two releases cannot share one.

Both send requests in function-calling mode by default: BFCL's
`OpenAICompletionsHandler`, with the checkpoint's own chat template and vLLM's tool
parser doing the formatting, and sampling sent only when set (the checkpoint's
generation_config applies otherwise). The one default is the output budget: when it serves
the model, the step starts vLLM with `--override-generation-config '{"max_new_tokens":16384}'`,
so an FC request that sends no `max_tokens` is capped at 16,384 tokens or the context left
after the prompt, whichever is smaller. That ends a checkpoint that loops instead of
answering (scored as its answer, not a timeout). It is a served default, not a request field,
because vLLM rejects a request whose prompt plus `max_tokens` exceeds the context. The step
applies it only when nothing else sets the output length, and `results.json` records the
effective cap and its source under `max_tokens`:

- not with `base_url` (an endpoint the step does not start: no budget applies);
- not with `handler=legacy-granite` (bfcl-v3), which sends its own budget (below); bfcl-v4
  has no `handler` option and always gets the budget;
- not with an explicit `max_tokens` option, which is sent with every request and wins;
- not when `vllm_args` set the generation config: `--override-generation-config` in any form
  vLLM's parser accepts (dotted keys, underscores, a unique prefix), `--generation-config`, or
  a `--config` file (recorded as unknown: the step does not read what they set);
- not when the checkpoint's `generation_config.json` sets `max_new_tokens` (vLLM applies it;
  the override would replace it, in either direction) or `max_length`, nor when that file
  cannot be read or is not a JSON object (recorded as unknown). An output dir generated
  before the step recorded this reports `not recorded`.

The evidence for 16,384 is v3 only: the largest per-request FC output in the four bfcl-v3
calibration runs (granite-4.0-micro and -h-small) was 8,697 tokens. There is none for
bfcl-v4 or other checkpoints; the v4 recipe's default, granite-4.2-3b, sets no output limit
in its generation_config (only `temperature` 1.0 and `top_p` 0.95). The step's run script
applies it (`granite_evals.benchmarks.bfcl_serving`), not the benchmark, so `granite-evals run`
outside the step needs it as a `--vllm-arg`. The step keeps its decision in the output dir's
`fc_budget.json`, which `results.json` reports under `max_tokens` in either phase.
`bfcl-v3` also has `--option
handler=legacy-granite`: BFCL prompt mode with the Granite 4.0 chat template's prompt
format rendered by the handler, sent as text to the completions endpoint, the format
Granite 4.0 BFCL v3 numbers were produced with. The handler itself parses the tool
calls from the model's `<tool_call>` blocks. It sends `temperature` and `top_p` only
when set, but it always sends `max_tokens`: unless that option is set, BFCL prompt
mode's budget, min(4096, context length − prompt tokens − 2), counted with vLLM's
`/tokenize` and `max_model_len`, so it needs a vLLM endpoint. To compare with the
Granite 4.0 numbers, use `handler=legacy-granite temperature=0.001` (BFCL's own default
temperature). For anything else, use the default handler, `fc`.

An entry whose inference failed for an infrastructure reason (a vLLM 5xx, a refused
connection, a timeout) is never scored as an answer: resume retries it, and an entry that
still fails is left out of the score and the run is marked incomplete (`entries_failed`,
`entries_total`, `incomplete` in `results.json`), as the other infra-dependent families
do. Above `max_failed_frac` (default 0.05), overall or in any one category (as tau applies it
per domain; a category may always lose one entry, since BFCL's smallest hold 16, where one
failure is already 6.25%), or with a category that has no entry left,
generate fails, so the target retry resumes it, and the score phase refuses; its message
names the recovery. A failed generate is retried by the build itself; to score without the
entries that still fail, start a new build with the same `RUN_NAME` and a higher
`max_failed_frac` (options are fixed when a build starts; a failed generate registered no
artifact, so its generate resumes in the same output dir). A score-only run whose generate
target already succeeded cannot retry the entries in place: a retry or `gb build restart` of
the same build reuses that target, and a new build with the same `RUN_NAME` emits the same
artifact URIs, which gbserver refuses to register for a second build ("Same artifact URI
found ... for another build"). Start a new build with a new `RUN_NAME`: a fresh output dir,
generated in full. A higher `max_failed_frac` is no shortcut there, since it needs that new
build too.

The code is the vendored granite-evals runtime (`steps/eval/src`), baked into this
family's image by `Dockerfile.gpu` / `Dockerfile.cpu` one directory up (built by
`steps/eval/Makefile`). Nothing is built from this step at run time.

```yaml
steps:
  - step_uri: space://steps/eval/bfcl
```

## Config

| Field | Default | Purpose |
|---|---|---|
| `benchmark` | `""` (required) | The benchmark id passed to `granite-evals run`: `bfcl-v4` or `bfcl-v3` (see above and `granite-evals list`). |
| `image` | `""` (required) | granite-evals bfcl image, pinned by tag. |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `""` | Full output dir, relative to `$GB_BUILD_WORKDIR`. Empty: composed from `output_root`. |
| `output_root` | `""` | With `output_dir` empty, the run writes `$output_root/$run_name/$benchmark`. |
| `run_name` | basename of `model_path` | Run directory under `output_root`. |
| `limit` | `""` (all: 4,951 for `bfcl-v4`'s default 21 categories, 4,441 for `bfcl-v3`) | **Smoke knob:** first N entries of every category. Memory entries bring their prerequisite conversations. |
| `repeats` | `""` (1) | Both versions are a single run. |
| `workers` | `8` | Concurrent test entries (BFCL `--num-threads`). |
| `dataset` / `dataset_revision` | `""` | Unused: the data ships inside the pinned bfcl-eval package. |
| `options` | `""` | Space-separated `key=value` options: `categories` (comma-separated BFCL categories or groups of that version; default `all_scoring` for v4, `all` for v3), `temperature`, `top_p`, `max_tokens` (default: 16,384 or the context left after the prompt, whichever is smaller, as the served vLLM's default, unless the checkpoint's generation_config sets an output limit or `base_url` is set; with `handler=legacy-granite`, min(4096, context length − prompt tokens − 2); an explicit value is sent with every request and wins), `max_failed_frac` (default `0.05`: infrastructure-failed entries above this fraction, overall or in any one category (each may lose one entry), fail the run; below it they are left out of the score and the run is marked incomplete), `include_input_log`; v4 only: `search_mcp_url`; v3 only: `handler` (`fc`, the default, or `legacy-granite`). |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `tool_call_parser` / `reasoning_parser` | `auto` / `auto` | vLLM parsers; `auto` detects them from the checkpoint, `""` disables one. |
| `vllm_args` | `""` | Space-separated extra `vllm serve` arguments, each passed as `--vllm-arg`. The run script expands them unquoted, so bash word-splits them and applies brace expansion (`{a,b}`) and quote removal: a JSON value with commas or quotes does not arrive intact; use vLLM's dotted form (`--override-generation-config.temperature=0.5`). When they set the generation config (`--override-generation-config`, dotted or not, `--generation-config`, or a `--config` file), the step serves no FC budget (it does not read what they set, and vLLM 0.27.1 keeps the last JSON override; dotted keys merge only with each other), and `results.json` records the cap as unknown: add `max_new_tokens` to your own override to keep one. |
| `base_url` | `""` | An existing OpenAI-compatible endpoint; skips starting vLLM (`served_model_name` must name the model there). |
| `seed` | `0` | Seed for the benchmark's own sampling. |
| `sandbox_cache` | `""` | Unused: this benchmark runs no sandbox. |
| `hf_home` | `""` | Overrides `HF_HOME`. v4's memory_vector category downloads `all-MiniLM-L6-v2` into it. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. A `bfcl-v4` job needs outbound HTTPS to the search MCP server, the web
pages the model fetches, and Hugging Face; a `bfcl-v3` job only Hugging Face (for a hub
model id).

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

BFCL's own result and score files are kept next to it under `output/bfcl/`, with
per-category accuracy in `results.json` under `per_category` and the group scores
under `groups`. `bfcl-v3` also records the request format under `handler` and the side
venv's package versions under `harness_versions`.
