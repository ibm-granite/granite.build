# nemoskills (SkyPilot)

Scores a checkpoint on one of the granite-evals **nemoskills** family's benchmarks:
each run through NVIDIA NeMo-Skills' own pipeline, locally (no Slurm, no NeMo-Run) —
data preparation (every upstream read pinned), prompt, generation, answer extraction
and metrics. `benchmark` selects which one.

The code is the vendored granite-evals runtime (`steps/eval/src`),
baked into this family's image by `Dockerfile.gpu` / `Dockerfile.cpu` one directory up (built by
`steps/eval/Makefile`). Nothing is built from this step at run time.

```yaml
steps:
  - step_uri: space://steps/eval/nemoskills
    config:
      benchmark: gpqa
```

## Benchmarks in this family

| `benchmark` | Metric | Notes |
|---|---|---|
| `aa-lcr` | pass@1 judge correct | `ArtificialAnalysis/AA-LCR`, 100 questions over ~100k-token documents; prompts reach ~130k tokens — set `max_model_len` to ~160k+ |
| `aime24` | pass@1 symbolic correct | |
| `aime25` | pass@1 symbolic correct | |
| `aime26` | pass@1 symbolic correct | `MathArena/aime_2026` (30 problems) |
| `arena-hard-v2` | win rate | NeMo-Skills' pairwise arena judge |
| `browsecomp` | mean reward | OpenAI simple-evals' 1266 questions (encrypted, pinned by sha256); web search via the IBM Google PSE MCP server (no key), page fetch; benchmark data itself is refused on fetch |
| `browsecomp-plus` | mean reward | Tevatron's BrowseComp+ (`Tevatron/browsecomp-plus`, 830 queries sourced from BrowseComp, canary-encrypted and decrypted at run time); the same search/fetch run as `browsecomp` with Tevatron's query template and grader |
| `critpt` | challenge accuracy | 70 physics research challenges, two-turn generation; graded by Artificial Analysis's CritPt API |
| `critpt-tools` | challenge accuracy | as `critpt`, plus a stateful python tool each turn in an enroot sandbox (network off) |
| `flores200` | en→xx COMET (XCOMET-XXL) | `openlanguagedata/flores_plus` devtest (1012 segments per language), default languages ar/cs/de/es/fr/it/ja/ko/nl/pt/zh (BCP-47 codes; `options: languages=` overrides); scored exactly as `wmt24pp` — XCOMET-XXL (`Unbabel/XCOMET-XXL`, gated, CC-BY-NC-SA-4.0) in the image's separate COMET env, phase `score` needs the gated model's access |
| `gpqa` | pass@1 symbolic correct | `Idavidrein/gpqa` (gpqa_diamond), 198 questions; **gated** — needs `HF_TOKEN` of an account that accepted its terms |
| `hle` | pass@1 judge correct | NeMo-Skills' `text` split of `cais/hle` (no-image questions); the official HLE judge prompt |
| `hle-tools` | pass@1 judge correct | as `hle`, plus a stateful python tool (enroot sandbox, network off) and web search via the IBM Google PSE MCP server |
| `hmmt-feb25` | pass@1 symbolic correct | |
| `human-eval` | pass@1 accuracy | `openai/openai_humaneval` (164 problems, MIT); scored by a port of HumanEval's own execution harness (openai/human-eval @ 37c4dd63) in an enroot sandbox (python:3.13-bookworm, network off) |
| `human-eval-tools` | pass@1 accuracy | as `human-eval`, plus a stateful python tool (enroot sandbox) for scratch work while writing the function |
| `imo-answerbench` | pass@1 judge correct | google-deepmind/superhuman's IMO AnswerBench (400 short-answer problems, CSV fetched from a commit-pinned URL); NeMo-Skills' IMO AnswerBench judge prompt |
| `livecodebench-v6` | pass@1 accuracy | 454 problems, release v6 (`test_v6_2408_2505`, contests 2024-08 to 2025-05); graded with the pinned `livecodebench` package in the job container |
| `math-500` | pass@1 symbolic correct | OpenAI's MATH-500 (the prm800k test subset, 500 problems), fetched from a commit-pinned URL at run time |
| `mbpp` | pass@1 accuracy | `google-research-datasets/mbpp` config `sanitized` (257 manually-verified problems, CC-BY-4.0); scored by the same ported HumanEval execution harness in an enroot sandbox (network off) |
| `mbpp-tools` | pass@1 accuracy | as `mbpp`, plus a stateful python tool (enroot sandbox) for scratch work while writing the function |
| `mmlu-pro` | 5-shot CoT symbolic correct | |
| `omniscience` | pass@1 judge correct | `ArtificialAnalysis/AA-Omniscience-Public`, 600 questions; judge: A correct, B incorrect, C partial, D not attempted |
| `omniscience-hallucination` | pass@1 judge_omni_hallucination (lower is better) | same pipeline as `omniscience`; hallucination rate = incorrect / (incorrect + partial + not attempted). `options: generations_from=<omniscience output dir>` re-scores that run instead of generating again |
| `ruler-64k` / `-128k` / `-256k` / `-512k` / `-1m` | accuracy (NeMo-Skills' `ruler_score`, mean over 13 tasks) | data generated for the model's own tokenizer (RULER source data baked into the image, sha256-checked), 100 samples/task; served at the matching `max_model_len` |
| `scicode` | pass@1 subtask accuracy | 65 problems / 288 subtasks, with background; graded in NeMo-Skills' local sandbox server (pinned Python 3.10 scientific env: numpy, scipy, sympy, h5py, matplotlib) |
| `wmt24pp` | en→xx COMET (XCOMET-XXL) | `google/wmt24pp`, default languages de_DE/es_MX/fr_FR/it_IT/ja_JP (998 segments each; `options: languages=` overrides); XCOMET-XXL (`Unbabel/XCOMET-XXL`, gated, CC-BY-NC-SA-4.0) runs in the image's separate COMET env — phase `score` needs the gated model's access |
| `nemoskills` | ns's own headline metric (`symbolic_correct` unless `metric=` names another ns metric key) | The generic bridge: `options: benchmark=<name>` (required; a dataset directory name under NeMo-Skills' `nemo_skills/dataset` at the pinned checkout, e.g. `benchmark=math-500`) runs that benchmark with ns's own defaults — its prepare, prompts and metrics — where the curated ids above freeze Granite's protocol on the same machinery. `dataset` is required and pins the data, in one of three forms: a local path to ns's prepared rows (a dir holding `<split>.jsonl`, or one `.jsonl` file; ns's prepare does not run); a hub id with `dataset_revision` a 40-hex commit (the HF dataset ns's `prepare.py` reads, pinned to it); or an `http(s)` URL with `dataset_revision: sha256:<hex>` (the file ns's `prepare.py` downloads, checked against the hash). In the prepare, `datasets.load_dataset` on hub ids, `hf_hub_download`, `snapshot_download` and `urllib.request.urlretrieve` are pinned (the curated machinery), and the bridge adds `urllib.request.urlopen` and a `wget` command (the images have none; its options are parsed, header values forwarded and never printed), both served only for the pinned URL, and refuses remote `data_files`/`data_dir` for the local `load_dataset` builders (`json`, `csv`, `parquet`, …). A checked read the recipe does not pin fails the prepare, naming only its http(s) URL (an uncaught prepare error prints with secret env values and Authorization values redacted), and the run fails unless the reads include the pin and the guards recorded that they loaded in the prepare (fail closed). The base class's own refusals print the full URL; the bridge re-raises them with every URL cut to scheme://host/path. Not intercepted: `requests`, `httpx`, `aiohttp`, `fsspec` reads (`hf://` or `https://` paths opened directly), the `wget` Python package, `git`, `curl` and other subprocesses, and remote `data_files` given to a pinned hub id (the revision pin covers the repo, not the URL) — a prepare that fetches that way can read beyond the pin, so give it a local `dataset`. The run's provenance records each checked read. `split=` picks which prepared `<split>.jsonl` the run reads (default: the ns module's `EVAL_SPLIT`, else `test`). The sampling and judge options, the phases and resume are the curated ids' own. Run dir: `$output_root/$run_name/nemoskills@benchmark-<name>@dataset-<pin>@dataset_revision-<pin>`, every value escaped into one path segment; a name over 200 bytes is cut to 180 plus `@h-` and 16 hex of its sha256, so `recipes/evals/nemoskills/ns-math-500` (a URL and a sha256 pin) writes `nemoskills@benchmark-math-500@dataset-https+3A+2F+2Fgithub.com+2Fopenai+2Fprm800k…@h-<hash>`. |

See `granite-evals list` for the current, authoritative set and sizes.

## Config

| Field | Purpose |
|---|---|
| `benchmark` | Required. One of the ids in the table above. |
| `image` | Required. The granite-evals `nemoskills` image, pinned by tag (`GRANITE_EVALS_IMAGE_NEMOSKILLS`). |
| `model_path` | Required. Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | Name vLLM serves under. Empty = basename of `model_path`. |
| `output_dir` | Full output dir, relative to `$GB_BUILD_WORKDIR`. Empty: composed from `output_root`. |
| `output_root` | With `output_dir` empty, the run writes `$output_root/$run_name/$benchmark`; a run that selects (`benchmark=`, `service=`, `task_suite=`, `domains=`, `categories=`, `tasks=`, `exclude=`, `instances=`, `split=`, or a dataset-pinned bridge's `dataset`) writes `…/<benchmark>@<key>-<value>…` instead, every selection value escaped into one Hydra-safe path segment, e.g. `…/lm-eval@tasks-hellaswag` (see steps/eval/README.md, The generic bridges). |
| `run_name` | Run directory under `output_root`; empty = basename of `model_path`. |
| `limit` | **Smoke knob:** score only the first N examples of the pinned/prepared data. Empty = the benchmark's full set. |
| `repeats` | Independent generations per example (avg-of-k). Empty = benchmark default (1). |
| `workers` | Concurrent requests to vLLM. Recipes set a per-benchmark default (32 for most; 64 for gpqa/livecodebench-v6/scicode; 128 for mmlu-pro). |
| `dataset` / `dataset_revision` | Override the dataset pinned in granite-evals for the selected benchmark. A hub id, a prepared `.jsonl` or a directory holding `<split>.jsonl` overrides it, where the benchmark supports one. |
| `options` | Space-separated `key=value` benchmark options. Common: `temperature`, `top_p`, `top_k`, `max_tokens`, `ns.<key>=<value>` (any NeMo-Skills generation override), `answers=gold` (reference answers through the same grading; `model_path` may be `none`). Benchmark-specific options exist for several of the above — see each benchmark's own granite-evals docs. |
| `phase` | `all` generates and scores in one job. `generate` serves the model and writes `generation.json` (`granite_generation`); `score` grades that output dir without a GPU (same `output_dir`, `limit`, `repeats`) and writes `results.json`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | vLLM. Several RULER sizes and `aa-lcr` need a non-default `max_model_len` — see the table above. |
| `tool_call_parser` / `reasoning_parser` | vLLM parsers (default `auto`: detected from the checkpoint; `""` disables one). |
| `vllm_args` | Space-separated extra `vllm serve` arguments, each passed as `--vllm-arg`. |
| `base_url` | An existing OpenAI-compatible endpoint; skips starting vLLM (`served_model_name` must name the model there). |
| `seed` | Seed for the benchmark's own sampling (default `0`). |
| `sandbox_cache` | Shared squashfs cache. Only `critpt-tools`, `hle-tools`, `human-eval*`, `livecodebench-v6`, `mbpp*`, `ruler-*` and `scicode` in this family sandbox; set this for those, leave empty for the rest. |
| `hf_home` | Overrides `HF_HOME`. |
| `hf_offline` | `1`/`true`/`yes`/`on` (empty/`0`/`false`/`no`/`off`: off; anything else fails the run). Warm-cache mode: `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` — every hub read (datasets; the model if it is a hub id) resolves from `hf_home`'s cache, never the hub. Warm it first (`HF_TOKEN=... make -C steps/eval prefetch HF_HOME=<same cache>`, which also prepares the `datasets.load_dataset` configs the benchmarks request; gated datasets need a token with access) and share it across runs; data the cache lacks fails fast. What runs offline (not terminal-bench; lm-eval bridge tasks only once prepared): steps/eval/README.md, "The warm HF cache". |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none.

Sampling defaults to the checkpoint's `generation_config` where the benchmark allows it
(NeMo-Skills' own default is greedy for some); `results.json` records the sampling used.

`answers=gold` (where supported) serves the reference answers through the same
NeMo-Skills generation and grading without a model (set `model_path: none`); it should
score at or near 100%. Use it to validate the data and grading on a new cluster.

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

The prepared data, per-repeat NeMo-Skills outputs (`generation/output-rs<k>.jsonl`) and
their logs are kept next to it.
