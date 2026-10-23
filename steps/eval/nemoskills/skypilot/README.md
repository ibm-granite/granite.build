# nemoskills (SkyPilot)

Scores a checkpoint on one of the granite-evals **nemoskills** family's benchmarks:
each run through NVIDIA NeMo-Skills' own pipeline, locally (no Slurm, no NeMo-Run) —
data preparation (every upstream read pinned), prompt, generation, answer extraction
and metrics. `granite_config.benchmark` selects which one.

The code is the external [granite-evals](https://github.com/laminair/granite-evals) runtime,
shipped as a prebuilt image. Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/eval/nemoskills
    config:
      granite_config:
        benchmark: gpqa
```

## Benchmarks in this family

| `benchmark` | Metric | Notes |
|---|---|---|
| `aa-lcr` | pass@1 judge correct | `ArtificialAnalysis/AA-LCR`, 100 questions over ~100k-token documents; prompts reach ~130k tokens — set `max_model_len` to ~160k+ |
| `aime25` | pass@1 symbolic correct | |
| `arena-hard-v2` | win rate | NeMo-Skills' pairwise arena judge |
| `browsecomp` | mean reward | OpenAI simple-evals' 1266 questions (encrypted, pinned by sha256); web search via the IBM Google PSE MCP server (no key), page fetch; benchmark data itself is refused on fetch |
| `critpt` | challenge accuracy | 70 physics research challenges, two-turn generation; graded by Artificial Analysis's CritPt API |
| `critpt-tools` | challenge accuracy | as `critpt`, plus a stateful python tool each turn in an enroot sandbox (network off) |
| `gpqa` | pass@1 symbolic correct | `Idavidrein/gpqa` (gpqa_diamond), 198 questions; **gated** — needs `HF_TOKEN` of an account that accepted its terms |
| `hle` | pass@1 judge correct | NeMo-Skills' `text` split of `cais/hle` (no-image questions); the official HLE judge prompt |
| `hle-tools` | pass@1 judge correct | as `hle`, plus a stateful python tool (enroot sandbox, network off) and web search via the IBM Google PSE MCP server |
| `hmmt-feb25` | pass@1 symbolic correct | |
| `livecodebench-v6` | pass@1 accuracy | 454 problems, release v6 (`test_v6_2408_2505`, contests 2024-08 to 2025-05); graded with the pinned `livecodebench` package in the job container |
| `mmlu-pro` | 5-shot CoT symbolic correct | |
| `omniscience` | pass@1 judge correct | `ArtificialAnalysis/AA-Omniscience-Public`, 600 questions; judge: A correct, B incorrect, C partial, D not attempted |
| `omniscience-hallucination` | pass@1 judge_omni_hallucination (lower is better) | same pipeline as `omniscience`; hallucination rate = incorrect / (incorrect + partial + not attempted). `options: generations_from=<omniscience output dir>` re-scores that run instead of generating again |
| `ruler-64k` / `-128k` / `-256k` / `-512k` / `-1m` | accuracy (NeMo-Skills' `ruler_score`, mean over 13 tasks) | data generated for the model's own tokenizer (RULER source data baked into the image, sha256-checked), 100 samples/task; served at the matching `max_model_len` |
| `scicode` | pass@1 subtask accuracy | 65 problems / 288 subtasks, with background; graded in NeMo-Skills' local sandbox server (pinned Python 3.10 scientific env: numpy, scipy, sympy, h5py, matplotlib) |
| `wmt24pp` | en→xx COMET (XCOMET-XXL) | `google/wmt24pp`, default languages de_DE/es_MX/fr_FR/it_IT/ja_JP (998 segments each; `options: languages=` overrides); XCOMET-XXL (`Unbabel/XCOMET-XXL`, gated, CC-BY-NC-SA-4.0) runs in the image's separate COMET env — phase `score` needs the gated model's access |

See `granite-evals list` for the current, authoritative set and sizes.

## Config (`granite_config`)

| Field | Purpose |
|---|---|
| `benchmark` | Required. One of the ids in the table above. |
| `image` | Required. The granite-evals `nemoskills` image, pinned by tag (`GRANITE_EVALS_IMAGE_NEMOSKILLS`). |
| `model_path` | Required. Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | Name vLLM serves under. Empty = basename of `model_path`. |
| `output_dir` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | **Smoke knob:** score only the first N examples of the pinned/prepared data. Empty = the benchmark's full set. |
| `repeats` | Independent generations per example (avg-of-k). Empty = benchmark default (1). |
| `workers` | Concurrent requests to vLLM. Recipes set a per-benchmark default (32 for most; 64 for gpqa/livecodebench-v6/scicode; 128 for mmlu-pro). |
| `dataset` / `dataset_revision` | Override the dataset pinned in granite-evals for the selected benchmark. A hub id, a prepared `.jsonl` or a directory holding `<split>.jsonl` overrides it, where the benchmark supports one. |
| `options` | Space-separated `key=value` benchmark options. Common: `temperature`, `top_p`, `top_k`, `max_tokens`, `ns.<key>=<value>` (any NeMo-Skills generation override), `answers=gold` (reference answers through the same grading; `model_path` may be `none`). Benchmark-specific options exist for several of the above — see each benchmark's own granite-evals docs. |
| `phase` | `all` generates and scores in one job. `generate` serves the model and writes `generation.json` (`granite_generation`); `score` grades that output dir without a GPU (same `output_dir`, `limit`, `repeats`) and writes `results.json`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | vLLM. Several RULER sizes and `aa-lcr` need a non-default `max_model_len` — see the table above. |
| `sandbox_cache` | Shared squashfs cache. Only `critpt-tools`, `hle-tools`, `livecodebench-v6`, `ruler-*` and `scicode` in this family sandbox; set this for those, leave empty for the rest. |
| `hf_home` | Overrides `HF_HOME`. |

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

## Developing

`make unit-tests` runs the template contract tests. The runtime has its own tests in
granite-evals. Every step under `steps/eval/` follows this template; the test checks this.
