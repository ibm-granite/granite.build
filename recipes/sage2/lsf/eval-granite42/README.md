# sage2 / eval-granite42

Scores one checkpoint on the Sage2 **Granite 4.2** suite on BlueVela. Each benchmark is
its own target and step (`space://steps/sage2-<benchmark>`), so a failure or a rerun
stays local to that benchmark. Each target writes a `results.json` (`sage2_results`).

The runtime is [sage2-evals](https://github.com/laminair/sage2-evals), shipped as one
prebuilt image per harness family (pinned in `parameters.yaml`). Nothing is built
here. Each step's `skypilot/README.md` lists every option.

## Before a run

| Needs | For | How |
|---|---|---|
| `HF_TOKEN` | `gpqa` (gated `Idavidrein/gpqa`, the account must have accepted its terms) | space secret in the job env |
| `SAGE2_JUDGE_API_KEY` | `arena-hard-v2`, `gdpval`, `profbench`, `tau3-*` (retail NL judge) | space secret; IBM LiteLLM key for `aws/claude-sonnet-5` |
| `SAGE2_USER_API_KEY` | `tau3-*` user simulator | space secret; same gateway |
| `SAGE2_SPEND_LEDGER`, `SAGE2_SPEND_BUDGET_USD` | every paid call is metered; at the budget the meter answers 402 and the run fails | job env; one ledger shared by every job |
| `SAGE2_MAVEN_MIRROR` (optional) | `swebench-multilingual` Java instances, where Maven Central answers the node's egress IP with 429 | job env |
| ICR pull access to `icr.io/tir-hew-sage2-evals` | every target | the space's enroot credentials |

Datasets are the public upstream ones, pinned by commit in sage2-evals and recorded in
`results.json`. Nothing is mirrored.

## Running

```bash
R=recipes/sage2/lsf/eval-granite42/build.yaml
# Whole suite with the favored configs (the parameters.yaml defaults):
gb build start -f $R --param MODEL_PATH=<hf dir or hub id> --param RUN_NAME=<name>
# Some targets only (positional):
gb build start -f $R aime25 gpqa --param MODEL_PATH=...
# Smoke: the first N examples by id, one repeat (results say smoke: true):
gb build start -f $R --param SAGE2_LIMIT=5 --param SAGE2_REPEATS=1 ...
```

Every target takes 1x H100 (`TENSOR_PARALLEL_SIZE`), `GPU_MEMORY`=256 GB and `QUEUE`.
Options are space-separated `key=value` in `<BENCH>_OPTIONS`.

**Check a target before scoring a model**: the reference mode serves the reference
answers (or patches, or oracle agents) through the full pipeline and should score 1.0
(`--param <BENCH>_OPTIONS=<gold>`; no GPU, no paid API):

| Gold option | Targets |
|---|---|
| `patch=gold` | `swebench-verified`, `swebench-pro`, `swebench-multilingual` |
| `agent=oracle` | `terminal-bench-2.1` |
| `sql=gold` | `birdbench` |
| `agent=gold` | `tau3-*` |
| `answers=gold` | `aime25`, `hmmt-feb25`, `gpqa`, `mmlu-pro`, `livecodebench-v6`, `scicode`, `mmlu-prox-lite`, `ruler-*` (still needs `MODEL_PATH` for the tokenizer) |
| `responses=o3 judge_model=human` | `profbench` (the dataset's human labels on o3's reports: 0.527, no key) |
| `deliverables=expert judge_model=self` | `gdpval` (expert vs expert: 1000) |

## Targets and favored configs

Unless noted, the favored config is the benchmark's protocol: the model's
`generation_config` sampling, thinking on (vLLM `--reasoning-parser auto`), the metric's
repeat count, and the suite's upstream harness at a pinned commit.

| Target | Metric | Image | Favored config / notes |
|---|---|---|---|
| `swebench-verified` | pass@1[avg-of-3] resolve rate | `swebench` | mini-swe-agent in enroot sandboxes, 8 workers |
| `swebench-pro` | pass@1[avg-of-3] resolve rate | `swebench` | V2 (642 tasks); `subset=hard` = HARD-51. Deviates from Scale's harness: agent sandbox has network, verifier runs in enroot, an empty patch is scored unresolved without running the verifier. Sandboxes share the node's network and resolve `localhost` to 127.0.0.1 only: NodeBB's test server listens on IPv4, and the node's `/etc/hosts` also maps `localhost` to `::1` |
| `swebench-multilingual` | pass@1[avg-of-3] resolve rate | `swebench` | 300 tasks. Java tests that resolve artifacts at test time fail on the environment without `SAGE2_MAVEN_MIRROR` (mirror URL still open) |
| `terminal-bench-2.1` | pass@1[avg-of-8] resolve rate | `tbench` | Scored out of 87 of 89: `configure-git-webserver`, `git-multibranch` excluded (they ssh to localhost:22, which on the shared host network is the node's own sshd; `exclude=none` runs them). Oracle also fails `build-pov-ray` (upstream download 403) and `build-cython-ext` (unpinned `planarity` 1.0) |
| `birdbench` | pass@1 execution match | `bird` | NeMo-Skills protocol, no evidence |
| `tau3-bench` | pass@1 (avg of 3): mean pass^1 of airline, retail, telecom | `tau` | user simulator `aws/claude-sonnet-5` (paid). Runs the three domains itself: running it with `tau3-airline`/`-retail`/`-telecom` pays twice |
| `tau3-airline` / `-retail` / `-telecom` / `-banking-knowledge` | pass@1 (pass^1, 4 trials) | `tau` | as above; banking uses BM25 + grep retrieval. `user_model=self judge_model=self` = no key, not comparable |
| `bfcl-v4` | overall_accuracy | `bfcl` | web search through the IBM search MCP, not SerpAPI; one run (`SAGE2_REPEATS` ignored) |
| `gdpval` | Elo (**approximation**) | `judged` | Elo-style score from the judged win rate vs the expert deliverables (expert = 1000), not GDPval-AA's Elo; judge `aws/claude-sonnet-5` (paid) |
| `profbench` | overall (ProfBench-lite, judged rubrics) | `judged` | judge `aws/claude-sonnet-5` at max effort: `PROFBENCH_OPTIONS` defaults to `judge_reasoning_effort=max judge_thinking=adaptive judge_effort=max` (paid; results record the judge request) |
| `aime25`, `hmmt-feb25` | pass@1[avg-of-4] symbolic correct | `nemoskills` | NeMo-Skills |
| `gpqa` (Diamond) | pass@1[avg-of-2] symbolic correct | `nemoskills` | needs `HF_TOKEN` |
| `livecodebench-v6` | pass@1[avg-of-2] accuracy | `nemoskills` | |
| `scicode` | pass@1[avg-of-2] subtask accuracy | `nemoskills` | `prefill_fixes` (default on): the harness-supplied steps 13.6/62.1 get their full class from SciCode's `eval/data` instead of ns's bare `__init__`, which NameErrors every later step |
| `mmlu-pro` | symbolic correct | `nemoskills` | |
| `mmlu-prox-lite` | exact match (custom-extract) | `lmeval` | lm-eval 5-shot CoT, chat mode, the 11 Granite languages MMLU-ProX has (`languages=` changes it); keep the default stop strings |
| `arena-hard-v2` | win rate | `nemoskills` | judge `aws/claude-sonnet-5`, not the official GPT-4.1, no style control (paid; a full run costs about $40-50) |
| `ifbench` | pass@1[avg-of-2] loose accuracy | `ifbench` | NeMo-Skills + IFBench verifiers |
| `ruler-64k` | accuracy | `nemoskills` | data generated in the job for `MODEL_PATH`'s tokenizer; thinking on (a thinking budget on top of ns's answer budgets, scored after the reasoning parser), served at 131072 |
| `ruler-128k` | accuracy | `nemoskills` | **placeholder**: `RULER_128K_OPTIONS=enable_thinking=false` (NeMo-Skills' RULER exactly), because a 128k sample plus the thinking budget does not fit 131072. The alternative is `sample_length=<131072 - budget>` with thinking on. `""` stops the run and names both. The experiment config is still open |

`results.json` records the options, dataset revision, image versions, the thinking mode
and scoring source (RULER), exclusions (Terminal-Bench) and the paid spend
(`details.api_spend`).
