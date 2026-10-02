# sage2 / eval-granite5

Scores one checkpoint on the Sage2 **Granite 5** suite on BlueVela: every benchmark of
`sage2-evals list --suite granite5` that is implemented (all but `toolathlon` and
`pinchbench`). The wiring (generate / score pairs, secrets, adding a benchmark,
validating, running end to end through a local gbserver) is the same as
[eval-granite42](../eval-granite42/README.md); this file covers what differs.

`MODEL_PATH` is a **placeholder** (`ibm-granite/granite-4.2-3b`) until a Granite 5
checkpoint is chosen. At 131072 tokens it cannot serve `aa-lcr` (163840) or
`ruler-256k` / `-512k` / `-1m`; those targets fail at vLLM start until `MODEL_PATH`
has the context.

Compared with eval-granite42: no `birdbench` or `tau3-bench` (the per-domain `tau3-*`
targets stay), and 17 new benchmarks: `browsecomp`, `mcpatlas`, `omniscience`, `hle`,
`hle-tools`, `critpt`, `critpt-tools`, `multi-challenge`, `fortress-adversity`,
`fortress-benign`, `strongreject`, `wmt24pp`, `aa-lcr`, `ruler-256k`, `ruler-512k`,
`ruler-1m`, `omniscience-hallucination`.

## Before a run

Everything in [eval-granite42's table](../eval-granite42/README.md#before-a-run), plus:

| Needs | For | How |
|---|---|---|
| Space secret `HF_TOKEN` | `hle`, `hle-tools` (gated `cais/hle`), `wmt24pp` score (gated XCOMET-XXL) | declared on those targets |
| Space secret `SAGE2_JUDGE_API_KEY` | score targets of `browsecomp`, `mcpatlas`, `omniscience`, `omniscience-hallucination`, `hle`, `hle-tools`, `multi-challenge`, `fortress-*`, `strongreject`, `aa-lcr` | declared there, with the spend ledger envs |
| Space secret `ARTIFICIAL_ANALYSIS_API_KEY` | score targets of `critpt`, `critpt-tools` | an approved Artificial Analysis key; declared there |
| Outbound HTTPS from the generate job | `browsecomp`, `hle-tools` (IBM Google PSE MCP), `mcpatlas` (its MCP servers; also port 1984) | the node's network |
| Enroot sandboxes | `hle-tools`, `critpt-tools`, `mcpatlas` | each step's `sandbox_cache` (`/proj/granite-build/g4os/sage2/enroot-cache`) |

## Running

```bash
R=recipes/sage2/lsf/eval-granite5/build.yaml
gb build start -f $R --param MODEL_PATH=<hf dir or hub id> --param RUN_NAME=<name>
# Smoke a few targets:
gb build start -f $R hle strongreject --param SAGE2_LIMIT=5 ...
# pass@k: pass@1[avg-of-k] + details.pass_at_k
gb build start -f $R multi-challenge --param SAGE2_REPEATS=8 ...
```

Resources are as in eval-granite42, except that the `wmt24pp` score target takes a GPU
(`GPU_MEMORY`) for XCOMET-XXL.

**Check a target before scoring a model** (gold options as in eval-granite42, plus):

| Gold option | Targets |
|---|---|
| `answers=gold` | `omniscience`, `omniscience-hallucination`, `hle`, `hle-tools`, `browsecomp`, `aa-lcr`, `wmt24pp`, `ruler-256k` / `-512k` / `-1m` (about 1.0; the judged ones still call the judge, paid) |
| `responses=refusal` | `fortress-adversity`, `strongreject` (score 1.0), `fortress-benign` (score 0.0) |
| `responses=<model>` | `multi-challenge`: grades the responses upstream ships (`claude-3-5-sonnet-20241022`, `gpt-4o-2024-08-06`, `o1-preview`); compare with `details.paper_reference` |
| `agent=gold` / `agent=replay` | `mcpatlas` |
| none | `critpt`, `critpt-tools` |

## Targets and favored configs

Unless noted, the favored config is the benchmark's protocol: the model's
`generation_config` sampling, thinking on (vLLM `--reasoning-parser auto`), one repeat
(pass@1; `SAGE2_REPEATS=k` reports pass@1[avg-of-k] plus `details.pass_at_k`), and the
suite's upstream harness at a pinned commit.

| Target | Metric | Image | Favored config / notes |
|---|---|---|---|
| `swebench-verified` | pass@1 resolve rate | `swebench` | mini-swe-agent in enroot sandboxes, 8 workers |
| `swebench-pro` | pass@1 resolve rate | `swebench` | V2 (642 tasks); `subset=hard` = HARD-51. Deviates from Scale's harness: agent sandbox has network, verifier runs in enroot, an empty patch is scored unresolved without running the verifier. Sandboxes share the node's network and resolve `localhost` to 127.0.0.1 only: NodeBB's test server listens on IPv4, and the node's `/etc/hosts` also maps `localhost` to `::1` |
| `swebench-multilingual` | pass@1 resolve rate | `swebench` | 300 tasks. Java tests that resolve artifacts at test time fail on the environment without `SAGE2_MAVEN_MIRROR` (mirror URL still open) |
| `terminal-bench-2.1` | pass@1 resolve rate | `tbench` | Scored out of 87 of 89: `configure-git-webserver`, `git-multibranch` excluded (they ssh to localhost:22, which on the shared host network is the node's own sshd; `exclude=none` runs them). Oracle also fails `build-pov-ray` (upstream download 403) and `build-cython-ext` (unpinned `planarity` 1.0) |
| `tau3-airline` / `-retail` / `-telecom` / `-banking-knowledge` | pass@1 (pass^1; `SAGE2_REPEATS=k` adds pass^k) | `tau` | as above; banking uses BM25 + grep retrieval. `user_model=self judge_model=self` = no key, not comparable |
| `bfcl-v4` | overall_accuracy | `bfcl` | web search through the IBM search MCP, not SerpAPI; one run (`SAGE2_REPEATS` ignored) |
| `browsecomp` | mean reward | `nemoskills` | search agent: `web_search` through the IBM Google PSE MCP server (outbound HTTPS from the generate job), answers judged by `aws/claude-sonnet-5` (paid) |
| `mcpatlas` | pass rate (coverage >= 0.75) | `judged` | the 30 tasks whose MCP servers need no API key, run in an enroot sandbox (outbound HTTPS, port 1984); claims coverage judged by `aws/claude-sonnet-5` (paid) |
| `gdpval` | Elo (**approximation**) | `judged` | Elo-style score from the judged win rate vs the expert deliverables (expert = 1000), not GDPval-AA's Elo; judge `aws/claude-sonnet-5` (paid) |
| `profbench` | overall (ProfBench-lite, judged rubrics) | `judged` | judge `aws/claude-sonnet-5` at max effort: `PROFBENCH_OPTIONS` defaults to `judge_reasoning_effort=max judge_thinking=adaptive judge_effort=max` (paid; results record the judge request) |
| `aime25`, `hmmt-feb25` | pass@1 symbolic correct | `nemoskills` | NeMo-Skills |
| `gpqa` (Diamond) | pass@1 symbolic correct | `nemoskills` | needs `HF_TOKEN` |
| `omniscience` | pass@1 judge_correct | `nemoskills` | judged by `aws/claude-sonnet-5` (paid) |
| `hle` (no tools) | pass@1 judge_correct | `nemoskills` | gated `cais/hle` (`HF_TOKEN`); judged by `aws/claude-sonnet-5` (paid) |
| `hle-tools` | pass@1 judge_correct | `nemoskills` | as `hle`, with a Python sandbox (enroot) and `web_search` (PSE MCP) |
| `critpt` (no tools) | Challenge Accuracy | `nemoskills` | graded by Artificial Analysis' API (`ARTIFICIAL_ANALYSIS_API_KEY` at score); answers are hidden, so no gold mode. The API grades all 70 submissions and allows 10 requests a day: with `SAGE2_LIMIT` the rest go as bare templates, score wrong, and accuracy is rescaled to the N submitted (`details.critpt`). Not metered |
| `critpt-tools` | Challenge Accuracy | `nemoskills` | as `critpt`, with a Python sandbox (enroot) |
| `livecodebench-v6` | pass@1 accuracy | `nemoskills` | |
| `scicode` | pass@1 subtask accuracy | `nemoskills` | `prefill_fixes` (default on): the harness-supplied steps 13.6/62.1 get their full class from SciCode's `eval/data` instead of ns's bare `__init__`, which NameErrors every later step |
| `mmlu-pro` | 5-shot CoT symbolic correct | `nemoskills` | ns per-category validation examples, `\boxed{}` answers; `shots=0` = zero-shot |
| `mmlu-prox-lite` | exact match (custom-extract) | `lmeval` | lm-eval 5-shot CoT, chat mode, the 11 Granite languages MMLU-ProX has (`languages=` changes it); keep the default stop strings |
| `arena-hard-v2` | win rate | `nemoskills` | judge `aws/claude-sonnet-5`, not the official GPT-4.1, no style control (paid; a full run costs about $40-50) |
| `ifbench` | pass@1 prompt loose / strict accuracy | `ifbench` | NeMo-Skills + IFBench verifiers |
| `multi-challenge` | pass@1 correct | `judged` | judged by `aws/claude-sonnet-5` (paid; Scale AI's own judge is GPT-4o); `SAGE2_REPEATS=8` = their pass@1[avg-of-8] |
| `fortress-adversity` | Adversity resilience (1 - ARS/100) | `judged` | adversarial prompts, rubric-judged by `aws/claude-sonnet-5` (paid) |
| `fortress-benign` | Benign compliance (1 - ORS/100) | `judged` | the benign counterparts: over-refusal |
| `strongreject` | Harmlessness (1 - StrongREJECT score) | `judged` | StrongREJECT rubric judged by `aws/claude-sonnet-5` (paid) |
| `wmt24pp` | en→xx COMET | `nemoskills` | XCOMET-XXL (gated, `HF_TOKEN`) in the score job, which takes a GPU; BLEU is always reported, `score_metric=bleu` makes it the value |
| `aa-lcr` | pass@1 judge correct | `nemoskills` | about 100k-token documents; served at `AA_LCR_MAX_MODEL_LEN`=163840; judged by `aws/claude-sonnet-5` (paid) |
| `ruler-64k` | accuracy | `nemoskills` | data generated in the job for `MODEL_PATH`'s tokenizer; served at 131072, thinking on up to the context cap (`max_tokens` = 131072 - prompt tokens), scored after the reasoning parser. Samples that do not fit score 0 and are logged to `ruler-64k/failures.jsonl` |
| `ruler-128k` | accuracy | `nemoskills` | as `ruler-64k`, but thinking gets only the slack a 128k sample leaves in 131072: expect `length_before_answer` failures (score 0, logged to `ruler-128k/failures.jsonl`, counted in `results.json`). `RULER_128K_OPTIONS=enable_thinking=false` = NeMo-Skills' RULER exactly |
| `ruler-256k` / `-512k` / `-1m` | accuracy | `nemoskills` | as `ruler-64k`, served at 262144 / 524288 / 1048576: needs a checkpoint with that context (granite-4.2-3b has 131072, so vLLM refuses them) |
| `omniscience-hallucination` | pass@1 judge_omni_hallucination | `nemoskills` | incorrect / (incorrect + partial + not attempted), **lower is better**; same pipeline as `omniscience` (judged, paid). `generations_from=<omniscience output dir>` reuses that run |

`results.json` records the options, dataset revision, image versions, the thinking mode
and scoring source (RULER), exclusions (Terminal-Bench) and the paid spend
(`details.api_spend`). RULER failures are in `OUTPUT_ROOT/RUN_NAME/ruler-<len>/failures.jsonl`.
