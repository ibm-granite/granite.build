# sage2 / eval-granite42

Scores one checkpoint on the Sage2 **Granite 4.2** suite on BlueVela. Each benchmark
runs its own step (`space://steps/sage2-<benchmark>`), so a failure or a rerun stays
local to that benchmark. This file is the full suite; to run a subset, name targets
(see [Running](#running)); to build another recipe, copy the targets you need (see
[How a benchmark is wired](#how-a-benchmark-is-wired)).

The runtime is [sage2-evals](https://github.com/laminair/sage2-evals), shipped as one
prebuilt image per harness family (pinned in `parameters.yaml`). Nothing is built
here. Each step's `skypilot/README.md` lists every option.

## How a benchmark is wired

A sage2 step runs one benchmark in one of three phases (`sage2_config.phase`):

| Phase | Serves the model | Does | Writes | Output |
|---|---|---|---|---|
| `generate` | yes (vLLM, GPU) | the model's work only: prompts, agent turns, user-simulator conversations | `generation.json` | `sage2_generation` |
| `score` | no (CPU) | grades `generation.json`: answer checks, sandboxed tests, verifiers, paid judges | `results.json` | `sage2_results` |
| `all` (step default) | yes | both, in one job | `results.json` | `sage2_results` |

granite.build has no sub-recipes or includes: a build file is one flat `targets` map,
and a binding can only name a target in the same file. A "generation + judgement"
benchmark is therefore **two targets that run the same step**, the second bound to
the first:

| Target | Resources | Phase |
|---|---|---|
| `<bench>-generate` | `TENSOR_PARALLEL_SIZE` x H100, `GPU_MEMORY` | `generate` |
| `<bench>` | `SCORE_CPUS`, `SCORE_MEMORY`, no GPU | `score` |

So no GPU sits idle while sandboxes, verifiers or judges grade. Benchmarks that grade
inside generation (`terminal-bench-2.1`: the agent's container is the test;
`mmlu-prox-lite`: lm-eval scores in-process) are **one target** with `phase: all`.

### The pair

`tau3-retail` from `build.yaml`, with the parts that make it a pair marked:

```yaml
    tau3-retail-generate:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      outputs:
        sage2_generation:                    # (1) the generate phase's artifact
          uri: "env://{{ binding.path }}"
          type: dataset
      steps:
        - step_uri: space://steps/sage2-tau3-retail   # (2) same step in both
          config:
            poll_interval_seconds: $${POLL_INTERVAL_SECONDS}
            log_retrieval_interval_seconds: $${LOG_RETRIEVAL_INTERVAL_SECONDS}
            sage2_config:                    # (3) identical in both but for phase
              image: "$${SAGE2_IMAGE_TAU}"
              model_path: "$${MODEL_PATH}"
              served_model_name: "$${SERVED_MODEL_NAME}"
              output_dir: "$${OUTPUT_ROOT}/$${RUN_NAME}/tau3-retail"
              limit: "$${SAGE2_LIMIT}"
              repeats: "$${SAGE2_REPEATS}"
              workers: "$${TAU_WORKERS}"
              dataset: "$${TAU_DATASET}"
              options: "$${TAU_OPTIONS}"
              phase: "generate"
              tensor_parallel_size: "$${TENSOR_PARALLEL_SIZE}"
              hf_home: "$${HF_HOME}"
            skypilot:                        # (4) only the keys this phase calls
              secrets:
                secret_names_to_use_as_env_variable:
                  - env_name: SAGE2_USER_API_KEY
            launcher_config:
              envs:                          # (5) plain env vars
                SAGE2_SPEND_LEDGER: "$${SPEND_LEDGER}"
                SAGE2_SPEND_BUDGET_USD: "$${SPEND_BUDGET_USD}"
              resources:                     # (6) GPU for generate ...
                accelerators: "H100:$${TENSOR_PARALLEL_SIZE}"
                cluster: "$${CLUSTER}"
                zone: "$${QUEUE}"
                memory: $${GPU_MEMORY}
    tau3-retail:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      inputs:
        generation:                          # (7) runs after, and only after, generate
          binding: tau3-retail-generate.sage2_generation
      outputs:
        sage2_results:
          uri: "env://{{ binding.path }}"
          type: dataset
      steps:
        - step_uri: space://steps/sage2-tau3-retail
          config:
            # ... poll intervals and sage2_config exactly as above, but:
            sage2_config:
              # ...
              phase: "score"
            skypilot:
              secrets:
                secret_names_to_use_as_env_variable:
                  - env_name: SAGE2_JUDGE_API_KEY
            launcher_config:
              envs:
                SAGE2_SPEND_LEDGER: "$${SPEND_LEDGER}"
                SAGE2_SPEND_BUDGET_USD: "$${SPEND_BUDGET_USD}"
              resources:                     # (6) ... CPU only for score
                cluster: "$${CLUSTER}"
                zone: "$${QUEUE}"
                memory: $${SCORE_MEMORY}
                cpus: "$${SCORE_CPUS}"
```

Rules:

1. **Same step, same `sage2_config`, different `phase`.** The two jobs meet on the
   shared filesystem: `output_dir` is where generate writes `generation.json` and score
   reads it. The binding (7) only orders the targets; the score job does not read the
   artifact path. The score job stops unless `generation.json` was made for its
   benchmark, model, limit, repeats and (when overridden) dataset, so a mismatch fails
   loudly instead of grading another run's outputs; differing options only warn (a
   scoring timeout or another judge may differ on purpose). It never generates.
2. **Outputs follow the phase.** `generate` declares `sage2_generation`; `score` and
   `all` declare `sage2_results`. The step emits exactly that artifact.
3. **Resources follow the phase.** `generate` and `all` take GPUs; `score` takes no
   `accelerators`. A score target on a GPU works but wastes it.
4. **Secrets per target, for the phase that calls them.** Only secrets listed under
   `config.skypilot.secrets.secret_names_to_use_as_env_variable` reach the job
   (`{env_name, secret_name}`; `secret_name` defaults to `env_name`). The space must
   hold a secret of that name: a declared but missing one fails the launch at once.
   Declare nothing a phase doesn't call, so a key never sits in a job that doesn't use it:

   | Benchmark | `-generate` | score target |
   |---|---|---|
   | `tau3-*` | `SAGE2_USER_API_KEY` (user simulator) | `SAGE2_JUDGE_API_KEY` (retail's NL judge; declared on every domain) |
   | `arena-hard-v2`, `gdpval`, `profbench` | — | `SAGE2_JUDGE_API_KEY` |
   | `gpqa` | `HF_TOKEN` (gated dataset) | `HF_TOKEN` |
   | everything else | — | — |

   A paid target also sets `SAGE2_SPEND_LEDGER` / `SAGE2_SPEND_BUDGET_USD` (5), so its
   calls are metered and capped (see [Before a run](#before-a-run)).
5. **Selecting `<bench>` runs both.** A positional target pulls in what it binds, in
   order. Rerunning `<bench>` alone also reruns `<bench>-generate`; that job skips every
   example already generated but still takes a GPU and starts vLLM briefly.

### One target instead of two

Use `phase: "all"`, declare `sage2_results`, request GPUs, and declare the union of
both phases' secrets (`terminal-bench-2.1` in `build.yaml` is the template). Do this
when grading needs the served model: options with `judge_model=self` grade with the
model itself, and the score job serves nothing, so it refuses them.

### Adding a benchmark

1. The benchmark must exist in sage2-evals (`sage2-evals list`) and its image be pushed.
2. Its step: `steps/sage2-<bench>/skypilot/` (copy a sibling with the same image
   family). Edit `step-template.yaml` only, then render and check, from the step dir:

   ```bash
   cd steps/sage2-<bench>/skypilot
   make publish-step PUBLISH_REQUIRE_IMAGE=false IMAGE_REF=<the image ref>
   make check-published   # the committed step.yaml/README.md match the template
   make test VENV_DIR=<a venv with granite.build installed>   # contract test
   ```

   `publish-step` renders to
   `configurations/assets/environments/skypilot/steps/sage2-<bench>/`, which is what
   `space://steps/sage2-<bench>` resolves to. Commit both. A stale published step is
   the one that runs, so run `check-published` after every template change.
3. Its targets: copy a pair (or a single target) above, rename, point `step_uri`,
   `output_dir` and the image param at the new benchmark, add `<BENCH>_WORKERS` /
   `<BENCH>_OPTIONS` to `parameters.yaml`, and declare its secrets per rule 4.
4. Validate (below), then smoke it with `SAGE2_LIMIT` and its gold option.

## Before a run

| Needs | For | How |
|---|---|---|
| Space secret `HF_TOKEN` | `gpqa` (gated `Idavidrein/gpqa`; the account must have accepted its terms) | declared on both gpqa targets |
| Space secret `SAGE2_JUDGE_API_KEY` | score targets of `arena-hard-v2`, `gdpval`, `profbench`, `tau3-*` | IBM LiteLLM key for `aws/claude-sonnet-5`; declared on those targets |
| Space secret `SAGE2_USER_API_KEY` | `tau3-*-generate` (user simulator) | same gateway; declared on those targets |
| `SPEND_LEDGER`, `SPEND_BUDGET_USD` params | every paid call is metered into the ledger; at the budget the meter answers 402 and the run fails | passed as `SAGE2_SPEND_LEDGER` / `SAGE2_SPEND_BUDGET_USD` to the paid targets. One ledger for every job makes the budget global. Empty = no ledger / no cap |
| `MAVEN_MIRROR` param (optional) | `swebench-multilingual` Java instances, where Maven Central answers the node's egress IP with 429 | passed as `SAGE2_MAVEN_MIRROR` to both swebench-multilingual targets |
| ICR pull access to `icr.io/tir-hew-sage2-evals` | every target | SkyPilot's LSF provider imports each image tag once into `/proj/granite-build/g4os/enroot/` with the **job account's default** enroot credentials (`~/.config/enroot/.credentials`); they must cover this namespace. Without it the job exits within seconds: `icr.io/oauth/token returned error code: 401` in `sky_logs/<job>.err`, and SkyPilot retries |

Only declared secrets reach a job; nothing else in the space does. Results record
the spend (`details.api_spend`), including earlier attempts on the same output dir
(`run_total_usd`).

Datasets are the public upstream ones, pinned by commit in sage2-evals and recorded in
`results.json`. Nothing is mirrored.

## Running

```bash
R=recipes/sage2/lsf/eval-granite42/build.yaml
# Whole suite with the favored configs (the parameters.yaml defaults):
gb build start -f $R --param MODEL_PATH=<hf dir or hub id> --param RUN_NAME=<name>
# Some targets only (positional; a score target brings its generate target):
gb build start -f $R aime25 gpqa --param MODEL_PATH=...
# Smoke: the first N examples by id, one repeat (results say smoke: true):
gb build start -f $R --param SAGE2_LIMIT=5 --param SAGE2_REPEATS=1 ...
```

Generate targets take 1x H100 (`TENSOR_PARALLEL_SIZE`) and `GPU_MEMORY`=256 GB; score
targets take `SCORE_CPUS`=16 and `SCORE_MEMORY`=64 GB. All use `QUEUE`. Options are
space-separated `key=value` in `<BENCH>_OPTIONS`. Outputs land in
`OUTPUT_ROOT/RUN_NAME/<bench>/`.

### Validating a change

From the granite.build checkout, cheapest first:

```bash
R=recipes/sage2/lsf/eval-granite42/build.yaml
export GB_ENVIRONMENT=STANDALONE   # no login needed; also what the local server expects
# 1. Parameters resolve ($${VAR}, client side): writes the rendered build file, submits
#    nothing, needs no server.
gb build start -f $R --dry-run --save-build-file /tmp/sage2-rendered.yaml
# 2. Against a local standalone gbserver: schema, plus every space:// step and
#    environment URI must resolve to the published assets.
gbserver standalone --host 127.0.0.1 --port 18080 --space-dir configurations/spaces/local &
export GBSERVER_HOST=http://127.0.0.1:18080
gb build validate -f $R --space standalone --verbose-validation
```

`--validation-type dynamic` adds nothing here: it skips SkyPilot targets.

### End to end through a local gbserver

The same standalone server submits to BlueVela over SSH (the environment's
`cluster_ssh_configs`: `granitebuild@login4` with `~/.ssh/ibm-bluevela.key`), so a
laptop can drive a real run:

```bash
gb build start -f $R aime25 \
  --param SAGE2_LIMIT=2 --param SAGE2_REPEATS=1 --param RUN_NAME=<smoke name>
gb build status <build id>
```

Each target becomes one LSF job (`gb-sage2-eval-granite42-<target>-...`) with its
logs in `<skypilot workdir>/<cluster name>/sky_logs/<job>.{out,err}` on BlueVela
(workdir `/proj/granite-build/g4os/skypilot`). The secrets come from the local space's
secret manager, so they must exist there under the declared names. Cancel with
`gb build cancel <build id>`.

**Check a target before scoring a model**: the reference mode serves the reference
answers (or patches, or oracle agents) through the full pipeline and should score 1.0
(`--param <BENCH>_OPTIONS=<gold>`; no paid API; the generate job needs no GPU in this
mode but the target still requests one):

| Gold option | Targets |
|---|---|
| `patch=gold` | `swebench-verified`, `swebench-pro`, `swebench-multilingual` |
| `agent=oracle` | `terminal-bench-2.1` |
| `sql=gold` | `birdbench` |
| `agent=gold` | `tau3-*` |
| `answers=gold` | `aime25`, `hmmt-feb25`, `gpqa`, `mmlu-pro`, `livecodebench-v6`, `scicode`, `mmlu-prox-lite`, `ruler-*` (still needs `MODEL_PATH` for the tokenizer) |
| `responses=o3 judge_model=human` | `profbench` (the dataset's human labels on o3's reports: 0.527, no key) |
| `deliverables=expert` | `gdpval` (expert vs expert: about 1000; the paid judge runs in the score job, a few cents at `SAGE2_LIMIT=5`) |

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
