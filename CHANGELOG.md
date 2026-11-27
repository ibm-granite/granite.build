# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **gbcli** — CLI client added to the monorepo under `src/gbcli/`, with entry points `gb`, `gbcli`, `llmbuild`, `llmb`, `lamb`
- Standalone mode — all-in-one server with SQLite storage and thread-based execution
- Docker environment — run build steps in containers with GPU support
- Bash environment — run build steps as local processes (macOS/Linux)
- Kubernetes environment — run build steps as K8s jobs
- RunPod environment (beta) — run build steps on RunPod GPU instances
- SkyPilot/AWS environment (beta) — run build steps on cloud instances via SkyPilot
- HuggingFace Hub integration — download models and datasets via `hf://` URIs
- REST API — FastAPI-based build management at `/api/v1`
- Pipeline orchestration — multi-step builds defined in `build.yaml`
- Built-in steps: `gbstep`, `hfpull`, `hfpush`, `lhpull`, `lhpush`, `cosrclone`
- **`openinstruct-sft` step (#229)**, `space://steps/train/openinstruct-sft`: SFT with upstream
  Open-Instruct `finetune.py` (v0.3.0, accelerate + DeepSpeed) on any SkyPilot endpoint, single
  node, with ClearML tracking (`auto` when `CLEARML_API_*` creds are present). Custom image
  `quay.io/granitebuild/steps/train/openinstruct-sft/skypilot:<version>`. Build tests on
  skypilot/aws and BlueVela SLURM. On SLURM the image is cached as a shared `.sqsh` by
  `slurm_image_cache`. The legacy hand-authored `space://steps/openinstruct-sft` assets (LSF, aws)
  and the granite4-350m recipes that use them are unchanged; moving those recipes over is a
  follow-up, #569 (they need pre-tokenized `--tokenized_data` input, which only the IBM
  fork's `finetune.py` has).
- **`hf-peft-sft` step (#585)**, `space://steps/train/hf-peft-sft`: LoRA, QLoRA and aLoRA
  supervised fine-tuning with Hugging Face PEFT 0.21.2 and TRL 1.17.1 `SFTTrainer` on any
  SkyPilot endpoint, single node, DDP or FSDP / DeepSpeed ZeRO-3 sharding, ClearML tracking
  (`auto` when `CLEARML_API_*` creds are present). Registers the PEFT adapter (not a merged
  model). Custom image `quay.io/granitebuild/steps/train/hf-peft-sft/skypilot:peft-v0.21.2-6`
  (torch 2.15.0 / CUDA 13.2) with a fully hashed dependency lock. Build tests on skypilot/aws
  (QLoRA, aLoRA) and BlueVela SLURM (LoRA/FSDP, QLoRA/DeepSpeed).
- **`mergekit` step (#238)**, `space://steps/merge/mergekit`, in a new `merge` category: one
  MergeKit (`mergekit-yaml`, v0.1.5, LGPL-3.0) merge on any SkyPilot endpoint, single node, on
  the CPU unless GPUs are requested. Public image: `setup` installs `mergekit==<mergekit_version>`
  with uv, as data-designer does, or `mk_image` provides it. `merge_config` is MergeKit's
  own recipe, passed through unchanged except that a model reference naming a bound input
  (`model_1`–`model_4`, `base_model`) gets the staged path. Hugging Face ids and node paths
  pass through, so MergeKit fetches them itself. An inline Jinja `chat_template` needs
  `{% raw %}`, because gbserver renders config strings. `args` takes further
  `mergekit-yaml` flags. Config errors fail at step creation. Because MergeKit reports a
  failed tokenizer copy as success, the step checks the merged output (tensors, tokenizer,
  requested chat template) before registering it. Build tests on local SLURM and local
  kind run MergeKit's `gradient-slerp` example on SmolLM2-135M + -Instruct, and
  `make test-mergekit` runs the real-MergeKit tests in a separate venv.
- **`fms-hf-tuning` step (#231)**, `space://steps/train/fms-hf-tuning`: full fine-tuning, LoRA and
  aLoRA with foundation-model-stack/fms-hf-tuning 3.2.2 (`tuning.sft_trainer`) on any SkyPilot
  endpoint, single node, DDP or FSDP (fms-hf-tuning's FSDP1 defaults), ClearML tracking (`auto`
  when `CLEARML_API_*` creds are present). Training data from a bound dataset, an HF id, or an
  fms-hf-tuning `data_config`; `args` is fms-hf-tuning's JSON config, checked against its own
  parser on the node before training starts. Registers the full model or the adapter. Custom
  image `quay.io/granitebuild/steps/train/fms-hf-tuning/skypilot:fms-v3.2.2-5` (torch 2.8.0 /
  CUDA 12.8, transformers 4.55.4 with the CVE-2026-5241 fix backported, flash-attn 2.8.3) with a
  fully hashed lock. Build tests on skypilot/aws (full, aLoRA) and BlueVela SLURM (LoRA/FSDP,
  full/FSDP with `data_config`). K8s, SkyPilot-K8s and LSF are follow-ups.
- **`make publish-image` refuses to overwrite an existing image tag (#229).** A tag must keep
  pointing at one image, and SLURM's `slurm_image_cache` keys its `.sqsh` by tag, so a re-pushed
  tag never reached it. `publish-image` now checks the registry first and pushes only a new tag;
  it also refuses when the check cannot tell. Bump `IMAGE_VERSION`, or pass
  `PUBLISH_OVERWRITE=true` to replace a tag on purpose. The check is the new
  `steps/scripts/registry-tag-status.sh` (`skopeo` when installed, else the registry's v2 API).
- **`make test` warns when it would test an outdated published image (#229).** A custom-image
  step's SkyPilot build tests pull the published `IMAGE_REF`, not the local build, so with a
  pinned `IMAGE_VERSION` a Dockerfile edit passed silently against the old image. `make test` now
  compares the local image ID with the published image's config digest (podman only) and prints a
  warning when they differ or can't be compared, then runs the tests. To test a change on a
  cluster, publish it under a new (or scratch) `IMAGE_VERSION` first; `TEST_PUBLISHED=true` skips
  the check and its registry query.
- **The registry checks read a stored `podman login` (#229).** `steps/scripts/registry-tag-status.sh`
  uses `REGISTRY_USER`/`REGISTRY_PASSWORD` when set, else the `podman login` in
  `$REGISTRY_AUTH_FILE` or `~/.config/containers/auth.json`, so a step's first publish to quay.io
  (which answers an anonymous check of a missing repository with 401) works after a plain
  `podman login`.

- **granite-evals** — benchmark evaluation suite: 41 benchmarks across 9 harness families
  (bfcl, bird, ifbench, judged, lmeval, nemoskills, swebench, tau, tbench), one image and
  one sourced step per family under `steps/eval/`, with dispatch recipes under
  `recipes/evals/` that split GPU generate from CPU score. No benchmark data is committed:
  dataset pins live in `steps/eval/VENDORED.md` and are fetched at run or image-build
  time (#460).

- **`single_node` step flag (#651)** — `environment_configs.Skypilot.single_node: true` makes
  gbserver resolve `num_nodes` from every layer the launch reads (`compute_config`, the step's
  launcher config, the build's `launcher_config`) and fail the step at creation unless it is 1.
  Other environments refuse the flag. The SkyPilot `openinstruct-sft`, `hf-peft-sft`, `nemo-rl`,
  `fm-tune`, `fms-hf-tuning` and `mergekit` steps use it instead of their own Jinja `num_nodes`
  checks: they now accept exactly what gbserver parses as 1 (hf-peft-sft newly accepts `"01"`);
  only the count the launch would use matters, so a `compute_config.num_nodes: 2` overridden by
  a build `launcher_config.num_nodes: 1` now passes; and nemo-rl newly checks the build's
  `launcher_config.num_nodes`. `gb step describe` shows the flag.
- **`num_nodes` parsing (#651)** — for every SkyPilot step, a boolean `num_nodes` (`true` used
  to count as one node) is now refused, and a malformed count fails at step creation, with an
  error that names the key it came from (`compute_config.num_nodes`,
  `launchers.<name>.config.num_nodes` or `launcher_config.num_nodes`).
- **Stateless SSO login for the CLI (#723).** `gb auth login --sso` receives the provider's
  browser redirect on a local listener, `http://127.0.0.1:48731/callback` (`--redirect-port`
  overrides the port). `gb auth login --sso --remote` is for a CLI that can't receive that
  redirect, e.g. over SSH: sign in on any device and paste the verification code shown. Both use
  new gbserver endpoints, `/api/v1/auth/oidc/{authorize,token,remote-callback}`, which keep no
  login state, so any worker or replica can serve them. **Operators:** register
  `http://127.0.0.1:48731/callback` and `https://<gbserver-host>/api/v1/auth/oidc/remote-callback`
  as redirect URIs with the SSO provider, and deploy the server before the CLI: the new CLI
  can't log in to an older gbserver, and says so before opening the browser. An older CLI
  still works with a new gbserver, through the legacy endpoints, which still need a single
  rest-server worker.

### Changed

- **distill: experiment tracking is one chosen tool per run (#752).** logit-distill and student-sft take a
  `tracking` block whose `tool` is `wandb`, `clearml` or `none` (the default; logit-distill had no tracking
  keys before), and the distill recipe takes `TRACKING_TOOL`. A config that sets a tool's keys without
  choosing it, or both tools' keys, is refused; before, each tool was switched on by its own keys. W&B now
  talks to `https://api.wandb.ai` unless `WANDB_BASE_URL` or wandb's settings file names another server
  (wandb 0.31's own default is `https://api.forge.coreweave.com`), and ClearML to `https://api.clear.ml`
  unless `CLEARML_API_HOST` or `clearml.conf` names one. The credential preflight looks up W&B's netrc entry
  the way wandb 0.31.2 does (the entry's own password, `default` entries, quoted names, the server's host,
  `WANDB_IDENTITY_TOKEN_FILE`) and ClearML's config files the way clearml 2.1.12 does.
- **distill images: wandb 0.31.2, filelock 4.2.11 (#752).** The GPU image moves from wandb 0.28.2 and
  filelock 3.32.7, and its `scan-ignore` lists no advisory any more. filelock is held below 4.2.12 until a
  two-account shared-cache test (#788). The CPU image no longer installs wandb or clearml, which nothing in
  it imports.

- **The granite-switch SLURM answerability-lora recipe runs on any SLURM cluster, from a
  site file (#737).** It shipped in v0.3.9 as `recipes/granite-switch/slurm/answerability-lora`,
  moved to `recipes/granite-switch/skypilot/slurm-bluevela/answerability-lora` in #639 (the
  v0.3.10 path), and now moves to `recipes/granite-switch/skypilot/slurm/answerability-lora`.
  Its environment is rendered by `scripts/setup-local-space.sh` from the SLURM template the
  distill recipes use, now shared at `recipes/environments/skypilot-slurm.template.yaml`,
  where the eval lane's `EVAL_QOS` and `EVAL_PARTITION` become optional. A
  `site.example.yaml` next to the recipe lists every site key with a fake value. The
  prepare target's partition is now `$${CPU_PARTITION}`, the template's key, not a fixed
  `cpu`. **Breaking:** `parameters.yaml` no longer has `ENVIRONMENT_URI`,
  `SOURCE_DATA_URI`, `DATA_ROOT` or `ADAPTER_ROOT`, and `ACCELERATORS` is split into the
  site's GPU model, `GPU_ACCELERATOR` (as in distill), and the recipe's count, `NUM_GPUS`
  (default 4). A run needs `--param-file <site.yaml>` with those site keys (and
  `CPU_PARTITION`), and fails to render without one; a smoke run's
  `--param ACCELERATORS=<model>:1` becomes `--param NUM_GPUS=1`. Neither old path is kept.
- **The fm-tune step moves from the `tune` category to `train` (#677).** Its URI changes from
  `space://steps/tune/fm-tune` to `space://steps/train/fm-tune`. The source is now
  `steps/train/fm-tune/`, and the published copies are
  `configurations/assets/environments/{bash,skypilot}/steps/train/fm-tune/`. The build tests,
  docs and unit tests are updated. **Breaking:** a build.yaml outside this repo that uses
  `space://steps/tune/fm-tune` must switch to the new URI. The old one is not kept as an alias.
- **The `misc` step category is now `aux` (#671).** The byoc step moves from
  `space://steps/misc/byoc` to `space://steps/aux/byoc`; its source is now `steps/aux/byoc/` and
  its published copy is `configurations/assets/environments/skypilot/steps/aux/byoc/`. The
  in-repo recipes, build tests and docs are updated. **Breaking:** a build.yaml outside this repo
  that uses `space://steps/misc/byoc` must switch to the new URI; the old one is not kept as an
  alias. Environment overrides keyed `steps.byoc:` still apply, because they match only the last
  part of the step URI.
- **Framework steps: share Jinja between templates with `gb-include` (#651).** A
  `{#- gb-include <path> -#}` line in a `step-template.yaml` is replaced by the file it
  names, after a `{#- included from steps/<file> -#}` comment, when `make space` /
  `make publish-step` render the step (`steps/scripts/render_step_template.py`, which
  replaces the `sed` render; steps without such lines render byte-identically). The
  SkyPilot `openinstruct-sft` and `hf-peft-sft` steps now share their `args` checks and
  `args` argv rendering (`steps/train/shared/trainer_args.j2`) and define their
  `validate` macro once (`macros/validate.j2`) instead of in both `setup:` and `run:`.
  `fms-hf-tuning` writes its `validate` once the same way (its `args` go to the job as
  JSON, so it keeps its own checks). All three share one shell-quoting `q()`
  (`steps/train/shared/q.j2`). Messages are unchanged; the one behaviour change is that
  a `null` config value (e.g. `clearml_project: null`, `dataset_config: null`) now
  reaches openinstruct-sft's and hf-peft-sft's scripts as empty, i.e. unset, instead of
  as the word `None`. The CI test that checks every published `step.yaml` against its
  source now renders the source as `make publish-step` does, so it also covers
  `${IMAGE_REF}` and `gb-include` steps.

- **Step resolution: `base_uris` order decides, and environments own steps by `environment.yaml`
  (#558).** `space://steps/<name>` now resolves to the first base, in `base_uris` order, that
  holds a copy serving the active environment, so an earlier base always outranks a later one.
  The space's own root `steps/<name>` still wins outright. Within one base, the copy in the
  active environment's own `steps/` comes first, then shared copies; among shared copies, the
  one whose `environment_configs` list the fewest classes wins. See
  [docs/environments/step-resolution.md](docs/environments/step-resolution.md).
  **Breaking changes:**
  - **A step's name is its full path below `steps/`.** Only an env-class folder directly under
    `steps/` (`bash`, `docker`, `k8s`, `lsf`, `runpod`, `skypilot`, `skypilot_managed`) may be
    left out, as with the builtins' `space://steps/command`. Any other folder is part of the
    name, so `space://steps/<leaf>` no longer reaches `steps/<category>/<leaf>`: write
    `space://steps/<category>/<leaf>`, e.g. `space://steps/distill/sft`, not
    `space://steps/sft`.
  - **A `steps/` folder next to an `environment.yaml` serves only that environment.** Steps
    there are no longer shared with other environments, of the same class or otherwise, and
    folder names (`environments/…`) no longer decide which environments a folder serves. Put a
    step meant for several environments in a folder with no `environment.yaml`, and limit it
    with `subtypes` if needed.
  - **`subtypes` apply wherever a step lives,** including an environment's own `steps/`. The
    env-owned steps under `skypilot/aws/steps/` and `skypilot/lsf/ibm-bluevela/steps/` no
    longer declare `subtypes`; their folder scopes them.
  - **New errors instead of a silent pick:** `Ambiguous space uri` when one base holds two
    equally specific copies of a name for the environment, and `Invalid step for its
    environment` when the copy that would be selected is in the environment's own `steps/` but
    doesn't admit it.
- **`gb build --param-file` (formerly `--parameters-path`) is repeatable and layers over the
  recipe's `parameters.yaml` instead of replacing it (#663).** `start`, `validate` and
  `describe` take the recipe's `parameters.yaml` (next to `build.yaml`) as the base, then each
  `--param-file` in the order given, then `--param`. The last layer wins per top-level key, and
  a list or map value is replaced whole. For example:
  `gb build start -f build.yaml --param-file site.yaml --param-file smoke.yaml`.
  Compatibility: a single `--parameters-path` used to replace `parameters.yaml` completely; now a
  key the file does not define falls back to `parameters.yaml`. The recipe's `parameters.yaml` is
  now always loaded, so a malformed one fails the build even when another `--param-file` is
  passed. A comments-only parameters file now counts as empty instead of failing, and the error
  for a file that is not a mapping names that file.
- **With podman, `make publish-image`'s `REGISTRY_USER`/`REGISTRY_PASSWORD` login is scoped to
  `$(REGISTRY)` instead of the whole registry host (#229),** so it no longer replaces your other
  logins on that host. docker still logs in per host.
- **Each custom step image gets its own repository, named after the step's directory (#229).**
  `steps/common.mk` now pushes `steps/<category>/<name>/<env>` as
  `$(REGISTRY)/steps/<category>/<name>/<env>:<IMAGE_VERSION>`, e.g.
  `quay.io/granitebuild/steps/train/openinstruct-sft/skypilot:oi-v0.3.0-6`, instead of
  `$(REGISTRY)/gb-step-<name>:<sha>`. The tag is the new `IMAGE_VERSION` (default: git short
  SHA); `IMAGE_NAME` and `IMAGE_TAG` can still be overridden, and an invalid repository name is
  refused. On quay.io, `publish-image` creates a step's new repository **public** through quay's
  API when `QUAY_API_TOKEN` is set (a push-created repository is private, and SkyPilot pulls
  anonymously); without the token it refuses a new repository rather than create a private one.
  Local image names change too, so retag or rebuild local images. No published
  step changes.
- **The `dpk` step moved into the `data` category: `space://steps/data/dpk` (#512).** This is
  the first step to move into a category directory under `steps/` (here `steps/data`).
  **Breaking change:** `space://steps/dpk` is no longer supported, so update builds to the
  new URI. The old URI is not blocked and may still resolve, but nothing
  guarantees which copy of the step it gets (for example, it does not prefer a space's own
  copy over the published one).

- **Lineage: artifact lineage now reads the local lineage index (#470).** The artifact
  page and `POST /lineage/graph` read `gb_lineage_index`, whatever
  `GBSERVER_LINEAGE_PROVIDER` is. **Upgrading a W&B deployment:** outside standalone the
  indexer indexes nothing until a checkpoint exists, and the Helm chart never seeds
  one: seed it by hand with `gbserver lineage-index-init --base-timestamp
  from-latest|all|<ISO-8601>` (`--force` to replace an existing one; on a fresh install
  with no jobs yet, add `--empty-from-beginning` to `from-latest`, which otherwise fails
  with "No job found in the source"). The chart value
  `deployments.lineageIndexer.seedOnStart` is removed: delete it from your overrides, as
  setting it now fails the render while the indexer is enabled. URI
  normalization also changed (bucket paths, LH revisions and hosts, git refs and
  subdirectories, trailing slashes), so an existing index must be rebuilt once, with
  the indexer stopped, by `gbserver lineage-index-init --rebuild` (deletes the index rows,
  keeping the table, and re-seeds at `all`; `--force` alone would keep the old-form rows
  beside the new). A running gbserver keeps working on the emptied table; a scan already
  in progress when the rows are deleted may write a few old-form rows back.
- **Lineage API (#470).** `POST /lineage/artifact` is removed; use `POST /lineage/graph`.
  `POST /lineage/search` is deprecated (flagged in the OpenAPI schema): it stays for
  existing external callers and will be removed once they move to
  `GET /lineage/jobs` / `POST /lineage/graph` with `job_ids`.
  `POST /lineage/graph` takes the query as a JSON body (`job_ids` included, so a grouped
  node expands in one walk), answers 422 for a bad `direction`,
  and limits callers who are not super admins to `max_depth <= 20` and the default
  per-level node ceiling. Nodes carry only cross-space fields; build and target-run
  ids appear on a run node only for callers who can read that job's space.
  `GET /lineage/jobs` answers 422 for `limit` outside 1–1000 or a negative `offset`, and
  503 when the index cannot be read. `GET /artifacts/?uri=...&match_lineage_uri=true`
  also matches an artifact by its normalized lineage URI (opt-in; without it `uri` is
  matched exactly), and each artifact, including `GET /artifacts/{id}`, carries a derived
  `lineage_uri`. `GET /lineage/jobs/{id}` adds `detail_denied`, true when the detail was
  withheld for lack of access rather than a store failure.
  `POST /lineage/graph` answers 422 (was 400) for out-of-range input, and both
  `POST /lineage/graph` and `GET /lineage/jobs` answer 503 rather than an empty 200 when
  the index cannot be read. `GET /lineage/jobs` caps `job_ids` at 1000, as the graph does.
- **Lineage index init (#470).** `gbserver lineage-index-init --rebuild` with provider
  `wandb` re-seeds at `all`, so the indexer back-fills the whole shared W&B project, not
  just this deployment's runs. A new `--empty-from-beginning` flag makes a fresh install
  with no history index its first jobs instead of skipping them.
- **`gb_targets` indexes (#470).** Schema adjust adds a `(status, finished_at)` index
  and, on SQLite 3.20+, an expression index on the JSON `finished_at`, so the standalone
  lineage indexer pages successful targets by keyset instead of scanning the table. On
  Postgres the index is built `CONCURRENTLY`, one process at a time. Index-only: no row
  is rewritten.
- **W&B lineage indexer (#470).** A W&B lineage run still `running` no longer stops
  indexing: the checkpoint waits behind it (up to 6 h, then it is indexed as it is),
  while later runs are indexed at once and recorded as `done_ahead` in the checkpoint
  (at most 5000; past that the scan waits as before). The W&B source has no settle
  grace: `createdAt` is set by the W&B server and nothing commits behind it, so a
  finished run is indexed on the next scan instead of ~6 minutes later
  (`GBSERVER_LINEAGE_INDEX_SETTLE_GRACE_S` applies to `gb_targets` only).
  With `provider=none` the index is fed from `gb_targets`, so artifact lineage appears
  about 6 minutes after a target finishes (it was ~30 s); see `docs/builds/lineage.md`.
  `gb_lineage_job.recorded_at` is stamped by the database inside the INSERT/UPDATE,
  so a retried write carries the time of the attempt that commits; it lives only in
  its column (not the JSON blob), and the `db` source's grace is 5 s instead of 360 s.
- **Lineage job owner and namespace (#470).** A job's owner and namespace are taken only
  when every index row of the job agrees; a job with conflicting (e.g. forged) rows has
  neither, and its detail fails closed.
- **Step drawer timestamps (#470).** `parseTimestamp` reads a zoneless ISO timestamp as
  UTC, so displayed times shift for users outside UTC.

- **The distillation steps moved to `steps/train/distill/` and were renamed so each name
  says what the step does.** Breaking: every `space://steps/distill/<step>` URI shipped in
  v0.3.8 and v0.3.9 is gone. Inputs and outputs are unchanged, and config is unchanged
  except in `logit-distill`, which:
  - drops the retired `ce_coef`, `log_student_entropy`, `entropy_guard_drop_frac`,
    `entropy_guard_baseline_steps`, `entropy_guard_patience` and `entropy_guard_action`
    keys and the `GOLD_MIN_TOKENS` env (all retired with the CE-anchor patch they
    belonged to);
  - **refuses `loss_arm: adaptive_kld`** (the vendored implementation does not compute
    the paper's objective; see `REJECTED_LOSS_ARMS` in `render_gold_config.py`);
  - **refuses `lmbda > 0` with `vllm_num_servers: 0`**, which used to generate through
    TRL's in-process path and, with `pad == eos`, masked every generated EOS out of the
    loss.

  Migrating a recipe means changing its `step_uri` to the new name below, dropping those
  keys, switching any `adaptive_kld` arm to `jsd` or `kl_interpolation`, and giving any
  on-policy run a vLLM server. The shipped recipes use the new URIs. Resume markers keep
  their recorded step names, so outputs written before the move are still skipped, not
  redone. `student-sft` and `teacher-logits` now download the pinned Mamba2 hub kernels
  (`mamba-ssm` and `causal-conv1d`, each pinned by commit) in-job on a cold cache by
  default (`GOLD_PREFETCH_KERNELS=1`); set it to `0` to require a pre-warmed cache
  instead.

  | Was | Now `space://steps/train/distill/…` |
  |---|---|
  | `distill/corpus-sources` | `data-mixture` |
  | `distill/corpus-prep` | `data-prep` |
  | `distill/corpus-pin-check` | `data-validate` |
  | `distill/tokenizer-align` | `tokenizer-align` |
  | `distill/sft` | `student-sft` |
  | `distill/gold` | `logit-distill` (GOLD, ULD, AKL, DistiLLM-2-style objectives) |
  | `distill/logit-precompute` | `teacher-logits` |
  | `distill/vllm-server` | `student-rollout-generator` |
  | `distill/eval` | `teacher-divergence` (not to be confused with the `steps/eval/` category) |
  | `distill/gen-smoke` | `repetition-check` |
  | `distill/probe` | `preflight-check` |

  The shared Accelerate/DeepSpeed launch config moved inside the GPU image from
  `steps/distill-gold-train/configs/` to `configs/`; recipes that set `DS_CONFIG` use
  `configs/deepspeed/accelerate_deepspeed_zero3.yaml`.
- **`hf-export` moved to `steps/aux/hf-export` with its own image.** Breaking:  `space://steps/distill/hf-export` is now `space://steps/aux/hf-export`. Config, inputs
  (`train_output`, `expected_tokenizer`) and the `hf_model` output are unchanged, and an
  export made by the old step is recognised and skipped, not redone. The step no longer runs
  on the distillation CPU image: it has its own CPU image,
  `quay.io/granitebuild/steps/aux/hf-export` (no torch), and
  imports nothing from the distillation collection, so any recipe that trains with HF Trainer
  or accelerate can use it.
- **The eval and teardown steps moved from the LSF `ibm-bluevela` subtype override to the
  generic skypilot tree** (`bfcl-eval`, `bfcl-export`, `sage-eval`, `sage-export`,
  `combined-export`, `skypilot-teardown`, under
  `configurations/assets/environments/skypilot/steps/`). They were BlueVela-only overrides
  before; they now resolve for every skypilot subtype — **including Kubernetes**, whose
  environments previously did not see them. Their step URIs are unchanged.
- **The distill GPU steps refuse a GPU undercount instead of rewriting
  `CUDA_VISIBLE_DEVICES` (#604).** `gold`, `sft`, `logit-precompute`, `eval`, `vllm-server`
  and `gen-smoke` compare `SKYPILOT_NUM_GPUS_PER_NODE` with `nvidia-smi -L` and fail fast,
  naming #604, when the container can access fewer GPUs than were allocated, as hf-peft-sft
  does.

- **Published steps declare named inputs and read them from bindings (#453).** Every
  `step.yaml` under `configurations/assets/` now has an `inputs:` block. A target that
  doesn't bind a required input now fails build validation before anything is queued.
  Before, the step failed on the cluster with an unresolved `{{ bindings… }}` string.
  Steps with no fixed set of input names declare `inputs: {allow_unknown: true}` and say
  why: `byoc`, `skypilot-teardown`, `corpus-sources`, `gen-smoke`, `probe`, the servers,
  the export gates, `hello`, `sage`, and the AWS S3-mounted evals.
  **Breaking change for the distill steps, which are not in any tag yet.** Their inputs
  are required, and the old config keys are no longer read. A recipe that still sets one
  of those keys fails at run time with a message naming the input to bind.

  | Step | Old config key | Input |
  |---|---|---|
  | `distill/tokenizer-align` | `align_config.teacher_model`, `.student_model` | `teacher`, `student`, `chat_template` (optional) |
  | `distill/corpus-prep` | `corpus_config.dataset`, `.tokenizer` | `source_dataset`, `tokenizer` |
  | `distill/corpus-pin-check` | `pin_check_config.tokenizer_dir` | `tokenizer` |
  | `distill/sft` | `sft_config.student_model_path`, `.corpus_path` | `student`, `corpus` |
  | `distill/gold` | `gold_config.model_name_or_path`, `.teacher_model_name_or_path`, `.dataset_name`, `.vllm_server_url` | `student`, `teacher`, `corpus`, `vllm` (optional) |
  | `distill/eval` | `eval_config.student_model`, `.teacher_model` | `student`, `teacher` (optional) |
  | `distill/hf-export` | `export_config.train_output_dir`, `.expect_tokenizer_from` | `train_output`, `expected_tokenizer` (optional) |
  | `distill/logit-precompute` | `precompute_config.corpus_path`, `.teacher_model_path`, `.teacher_tokenizer_path` | `corpus`, `teacher`, `teacher_tokenizer` |
  | `distill/vllm-server` | `vllm_config.model_path` | `model` |

  The shipped recipes, examples and step READMEs bind the new names. See
  [Declaring inputs](docs/steps/README.md#declaring-inputs).

- **SkyPilot HPC (SLURM/LSF) — `cluster_ssh_configs` may list several candidate login
  nodes for one cluster, with automatic failover.** A `Host` block's `HostName` may now be a
  list of interchangeable login hostnames. gbserver picks one per launch and, if provisioning
  fails with a *transient SSH control-plane* error (a late banner, a wedged session, a
  key-exchange reset — the class behind SkyPilot's opaque `Failed to get partitions for
  cluster …`), rewrites `~/.<cloud>/config` to the next candidate before the launch is
  retried, reusing the ordinary provision retry budget. Capacity failures and SSH *auth*
  rejections do not trigger failover. Failover is **launch-time only**: SkyPilot pins the
  chosen login node into the provisioned cluster's handle, so once a cluster is up its status
  polling, log streaming, and teardown all stay on that node (unlike the native-LSF SSH
  tunnel, which can re-pick per operation). The per-launch pick is **sticky** — a candidate
  already written for the cluster (e.g. by a parallel launch that just failed over) is kept
  rather than re-randomized back onto a wedged node; otherwise a random candidate is chosen to
  spread load. When the infra names no cluster (a bare `lsf`/`slurm` infra) but the env
  declares exactly one host for that cloud, that host is the launch target, so its candidates
  still fail over. There is **no** up-front reachability probe: the earlier probe held a
  login-node SSH slot waiting on slow banners and starved the control SSH SkyPilot opens next,
  failing healthy clusters, so it (and `GBSERVER_SKYPILOT_SSH_PROBE_TIMEOUT_S` / the per-host
  `ssh_probe_timeout_s` key) has been removed. A single-node cluster simply retries the same
  node, so a genuine outage still surfaces the real error.
- **SkyPilot — a GPU count without a GPU request now logs a warning (#536).** The launcher does
  not read `compute_config.num_gpus_per_node`; GPUs come only from
  `launcher_config.resources.accelerators` (or an `instance_type` that has them). A step that
  sets `num_gpus_per_node > 0` with neither used to launch on a node with no GPU, silently. It
  still launches, but gbserver now warns, once per step. `Skypilot_managed` now layers resources
  like the unmanaged launcher, through the same helpers: the build's `launcher_config.resources`
  wins over the step launcher's, and `cloud`/`infra`/`zone`, `instance_type` and `use_spot` are
  honored (it used to drop the build's resources and those keys, so a managed GPU request could
  silently land on a default CPU node). It logs the same warning. The `nemo-rl` step no longer renders `accelerators` or
  defaults `num_gpus_per_node: 8`: a build sets `launcher_config.resources.accelerators` (e.g.
  `"A10G:1"`) or a GPU `instance_type`, and the step pins NeMo-RL's `cluster.gpus_per_node` from
  `SKYPILOT_NUM_GPUS_PER_NODE` (override with the new `nemo_rl_config.gpus_per_node`). A `nemo-rl`
  build with neither fails at step creation, before anything is provisioned. **Breaking change
  for `nemo-rl` builds:** `compute_config` `gpu_model` / `num_gpus_per_node` are no longer read;
  move the GPU request to `accelerators`.
- **SkyPilot — the post-build workdir-cleanup VM is right-sized to 1 CPU.** `teardown_skypilot`
  launches a throwaway VM that mounts the shared filesystem and `rm -rf`s the run's per-run
  workdir after the build has finished. It now requests a single vCPU (the smallest
  schedulable allocation, and the easiest to place when an HPC cluster is near capacity)
  instead of SkyPilot's oversized default; on non-HPC clouds it adds a 2-GiB memory floor so
  the pick does not land on a `t2.nano`/`micro` too small for SkyPilot's Ray runtime (SLURM
  and LSF match CPUs directly and do not track memory, so the floor is skipped there). The
  warning logged when this cleanup fails now states explicitly that it does **not** affect the
  build outcome — the build has already completed — and that the only consequence is a leaked
  per-run tree to be reaped separately, so its `Failed to provision …` detail is not misread
  as the build failing.
- **SkyPilot environment — secrets are now injected least-privilege (declared-only).**
  SkyPilot previously dumped the entire resolved space/user secret bag into the launched
  task environment. It now injects **only** the secrets a step declares under
  `config.skypilot.secrets.secret_names_to_use_as_env_variable`, matching the LSF and
  Kubernetes environments. **Migration:** a SkyPilot build that relied on an *undeclared*
  secret reaching the task env will now find that variable unset at runtime, with no
  launch-time error — add the secret to `secret_names_to_use_as_env_variable` to restore
  it. (Builds whose secret bag contained a non-identifier name were already failing to
  launch on SkyPilot before this change; see Fixed.)
- **SSO settings are now `GBSERVER_OIDC_*` (#723):** `CLIENT_ID`, `CLIENT_SECRET`,
  `CALLBACK_URL`, `REMOTE_CALLBACK_URL`, `ISSUER`, `JWKS_URI`, `AUTHORIZE_URL`, `TOKEN_URL`,
  `USERINFO_URL`. The legacy `GBSERVER_IBMID_*` names are still accepted; the new name wins if
  both are set, and an empty value counts as unset.
- **Plain `gb auth login --sso` exits right away when a local browser can't take the redirect
  (#723)**: in an SSH session, or with no display or browser. It points to `--remote`. Set
  `BROWSER` to skip the check, e.g. for VS Code Remote-SSH or SSH port forwarding.

### Deprecated

- **`gb build --parameters-path`, in favor of `--param-file` (#663).** The option now adds a
  layer instead of naming the one parameters file, and `--param-file` pairs with `--param`.
  `--parameters-path` still works on `start`, `validate` and `describe` as an alias, and the two
  names can be mixed; layers apply in command-line order. It will be removed in a future
  release.
- **Config keys that carried an input path in steps shipped in v0.3.x (#453).** These
  steps now declare the input as optional and read the binding when it is bound. A bound
  input wins over the key. When the input isn't bound, the step falls back to the old
  key and prints a `DEPRECATED` warning. A key set to the same value as the bound input
  also gets a `DEPRECATED` note asking the recipe to drop it. The fallback will be removed
  in a future release.

  | Step | Deprecated key | Input |
  |---|---|---|
  | `sage-eval` (LSF) | `sage_eval_config.model_path` | `model` |
  | `bfcl-eval` (LSF) | `bfcl_config.model_path` | `model` |
  | `openinstruct-sft` (LSF) | `sft_config.model_path` | `model` |
  | `openinstruct-rl` (LSF) | `rl_config.rm_server_url`, `rl_config.code_server_url` | `rm_url`, `code_url` (`mem://`) |
  | `dpk` | `dpk_config.input_path` | `docs` |

  The AWS `openinstruct-sft` also declares an optional `model` input, but its
  `sft_config.model_path` is **not** deprecated: it is the Hugging Face id the trainer
  downloads when `model` is not bound, so it stays as a real default with no warning.

### Removed

- **The granite-switch LSF answerability-sr recipe (#725).** It shipped in v0.3.9 as
  `recipes/granite-switch/lsf/answerability-sr` and moved to
  `recipes/granite-switch/skypilot/lsf-bluevela/answerability-sr` in #639. It ran on SkyPilot
  LSF, and since #617 the SkyPilot release `pyproject.toml` pins has no LSF cloud (see
  `docs/environments/skypilot-lsf.md`); #631 tracks retiring LSF from the docs.
  `skypilot/aws/answerability-sr` builds the same kind of Shadow Residual adapter, with the same
  training config, from data you bring. The last version of the recipe (`README.md`,
  `build.yaml`, `parameters.yaml`) is at commit `ebabfa44`. From the repo root, read a file with
  `git show ebabfa44:recipes/granite-switch/skypilot/lsf-bluevela/answerability-sr/<file>`, or
  restore the directory with
  `git checkout ebabfa44 -- recipes/granite-switch/skypilot/lsf-bluevela`.
  The restored recipe runs only with a SkyPilot build that has the LSF cloud, such as the fork
  that tag `pre-slurm-only-2026-11-02` pins.
- **The LSF and single-pair distillation recipes (#664).** Replaced by
  [`recipes/distill/`](recipes/distill/README.md): one `distill` pipeline with off-policy,
  ramp and on-policy λ modes (`LMBDA_MODE`) and a `smoke.yaml` layer, plus `distill-eval`
  for benchmarking existing exports. LSF is being decommissioned, so the LSF recipes are
  deleted rather than migrated. Removed, with their unit tests:
  - `recipes/granite4-gold-distillation/` (all of it: `distill-gold`, `distill-pipeline-smoke`,
    `distill-sft-gold`, `gold-blend-smoke`, `gold-onpolicy-smoke`, `gold-onpolicy-full-smoke`,
    `gold-smoke`, `gold-sweep-100`, `gold-sweep-smoke`);
  - `recipes/granite4-350m/lsf/distill-{probe,smoke,stage1,stage1-v2,onpolicy-v2,checkpoint-eval}`.

### Fixed

- **logit-distill: `loss_arm: uld` trains on what it aligns, or fails loudly (#754, #760).**
  It had never run end to end, and at the step's defaults it could not align a single
  multi-turn row (or, on a ChatML template, any row).
  - **New render refusal:** `loss_arm: uld` requires `last_message_only: true`. The ULD
    teacher is the last assistant turn alone, so with `false` the student's span starts at
    the first turn, and the loss is exactly zero. The step's renderer refuses `false` or
    unset, and, for this arm alone, now emits `last_message_only` off-policy (before, it
    was dropped whenever `vllm_num_servers` was 0). `CustomGOLDConfig` refuses it at
    startup for a config that bypasses the renderer. The distill recipe refuses
    `LOSS_ARM=uld` without `--param LAST_MESSAGE_ONLY=true` before submission, and the
    sampled-OPD loss without it likewise. Every other arm renders byte-identically.
  - **Changed numerics:** the non-hybrid ULD loss (trl 0.26.2's, vendored) read each
    completion token's distribution at the token itself, one position late. It now reads
    the logits one position earlier (start-1), as the hybrid ULD path and the JSD path do,
    so ULD losses differ from earlier runs. A row whose alignment yields no groups now
    contributes no loss; before, it raised a shape `RuntimeError` or scored unaligned
    positions against each other.
  - **New runtime `RuntimeError` and metrics:** the trainer now sums ULD rows over all ranks
    at every optimizer step. It raises when every row of a step is unaligned, or when more
    than half the rows of a logging window are (`uld_max_unaligned_fraction`, default 0.5;
    the final partial window is judged at train end). The log line carries the new
    `train/alignment_rows`, `train/alignment_unaligned_rows` and
    `train/alignment_unaligned_rate`, all summed over ranks.
  - A completion starting at position 0 now raises `ValueError` on both ULD paths.
    `masking.py check-config` now requires `last_message_only: true` for `uld` and
    `sampled_opd` instead of reporting it.
  - Separately, the checkpoint watcher's `sleep` no longer holds rank 0's output open for
    up to 60 s after the script exits.
- **Bash, Docker and SkyPilot: `hf://` inputs that differ only in type, host or bucket path no
  longer share a pull dir (#715).** `<cache>/<owner>/<repo>/<revision>` ignored all three, so a
  model and a same-named dataset overwrote each other's files, two paths of one bucket synced into
  one dir, and on Docker both were mounted at one container path. A dataset or space is now pulled
  under `datasets/` or `spaces/`, each bucket input under `buckets/<owner>/<bucket>/<hash of its
  path>`, and content on another host under `@<host>/`; models on huggingface.co keep their dirs,
  so their caches stay valid. Docker refuses an input whose content another `cache_path` already
  mounted, instead of failing every later container on a duplicate mount point, and passes mounts
  as structured bind mounts, so a `:` in a path in a repo no longer breaks the volume spec. K8s
  and LSF are unchanged. **Behavior change:** cached datasets, spaces and buckets are downloaded
  once more into their new dirs. A dataset or space pulled before this change still sits in the
  model layout, so if one shared an owner/repo with a model you use, delete
  `<cache>/<owner>/<repo>/<revision>` once, or the model pull adds to its files. The cache is the
  store's `cache_path`, else `~/.cache/gbserver/hf` (Bash, Docker) or `<shared_workdir>/hf_cache`
  (SkyPilot).
- **openinstruct-sft, hf-peft-sft: an `args` value that starts with `-` is refused at step
  creation (#613).** Each value is its own argv word after the step's flags, so a value
  such as `--output_dir=/x`, or a list item `--output_dir`, was read by the trainer's
  argparse as a flag and could override a step-owned one (`output_dir`, `report_to`,
  `push_to_hub`). A value or list item that starts with `-` followed by anything but a
  digit or `.` (`-x`, `--x`, `--`) is now refused; negative numbers (`-1`, `-0.5`, `-.5`)
  and a lone `-` pass. The check lives in `validate_args` in
  `steps/train/shared/trainer_args.j2`.

- **Legacy `space://steps/openinstruct-sft` on SkyPilot/AWS refuses multi-node builds
  (#683).** The hand-authored step trains on one machine (`accelerate launch
  --num_machines 1`), but nothing checked `num_nodes`, so with `num_nodes: 2` SkyPilot
  provisioned 2 nodes that each trained on their own (the #609 bug, fixed for the framework
  step in #625). It now sets `single_node: true`, and gbserver refuses any resolved
  `num_nodes` other than 1 at step creation. The `skypilot/lsf/ibm-bluevela` copy is
  unchanged: the pinned SkyPilot 0.14.2 has no LSF cloud, so it can't launch (LSF is being
  retired, #631).

- **SkyPilot SLURM: a step's `total_memory_per_node` is sent as `--mem` on clusters that track
  memory (#657).** Before, it was always dropped on SLURM, so jobs got SkyPilot's 1 GB per CPU.
  The built-in `hfpull`/`hfpush` (10Gi) were OOM-killed on BlueVela. gbserver now reads the
  cluster's `SelectTypeParameters` and sends the step's memory only on `CR_*_Memory` clusters;
  `CR_CPU`/`CR_Core` clusters and LSF are unchanged. **Behavior change:** on a memory-tracking
  cluster a step now requests its declared memory, so one that declares more than the partition
  allows sits PENDING instead of running. Set `resources.memory` to override it per step
  (`memory: 0` omits `--mem`). The BlueVela environment's `resources: { memory: 8 }` stopgap
  for hfpull/hfpush is removed. See
  [skypilot-slurm.md](docs/environments/skypilot-slurm.md#step-memory-total_memory_per_node).
- **K8s steps no longer fail for email usernames (ibmid auth) (#630).** The gbstep chart rendered
  `run_metadata.username` verbatim as a label, so the API server rejected the AppWrapper.
  `launch_helm` now passes a sanitized username (`user@x.com` -> `user_x.com`) chart-wide.
- **`space://steps/train/openinstruct-sft` refuses multi-node builds set through
  `launcher_config.num_nodes`, and abbreviated `args` keys (#609).** Before,
  `launcher_config.num_nodes: 2` passed the step's single-node check, so SkyPilot ran two
  independent single-node trainings, and `1.0` was refused. Now both `compute_config` and
  `launcher_config` must be 1 (`1.0` and strings like `"01"` count, as gbserver reads them).
  An `args` key that abbreviates a step-owned flag (`output_d`), is a `no_*` key (it takes no
  value), or is not a bare name is refused at step creation, and the image (`oi-v0.3.0-9`)
  parses `finetune.py`'s flags with prefix matching off, so `learning_rat` fails the run
  instead of setting `learning_rate`. `dataset_cache_mode` is now step-owned and pinned to
  `local`: `hf` pushed the tokenized dataset to the Hub whenever `HF_TOKEN` was set. The
  `Step:` status message now carries the whole progress line (LR, loss, throughput), not just
  `Step: <n>`. The image also moves `Authlib` from 1.9.0 to 1.9.5 for GHSA-8m4g-j4wp-wgcg
  (JWS/JWT forgery with an empty HMAC key). The legacy hand-authored
  `space://steps/openinstruct-sft` assets are unchanged; see #569.

- **Workload-status events now carry the `message` field step monitor rules send (#598).**
  Before, the `Step:` progress events were dropped and gbserver logged a traceback for each
  matching line. These events are now stored and published once per matching log line, and the
  build runner logs the message with the status.

- **`make publish-step` no longer reports a published single-platform image as missing (#568).**
  Without `skopeo`, its image check used `podman manifest inspect`, which fails on a
  single-platform image. It now uses `steps/scripts/registry-tag-status.sh`, the same check as
  `publish-image`.

- **SkyPilot launch crash on non-identifier secret names.** Whole-bag secret injection
  turned every secret name into a task env-var key, so a name such as `rits-access`
  violated SkyPilot's env-key naming rules and failed the launch. Declared secrets are
  mapped to valid env-var names via `secret_names_to_use_as_env_variable`, so hyphenated
  (and otherwise non-identifier) secret names no longer break SkyPilot launches.
- **SSO login with several rest-server workers (#723).** Pending logins lived in one worker's
  memory, so the browser could show "Session not found or expired" or the CLI could poll
  `/status` without end. The new CLI uses the stateless `/oidc/*` endpoints instead.

### Security

- **Reflected XSS on the legacy SSO `/api/v1/auth/callback` page (#723).** The provider's
  `error_description` was put into the HTML unescaped; it is now escaped.
