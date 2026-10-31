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
  model). Custom image `quay.io/granitebuild/steps/train/hf-peft-sft/skypilot:peft-v0.21.2-2`
  (torch 2.15.0 / CUDA 13.2) with a fully hashed dependency lock. Build tests on skypilot/aws
  (QLoRA, aLoRA) and BlueVela SLURM (LoRA/FSDP, QLoRA/DeepSpeed).
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

### Changed

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

### Deprecated

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

### Fixed

- **`make publish-step` no longer reports a published single-platform image as missing (#568).**
  Without `skopeo`, its image check used `podman manifest inspect`, which fails on a
  single-platform image. It now uses `steps/scripts/registry-tag-status.sh`, the same check as
  `publish-image`.

- **SkyPilot launch crash on non-identifier secret names.** Whole-bag secret injection
  turned every secret name into a task env-var key, so a name such as `rits-access`
  violated SkyPilot's env-key naming rules and failed the launch. Declared secrets are
  mapped to valid env-var names via `secret_names_to_use_as_env_variable`, so hyphenated
  (and otherwise non-identifier) secret names no longer break SkyPilot launches.
