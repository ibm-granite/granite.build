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

### Changed

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

- **SkyPilot launch crash on non-identifier secret names.** Whole-bag secret injection
  turned every secret name into a task env-var key, so a name such as `rits-access`
  violated SkyPilot's env-key naming rules and failed the launch. Declared secrets are
  mapped to valid env-var names via `secret_names_to_use_as_env_variable`, so hyphenated
  (and otherwise non-identifier) secret names no longer break SkyPilot launches.
