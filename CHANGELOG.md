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

- **SkyPilot HPC (SLURM/LSF) — the login node is now chosen by a mandatory reachability
  probe; `GBSERVER_SKYPILOT_SSH_PROBE_TIMEOUT_S=0` no longer disables probing.**
  `cluster_ssh_configs` may now list several candidate `HostName`s for one cluster; before
  an HPC launch gbserver SSH-probes the candidates (a trivial `echo`, in random order) and
  configures SkyPilot to use the first reachable one, failing fast with a clear error naming
  the alias and the hostnames tried when none answers — instead of SkyPilot's opaque
  `Failed to get partitions for cluster …`. Because the probe now *selects* the login node
  the launch uses, it always runs and can no longer be switched off. **Migration:** a
  deployment that set `GBSERVER_SKYPILOT_SSH_PROBE_TIMEOUT_S=0` to opt out of the probe will
  now probe with the built-in default `ConnectTimeout` (30s); the variable (and the per-host
  `ssh_probe_timeout_s` synthetic key) now only sets that timeout, and a non-positive value
  falls back to 30s. A login node that was genuinely unreachable — and previously slipped
  through to an opaque post-launch failure — will now fail the launch up front.
- **SkyPilot environment — secrets are now injected least-privilege (declared-only).**
  SkyPilot previously dumped the entire resolved space/user secret bag into the launched
  task environment. It now injects **only** the secrets a step declares under
  `config.skypilot.secrets.secret_names_to_use_as_env_variable`, matching the LSF and
  Kubernetes environments. **Migration:** a SkyPilot build that relied on an *undeclared*
  secret reaching the task env will now find that variable unset at runtime, with no
  launch-time error — add the secret to `secret_names_to_use_as_env_variable` to restore
  it. (Builds whose secret bag contained a non-identifier name were already failing to
  launch on SkyPilot before this change; see Fixed.)

### Fixed

- **SkyPilot launch crash on non-identifier secret names.** Whole-bag secret injection
  turned every secret name into a task env-var key, so a name such as `rits-access`
  violated SkyPilot's env-key naming rules and failed the launch. Declared secrets are
  mapped to valid env-var names via `secret_names_to_use_as_env_variable`, so hyphenated
  (and otherwise non-identifier) secret names no longer break SkyPilot launches.
