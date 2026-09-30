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

- **SkyPilot HPC (SLURM/LSF) — `cluster_ssh_configs` may list several candidate login
  nodes for one cluster, with automatic failover.** A `Host` block's `HostName` may now be a
  list of interchangeable login hostnames. gbserver picks one at random per launch (spreading
  load), and if provisioning fails with a *transient SSH control-plane* error (a late banner,
  a wedged session, a key-exchange reset — the class behind SkyPilot's opaque `Failed to get
  partitions for cluster …`) it rewrites `~/.<cloud>/config` to the next candidate before the
  launch is retried, reusing the ordinary provision retry budget. Capacity failures and SSH
  *auth* rejections do not trigger failover. There is **no** up-front reachability probe: the
  earlier probe held a login-node SSH slot waiting on slow banners and starved the control
  SSH SkyPilot opens next, failing healthy clusters, so it (and
  `GBSERVER_SKYPILOT_SSH_PROBE_TIMEOUT_S` / the per-host `ssh_probe_timeout_s` key) has been
  removed. A single-node cluster simply retries the same node, so a genuine outage still
  surfaces the real error.
- **SkyPilot HPC (SLURM/LSF) — the provision-retry cleanup VM is right-sized to 1 CPU.** The
  bounded teardown that runs between provision retries now requests a single CPU (with a
  cloud-only memory floor), matching the EFS/cleanup-VM right-sizing in #427/#430, and the
  warning logged when that teardown does not finish in time was reworded to say the launch is
  retried anyway rather than implying the cluster leaked permanently.
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
