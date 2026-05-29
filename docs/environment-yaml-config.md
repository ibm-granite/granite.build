# Environment YAML Configuration Reference

Two YAML files control how environments and steps interact:

| File | Lives in | Purpose |
|------|----------|---------|
| `environment.yaml` | Environment asset repo | Declares the environment type, credentials, retry behaviour, and asset stores |
| `step.yaml` (the `environment_configs` section) | Step asset repo | Declares how a step should be launched and monitored per environment type |

The `config` section of `step.yaml` also carries per-environment fields (`lsf`, `k8s`) that are read by the environment at launch time.

---

## `environment.yaml`

### Top-level structure

```yaml
name: <string>          # Human-readable name (informational)
type: <string>          # Environment class to instantiate: K8s, Lsf, Bash, Docker, Skypilot, etc.
config:                 # Environment-type-specific config (see sections below)
  ...
assetstores:            # Asset stores accessible from this environment
  - store_uri: <uri>
    load:
      - mode: <mode>
        config: {}
    push:
      - mode: <mode>
        config: {}
```

---

### `K8s` environment config

```yaml
name: my-k8s-env
type: K8s
config:
  namespace: granite-build          # Required. Kubernetes namespace for all resources.

  authentication:
    kube_config: my_kubeconfig      # Secret name whose value is a kubeconfig YAML string.
                                    # If omitted, falls back to the in-cluster or default
                                    # kubeconfig location on the server.
    kube_context: my-context        # Secret name whose value is the kubeconfig context to use.
                                    # Optional; uses the kubeconfig default context if absent.
    ssl_verification: true          # Whether to verify the K8s API server TLS certificate.
                                    # Default: true. Set to false for self-signed clusters.

  messaging:
    authentication_secret_name: rabbitmq_secret
                                    # Secret name whose value is a JSON RabbitMQ credentials
                                    # object. Required when using sidecar_monitor or
                                    # event_monitor.

  retry:
    enabled: true                   # Master switch. Default: true.
    max_retries: 3                  # Maximum number of retry attempts. Default: 3.
    strategies:                     # Optional override. When absent, uses the K8s defaults:
      - type: UnhealthyInsufficientPods  # UnhealthyInsufficientPods, PodEviction, NCCLError.
      - type: PodEviction
        object_types: [AppWrapper]
      - type: NCCLError

  targetsteprun_assets_dir: /gb-read-write
                                    # Mount path inside the pod where step assets are copied.
                                    # Default: /gb-read-write.

assetstores:
  - store_uri: lh://my-namespace    # Lakehouse store URI
    load:
      - mode: dmf_pull              # Supported modes: dmf_pull (default for LH in K8s)
        config:
          step_uri: lh://steps/lhpull   # Optional: override the lhpull built-in step URI
    push:
      - mode: dmf_push
        config:
          step_uri: lh://steps/lhpush

  - store_uri: cos://my-bucket      # IBM COS store URI
    load:
      - mode: cos_rclone
        config:
          step_uri: lh://steps/cosrclone
    push:
      - mode: cos_rclone

  - store_uri: hf://my-org/my-model # HuggingFace store URI
    load:
      - mode: hf_pull
```

---

### `Lsf` environment config

```yaml
name: my-lsf-env
type: Lsf
config:
  workspace:
    local_dir: /tmp/gbserver/lsf    # Local staging directory (non-SSH mode only).
                                    # Default: <DEFAULT_ROOT_WORKSPACE_DIR>/env_lsf
    remote_dir: /gpfs/workspace     # Remote directory on the LSF cluster.
                                    # Required in SSH mode; sets the base path for all
                                    # copied job scripts and outputs.

  authentication:
    use_ssh: true                   # Whether to use SSH to reach the LSF cluster.
                                    # Default: true.
    copy_method: scp                # File transfer method for copying assets to the
                                    # remote LSF node. One of: "scp", "rsync".
                                    # Default: scp.
    ssh_port: 22                    # SSH port. Default: 22.
    ssh_max_sessions: 10            # Maximum concurrent SSH multiplexed sessions.
                                    # Default: 10.
    login_nodes:                    # List of SSH login nodes. At least one required
      - login1.cluster.example.com  # when use_ssh: true. Nodes are tried round-robin;
      - login2.cluster.example.com  # unreachable nodes are skipped automatically.
    login_node_username: myuser     # SSH username.
    login_node_ssh_key: my_ssh_key  # Secret name whose value is the SSH private key
                                    # (PEM format). Required when use_ssh: true.
    ssh_host_key_verification: true # Whether to verify the SSH host key.
                                    # Default: true. Set to false for dev/test clusters
                                    # with self-signed host keys.
    ssh_timeout: 5                  # Timeout in seconds for SSH reachability probe.
                                    # Default: 5.

  retry:
    enabled: true                   # Master switch. Default: true.
    max_retries: 3                  # Default: 3.
    strategies:                     # Optional override. When absent, uses the Lsf default:
      - type: LsfTransientError     # LsfTransientErrorRetryStrategy.

assetstores:
  - store_uri: lh://my-namespace
    load:
      - mode: dmf_pull              # Injects a lhpull built-in step before the main job.
        config:
          cache_path: /gpfs/cache/lh    # Required. Path on the cluster where LH data is
                                        # cached after dmf_pull.
          use_aspera: false             # Optional. Use Aspera for transfer. Default: false.
          step_uri: lh://steps/lhpull  # Optional: override the lhpull step URI.
    push:
      - mode: dmf_push
        config:
          step_uri: lh://steps/lhpush

  - store_uri: cos://my-bucket
    load:
      - mode: cos_pull              # Injects a cosrclone built-in step.
        config:
          cache_path: /gpfs/cache/cos   # Required. Path on the cluster where COS data
                                        # is downloaded.
```

---

### `Skypilot` environment config

The `Skypilot` environment provisions a fresh SkyPilot cluster for each step
via `sky.launch()` and tears it down per-step on cleanup. The `config:` block
is intentionally small — most knobs live on the per-step launcher (see the
[Skypilot launcher](#skypilot-launcher-and-monitor-types) section below).

```yaml
name: my-skypilot-env
type: Skypilot
config:
  default_cloud: k8s                # SkyPilot infra to provision on when a step does
                                    # not override it. Forwarded as the `infra` arg
                                    # to `sky.Resources`. Common values: "k8s",
                                    # "slurm", "aws", "gcp", "runpod".
                                    # Default: "k8s".

  idle_minutes_to_autostop: 10      # Stop the SkyPilot cluster after N idle minutes
                                    # (success or failure). Default: 10. Set to 0
                                    # for near-immediate autostop, or null to
                                    # disable autostop entirely. Per-step cleanup
                                    # already runs `sky down` after each step
                                    # finishes, so this is only a safety net for
                                    # crashed processes. SLURM does not support
                                    # autostop — gbserver ignores this value when
                                    # the resolved cloud is `slurm` to avoid a
                                    # `sky.launch` provisioning failure.

  cluster: <slurm-cluster-name>     # Optional, SLURM-only convenience field.
                                    # When `default_cloud: slurm`, this name is
                                    # composed into `infra=slurm/<cluster>` for
                                    # every step that does not set its own
                                    # `resources.infra`. Other clouds: ignored —
                                    # use `resources.infra` on the launcher instead.

  zone: <zone>                      # Optional. Forwarded to `sky.Resources(zone=...)`
                                    # for steps that don't set `resources.zone`.

  shared_workdir: <path>            # Optional. Path to a filesystem mounted on
                                    # *every* worker the Skypilot env launches
                                    # against. Used as the default base directory
                                    # for gbserver-managed cross-step caches
                                    # (currently the HF asset cache; other stores
                                    # may follow). Each `sky launch` is a fresh
                                    # allocation, so cross-step state requires a
                                    # shared FS provisioned by the cluster admin —
                                    # gbserver does not create or mount it.
                                    # Examples per backend:
                                    #   slurm:  /shared             (NFS / Lustre / GPFS)
                                    #   k8s:    /mnt/shared         (RWX PVC)
                                    #   aws:    /mnt/efs            (EFS / FSx)
                                    #   gcp:    /mnt/filestore      (Filestore)
                                    # When unset, gbserver-managed caches fall
                                    # back to `~/.cache/gbserver/<store>` on the
                                    # worker, which only works when consecutive
                                    # steps land on the same machine.
                                    # When set, the same path is also exported to
                                    # every step's `run` command as the
                                    # `GB_SHARED_WORKDIR` environment variable, so
                                    # step yamls can stage cross-step state
                                    # without hard-coding the cluster path
                                    # (e.g. `mkdir -p "$GB_SHARED_WORKDIR/outputs"`).
                                    # gbserver also provisions a per-target-run
                                    # subdir under
                                    #   ${shared_workdir}/builds/<build_id>/runs/<targetrun_id>/
                                    # which is exported as `GB_BUILD_WORKDIR` and
                                    # set as the *initial CWD* of every step's
                                    # `run` command — step authors can write
                                    # outputs with relative paths and get implicit
                                    # per-run isolation. The dir is created lazily
                                    # before the first step runs and `rm -rf`'d at
                                    # target-run teardown. Retries get a fresh dir.

assetstores:
  - store_uri: space://assetstores/hf      # HuggingFace Hub asset store.
    load:
      - mode: hf_pull                       # The only supported load mode.
        config:
          step_uri: space://steps/my_hfpull      # Optional override of the builtin
                                                  # hfpull step URI. The default is
                                                  # the gbserver builtin (which
                                                  # works for SkyPilot bare-node
                                                  # SLURM/cloud setups out of the
                                                  # box).
          cache_path: /tmp/hf_cache              # Optional cache dir on the
                                                  # SkyPilot worker where pulled
                                                  # snapshots are written. When
                                                  # unset, defaults to
                                                  # `{shared_workdir}/hf_cache` if
                                                  # the env declares a
                                                  # `shared_workdir`, otherwise
                                                  # `~/.cache/gbserver/hf` on the
                                                  # worker. Set explicitly only to
                                                  # override that default for this
                                                  # one assetstore.
    push:
      - mode: hf_push
        config:
          step_uri: space://steps/my_hfpush      # Optional override of the builtin
                                                  # hfpush step URI.

  - store_uri: space://assetstores/cos     # IBM COS / S3-compatible store.
    load: [...]
    push:
      - mode: cos_rclone
        config:
          step_uri: space://steps/s3push         # Optional override of s3push.
```

**Notes**

- `assetstores` resolution dispatches to `pullasset_hfstore` /
  `pushasset_hfstore` / `pushasset_cosstore` in
  [skypilot.py](../src/gbserver/environment/skypilot.py), which queue the
  configured `step_uri` as a separate target step on the SkyPilot cluster.
  HF auth is taken from the asset store's resolved token; COS auth from the
  COS store's `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` secrets (or env
  vars) — both are injected as launcher `envs`.
- The Skypilot env has no `messaging:` or `retry:` block. There is no
  RabbitMQ dependency — monitoring is done by polling
  `sky.job_status()` per launch (see `skypilot_monitor` below). Retry is
  per-step via `retry_enabled` / `retry_transparently` on the step config.
- Secrets configured on the environment via `secret_refs` are merged into
  every launched step's environment variables before launcher-supplied
  `envs` are layered on top.

---

## `step.yaml` — `environment_configs` section

`environment_configs` declares, per environment type, which launchers and monitors to use.

### Structure

```yaml
environment_configs:
  K8s:              # or Lsf, Bash, Docker, etc. Case-insensitive match.
    launchers:
      <launcher_name>:          # Logical name; used as the launcher suffix key.
        type: <suffix>          # Maps to launch_<suffix>() on the Environment class.
        monitors:               # List of monitor names (defined under monitors:) to
          - <monitor_name>      # run concurrently with this launcher.
        config:                 # Launcher-specific config passed as kwargs to launch_<suffix>().
          ...
    monitors:
      <monitor_name>:           # Logical name referenced in launchers[].monitors.
        type: <suffix>          # Maps to monitor_<suffix>() on the Environment class.
        config:                 # Monitor-specific config passed as kwargs to monitor_<suffix>().
          event_configs: [...]  # Log line parsing rules (see below).
```

### K8s launcher and monitor types

| `type` | Method called | When to use |
|--------|--------------|-------------|
| `helm` (launcher) | `launch_helm` | Standard: submits workload via Helm + AppWrapper |
| `sidecar_monitor` | `monitor_sidecar_monitor` | Recommended: AppWrapperMonitor + RabbitMQ monitor |
| `appwrapper_only` | `monitor_appwrapper_only` | AppWrapper polling only, no RabbitMQ |
| `event_monitor` | `monitor_event_monitor` | RabbitMQ events only, no AppWrapper polling |
| `log_monitor` | `monitor_log_monitor` | Direct K8s API log streaming (no RabbitMQ required) |

Helm launcher `config` fields:

```yaml
launchers:
  my-launcher:
    type: helm
    monitors:
      - log_monitor
    config:
      chart: helm-charts/my-chart   # Required. Path to the Helm chart relative to the
                                    # step asset root directory.
```

### Lsf launcher and monitor types

| `type` | Method called | When to use |
|--------|--------------|-------------|
| `bsub` (launcher) | `launch_bsub` | Standard: submits job via bsub |
| `bsub_monitor` | `monitor_bsub_monitor` | Recommended: polls bjobs + tails log file |
| `logfile_monitor` | `monitor_logfile_monitor` | Deprecated. Was a separate log tail; now a no-op. Move `event_configs` to `bsub_monitor`. |

Bsub launcher `config` fields (all optional):

```yaml
launchers:
  my-launcher:
    type: bsub
    monitors:
      - bsub_monitor
    # config:
    #   No launcher-level config is currently used for bsub.
    #   Job submission options come from the step config section (see below).
```

### Skypilot launcher and monitor types

| `type` | Method called | When to use |
|--------|--------------|-------------|
| `skypilot` (launcher) | `launch_skypilot` | The only launcher type. Builds a `sky.Task` from the launcher config and calls `sky.launch()`. |
| `skypilot_monitor` | `monitor_skypilot_monitor` | The only monitor type. Polls `sky.job_status()` and downloads job logs once the job reaches a terminal state. |

The `skypilot` launcher maps directly onto SkyPilot's
[`sky.Resources`](https://docs.skypilot.co/en/latest/reference/api.html#sky.Resources)
and [`sky.Task`](https://docs.skypilot.co/en/latest/reference/api.html#sky.Task).
Only the fields listed below are passed through; anything else is ignored.

```yaml
environment_configs:
  Skypilot:
    default_launcher: <launcher_name>   # Used when no launcher is named for the step.
    launchers:
      <launcher_name>:
        type: skypilot
        monitors:
          - skypilot_monitor
        config:
          # ---- maps onto sky.Resources ----
          image_id: docker:python:3.11-slim
                                  # Optional. Docker image to run the task in. On a
                                  # SLURM cluster this REQUIRES the Pyxis SPANK
                                  # plugin; omit on bare-host SLURM clusters or the
                                  # launch will fail with NotSupportedError.
          resources:
            cloud: <cloud>        # Optional. Per-step override of the env's
                                  # default_cloud.
            cpus: "2+"            # SkyPilot resource string. Recommended on every
                                  # step. "2+" means "2 or more vCPUs".
            memory: "4+"          # SkyPilot resource string. "4+" = 4 GiB or more.
            accelerators: A100:1  # Optional. SkyPilot accelerator string,
                                  # e.g. "A100:8", "H100:1".
            disk_size: 50         # Optional. Disk size in GB.
            infra: <infra-string> # Optional. Full SkyPilot infra spec, e.g.
                                  # "slurm/cluster/partition" or "k8s/my-context".
                                  # If unset and `cluster` is set below, gbserver
                                  # builds it as "<cloud>/<cluster>[/<zone>]".
            cluster: <name>       # Optional. SLURM cluster name; combined with
                                  # `cloud` to produce `infra` if `infra` is unset.
            zone: <zone>          # Optional. Cloud zone.

          # ---- maps onto sky.Task ----
          setup: |                # Optional. Bash run once at cluster bring-up
            pip install foo bar   # (cached across reuse of the same cluster).
          run: |                  # Required. Bash run as the actual job each launch.
            echo "LLMB_ARTIFACT_ID:my_out LLMB_ARTIFACT_PATH:/tmp/out.json"
          envs:                   # Optional. Extra env vars exposed to setup/run.
            FOO: bar              # Merged AFTER env-level secrets and BEFORE
                                  # `config.launcher_config.envs` (see step config
                                  # section below). Several `GB_*` vars are injected
                                  # automatically — see the table below.
          file_mounts:            # Optional. Two forms are supported:
            /remote/path: /local/path
                                  # String value → local-to-remote file/directory
                                  # copy via `task.set_file_mounts()`.
            /remote/bucket-path:  # Dict value → SkyPilot Storage mount via
              source: s3://bucket/prefix     # `task.set_storage_mounts()`.
              mode: MOUNT         # MOUNT or COPY (default MOUNT). Bucket-only
                                  # sources with a sub-path are auto-split: the
                                  # path part becomes `_bucket_sub_path` so the
                                  # mount can target a single prefix.

          idle_minutes_to_autostop: 10
                                  # Optional. Per-step override of the env-level
                                  # value. Same semantics: 0 = ASAP, null =
                                  # disable, positive int = idle minutes.
    monitors:
      skypilot_monitor:
        type: skypilot_monitor
        config:
          poll_interval_seconds: 15
                                  # How often to poll `sky.job_status()`. Default 15.
          event_configs:          # Same schema as K8s/Lsf event_configs (see the
            - event_type: NEWARTIFACT_IN_ENVIRONMENT_EVENT
              line_regex: "LLMB_ARTIFACT_ID:.* LLMB_ARTIFACT_PATH:.*"
              is_json: false      # event_configs section below). Lines are matched
              event_fields:       # by walking the downloaded job log AFTER the job
                - field_name: binding_id              # reaches a terminal status —
                  field_regex: "(?<=LLMB_ARTIFACT_ID:)[^ ]+"   # see the note below.
                - field_name: path
                  field_regex: "(?<=LLMB_ARTIFACT_PATH:).*"
                  is_data: true
                - field_name: binding
                  field_value_template: '{ "path": "{{ fields.data.path }}" }'
                  is_json: true
```

**Auto-injected environment variables**

These are added unconditionally on top of (and override) anything the user
puts in `envs`:

| Env var                    | Source                                            |
|----------------------------|---------------------------------------------------|
| `GB_SKYPILOT_LAUNCH_ID`    | The targetsteprun launch id (UUID)                |
| `GB_SKYPILOT_CLUSTER_NAME` | `gb-<launch_id_prefix>` — the actual SkyPilot cluster name |
| `GB_TARGETRUN_ID`          | The enclosing target run id, when present         |
| `GB_BUILD_ID`              | The build id, when present                        |
| `GB_SHARED_WORKDIR`        | The env-level `shared_workdir` path (when set on the environment.yaml) — same path on every worker, suitable for cross-step state |
| `GB_BUILD_WORKDIR`         | Per-target-run subdir `${shared_workdir}/builds/<build_id>/runs/<targetrun_id>/` (when `shared_workdir` is set). Created lazily, also set as the **initial CWD** of the run script, and `rm -rf`'d on target-run teardown. Retries get a fresh dir. |
| `<env secrets>`            | All secrets resolved from the env's `secret_refs` |

**Monitoring & artifact events: timing caveat**

`skypilot_monitor` does not stream logs in real time. It polls
`sky.job_status()` on `poll_interval_seconds`, and only **after** the job
reaches a terminal status does it download the full job log and re-walk
every line through the configured `event_configs`. Two consequences:

1. Artifact-emitting log lines (`LLMB_ARTIFACT_ID:...`) are captured even
   if they scroll past a poll interval — log download is offline, not
   tail-based.
2. Artifacts are not registered in the build until the job *completes*.
   There is no live event_monitor mode (unlike the K8s sidecar_monitor),
   so a long-running step's artifact events are batched at the end.

If a step exits with a non-`SUCCEEDED` JobStatus, the monitor emits a
`WORKLOAD_STATUS_EVENT` with status `FAILED` so the build is marked failed
even when the user did not write a status line to the log.

---

## `step.yaml` — `config` section fields read by environments

The `config` section of `step.yaml` (and per-step `config` overrides in `build.yaml`) carries fields that environments read at launch time.

### Common fields

```yaml
config:
  retry_enabled_default: false      # Whether retry is enabled for this step type by default.
                                    # Can be overridden per-run in build.yaml.
                                    # Default: false (when absent).
  retry_transparently_default: true # Whether to deduplicate NEWARTIFACT events across retries.
                                    # Default: true.

  gb:
    step_contents_in_env: true      # K8s only. Whether to copy the step asset directory
                                    # into the running pod. Default: true.
                                    # Set to false for steps that don't need the step files
                                    # inside the pod (e.g. upload/download steps).

  workload:                         # Used by Lsf to derive workspace and log paths.
    path: ""                        # Path to the workload entry point.
    args: ""                        # Command-line args for the workload.
    workspace_dir: ""               # Base workspace directory inside the cluster.
    output_dir: ""                  # Output directory (defaults to <workspace_dir>/outputs).
                                    # Job log is written to <output_dir>/job.log.
    python_env:
      env_dirs: []                  # Additional directories to add to PYTHONPATH.
      venv: ""                      # Name of a virtualenv to activate.
      conda: ""                     # Name of a conda environment to activate.

  lsf:                              # LSF-specific overrides for a single step run.
    bsub:
      jobid: ""                     # If set, adopt this pre-existing job ID instead of
                                    # submitting a new one.
      log_path: ""                  # Log file path to use with a pre-existing jobid.
      args: ""                      # Full bsub argument string (managed externally).
      additional_args: ""           # Extra args appended to the generated bsub command.
      queue: ""                     # LSF queue name.
      jobs_group: ""                # LSF jobs group.
      job_name: ""                  # LSF job name.

  k8s:                              # K8s-specific fields for a single step run.
    secrets:
      secret_names_to_use_as_pull_secret:
        - my_dockerconfig_secret    # Secret name whose value is a dockerconfigjson string.
                                    # Creates a K8s image pull secret in the namespace.
      secret_names_to_use_as_env_variable:
        - env_name: MY_ENV_VAR      # Environment variable name injected into the pod.
          secret_name: my_secret    # Secret name in the space secrets to read the value from.
                                    # Falls back to env_name.lower() if secret_name is absent.
    app_wrapper_config:
      warmupGracePeriodDuration: 30m  # AppWrapper-specific settings passed through to the
      retryLimit: 2                   # Helm chart values.
    affinity:                         # Kubernetes affinity rules, merged into Helm values.
      nodeAffinity: {}
```

### Skypilot-specific top-level config fields

The `Skypilot` launcher does **not** read a top-level `skypilot:` block. It
reads only two fields from `config:` in `step.yaml` / `build.yaml`:

```yaml
config:
  launcher_config:
    envs:                             # Extra env vars merged on top of the
      MY_VAR: my-value                # launcher's own `envs:`. This is the
                                      # primary way auto-queued steps (hfpull,
                                      # hfpush, s3push) inject HF_TOKEN /
                                      # AWS_ACCESS_KEY_ID etc. without modifying
                                      # the step.yaml.

  file_mounts:                        # Same schema as
    /remote: /local                   # `launcher_config.file_mounts` above.
                                      # Used as a fallback when the launcher
                                      # itself does not declare `file_mounts`.
```

**`compute_config` is not honored by the Skypilot launcher.** K8s and Lsf
launchers translate `compute_config.num_gpus_per_node` /
`total_memory_per_node` into pod / LSF resource specs. SkyPilot reads
`resources` directly from
`environment_configs.Skypilot.launchers.<name>.config.resources`. If a step
needs GPU/memory specs, set `resources.accelerators` and `resources.memory`
in the step.yaml — you can template those off `{{ config.compute_config.* }}`
if you want a single source of truth, but the launcher will never reach
into `compute_config` on its own.

The K8s-only `gb.step_contents_in_env`, `k8s.*`, and `lsf.*` blocks are
likewise ignored by the SkyPilot launcher. Step-asset code does not get
copied into the SkyPilot pod automatically; if the `run:` script needs
files, use `file_mounts` or fetch them inside `setup:` / `run:`.

---

## `event_configs` — log line parsing rules

`event_configs` appear under monitor `config` sections for both K8s and Lsf environments. Each rule parses a log line and emits a `BuildEvent` when matched.

```yaml
event_configs:
  - event_type: <BuildEventType>    # Required. One of the BuildEventType enum values:
                                    #   NEWARTIFACT_IN_ENVIRONMENT_EVENT
                                    #   MESSAGE_EVENT
                                    #   WORKLOAD_STATUS_EVENT
                                    #   VALIDATION_DATA_EVENT
                                    #   ARTIFACT_PUSHED_EVENT

    line_regex: "<regex>"           # Required. Regex matched against each log line.
                                    # If it matches, this rule fires. Only the matched
                                    # portion is used for subsequent field extraction.

    is_json: false                  # If true, the matched portion is parsed as JSON and
                                    # its contents placed in event_data["data"].

    event_fields:                   # List of fields to extract from the matched text.
      - field_name: <name>          # Required. Key used in the event payload.

        field_regex: "<regex>"      # Extract the value by running this regex against the
                                    # matched line. The full match (group 0) is used.
                                    # Mutually exclusive with field_value_template.

        field_value_template: "..." # Jinja2 template for the value. Available context:
                                    #   {{ fields.<field_name> }}   previously-extracted fields
                                    #   {{ fields.data.<key> }}     fields with is_data: true
                                    # Mutually exclusive with field_regex.

        is_json: false              # If true, parse the extracted value as JSON before
                                    # storing it in the event payload.

        is_data: false              # If true, store this field in event_data["data"]
                                    # instead of the top-level event payload. Useful for
                                    # intermediate values referenced by field_value_template.
```

### Event type conventions

| `event_type` | Typical log trigger | Common fields |
|-------------|---------------------|---------------|
| `NEWARTIFACT_IN_ENVIRONMENT_EVENT` | Workload writes an output file | `binding_id` (matches an output name in `build.yaml`), `binding` (JSON with `"path"`) |
| `MESSAGE_EVENT` | Informational line to surface in the build UI | `msg` |
| `WORKLOAD_STATUS_EVENT` | Periodic progress update | `status` |
| `VALIDATION_DATA_EVENT` | Structured metrics/validation results | `data` |
| `ARTIFACT_PUSHED_EVENT` | Upload step confirms a push | `uri`, `binding_id` |

### Example: artifact detection from a log line

```
# Log line emitted by the workload:
Final checkpoint saved in /gpfs/workspace/output/checkpoint-final

# Matching rule:
- event_type: NEWARTIFACT_IN_ENVIRONMENT_EVENT
  line_regex: "Final\\scheckpoint\\ssaved\\sin\\s.*"
  is_json: false
  event_fields:
    - field_name: binding_id
      field_value_template: final_checkpoint   # Static value; matches an output name in build.yaml
    - field_name: path
      field_regex: "/.*"
      is_data: true                            # Store in data dict for use by binding template
    - field_name: binding
      field_value_template: '{ "path": "{{ fields.data.path }}" }'
      is_json: true
```

### Example: JSON log line

```
# Log line emitted by the workload (JSON):
{"gb_new_artifact": {"name": "my_output", "path": "/workspace/output/result"}}

# Matching rule:
- event_type: NEWARTIFACT_IN_ENVIRONMENT_EVENT
  line_regex: "{.*gb_new_artifact.*path.*}"
  is_json: true                                # Parse the whole matched portion as JSON
  event_fields:
    - field_name: binding_id
      field_value_template: "{{ fields.data.gb_new_artifact.name }}"
    - field_name: binding
      field_value_template: '{ "path": "{{ fields.data.gb_new_artifact.path }}" }'
      is_json: true
```

---

## Complete annotated examples

### K8s `environment.yaml`

```yaml
name: vela-production
type: K8s
config:
  namespace: granite-build
  authentication:
    kube_config: prod_kubeconfig        # Space secret containing the kubeconfig YAML
    kube_context: prod-context
    ssl_verification: true
  messaging:
    authentication_secret_name: rabbitmq_prod
  retry:
    enabled: true
    max_retries: 3
assetstores:
  - store_uri: lh://granite_dot_build.public
    load:
      - mode: dmf_pull
    push:
      - mode: dmf_push
  - store_uri: cos://my-cos-bucket
    load:
      - mode: cos_rclone
    push:
      - mode: cos_rclone
```

### Lsf `environment.yaml`

```yaml
name: frontier-lsf
type: Lsf
config:
  workspace:
    local_dir: /tmp/gbserver/lsf
    remote_dir: /gpfs/projects/myteam/gbserver
  authentication:
    use_ssh: true
    copy_method: scp
    ssh_port: 22
    ssh_max_sessions: 10
    login_nodes:
      - frontier-login1.example.com
      - frontier-login2.example.com
    login_node_username: gbsvcuser
    login_node_ssh_key: frontier_ssh_key
    ssh_host_key_verification: true
    ssh_timeout: 5
  retry:
    enabled: true
    max_retries: 3
assetstores:
  - store_uri: lh://granite_dot_build.public
    load:
      - mode: dmf_pull
        config:
          cache_path: /gpfs/cache/lakehouse
    push:
      - mode: dmf_push
```

### `step.yaml` for a K8s training step

```yaml
name: my-training-step
version: 1.0.0
type: custom
config:
  retry_enabled_default: false
  gb:
    step_contents_in_env: false
  k8s:
    secrets:
      secret_names_to_use_as_pull_secret:
        - my_registry_secret
      secret_names_to_use_as_env_variable:
        - env_name: HF_TOKEN
          secret_name: huggingface_token
  compute_config:
    num_nodes: 2
    num_gpus_per_node: 8

environment_configs:
  K8s:
    launchers:
      training:
        type: helm
        monitors:
          - log_monitor
        config:
          chart: helm-charts/my-training-step
    monitors:
      log_monitor:
        type: sidecar_monitor
        config:
          event_configs:
            - event_type: NEWARTIFACT_IN_ENVIRONMENT_EVENT
              line_regex: "Final checkpoint saved in .*"
              is_json: false
              event_fields:
                - field_name: binding_id
                  field_value_template: final_checkpoint
                - field_name: path
                  field_regex: "/.*"
                  is_data: true
                - field_name: binding
                  field_value_template: '{ "path": "{{ fields.data.path }}" }'
                  is_json: true
            - event_type: MESSAGE_EVENT
              line_regex: "^GB_MESSAGE.*"
              is_json: false
              event_fields:
                - field_name: msg
                  field_regex: ".*"
            - event_type: WORKLOAD_STATUS_EVENT
              line_regex: "^LLMB_EVENT_WORKLOAD_STATUS:.+"
              is_json: false
              event_fields:
                - field_name: status
                  field_regex: "(?<=LLMB_EVENT_WORKLOAD_STATUS:).+"
```

### `step.yaml` for an Lsf training step

```yaml
name: my-lsf-step
version: 1.0.0
type: custom
config:
  retry_enabled_default: false
  workload:
    workspace_dir: ""    # derived from remote_dir + launch hierarchy at runtime
    output_dir: ""       # defaults to <workspace_dir>/outputs; job.log written here

environment_configs:
  Lsf:
    launchers:
      training:
        type: bsub
        monitors:
          - bsub_monitor
    monitors:
      bsub_monitor:
        type: bsub_monitor
        config:
          event_configs:
            - event_type: NEWARTIFACT_IN_ENVIRONMENT_EVENT
              line_regex: "LLMB_ARTIFACT_ID:.* LLMB_ARTIFACT_PATH:.*"
              is_json: false
              event_fields:
                - field_name: binding_id
                  field_regex: "(?<=LLMB_ARTIFACT_ID:)[^ ]+"
                - field_name: path
                  field_regex: "(?<=LLMB_ARTIFACT_PATH:).*"
                  is_data: true
                - field_name: binding
                  field_value_template: '{ "path": "{{ fields.data.path }}" }'
                  is_json: true
            - event_type: WORKLOAD_STATUS_EVENT
              line_regex: "^LLMB_EVENT_WORKLOAD_STATUS:.+"
              is_json: false
              event_fields:
                - field_name: status
                  field_regex: "(?<=LLMB_EVENT_WORKLOAD_STATUS:).+"
```

### Skypilot `environment.yaml` (bare-host SLURM)

This is the pattern used by the
[`skypilot_slurm` integration test](../test/integration/standalone/buildrunner/skypilot_slurm/)
against the local Docker SLURM cluster from
[`docs/skypilot/local-infrastructure-setup.md`](skypilot/local-infrastructure-setup.md).
No `image_id` is set on the launchers because the local cluster has no Pyxis
SPANK plugin.

```yaml
name: slurm-local
type: Skypilot
config:
  default_cloud: slurm
  cluster: slurm-docker
  zone: normal
  idle_minutes_to_autostop: 0   # ignored on SLURM (autostop unsupported);
                                # per-step `sky down` handles teardown.
  shared_workdir: /shared       # Path shared across slurmctld/c1/c2 in the
                                # local Docker fixture (the `slurm_shared_fs`
                                # named volume). HF cache defaults to
                                # /shared/hf_cache via this declaration —
                                # no per-store `cache_path` override needed.
assetstores:
  - store_uri: space://assetstores/hf
    load:
      - mode: hf_pull
    push:
      - mode: hf_push
```

### `step.yaml` for a Skypilot bash step (no Pyxis)

```yaml
name: bash
version: 1.0.0
type: exec
config:
  bash_config:
    command: ""              # Filled in from build.yaml — see below.
  compute_config:
    num_gpus_per_node: 0     # NOT read by the Skypilot launcher; left here for
    total_memory_per_node: 4Gi  # parity with K8s/Lsf step authoring.

environment_configs:
  Skypilot:
    default_launcher: bash
    launchers:
      bash:
        type: skypilot
        monitors:
          - skypilot_monitor
        config:
          # No image_id — runs directly on the SLURM compute node.
          resources:
            cpus: "1+"
            memory: "1+"
          run: |
            {{ config.bash_config.command }}
    monitors:
      skypilot_monitor:
        type: skypilot_monitor
        config:
          poll_interval_seconds: 5
          event_configs:
            - event_type: NEWARTIFACT_IN_ENVIRONMENT_EVENT
              line_regex: "LLMB_ARTIFACT_ID:.* LLMB_ARTIFACT_PATH:.*"
              is_json: false
              event_fields:
                - field_name: binding_id
                  field_regex: "(?<=LLMB_ARTIFACT_ID:)[^ ]+"
                - field_name: path
                  field_regex: "(?<=LLMB_ARTIFACT_PATH:).*"
                  is_data: true
                - field_name: binding
                  field_value_template: '{ "path": "{{ fields.data.path }}" }'
                  is_json: true
```

### `build.yaml` invoking a Skypilot step

The corresponding `build.yaml` provides the per-run config that the launcher
template renders against (`{{ config.bash_config.command }}`):

```yaml
granite.build:
  name: skypilot-bash-example
  targets:
    image-run:
      environment_uri: space://environments/slurm
      inputs:
        input_model:
          uri: hf:///datasets/ibm-research/some-dataset
      outputs:
        output_model:
          uri: hf://huggingface.co/datasets/my-org/out_{{ binding.path | short_hash }}
      steps:
        - step_uri: space://steps/bash
          config:
            bash_config:
              command: >-
                mkdir -p /tmp/gb-outputs;
                file=/tmp/gb-outputs/out.json; touch $file;
                echo "LLMB_ARTIFACT_ID:output_model LLMB_ARTIFACT_PATH:$file"
            compute_config:
              num_gpus_per_node: 0
              total_memory_per_node: 1Gi
```

Per-step lifecycle for this build:

1. `pullasset_hfstore` queues the builtin `hfpull` step on its own SkyPilot
   cluster, which runs `hf download` into `/tmp/hf_cache/...`.
2. The `bash` step runs on a fresh SkyPilot cluster, emits the
   `LLMB_ARTIFACT_ID:` line, and exits.
3. `pushasset_hfstore` queues the builtin `hfpush` on a third SkyPilot
   cluster: it calls `huggingface_hub.HfApi.create_repo(..., exist_ok=True)`
   to ensure the HF repo exists, then `hf upload`'s the file.

Each step's cluster is torn down by `cleanup_skypilot()` once its
`skypilot_monitor` sees a terminal job status. On a SLURM backend the
torn-down cluster releases its node allocation; if you queue more parallel
steps than the cluster has nodes, the surplus steps stay PENDING until
earlier ones finish and free a node.
