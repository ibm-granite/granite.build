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
