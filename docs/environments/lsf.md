# LSF (`Lsf`) environment

> **Audience:** operators configuring a native `Lsf` environment and step authors targeting it.
> For the common schema and `event_configs` see [Environment overview](README.md).

## Compute environment

The **Lsf** environment submits each step to an **IBM LSF** cluster with `bsub`, reaching the cluster
over **SSH** (login nodes), staging job scripts and assets to a remote workspace, and tailing the job
log to emit build events. This is the *native* LSF path — gbserver builds and submits the `bsub`
command itself.

> LSF can also be reached **through SkyPilot** ([skypilot-lsf.md](skypilot-lsf.md)), where SkyPilot's
> LSF provisioner submits the job. Use the native `Lsf` type when you want gbserver's SSH workspace
> management and `bsub` lifecycle directly; use SkyPilot-LSF for a single environment definition that
> shares a launcher shape with other clouds.

The implementation is [`Lsf`](../../src/gbserver/environment/lsf.py).

## `environment.yaml`

```yaml
name: my-lsf-env
type: Lsf
config:
  workspace:
    local_dir: /tmp/gbserver/lsf    # Local staging directory (non-SSH mode only).
                                    # Default: <DEFAULT_ROOT_WORKSPACE_DIR>/env_lsf
    remote_dir: /gpfs/workspace     # Remote directory on the LSF cluster. Required in SSH mode;
                                    # base path for all copied job scripts and outputs.

  authentication:
    use_ssh: true                   # Use SSH to reach the LSF cluster. Default: true.
    copy_method: scp                # Asset transfer method: "scp" or "rsync". Default: scp.
    ssh_port: 22                    # SSH port. Default: 22.
    ssh_max_sessions: 10            # Max concurrent multiplexed SSH sessions. Default: 10.
    login_nodes:                    # SSH login nodes. At least one required when use_ssh: true.
      - login1.cluster.example.com  # Tried round-robin; unreachable nodes are skipped.
      - login2.cluster.example.com
    login_node_username: myuser     # SSH username.
    login_node_ssh_key: my_ssh_key  # Secret name whose value is the SSH private key (PEM).
                                    # Required when use_ssh: true.
    ssh_host_key_verification: true # Verify the SSH host key. Default: true. Set false for
                                    # dev/test clusters with self-signed host keys.
    ssh_timeout: 5                  # SSH reachability probe timeout (seconds). Default: 5.

  retry:
    enabled: true                   # Master switch. Default: true.
    max_retries: 3                  # Default: 3.
    strategies:                     # Optional override. Default: LsfTransientError.
      - type: LsfTransientError

assetstores:
  - store_uri: hf://huggingface.co/my-org
    load:
      - mode: hf_pull               # Injects an hfpull built-in step before the main job.
        config:
          cache_path: /gpfs/cache/hf    # Required. Cluster path where HF data is cached.
          step_uri: space://steps/hfpull  # Optional override of the hfpull step URI.
    push:
      - mode: hf_push
        config:
          step_uri: space://steps/hfpush
  - store_uri: cos://my-bucket
    load:
      - mode: cos_pull              # Injects a cosrclone built-in step.
        config:
          cache_path: /gpfs/cache/cos   # Required. Cluster path where COS data is downloaded.
    push:
      - mode: cos_push
```

## `step.yaml` — launcher and monitor types

| `type` | Method | When to use |
|--------|--------|-------------|
| `bsub` (launcher) | `launch_bsub` | Standard: submits the job via `bsub`. |
| `bsub_monitor` | `monitor_bsub_monitor` | Recommended: polls `bjobs` and tails the job log. |
| `logfile_monitor` | `monitor_logfile_monitor` | Deprecated. Was a separate log tail; now a no-op. Move `event_configs` to `bsub_monitor`. |

The `bsub` launcher takes no launcher-level config — job-submission options come from the step
`config` section below.

## Step `config` blocks read by Lsf

```yaml
config:
  workload:                         # Used to derive workspace and log paths.
    path: ""                        # Path to the workload entry point.
    args: ""                        # Command-line args for the workload.
    workspace_dir: ""               # Base workspace directory inside the cluster.
    output_dir: ""                  # Output dir (defaults to <workspace_dir>/outputs).
                                    # Job log is written to <output_dir>/job.log.
    python_env:
      env_dirs: []                  # Extra directories added to PYTHONPATH.
      venv: ""                      # Virtualenv to activate.
      conda: ""                     # Conda environment to activate.

  lsf:                              # LSF-specific overrides for a single step run.
    bsub:
      jobid: ""                     # Adopt this pre-existing job ID instead of submitting a new one.
      log_path: ""                  # Log file path to use with a pre-existing jobid.
      args: ""                      # Full bsub argument string (managed externally).
      additional_args: ""           # Extra args appended to the generated bsub command.
      queue: ""                     # LSF queue name.
      jobs_group: ""                # LSF jobs group.
      job_name: ""                  # LSF job name.
```

## Complete example

### `environment.yaml`

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
  - store_uri: hf://huggingface.co/my-org
    load:
      - mode: hf_pull
        config:
          cache_path: /gpfs/cache/hf
    push:
      - mode: hf_push
```

### `step.yaml`

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

## See also

- [Environments overview](README.md) and the shared [event_configs schema](README.md#event_configs--log-line-parsing-rules)
- [SkyPilot on LSF](skypilot-lsf.md) — the same cluster, fronted by SkyPilot
- [Build retry](../builds/build-retry.md)
- [Troubleshooting](../help/troubleshooting.md)
