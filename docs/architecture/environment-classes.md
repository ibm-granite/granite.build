# Environment Classes

This document covers the `Environment` abstract base class and its two primary implementations: `Lsf` and `K8s`.

**Source files:**
- `src/gbserver/environment/environment.py` — `Environment` base
- `src/gbserver/environment/lsf.py` — `Lsf`
- `src/gbserver/environment/k8s.py` — `K8s`

---

## `Environment` (Abstract Base Class)

`Environment` is the abstract base for all compute backends. It manages the full lifecycle of a workload: setup, launch, monitoring, cleanup, and teardown.

### Core Concepts

#### IDs

| ID | Scope | Groups |
|----|-------|--------|
| `setup_id` | One per environment session | `setup_*` / `teardown_*` calls |
| `launch_id` | One per workload submission | `launch_*` / `cleanup_*` / `monitor_*` calls |

Multiple launches can occur between a single setup/teardown pair.

#### Suffix-Based Dispatch

Methods are discovered by prefix using `get_fns_with_prefix()`. The suffixes (e.g. `helm`, `bsub`) are referenced in `environment.yaml` and `step.yaml` to select which methods to call:

| Prefix | Called via | Purpose |
|--------|-----------|---------|
| `setup_SUFFIX` | `setup(type, ...)` | One-time environment initialization |
| `teardown_SUFFIX` | `teardown(type, setup_id, ...)` | One-time environment teardown |
| `launch_SUFFIX` | `launch(launcher_type, setup_ids, ...)` | Submit a workload |
| `cleanup_SUFFIX` | `cleanup(launch_type, launch_id, ...)` | Remove/kill a submitted workload |
| `monitor_SUFFIX` | `monitor(type, launch_id, ...)` | Watch workload status and logs |
| `pullasset_SUFFIX` | `pullasset(task_group, uri, ...)` | Pull an asset into the environment |
| `pushasset_SUFFIX` | `pushasset(task_group, binding, uri, ...)` | Push an artifact to an asset store |

#### Lifecycle Ordering

The base class enforces strict ordering via `asyncio.Event` coordination:

```
setup_xyz(setup_id)
    └─► launch_xyz(launch_id)  ← blocked until setup done
            └─► release_monitors(launch_id)  ← called by launch to unblock monitors
            └─► monitor_abc(launch_id)  ← blocked until release_monitors()
        cleanup_xyz(launch_id)  ← blocked until launch done
teardown_xyz(setup_id)  ← blocked until all launches + cleanups done
```

If `launch_xyz()` raises an exception, monitors are not triggered for that `launch_id`.

### Key Methods

#### `get_environment(environment_uri, event_q, ...)` (classmethod)

Resolves an environment URI to an `Environment` instance. Downloads the asset, finds `environment.yaml`, reads the `type` field, and instantiates the matching subclass. Results are cached per URI in thread-local storage.

#### `release_monitors(launch_id)`

Called from within `launch_*` once the workload is running and ready to be monitored. Unblocks all `monitor_*` tasks waiting on this `launch_id`.

#### `_get_launch_stopped_event(launch_id) → asyncio.Event`

Returns a shared event for `launch_id`. Monitors check this event to know when they should stop. Typically set by a monitor that detects workload completion (e.g. AppWrapper reaching terminal state).

#### `dispatch_event(event)`

Puts a `BuildEvent` onto the `event_q` synchronously (non-blocking). Used by all subclasses to emit status/artifact/message events.

#### `get_events_from_log_line(log_line, event_configs, event_q, ...)` (static)

Parses a log line against a list of `EventLogLineParserConfig` rules. Each rule has a primary `line_regex` and optional `event_fields` for extracting structured data. Matching lines produce `BuildEvent` objects placed on `event_q`.

### Retry Support

The `Environment` base provides a full retry framework that subclasses hook into:

| Method | Purpose |
|--------|---------|
| `_get_default_retry_strategies()` | Override to return environment-specific `RetryStrategy` list |
| `_get_retry_test_scenario()` | Override to name a simulation scenario for testing |
| `retry_workload(launch_id, ...)` | Override to implement the actual re-launch logic |
| `with_retry_handler(launch_id, event_q, ...)` | Async context manager; wraps the event queue with a `RetryHandler` when retry is enabled |

Retry is controlled at two levels:
1. **Environment level** — `retry.enabled` / `retry.strategies` / `retry.max_retries` in `environment.yaml`
2. **Step level** — `retry_enabled` / `retry_enabled_default` / `retry_transparently` in `step.yaml` or `build.yaml`

`retry_transparently=True` deduplicates `NEWARTIFACT` events across retry iterations using `RetryArtifactFilterQueue`, so downstream consumers don't see duplicate artifact events.

### Abstract Method

```python
@abstractmethod
def get_step_env_config(self, config: dict) -> StepEnvConfig:
    ...
```

Subclasses must implement this to extract their environment-specific section from a step's config dict.

---

## `Lsf` — Load Sharing Facility

`Lsf` runs workloads on HPC clusters managed by IBM Spectrum LSF. Jobs are submitted using `bsub`, optionally via SSH.

**Suffix used:** `bsub`
- `setup_bsub` / `teardown_bsub`
- `launch_bsub` / `cleanup_bsub`
- `monitor_bsub_monitor`

### Configuration (`environment.yaml`)

```yaml
type: Lsf
workspace:
  local_dir: /tmp/lsf_workspace     # local staging dir (non-SSH mode)
  remote_dir: /remote/workspace      # remote dir on the cluster

authentication:
  use_ssh: true
  ssh_port: 22
  ssh_max_sessions: 10
  login_nodes:
    - login1.cluster.example.com
    - login2.cluster.example.com
  login_node_username: myuser
  login_node_ssh_key: my_ssh_key_secret   # secret name in space secrets
  ssh_host_key_verification: true
  ssh_timeout: 5
```

### Lifecycle

#### `setup_bsub(setup_id, space_secrets, ...)`

1. Retrieves the SSH private key from `space_secrets` and writes it to a temporary file (`chmod 600`).
2. Opens a persistent `SshTunnel` (multiplexed SSH connection with port forwarding) to the first reachable login node.
3. Returns the key file path and space secrets in the setup config, which is forwarded to subsequent `launch_bsub` calls.

#### `launch_bsub(launch_id, targetsteprun_asset_dir, ...)`

1. Reads `step.yaml` config for job details. If an existing `jobid` is specified, adopts it passively without submitting.
2. Copies the step asset directory to the cluster:
   - **SSH mode**: prepares the job submission script locally (template variable replacement), then `rsync`s everything to the remote `workspace_remote_dir` via the `SshTunnel`.
   - **Non-SSH mode**: copies locally to `workspace_local_dir`, applies variable replacement in place.
3. Runs the `llmb_lsf_jobsub.sh` script (directly or via SSH tunnel) to call `bsub`.
4. Parses the `Job <JOBID> is submitted` output and records the job ID.
5. Calls `release_monitors(launch_id)`.

Template variables injected into `llmb_lsf_jobsub.sh`:
- `LLMB_LSF_REPLACE_THIS_LAUNCH_ID` → `launch_id`
- `LLMB_LSF_REPLACE_THIS_ASSET_DIR` → path to the copied step assets

#### `monitor_bsub_monitor(launch_id, event_q, ...)`

Runs `LSFBsubMonitor` and `LogFileMonitor` concurrently within an optional `RetryHandler` context:

```
while True:
    asyncio.gather(
        LSFBsubMonitor.monitor(),   # polls bjobs, sets stop_event on completion/failure
        LogFileMonitor.monitor(),   # tails job.log, emits artifact events
    )
    if retry_complete_event.is_set():
        continue   # RetryHandler triggered; loop for next attempt
    break
```

The log file is streamed either directly from the filesystem (non-SSH) or via `SshTunnel` (SSH mode).

Default retry strategy: `LsfTransientErrorRetryStrategy` — detects transient LSF errors (e.g. `TERM_RUNLIMIT`, scheduler failures) and retriggers `retry_workload`.

#### `retry_workload(launch_id, ...)`

1. Signals `LSFBsubMonitor` to exit cleanly by setting the stop event.
2. Calls `cleanup_bsub` → `bkill <jobid>`.
3. Clears the stop event.
4. Re-calls `launch_bsub` with the original kwargs.
5. Sets `_lsf_retry_complete_events[launch_id]` to signal `monitor_bsub_monitor` to loop.

#### `cleanup_bsub(launch_id, ...)`

Calls `bkill <jobid>` (via SSH tunnel or local subprocess). Skips jobs that were pre-existing (passively adopted).

#### `teardown_bsub(setup_id)`

1. Deletes all `_pending_cleanup_dirs` (remote via `SshTunnel` or local via `shutil.rmtree`).
2. Closes the `SshTunnel`.
3. Deletes the temporary SSH key file.

### SSH Login Node Management

- Login nodes are checked for reachability at launch time (`_get_reachable_ssh_node()`).
- Unreachable nodes are tracked in `unreachable_ssh_nodes` and skipped (with a full reset if all become unreachable).
- Connections are multiplexed through `SshTunnel` (one persistent connection per setup), avoiding repeated SSH handshakes.

### Asset Stores

| Method | Store type | Mode |
|--------|-----------|------|
| `pullasset_lhstore` | Lakehouse | `dmf_pull` — injects a `lhpull` built-in step |
| `pushasset_lhstore` | Lakehouse | Injects a `lhpush` built-in step |
| `pullasset_cosstore` | IBM COS | `cos_pull` — injects a `cosrclone` built-in step |
| `pushasset_cosstore` | IBM COS | Injects a `cosrclone` built-in step |
| `pullasset_envstore` | Env (local path) | Returns the path directly as a binding |

For Lakehouse and COS stores, loading/pushing is delegated to built-in steps that run as additional LSF jobs in the pipeline.

---

## `K8s` — Kubernetes / OpenShift

`K8s` runs workloads on Kubernetes or OpenShift clusters using Helm charts and CodeFlare `AppWrapper` resources. Monitoring is event-driven via RabbitMQ sidecar or direct K8s API log streaming.

**Suffix used:** `helm`
- `setup_helm` / `teardown_helm`
- `launch_helm` / `cleanup_helm`
- `monitor_sidecar_monitor` / `monitor_appwrapper_only` / `monitor_event_monitor` / `monitor_log_monitor`

### Configuration (`environment.yaml`)

```yaml
type: K8s
namespace: my-namespace

authentication:
  kube_config: my_kubeconfig_secret    # secret name (base64 kubeconfig YAML)
  kube_context: my-context             # optional context override
  ssl_verification: true

messaging:
  authentication_secret_name: my_rabbitmq_secret  # required for sidecar/event monitors

retry:
  enabled: true
  max_retries: 3
  # strategies: [...]   # optional override
```

### `AtomicApiClient`

A thread-safe and process-safe factory for `kubernetes_asyncio.client.ApiClient`. Uses a `multiprocessing.Lock` to serialize kubeconfig loading. Created fresh for each API operation via `async with await AtomicApiClient.create_api_client(...) as api:`.

### Lifecycle

#### `setup_helm(setup_id, space_secrets, ...)`

Creates a K8s `Secret` in the target namespace containing all space secrets (base64-encoded). The secret name is a hash of `setup_id`. This secret is later referenced by `launch_helm` to inject credentials into the workload pod (as image pull secrets or environment variables).

#### `launch_helm(launch_id, targetsteprun_asset_dir, ...)`

1. Builds the `helm install` command:
   - Resolves `values-default.yaml`, `values.yaml`, `values-config.yaml` from the chart directory.
   - Injects run metadata (build ID, target name, etc.) via `--set`.
   - Configures image pull secrets and environment variables from `space_secrets` via `--set`/`--set-file`.
2. Performs a `helm install --dry-run --debug` first; raises on failure.
3. Runs the actual `helm install`. On `resourcequotas` conflict, retries up to 3 times.
4. If `config.gb.step_contents_in_env` is `true` (default): copies the merged step directory into the running pod via `kubectl cp` (or `oc cp` as fallback), placing it at `<targetsteprun_assets_dir>/llmb-targetsteprun-assets/<launch_id>/`.
5. Calls `release_monitors(launch_id)`.
6. Stores launch params in `self.launch_params[launch_id]` for retry.

#### `cleanup_helm(launch_id, ...)`

Runs `helm uninstall <release_name>`. The Helm release name is derived from a hash of `launch_id` with the prefix `gb`.

#### `teardown_helm(setup_id)`

Deletes the K8s `Secret` created in `setup_helm`.

### Monitors

#### `monitor_sidecar_monitor` (preferred)

Runs `AppWrapperMonitor` and `RabbitMQEventMonitor` concurrently within a `RetryHandler` context:

```
asyncio.gather(
    AppWrapperMonitor.monitor(),      # watches AppWrapper phase via K8s API
    RabbitMQEventMonitor.monitor(),   # receives events from workload sidecar
)
```

`AppWrapperMonitor` sets the `stop_event` when the AppWrapper reaches a terminal state, which stops both monitors. `RabbitMQEventMonitor` receives structured events (artifacts, logs, metrics) published by the `gbstep` sidecar inside the pod.

#### `monitor_appwrapper_only`

Runs only `AppWrapperMonitor`. Used for workloads without a RabbitMQ sidecar. Supports the full retry handler path.

#### `monitor_event_monitor`

Runs only `RabbitMQEventMonitor`. Used standalone when AppWrapper status is tracked elsewhere.

#### `monitor_log_monitor`

Directly streams pod logs via the K8s API (`kubernetes_asyncio` watch):
- `watch_for_pods()` — discovers new/restarted pods belonging to the AppWrapper, enqueues their names.
- `_stream_pod_logs()` — streams log lines from each pod, parsing them through `EventLogLineParserConfig` rules.

This monitor does not use RabbitMQ and does not support retry.

### Retry

Default strategies (applied when `retry.enabled: true` in `environment.yaml`):

| Strategy | Trigger |
|----------|---------|
| `UnhealthyInsufficientPodsRetryStrategy` | AppWrapper stuck with insufficient healthy pods |
| `PodEvictionRetryStrategy` | Pod evicted by the node |
| `NCCLErrorRetryStrategy` | NCCL communication error in distributed training |

#### `retry_workload(launch_id, nodes_to_avoid, ...)`

1. Pauses the `AppWrapperMonitor` (stops processing events while the pod is being replaced).
2. Calls `cleanup_helm` → `helm uninstall`.
3. Waits for the AppWrapper to be fully deleted (polls K8s API up to 5 minutes).
4. If `nodes_to_avoid` is given, adds `kubernetes.io/hostname NotIn [...]` node anti-affinity to the config.
5. Re-calls `launch_helm` with updated config.
6. Unpauses the `AppWrapperMonitor` to resume monitoring the new pod.

Test scenario name: `"pod_eviction"` (used with `GBTEST_SIMULATE_FAILURE_SCENARIO`).

### Asset Stores

| Method | Store type | Mode |
|--------|-----------|------|
| `pullasset_lhstore` | Lakehouse | `dmf_pull` — injects a `lhpull` step (runs as a Helm job before the main workload) |
| `pushasset_lhstore` | Lakehouse | Injects a `lhpush` step |
| `pullasset_cosstore` | IBM COS | `cos_rclone` or custom step |
| `pushasset_cosstore` | IBM COS | Custom rclone step |
| `pullasset_hfstore` | HuggingFace | Downloads model/dataset into a PVC path |

---

## Adding a New Environment

1. Create `src/gbserver/environment/myenv.py` with a class named `Myenv(Environment)`.
2. Implement `get_step_env_config(config)`.
3. Add `setup_*`, `teardown_*`, `launch_*`, `cleanup_*`, and `monitor_*` methods with your chosen suffix.
4. Override `_get_default_retry_strategies()` and `retry_workload()` if retry is relevant.
5. Create an `environment.yaml` asset with `type: Myenv`.

The `Environment._load_environment_types()` classmethod auto-discovers all `Environment` subclasses in the `gbserver/environment/` package at startup. No registration is required.
