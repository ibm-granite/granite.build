# Kubernetes (`K8s`) environment

> **Audience:** operators configuring a `K8s` environment and step authors targeting it.
> For the common schema, asset stores, and `event_configs` see [Environment overview](README.md).

## Compute environment

The **K8s** environment runs each step as a Kubernetes/OpenShift workload, submitted via **Helm** as an
**AppWrapper** (Kueue-managed). It is gbserver's first-class cluster backend: it can stream live build
events over RabbitMQ, poll AppWrapper/pod status, and apply Kubernetes-aware retry strategies (pod
eviction, NCCL errors, insufficient-pods).

The implementation is [`K8s`](../../src/gbserver/environment/k8s.py). For provisioning a cluster for
SkyPilot-on-Kubernetes instead, see [skypilot-kubernetes.md](skypilot-kubernetes.md).

## `environment.yaml`

```yaml
name: my-k8s-env
type: K8s
config:
  namespace: granite-build          # Required. Kubernetes namespace for all resources.

  authentication:
    kube_config: my_kubeconfig       # Secret name whose value is a kubeconfig YAML string. If omitted,
                                     # falls back to the in-cluster or default kubeconfig on the server.
    kube_context: my-context         # Secret name whose value is the kubeconfig context. Optional.
    ssl_verification: true           # Verify the K8s API server TLS cert. Default: true. Set false for
                                     # self-signed clusters.

  messaging:
    authentication_secret_name: rabbitmq_secret
                                     # Secret name whose value is a JSON RabbitMQ credentials object.
                                     # Required when using sidecar_monitor or event_monitor.

  retry:
    enabled: true                    # Master switch. Default: true.
    max_retries: 3                   # Default: 3.
    strategies:                      # Optional override. When absent, uses the K8s defaults below.
      - type: UnhealthyInsufficientPods
      - type: PodEviction
        object_types: [AppWrapper]
      - type: NCCLError

  targetsteprun_assets_dir: /gb-read-write
                                     # Mount path inside the pod where step assets are copied.
                                     # Default: /gb-read-write.

  umask: "0002"                      # umask applied in every step container before the workload
                                     # runs, so files on the shared PVC are group-writable.
                                     # Default: "0002". Quote the value.

assetstores:
  - store_uri: cos://my-bucket
    pull:
      - mode: cos_rclone
        config:
          step_uri: space://steps/cosrclone
    push:
      - mode: cos_rclone
  - store_uri: hf://huggingface.co/my-org/my-model
    pull:
      - mode: hf_pull
    push:
      - mode: hf_push
```

### Retry strategies

The K8s environment ships Kubernetes-aware retry strategies, applied when `retry.enabled` is true:

| Strategy | Handles |
|----------|---------|
| `UnhealthyInsufficientPods` | Pods that never become healthy / insufficient scheduled pods. |
| `PodEviction` | Pod evictions (preemption, node pressure). |
| `NCCLError` | NCCL / distributed-training communication failures. |
| `Aspera` | Aspera asset-transfer failures (when `dmf.use_aspera` is enabled). |

See [builds/build-retry.md](../builds/build-retry.md) and
[builds/step-retry-configuration.md](../builds/step-retry-configuration.md) for how retry is
configured and how environment, build, and step retry interact.

### File permissions on the shared PVC (`umask`)

Step pods run as an **arbitrary UID** on OpenShift: the UID comes from the namespace's SCC
range and also depends on each step image's own `USER`, so it varies between pods and between
steps (`1002`, `1001060000`, `root`, …). What the pods *do* share is a group — the chart sets
`runAsGroup: 0` (see `run_as_root_group` in the step chart's `values-default.yaml`), following
[Red Hat's arbitrary-UID guidance](https://docs.redhat.com/en/documentation/openshift_container_platform/4.20/html/images/creating-images#use-uid_create-images).

A shared group only helps if the permission bits allow group writes. With the default `umask`
of `0022`, directories are created `0755` — group-readable but **not** group-writable — so a
later pod running as a different UID cannot write into a tree an earlier pod created. The K8s
step container therefore sets `umask` (default `0002`) before the workload runs, which makes new
directories `0775` and new files `0664`.

This applies to the containers that run step workloads (single- and multi-container steps). The
monitoring sidecar is not covered — it only writes its own config inside the container
filesystem, never the shared PVC.

This matters most for the shared HuggingFace cache (the `cache_path` of an `hf` assetstore, e.g.
`/gb-read-write/hfcache`). `huggingface_hub` writes its download lock files *inside* the snapshot
directory, at `<cache_path>/<owner>/<repo>/<hash>/.cache/huggingface/download/*.lock`, so the
locks are unavoidably on the shared PVC — pointing `HF_HOME` elsewhere does not move them. When
the cache directory is not group-writable, a second pull fails with a `PermissionError` on the
lock file.

Override per environment if a site needs different bits:

```yaml
config:
  umask: "0002"    # must be quoted, and a valid 3-4 digit octal
```

> [!IMPORTANT]
> **Quote the value.** Unquoted `0002` is parsed by YAML as the integer `2`, and `0027` becomes
> `23` — which bash would read as octal `0023`, *loosening* permissions instead of tightening
> them. The step prologue validates the value and falls back to `0002` with a warning in the pod
> log rather than applying a wrong mask, so a misconfiguration is visible instead of silent.

Keep the group-write bit clear (the `2` in `0002` masks only "other" writes). A umask that masks
group writes — `0022`, the default when unset — is what causes the failure described above.

**One-time cleanup for existing caches.** A `umask` only affects newly created files, so trees
created before this setting existed stay `0755`. Fix them once per cluster from a pod that mounts
the PVC:

```bash
chmod -R g+rwX /gb-read-write/hfcache
```

A directory's mode can only be changed by its owner or root, so run this with sufficient
privilege; anything it cannot fix is recreated group-writable the next time a pull recreates it.

### Artifact permissions on step output (post-workload `chmod`)

A `umask` sets the *default* mode for newly created files, but it can only mask bits **off** the
mode a writer asks for — it cannot add any. A workload that explicitly creates a file `0600` gets
`0600` under any umask. `safetensors` does exactly this (it writes via `mkstemp`, which is `0600`
by design), so a training step can leave `adapter_model.safetensors` unreadable to every UID but
its own, sitting beside `0644` siblings.

Because the next step runs in a different pod on a different UID, a subsequent `hfpush` then fails
on a file that plainly exists:

```
[Errno 13] Permission denied: '/gb-read-write/custom_output_xxx/model/adapter_model.safetensors'
```

BYOI steps can run arbitrary code and set arbitrary modes, so the only reliable point of control
is *after* the workload exits. The step container normalizes the output tree once the workload is
done:

```bash
chmod -R g+rwX "${OUTPUT_PATH}"
```

The block lives in one place — the `gbstepbase.normalizeOutputPermissions` define in
`charts/gbstepbase/templates/_utils.tpl` — and both the single- and multi-container templates
`include` it. It references no chart values, so the same copy serves both contexts.

- It runs **before** the exit-code check, so a failed run's partial output is normalized too —
  later pods still read and retry against it.
- `g+rwX` (capital `X`) adds group-execute only to directories and files that are *already*
  executable, never to data files.
- It is a no-op when `OUTPUT_PATH` is unset or absent, which is the case for the built-in steps
  (`hfpush`, `hfpull`, `lhpush`, …) — they consume artifacts rather than produce them.

**This is a guard, not a gate.** The artifact is often already readable, and some environments
forbid `chmod` outright — a read-only or root-squashed mount, or files owned by another UID — so a
`chmod` failure says nothing about whether the step succeeded and **never fails the step**. It is
not silent either, since it predicts a later push failure: the step logs a `WARNING` naming the
paths it could not fix. `chmod -R` continues past individual errors, so a partial pass still does
its work.

The warning goes to **stdout**, not stderr, and the listing is capped at 10 paths (with the true
total reported). Both are deliberate: only `command.sh` is tee'd into `/logs/output.log` — the file
the sidecar monitor tails — and this block runs after that pipeline, so a bare stderr write would
never reach the log an operator actually reads. Uncapped, a wholly root-squashed tree would emit one
line per file and flood both the log and the event stream.

`hfpush` also checks readability *before* it starts uploading, so an artifact that is still
unreadable names every offending path up front instead of failing mid-commit with a
`PermissionError` that carries no HTTP status (which reads like a Hub outage rather than a local
`EACCES`).

> [!NOTE]
> `fsGroup` in a pod `securityContext` is the other way to solve this, but it depends on the
> volume's CSI driver honouring ownership management — many NFS-backed RWX volumes ignore it — and
> these charts never see the PVC object (`/gb-read-write` is a path assumed to be pre-mounted).
> It also cannot fix files that already exist. The `chmod` pass has neither limitation.

## `step.yaml` — launcher and monitor types

| `type` | Method | When to use |
|--------|--------|-------------|
| `helm` (launcher) | `launch_helm` | Standard: submits the workload via Helm + AppWrapper. |
| `sidecar_monitor` | `monitor_sidecar_monitor` | Recommended: AppWrapper polling + RabbitMQ event monitor. |
| `appwrapper_only` | `monitor_appwrapper_only` | AppWrapper polling only, no RabbitMQ. |
| `event_monitor` | `monitor_event_monitor` | RabbitMQ events only, no AppWrapper polling. |
| `log_monitor` | `monitor_log_monitor` | Direct K8s API log streaming (no RabbitMQ required). |

Helm launcher `config`:

```yaml
launchers:
  training:
    type: helm
    monitors:
      - log_monitor
    config:
      chart: helm-charts/my-chart   # Required. Path to the Helm chart, relative to the step asset root.
```

## Step `config` blocks read by K8s

```yaml
config:
  gb:
    step_contents_in_env: true      # Copy the step asset directory into the pod. Default: true.
                                    # Set false for steps that don't need step files inside the pod.

  k8s:
    secrets:
      secret_names_to_use_as_pull_secret:
        - my_dockerconfig_secret    # Secret name whose value is a dockerconfigjson; creates an image
                                    # pull secret in the namespace.
      secret_names_to_use_as_env_variable:
        - env_name: HF_TOKEN        # Env var injected into the pod.
          secret_name: huggingface_token  # Space secret to read; falls back to env_name.lower().
    app_wrapper_config:
      warmupGracePeriodDuration: 30m  # Passed through to the Helm chart values.
      retryLimit: 2
    affinity:                         # Kubernetes affinity rules, merged into Helm values.
      nodeAffinity: {}
```

`compute_config.num_gpus_per_node` / `total_memory_per_node` are translated into pod resource specs by
the Helm chart values.

## Complete example

### `environment.yaml`

```yaml
name: vela-production
type: K8s
config:
  namespace: granite-build
  authentication:
    kube_config: prod_kubeconfig
    kube_context: prod-context
    ssl_verification: true
  messaging:
    authentication_secret_name: rabbitmq_prod
  retry:
    enabled: true
    max_retries: 3
assetstores:
  - store_uri: hf://huggingface.co/my-org
    pull:
      - mode: hf_pull
    push:
      - mode: hf_push
  - store_uri: cos://my-cos-bucket
    pull:
      - mode: cos_rclone
    push:
      - mode: cos_rclone
```

### `step.yaml`

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
            # Markers standardized on GB_; the legacy LLMB_ prefix is dual-accepted.
            - event_type: WORKLOAD_STATUS_EVENT
              line_regex: "^(?:GB_|LLMB_)EVENT_WORKLOAD_STATUS:.+"
              is_json: false
              event_fields:
                - field_name: status
                  field_regex: "(?:(?<=GB_EVENT_WORKLOAD_STATUS:)|(?<=LLMB_EVENT_WORKLOAD_STATUS:)).+"
```

## See also

- [Environments overview](README.md) and the shared [event_configs schema](README.md#event_configs--log-line-parsing-rules)
- [SkyPilot on Kubernetes](skypilot-kubernetes.md) — the same cluster, fronted by SkyPilot
- [Build retry](../builds/build-retry.md) · [Step retry](../builds/step-retry-configuration.md)
- [Bring your own image](../steps/bring-your-own-image.md)
- [Troubleshooting](../help/troubleshooting.md)
