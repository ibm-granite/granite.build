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

assetstores:
  - store_uri: cos://my-bucket
    load:
      - mode: cos_rclone
        config:
          step_uri: space://steps/cosrclone
    push:
      - mode: cos_rclone
  - store_uri: hf://huggingface.co/my-org/my-model
    load:
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
    load:
      - mode: hf_pull
    push:
      - mode: hf_push
  - store_uri: cos://my-cos-bucket
    load:
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
            - event_type: WORKLOAD_STATUS_EVENT
              line_regex: "^LLMB_EVENT_WORKLOAD_STATUS:.+"
              is_json: false
              event_fields:
                - field_name: status
                  field_regex: "(?<=LLMB_EVENT_WORKLOAD_STATUS:).+"
```

## See also

- [Environments overview](README.md) and the shared [event_configs schema](README.md#event_configs--log-line-parsing-rules)
- [SkyPilot on Kubernetes](skypilot-kubernetes.md) — the same cluster, fronted by SkyPilot
- [Build retry](../builds/build-retry.md) · [Step retry](../builds/step-retry-configuration.md)
- [Bring your own image](../steps/bring-your-own-image.md)
- [Troubleshooting](../help/troubleshooting.md)
