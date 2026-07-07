# SkyPilot on Kubernetes

> **Audience:** operators configuring a `Skypilot` environment whose `default_cloud` is `kubernetes`.
> Read [skypilot.md](skypilot.md) first for the compute model and config common to all clouds; this
> page covers only what is Kubernetes-specific. For the *native* Kubernetes backend (gbserver submits
> via Helm + AppWrapper), see [k8s.md](k8s.md) instead.

## Compute environment

With `default_cloud: kubernetes` (alias `k8s`), SkyPilot launches each step as a pod on an **existing
Kubernetes cluster**. SkyPilot uses the cluster reachable through your kube context — there is **no
SSH config** to materialize and no SkyPilot-managed cluster provisioning; the cluster already exists.

This differs from the native [K8s environment](k8s.md): that path submits an AppWrapper via Helm and
can stream live RabbitMQ events; the SkyPilot path launches a SkyPilot pod and uses the polling
`skypilot_monitor`. Choose SkyPilot-on-Kubernetes when you want one environment definition that targets
Kubernetes alongside other clouds with a uniform launcher.

## Kubernetes-specific configuration

### Credentials: `~/.kube/config`

SkyPilot reads the kube context from `~/.kube/config` on the gbserver host. This must be provisioned
out-of-band (it is not one of the inline materialized blocks). For a full setup walkthrough — including
the SkyPilot SSH-node-pool / RBAC requirements — see
[setup/skypilot-kubernetes-setup.md](setup/skypilot-kubernetes-setup.md).

### No `cluster_ssh_configs` or `aws_credentials`

The Kubernetes backend uses neither inline block. If you need to tune SkyPilot's Kubernetes behaviour,
use a `cloud_config` with a `kubernetes:` block (deep-merged into `~/.sky/config.yaml`); otherwise the
`config:` block is minimal.

### Autostop

Kubernetes supports autostop, but per-step `cleanup_skypilot()` already runs `sky down` after each step.
`idle_minutes_to_autostop` (default 10) is a safety net for crashed processes; set `0` for near-immediate
autostop or `null` to disable.

### `shared_workdir`

For cross-step state, point `shared_workdir` at a path backed by a **ReadWriteMany PVC** mounted on
every worker (e.g. `/mnt/shared`). See [skypilot.md](skypilot.md#shared_workdir).

## Example `environment.yaml`

```yaml
name: sky-kube
type: Skypilot
config:
  default_cloud: kubernetes
  idle_minutes_to_autostop: 0
assetstores:
  - store_uri: space://assetstores/env-local
    load:
      - mode: env_local
        config: {}
    push:
      - mode: env_local
        config: {}
  - store_uri: space://assetstores/hf
    load:
      - mode: hf_pull
        config:
          cache_path: /tmp/hf_cache
    push:
      - mode: hf_push
        config: {}
```

Steps may set `image_id` freely (Kubernetes runs containers natively — no Pyxis/enroot caveat), and
`resources.accelerators` / `resources.memory` map onto the pod's resource requests.

## See also

- [SkyPilot overview](skypilot.md) — compute model, launcher fields, inline-config rules
- [Native Kubernetes (`K8s`) environment](k8s.md) — Helm + AppWrapper, live RabbitMQ events
- [SkyPilot on Kubernetes setup](setup/skypilot-kubernetes-setup.md) — cluster prerequisites
