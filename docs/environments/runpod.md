# RunPod (`Runpod`) environment

> **Audience:** operators configuring a `Runpod` environment and step authors targeting it.
> For the common schema see [Environment overview](README.md). To run gbserver *itself* as a persistent
> orchestrator on RunPod, see [runpod-setup.md](setup/runpod-setup.md).

## Compute environment

The **Runpod** environment launches each step on a [RunPod](https://www.runpod.io/) **GPU pod** — a
persistent GPU VM created on demand via the RunPod API, running the step's Docker image, then
terminated on cleanup. It is well suited to a CPU orchestrator that spins up GPU pods only for the
compute-heavy steps (fine-tuning, evaluation).

The implementation is [`Runpod`](../../src/gbserver/environment/runpod.py). The `runpod` SDK is
lazy-imported (`pip install runpod`).

## `environment.yaml`

```yaml
name: runpod-standalone
type: Runpod
config:
  authentication:
    api_key: RUNPOD_API_KEY       # Secret name (or env var name) holding the RunPod API key.
                                  # Default: RUNPOD_API_KEY. Resolved from secrets first, then env.
  defaults:
    gpu_type: "A100-80GB"         # Default GPU. A normalized name (see table) or a RunPod-native id.
    cloud_type: "SECURE"          # RunPod cloud type. Default: SECURE.
    container_disk_gb: 50         # Container disk size (GB). Default: 50.
    volume_gb: 100                # Persistent volume size (GB). Default: 100.
    volume_mount_path: "/workspace"  # Where the volume is mounted in the pod. Default: /workspace.
```

### GPU type normalization

`gpu_type` accepts either a RunPod-native id (e.g. `NVIDIA A100 80GB PCIe`) or one of these normalized
shorthands, which are mapped to the native id:

| Normalized | RunPod-native id |
|------------|------------------|
| `A100-80GB` | NVIDIA A100 80GB PCIe |
| `A100-40GB` | NVIDIA A100-SXM4-40GB |
| `H100-80GB` | NVIDIA H100 80GB HBM3 |
| `H100-SXM` | NVIDIA H100 SXM |
| `L40S` | NVIDIA L40S |
| `RTX-4090` | NVIDIA GeForce RTX 4090 |
| `RTX-A6000` | NVIDIA RTX A6000 |
| `A40` | NVIDIA A40 |

An unrecognized value raises `UnknownGPUType`.

## `step.yaml` — launcher and monitor types

| `type` | Method | Notes |
|--------|--------|-------|
| `runpod` (launcher) | `launch_runpod` | The only launcher. Creates the pod, waits until RUNNING (up to 10 min), then releases monitors. |
| `pod_status_monitor` | `monitor_pod_status_monitor` | Polls pod status; emits a `MESSAGE_EVENT` on each status change; exits on a terminal status (`EXITED`, `TERMINATED`, `ERROR`). |

Launcher `config` (`launcher_config`):

```yaml
launchers:
  train:
    type: runpod
    monitors:
      - pod_status
    config:
      image: "docker.io/my-namespace/fms-tuning:latest"  # Required. Pod Docker image.
      command: ""                                         # Optional. docker_args for the pod.
      env:                                                # Optional. Extra env vars for the pod.
        HF_TOKEN: <token>
```

Monitor `config`:

```yaml
monitors:
  pod_status:
    type: pod_status_monitor
    config:
      poll_interval: 10     # Seconds between status polls. Default: 10.
```

## Resources

The launcher reads `compute_config`:

| `compute_config` field | Effect |
|------------------------|--------|
| `gpu_type` | Overrides `defaults.gpu_type` for this step. |
| `num_gpus_per_node` | GPU count for the pod. Default: 1. |

## Monitoring caveat

`pod_status_monitor` tracks **pod status only** — it does not stream the container's logs and does not
apply `event_configs`. Artifacts from a RunPod step are therefore not registered by tailing
`LLMB_ARTIFACT_ID:` lines; have the step push its outputs to an asset store (e.g. an `s3push` step to a
bucket the orchestrator can read) rather than relying on log-line artifact detection.

## Built-in environment variables

Every pod gets `LLMB_RUNPOD_LAUNCH_ID` and `LLMB_RUNPOD_POD_NAME` (`gb-<step>-<launch_id[:8]>`) on top
of the environment- and launcher-level `env`.

## See also

- [RunPod setup](setup/runpod-setup.md) — running gbserver on RunPod with on-demand GPU pods
- [Environments overview](README.md)
- [SkyPilot environment](skypilot.md) — multi-cloud GPU orchestration (can also target RunPod)
