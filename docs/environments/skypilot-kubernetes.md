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

### Idle timeout (autodown, not autostop)

A pod cannot be stopped, only deleted, so SkyPilot supports **autodown** on Kubernetes but not autostop —
a plain autostop request fails every launch with "Auto-stop is not supported on Kubernetes". gbserver
therefore applies `idle_minutes_to_autostop` on Kubernetes as autodown (`down=True`): after that many idle
minutes the pod is deleted.

Per-step `cleanup_skypilot()` already runs `sky down` after each step, so the idle timeout (default 10) is
only a safety net that reaps a pod orphaned by a crashed gbserver. Keep it small but not `0` — `0` means
"delete as soon as idle", which SkyPilot rounds up to 1 minute. Set `null` to disable it entirely.

> **`sbatch_options` is a no-op on Kubernetes.** The per-step `sbatch_options`
> field ([skypilot.md](skypilot.md#config-overrides-docker-sbatch_options)) is a
> **SLURM-only** knob; SkyPilot exposes no per-task equivalent on Kubernetes, so
> a value set here is ignored (a WARNING is logged). Bound job runtime inside the
> `run:` command or via a Kubernetes-level policy instead.

### `shared_workdir`

For cross-step state, point `shared_workdir` at a path backed by a **ReadWriteMany PVC** mounted on
every worker (e.g. `/mnt/shared`). See [skypilot.md](skypilot.md#shared_workdir).

Kubernetes has no host/container split to worry about: the step's image **is** the pod, and the PVC is
attached as a pod volume, so it is already the container's filesystem. There is no separate `workdir`
mount to configure (unlike SLURM) — mounting the RWX PVC at your `shared_workdir` path makes the per-run
workdir visible to every step, bare or containerized.

## Example `environment.yaml`

The `env://` store is registered implicitly for **every** environment, so it needs no `assetstores`
entry — declare a store below only for the schemes you actually configure (e.g. `hf`).

```yaml
name: sky-kube
type: Skypilot
config:
  default_cloud: kubernetes
  idle_minutes_to_autostop: 5   # applied as autodown on Kubernetes
assetstores:
  - store_uri: space://assetstores/hf
    pull:
      - mode: default
        config:
          cache_path: /tmp/hf_cache
          inline: true
    push:
      - mode: default
        config: {}
```

**`inline: true` is required without a `shared_workdir`.** Every step runs in its own pod, which
is torn down when the step finishes. Without `inline`, an `hf://` input is downloaded by a separate
`hfpull` step into *its* pod's `/tmp`, and the consuming step's fresh pod finds an empty directory.
`inline: true` injects the `hf download` into the consuming step's own `setup` instead — the same
arrangement as `skypilot/aws`. If you mount a ReadWriteMany PVC as `shared_workdir`, drop both
`cache_path` and `inline` so `hfpull` runs as its own step and caches to `${shared_workdir}/hf_cache`.

For the same reason this example has no cross-target handoff: `env://` outputs stay in the
producing pod. Add an `s3` assetstore or a `shared_workdir` PVC if one target must read another's
output.

Steps may set `image_id` (Kubernetes runs containers natively — no Pyxis/enroot caveat), and
`resources.accelerators` / `resources.memory` map onto the pod's resource requests.

### Custom images

Two requirements, both checked on a local kind cluster:

- **The image must be Debian/Ubuntu-based, or already contain SkyPilot's bootstrap packages.** Before a
  step runs, SkyPilot's pod startup (`sky/templates/kubernetes-ray.yml.j2`) needs `rsync curl wget netcat
  gcc patch pciutils openssh-server`; it installs any that are missing with **`apt-get`**, and the pod
  exits if it cannot. Debian/Ubuntu images (including the `python:*-slim` family) work as-is; a Fedora,
  RHEL or Alpine image works only if all of those are preinstalled. `quay.io/fedora/fedora-minimal`, for
  example, fails at startup. To avoid Docker Hub pull-rate limits, `public.ecr.aws/docker/library/<image>`
  mirrors the Docker official images.
- **An `hf://` input is downloaded inside the image.** With `inline: true` the download runs in the
  consuming step's own setup — in its image when it sets one. gbserver obtains the `hf` client itself:
  it uses `pip` when the image has one (unchanged from before), else a preinstalled `hf`, else it
  bootstraps one with `uv` via `curl` (which SkyPilot's startup guarantees). So images without pip work,
  but that last route needs outbound access to `astral.sh` and PyPI; on an air-gapped cluster use an image
  with `pip` or `hf` installed.

## See also

- [SkyPilot overview](skypilot.md) — compute model, launcher fields, inline-config rules
- [Native Kubernetes (`K8s`) environment](k8s.md) — Helm + AppWrapper, live RabbitMQ events
- [SkyPilot on Kubernetes setup](setup/skypilot-kubernetes-setup.md) — cluster prerequisites
