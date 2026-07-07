# Environment setup

Guides, scripts, and configuration for setting up the compute infrastructure gbserver
runs against. For the `environment.yaml` reference for each backend, see the
[environments overview](../README.md).

## Setup guides

- [skypilot-kubernetes-setup.md](skypilot-kubernetes-setup.md) — configure SkyPilot against a
  Kubernetes or OpenShift cluster (RBAC, GPU setup, troubleshooting). See also [SkyPilot on
  Kubernetes](../skypilot-kubernetes.md).
- [skypilot-slurm-setup.md](skypilot-slurm-setup.md) — bring up a local Docker SLURM cluster + MinIO
  for development and integration testing. See also [SkyPilot on SLURM](../skypilot-slurm.md).
- [runpod-setup.md](runpod-setup.md) — run gbserver as a persistent CPU orchestrator on RunPod with
  on-demand GPU pods. See also the [RunPod environment](../runpod.md).

## SkyPilot on Kubernetes

- [skypilot-kubernetes-setup.md](skypilot-kubernetes-setup.md) — full guide for
  configuring SkyPilot against a Kubernetes or OpenShift cluster, including RBAC,
  GPU setup, and troubleshooting.
- [setup-skypilot.sh](setup-skypilot.sh) — idempotent setup script that creates
  the namespace, service account, RBAC, and `~/.sky/config.yaml`, and optionally
  deploys the SkyPilot API server.
- [skypilot-setup-config.yaml](skypilot-setup-config.yaml) — schema mapping each
  configuration value to its environment variable and default. Not edited
  directly; consumed by `setup-skypilot.sh`.
- [skypilot-setup.env.template](skypilot-setup.env.template) — copy this to
  `.env.skypilot`, edit, and `source` before running `setup-skypilot.sh`.

## Build-time secrets

The build steps reach into the SkyPilot namespace for credentials. These scripts
create the Kubernetes secrets they expect:

- [create-ghe-secret.sh](create-ghe-secret.sh) — creates `ghe-credentials` from
  `$GHE_TOKEN` for GitHub Enterprise access.
- [create-hf-secret.sh](create-hf-secret.sh) — creates `hf-credentials` from
  `$HF_TOKEN` for HuggingFace pulls and pushes.

Both default to namespace `skypilot`; override with `SKYPILOT_NAMESPACE`.
