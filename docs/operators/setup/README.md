# Operator setup scripts

Scripts and configuration for setting up the infrastructure gbserver depends on.
Today this directory is focused on SkyPilot Kubernetes setup and the Kubernetes
secrets the build steps consume.

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
