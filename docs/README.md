# Granite.Build documentation

Topic index. The top-level [`README.md`](../README.md) is the project overview and
quickstart; everything below is reference material organized by audience.

## Reading paths

### I'm writing a build

You're authoring a `build.yaml`, picking environments, and submitting builds with `gb`.

- [Getting started](getting-started.md) — first build on the standalone server
- [`build.yaml` reference](users/build-yaml-reference.md) — full schema
- [CLI reference](users/cli-reference.md) — `gb` subcommands
- [HuggingFace push](users/hf-push.md) — `hf://` URIs and `store_push`
- [Bring your own image](users/bring-your-own-image.md) — custom container images
- [Try the demos](demos.md) — TRL fine-tuning and unitxt evaluation, standalone or on SLURM
- Working examples live in [`samples/`](../samples/) and [`examples/`](../examples/)

Cross-cutting features you'll reach for:

- [Build retry](features/build-retry.md) and [target reuse](features/target-reuse.md) — restart failed builds without re-doing successful targets
- [Step retry](features/step-retry-configuration.md) — retry a single step within one build
- [`gbtest`](features/gbtest.md) — YAML-driven assertions for your builds
- [Retry overview](features/retry.md) — how build- and step-level retry fit together

### I'm running gbserver

You're deploying gbserver, configuring environments, and keeping it healthy in production.

- [`environment.yaml` reference](operators/environment-yaml-config.md) — Kubernetes, LSF, SkyPilot, RunPod
- [Setup scripts and SkyPilot Kubernetes setup](operators/setup/)
- [Local SkyPilot infrastructure](operators/skypilot-local-infrastructure.md) — Docker SLURM + MinIO for local testing
- [RunPod orchestrator](operators/runpod-orchestrator.md) — gbserver as a persistent CPU orchestrator with on-demand GPU pods
- [Local secrets manager](operators/local-secrets-manager.md) — file-backed secrets with optional remote sync
- [Multi-provider authentication](operators/multi-provider-authentication.md) — GitHub, IBMid, API key
- [Troubleshooting](operators/troubleshooting.md) — common failures and where to look

Planned (not yet written): a "secrets and credentials" decision guide, and
a "monitoring and lineage" operator-facing summary.

### I'm changing gbserver

You're modifying gbserver internals — adding an environment, a step, an asset store, or fixing the build engine.

- [Architecture diagram](architecture/arch-diagram.md) — the big picture
- [Environment classes](architecture/environment-classes.md) — the `Environment` base class and concrete implementations

Planned (not yet written): a narrative architecture tour, a build-object-model
reference (`Build`/`Target`/`Step` lifecycles), and design docs for the asset
stores, storage layer, API layer, messaging, and resilience modules.

## Other

- [Dependency licenses](compliance/dependency-licenses.md) — Apache 2.0 audit
