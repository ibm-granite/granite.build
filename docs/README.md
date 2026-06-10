# Granite.Build documentation

Topic index. The top-level [`README.md`](../README.md) is the project overview and
quickstart; everything below is reference material organized by audience.

## Reading paths

### I'm writing a build

You're authoring a `build.yaml`, picking environments, and submitting builds with `gb`.

- [Getting started](getting-started.md) — first build on the standalone server
- [`build.yaml` reference](users/build-yaml-reference.md) — full schema
- [CLI reference](users/cli-reference.md) — `gb` subcommands
- [Templates](templates/README.md) — reusable build.yaml patterns and how to create your own
- [Steps](steps/README.md) — built-in steps, step.yaml structure, and custom step creation
- [HuggingFace push](users/hf-push.md) — `hf://` URIs and `store_push`
- [Bring your own step](users/bring-your-own-step.md) — custom code from a Git repo
- [Custom code steps](users/custom-code-steps.md) — inline commands without step definitions
- [Bring your own image](users/bring-your-own-image.md) — custom container images
- [FAQ](users/faq.md) — common questions and troubleshooting
- [Glossary](glossary.md) — key terms and abbreviations
- [Try the demos](demos.md) — TRL fine-tuning and unitxt evaluation, standalone or on SLURM
- Working examples live in [`samples/`](../samples/) and [`examples/`](../examples/)

Cross-cutting features you'll reach for:

- [Build retry](features/build-retry.md) and [target reuse](features/target-reuse.md) — restart failed builds without re-doing successful targets
- [Step retry](features/step-retry-configuration.md) — retry a single step within one build
- [`gbtest`](features/gbtest.md) — YAML-driven assertions for your builds
- [Retry overview](features/retry.md) — how build- and step-level retry fit together
- [Lineage tracking](features/lineage.md) — OpenLineage/W&B backend for build provenance

### I'm running gbserver

You're deploying gbserver, configuring environments, and keeping it healthy in production.

- [`environment.yaml` reference](operators/environment-yaml-config.md) — Kubernetes, LSF, SkyPilot, RunPod
- [Step resolution](operators/step-resolution.md) — how `space://steps/<name>` URIs route to the right impl per env
- [Setup scripts and SkyPilot Kubernetes setup](operators/setup/)
- [Local SkyPilot infrastructure](operators/skypilot-local-infrastructure.md) — Docker SLURM + MinIO for local testing
- [RunPod orchestrator](operators/runpod-orchestrator.md) — gbserver as a persistent CPU orchestrator with on-demand GPU pods
- [Local secrets manager](operators/local-secrets-manager.md) — file-backed secrets with optional remote sync
- [Multi-provider authentication](operators/multi-provider-authentication.md) — GitHub, IBMid, API key
- [Troubleshooting](operators/troubleshooting.md) — common failures and where to look

### I'm changing gbserver

You're modifying gbserver internals — adding an environment, a step, an asset store, or fixing the build engine.

- [Architecture diagram](architecture/arch-diagram.md) — the big picture
- [Environment classes](architecture/environment-classes.md) — the `Environment` base class and concrete implementations

## Other

- [Dependency licenses](compliance/dependency-licenses.md) — Apache 2.0 audit
