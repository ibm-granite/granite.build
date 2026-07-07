# Granite.Build documentation

Topic index. The top-level [`README.md`](../README.md) is the project overview and
quickstart; everything below is reference material organized by audience.

## Reading paths

### I'm writing a build

You're authoring a `build.yaml`, picking environments, and submitting builds with `gb`.

- [Getting started](getting-started.md) — first build on the standalone server
- [Builds](builds/README.md) — the build definition: targets, steps, environments, and artifacts
- [`build.yaml` reference](builds/build-yaml-reference.md) — full schema
- [`gb` CLI reference](cli/gb-cli-reference.md) — client subcommands (submit and monitor builds)
- [Templates](templates/README.md) — reusable build.yaml patterns and how to create your own
- [Steps](steps/README.md) — built-in steps, step.yaml structure, and custom step creation
- [HuggingFace push](builds/hf-push.md) — `hf://` URIs and `store_push`
- [Bring your own step](steps/bring-your-own-step.md) — custom code from a Git repo
- [Custom code steps](steps/custom-code-steps.md) — inline commands without step definitions
- [Bring your own image](steps/bring-your-own-image.md) — custom container images
- [FAQ](help/faq.md) — common questions and troubleshooting
- [Glossary](glossary.md) — key terms and abbreviations
- [Try the demos](demos/README.md) — TRL fine-tuning and unitxt evaluation (Docker or SLURM), plus Granite 4.0 Nano on AWS
- Working examples live in [`samples/`](../samples/) and [`examples/`](../examples/)

Cross-cutting [build features](builds/README.md#advanced) you'll reach for:

- [Build retry](builds/build-retry.md) and [target reuse](builds/target-reuse.md) — restart failed builds without re-doing successful targets
- [Step retry](builds/step-retry-configuration.md) — retry a single step within one build
- [`gbtest`](cli/gbtest-cli-reference.md) — YAML-driven assertions for your builds
- [Retry overview](builds/retry.md) — how build- and step-level retry fit together
- [Lineage tracking](builds/lineage.md) — OpenLineage/W&B backend for build provenance
- [Event notifications](builds/event-notifications.md) — real-time build event streaming and notifications

### I'm running gbserver

You're deploying gbserver, configuring environments, and keeping it healthy in production.

- [`gbserver` CLI reference](cli/gbserver-cli-reference.md) — running the REST API, watchers, build runner, standalone, and admin
- [Configuration](configuration/README.md) — runtime settings: env vars, config files, and `GB_ENVIRONMENT`
- [Spaces](spaces/README.md) — what a space is and the full `space.yaml` schema (`secret_manager`, `base_uris`, `variables`)
- [Environments](environments/README.md) — the compute-endpoint map, per-type `environment.yaml` reference (Bash, Docker, Kubernetes, LSF, RunPod, SkyPilot and its clouds), and the setup guides (SkyPilot Kubernetes/SLURM, RunPod)
- [Step resolution](environments/step-resolution.md) — how `space://steps/<name>` URIs route to the right impl per env
- [Asset stores](asset-stores/README.md) — how artifacts are located and reached by URI scheme (file, git, COS/S3, HF, Lakehouse, env-local)
- [Secrets](secrets/README.md) — secret-manager backends (local, env, hybrid, IBM Cloud) and how secrets are resolved
- [REST API](rest-api/README.md) — the `/api/v1` HTTP API, its interactive OpenAPI docs, and authentication (GitHub, IBMid, API key)
- [Troubleshooting](help/troubleshooting.md) — common failures and where to look

### I'm changing gbserver

You're modifying gbserver internals — adding an environment, a step, an asset store, or fixing the build engine.

- [Architecture](architecture/README.md) — the internals section index
- [Architecture diagram](architecture/arch-diagram.md) — the big picture
- [Environment classes](architecture/environment-classes.md) — the `Environment` base class and concrete implementations
- [Testing](testing/README.md) — running the suites via the Makefile, markers, and the `gbtest` CLI

## Other

- [Dependency licenses](dependency-licenses.md) — Apache 2.0 audit
