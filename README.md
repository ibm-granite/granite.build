# Granite.Build

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Build orchestration for LLM pipelines. Define multi-step model workflows in YAML — download, fine-tune, evaluate, and deploy — and run them locally or on cloud infrastructure.

_This repository is currently in alpha. The code and documentation are under active development and may change frequently as we work to improve usability and reliability. Contributions and feedback are welcome, but please be aware that breaking changes may occur._

## Contents

- [What is Granite.Build?](#what-is-granitebuild)
- [Quick start](#quick-start)
- [Example `build.yaml`](#example-buildyaml)
- [Repository layout](#repository-layout)
- [Features](#features)
- [Supported environments](#supported-environments)
- [CLI](#cli)
- [REST API](#rest-api)
- [Documentation](#documentation)
- [Try the demos](#try-the-demos)
- [Contributing](#contributing)
- [License](#license)

## What is Granite.Build?

Granite.Build orchestrates LLM build pipelines. You describe your workflow in a `build.yaml` file — which models to download, how to fine-tune them, what evaluations to run — and Granite.Build executes each step in the environment you choose: a local Docker container, a Kubernetes cluster, a cloud GPU instance, or a plain bash process on your laptop.

The system has three main components:

- **gbserver** — the orchestration server. It provides a REST API (`/api/v1`) for build management and a build watcher that polls for pending builds and dispatches them to execution environments. It stores build metadata in SQLite (standalone) or PostgreSQL (production).

- **gb** (gbcli) — the command-line client. It talks to the server's REST API to submit builds, list status, manage artifacts, and more.

- **build.yaml** — the pipeline definition. Each file declares a set of named **targets** (logical stages like "download", "fine-tune", "evaluate"). Each target specifies an execution environment, input/output artifacts, and one or more **steps** to run. Targets can depend on each other through artifact **bindings** — when an upstream target produces an output, downstream targets that reference it are automatically dispatched.

### How the pieces fit together

```
build.yaml ──→ gb build start ──→ gbserver REST API
                                       │
                                  BuildWatcher
                                       │
                                  BuildRunner
                                       │
                          ┌────────────┼────────────┐
                          │            │            │
                       Docker     Kubernetes     Bash
                       RunPod      SkyPilot
                          │            │            │
                          └────────────┼────────────┘
                                       │
                              Artifact stores
                          (HuggingFace, file://, git://)
```

The **BuildWatcher** polls storage for pending builds and creates a **BuildRunner** for each one. The runner walks the target graph, resolving dependencies and launching steps through the configured **Environment** (Docker, Kubernetes, Bash, RunPod, or SkyPilot). Each step can pull inputs from and push outputs to **artifact stores** selected by URI scheme (`hf://`, `file://`, `git://`, `cos://`).

## Quick start

Five commands to a running build, using the bundled `standalone-quickstart` sample.

```bash
# 1. Clone and enter the repo
git clone git@github.com:ibm-granite/granite.build.git
cd granite.build

# 2. Create the venv and install (no Artifactory or cloud creds needed)
make standalone-venv PYTHON=python3.13
source .venv/bin/activate

# 3. Start the standalone server, pointed at the bundled sample space
gbserver standalone --space-dir samples/standalone/standalone-quickstart

# 4. In another terminal, activate the venv and submit the sample build
source .venv/bin/activate
export GB_ENVIRONMENT=STANDALONE
gb build start -f samples/standalone/standalone-quickstart/build.yaml

# 5. Watch progress
gb build status <build-id>
gb build log <build-id>
```

The sample runs a single step in a local bash process — no Docker required. To switch backends, edit the `environment_uri` line in
[`samples/standalone/standalone-quickstart/build.yaml`](samples/standalone/standalone-quickstart/build.yaml); the file has `bash`, `docker`, `runpod`, and `skypilot` options pre-commented.

> **Auth note (skip for localhost):** when the client and server are both on the same host, `gbserver` allows unauthenticated access from `127.0.0.1` / `::1` and the quickstart above just works. If you're running `gbserver` on a remote box (or hitting auth errors), set a shared secret in both terminals before running steps 3 and 4:
>
> ```bash
> export GBSERVER_API_KEY="my-secret-key"   # same value in both terminals
> ```

For a longer walkthrough of the same path, see [`docs/getting-started.md`](docs/getting-started.md).

## Example `build.yaml`

A minimal pipeline that runs a single step in a Docker container:

```yaml
llm.build:                   # alias: granite.build (both keys are accepted)
  name: my-build
  targets:
    download:
      environment_uri: space://environments/docker
      inputs:
        model:
          uri: hf://huggingface.co/ibm-granite/granite-3.3-2b-instruct
      outputs:
        model:
          uri: file:workspace/model
      steps:
        - step_uri: space://steps/somestep
```

A multi-target pipeline chains stages through bindings:

```yaml
llm.build:
  name: tune-and-eval
  targets:
    download:
      environment_uri: space://environments/docker
      outputs:
        model: { uri: file:workspace/model }
      steps:
        - step_uri: space://steps/somestep
    fine-tune:
      environment_uri: space://environments/docker
      inputs:
        model: { binding: download.model }
      outputs:
        checkpoint: { uri: file:workspace/checkpoint }
      steps:
        - step_uri: space://steps/sft
    evaluate:
      environment_uri: space://environments/docker
      inputs:
        model: { binding: fine-tune.checkpoint }
      steps:
        - step_uri: space://steps/eval
```

For the full schema, see [`docs/users/build-yaml-reference.md`](docs/users/build-yaml-reference.md).

## Repository layout

| Path | Description |
|------|-------------|
| `src/gbserver/` | Build orchestration server (REST API, build engine, storage). |
| `src/gbcli/` | CLI client (`gb`) for interacting with gbserver. |
| `src/gbcommon/` | Shared types and utilities. |
| `docs/` | User, operator, and contributor docs — start at [`docs/README.md`](docs/README.md). |
| `samples/` | Sample build configs, environments, and steps. The [`standalone-quickstart`](samples/standalone/standalone-quickstart/) is the canonical first build. |
| `examples/` | Worked examples for specific scenarios. |
| `configurations/` | Space, environment, step, and assetstore configurations consumed by builds. [`configurations/assets/`](configurations/assets/) holds the reusable assetstores, environments, and steps; [`configurations/spaces/standalone/public/`](configurations/spaces/standalone/public/) is the user-facing space for `GB_ENVIRONMENT=STANDALONE` and ships the build templates. |
| `test/` | Test suites for all components. |
| `scripts/` | Helper scripts including the standalone and SLURM demos. |
| `k8s/` | Helm charts for production Kubernetes deployment. |
| `Makefile` | `make standalone-venv`, `make demo-venv`, `make image`, format/lint targets. |

## Features

- **Multi-environment execution** — Docker, Kubernetes, RunPod, SkyPilot/AWS, or local bash
- **HuggingFace Hub integration** — download and push models and datasets via `hf://` URIs
- **Pipeline orchestration** — chain steps with artifact bindings in a single `build.yaml`
- **CLI client** — `gb` command for build management, artifact handling, model operations, and more
- **REST API** — FastAPI-based build management at `/api/v1`
- **Standalone mode** — SQLite + thread-based execution, no external services needed
- **Lineage tracking** — records data provenance of builds, targets, and artifacts

## Supported environments

| Environment | Platform | GPU Support | Status |
|-------------|----------|-------------|--------|
| Docker | Linux, macOS | Yes (nvidia-container-toolkit) | Stable |
| Bash | macOS / Linux | CPU only | Stable |
| Kubernetes | Linux | Yes | Stable |
| SLURM (via SkyPilot) | Linux | Yes (auto-detected) | Beta |
| RunPod | Cloud | Yes | Beta |
| SkyPilot / AWS | Cloud | Yes | Beta |

## CLI

The CLI client is available as multiple equivalent entry points: `gb`, `gbcli`, `llmbuild`, `llmb`, `lamb`. The test harness ships as `gbtest`, and the server ships as `gbserver`.

```
gb               # client (build, artifact, space, secret, model, ...)
gbserver         # server (standalone, rest-server, build-watch, build, ...)
gbtest           # YAML-driven build assertions for tests
```

Run `gb --help` or `gbserver --help` for top-level usage. Common flows:

```bash
gb build start -f build.yaml      # submit a build
gb build list                     # list recent builds
gb build status <build-id>        # show build state and per-step status
gb build log <build-id>           # stream logs
gb build cancel <build-id>        # cancel a running build
gb artifact list                  # list artifacts
```

For the full subcommand reference, see [`docs/users/cli-reference.md`](docs/users/cli-reference.md).

## REST API

The REST API is available at `/api/v1` when the server is running. Start with `gbserver standalone` or `gbserver rest-server` and visit `http://localhost:8080/docs` for the interactive OpenAPI documentation. Authentication options (GitHub, IBMid, API key) are documented in [`docs/operators/multi-provider-authentication.md`](docs/operators/multi-provider-authentication.md).

## Documentation

The [`docs/`](docs/) directory has complete reference material. Three reading paths from the [docs index](docs/README.md):

- **Writing a build** → [`build.yaml` reference](docs/users/build-yaml-reference.md), [CLI reference](docs/users/cli-reference.md), [HuggingFace push](docs/users/hf-push.md), [features](docs/features/) (retry, target reuse, lineage, gbtest).
- **Running gbserver** → [`environment.yaml` reference](docs/operators/environment-yaml-config.md), [setup scripts](docs/operators/setup/), [SkyPilot Kubernetes setup](docs/operators/setup/skypilot-kubernetes-setup.md), [troubleshooting](docs/operators/troubleshooting.md).
- **Changing gbserver** → [architecture diagram](docs/architecture/arch-diagram.md), [environment classes](docs/architecture/environment-classes.md).

## Try the demos

End-to-end demos with TRL fine-tuning and unitxt evaluation. Each runs locally and tears down cleanly. Full setup instructions in [`docs/demos.md`](docs/demos.md).

```bash
# Standalone Docker — fine-tune + eval in containers on this machine
make demo-venv PYTHON=python3.13 && source .venv/bin/activate
bash scripts/demo-standalone.sh

# SLURM via SkyPilot — same workload on a local Docker SLURM cluster, with MinIO push
make g4os-skypilot-venv PYTHON=python3.13 && source .venv/bin/activate
make minio-setup && make slurm-setup
bash scripts/demo-slurm.sh
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup, code style, and pull request guidelines. This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md). To report a vulnerability, see [`SECURITY.md`](SECURITY.md).

## License

[Apache License 2.0](LICENSE)
