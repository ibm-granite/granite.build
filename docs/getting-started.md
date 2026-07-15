# Getting started

Walks you through running your first build on the standalone server. The
top-level [`README.md`](../README.md) has a faster overview; this guide
fills in the *why* and points to the right reference docs.

> **Audience:** users authoring `build.yaml` files. If you're deploying gbserver
> for a team, start with the [running-gbserver docs](README.md#im-running-gbserver) instead.

## Prerequisites

- Python 3.11+ (3.12 or 3.13 recommended)
- Docker or Podman with a running daemon (only if you want to use the Docker environment)

## Install

```bash
git clone git@github.com:ibm-granite/granite.build.git
cd granite.build

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[standalone,thirdparty]"
```

This installs both the server (`gbserver`) and the CLI client (`gb`).

The repo is private; clone over SSH (the HTTPS URL above will fail unless you
have HTTPS credentials configured for github.com).

## Run the server

```bash
gbserver standalone --space-dir configurations/spaces/local
```

The server listens on port 8080. It uses SQLite for metadata and runs builds in
threads, so no Kubernetes or PostgreSQL is required.

The `--space-dir` flag points at a space directory whose `space.yaml` chains
(via `base_uris`) into the shared *environments*, *steps*, and *asset stores*
under [`configurations/assets/`](../configurations/assets/). The
[`configurations/spaces/local/`](../configurations/spaces/local/) space is the
in-repo canonical example — read its `space.yaml` to see how a space is laid out,
or see [Spaces and `space.yaml`](spaces/README.md) for the full schema.
The build you submit below lives in
[`samples/standalone/standalone-quickstart/`](../samples/standalone/standalone-quickstart/).

> **Auth note (skip for localhost):** `gbserver` allows unauthenticated access
> from `127.0.0.1` / `::1` when `GBSERVER_API_KEY` is unset, so this localhost
> walkthrough just works. If you're running `gbserver` on a remote box, or
> the client and server are on different hosts, set a shared secret in both
> terminals before starting the server and submitting the build:
>
> ```bash
> export GBSERVER_API_KEY="my-secret-key"   # same value in both terminals
> ```

## Submit a build

In a second terminal:

```bash
source .venv/bin/activate
export GB_ENVIRONMENT=STANDALONE
gb build start -f samples/standalone/standalone-quickstart/build.yaml
```

The command prints a build ID. Use it to inspect progress:

```bash
gb build status <build-id>
gb build log <build-id>
gb build list
```

The quickstart `build.yaml` runs a single step in a local bash process. Edit
the `environment_uri` line to switch backends — the file has `bash`, `docker`,
`runpod`, and `skypilot` options pre-commented.

## What just happened

```
build.yaml ──→ gb build start ──→ gbserver REST API
                                       │
                                  BuildWatcher (polls for pending builds)
                                       │
                                  BuildRunner (walks the target graph)
                                       │
                                  Environment (bash | docker | k8s | runpod | skypilot)
                                       │
                                       └─→ runs your step, captures artifacts
```

For the longer version of this story, see
[`architecture/arch-diagram.md`](architecture/arch-diagram.md).

## Where to next

- Build something real → [`builds/build-yaml-reference.md`](builds/build-yaml-reference.md) for the full schema
- Use a different backend → [`environments/README.md`](environments/README.md)
- Push artifacts to HuggingFace → [`builds/hf-push.md`](builds/hf-push.md)
- Validate a build with assertions → [`cli/gbtest-cli-reference.md`](cli/gbtest-cli-reference.md)
- Hit a problem → [`help/troubleshooting.md`](help/troubleshooting.md)
