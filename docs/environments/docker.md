# Docker environment

> **Audience:** operators configuring a `Docker` environment and step authors targeting it.
> For the common schema see [Environment overview](README.md).

## Compute environment

The **docker** environment runs each step in a local **Docker container** (or **Podman** via its
Docker-compatible API socket). It is the containerized analogue of the [bash](bash.md) environment:
same local machine, but the step runs inside an image you choose, with bind-mounted inputs/outputs and
optional CPU/memory/GPU limits.

The implementation is [`Docker`](../../src/gbserver/environment/docker.py). The `docker` SDK is
lazy-imported, so gbserver only requires it when a Docker environment is actually configured
(`pip install 'gbserver[docker]'`). For Podman, point `DOCKER_HOST` at its socket — the environment
also auto-discovers a running Podman machine socket:

```shell
export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock
```

## `environment.yaml`

```yaml
name: docker-image-hf
type: Docker
config:
  defaults:
    image: ""                  # Optional. Fallback image when neither the launcher nor the step
                               # config specify one (see image resolution below).
    env: {}                    # Optional. Default env vars for every container (lowest precedence).
assetstores:
  - store_uri: space://assetstores/hf/
    load:
      - mode: default          # HF snapshot is downloaded and bind-mounted into the container.
    push:
      - mode: default
  - store_uri: space://assetstores/local/
    load:
      - mode: default
    push:
      - mode: default
```

`config.defaults` holds environment-wide fallbacks (`image`, `env`). Most knobs are set per step.

## `step.yaml` — launcher and monitor types

| `type` | Method | Notes |
|--------|--------|-------|
| `docker` (launcher) | `launch_docker` | The only launcher. Pulls the image per `pull_policy`, then runs a detached container. |
| `docker_log` | `monitor_docker_log` | Streams container logs, applies `event_configs`, and reports the container exit code (and OOMKilled). A non-zero exit fails the step. |

Launcher `config` (`launcher_config`) fields:

```yaml
launchers:
  run:
    type: docker
    monitors:
      - docker_log
    config:
      image: docker.io/library/python:3.11-slim   # Optional. Highest-priority image source.
      command: "python run.py"                     # Optional. Container command (CMD override).
      env:                                         # Optional. Highest-priority env layer.
        FOO: bar
```

## Step `config.docker` block

```yaml
config:
  docker:
    image: ""                  # Image (middle priority — see resolution order).
    pull_policy: if-not-present # always | if-not-present (default) | never.
    registry_auth:             # Optional. Private-registry credentials.
      username: <user>
      password: <secret-or-token>
    env:                       # Env vars (middle precedence). A value may be a literal or {value: ...}.
      HF_TOKEN: <token>
```

**Image resolution** (first non-empty wins): `launcher_config.image` → `config.docker.image` →
`config.defaults.image`. If none is set, the launch fails with a clear error.

**Environment-variable precedence** (later wins): `config.defaults.env` → `config.docker.env` →
`launcher_config.env` → built-ins (`LLMB_DOCKER_LAUNCH_ID`, `LLMB_DOCKER_CONTAINER_NAME`).

## Resources

The launcher reads the generic [`compute_config`](README.md#stepyaml-config--common-fields-read-by-environments)
and translates it to Docker container limits:

| `compute_config` field | Container setting |
|------------------------|-------------------|
| `total_memory_per_node` (e.g. `32Gi`, `512Mi`) | `mem_limit` (`g`/`m` suffix) |
| `num_cpus_per_node` | `nano_cpus` (`n × 1e9`) |
| `num_gpus_per_node` | `device_requests` with `capabilities=[["gpu"]]` (requires the NVIDIA container toolkit) |

## How inputs and outputs are wired

- **Inputs.** Unlike [bash](bash.md), the Docker environment does **not** auto-export input paths as
  env vars. Wire a resolved binding into the container explicitly, e.g. via `launcher_config.env` using
  `{{ bindings.<name>.binding.path }}`. HuggingFace inputs loaded via the `hf` asset store are
  bind-mounted read-only under `/gb-hf-models/...` (the whole `models--org--repo` dir is mounted so HF
  cache symlinks resolve), and the in-container path is supplied as the binding.
- **Workspace.** The step's asset directory is bind-mounted at `/gb-workspace` (read-write); write
  outputs there.
- **Outputs.** Register artifacts with the standard `LLMB_ARTIFACT_ID:<id> LLMB_ARTIFACT_PATH:<path>`
  log line (see the [event_configs schema](README.md#event_configs--log-line-parsing-rules)). On push,
  container paths under `/gb-workspace` are translated back to their host path automatically.

## See also

- [Environments overview](README.md)
- [Bash environment](bash.md) — the non-containerized local backend
- [Bring your own image](../steps/bring-your-own-image.md)
- [build.yaml reference](../builds/build-yaml-reference.md)
