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
    pull:
      - mode: default          # HF snapshot is downloaded and bind-mounted into the container.
    push:
      - mode: default
  # Local filesystem (file: URIs) — the builtin file store. Docker implements
  # push only (pushasset_filestore); it has no pullasset_filestore, so declare
  # push and NOT load — otherwise a file: input would resolve here and fail.
  - store_uri: space://assetstores/file/
    push:
      - mode: default
```

> `space://assetstores/file/` is the bundled builtin File store (resolved via the
> builtins base_uri). Unlike `env://`/`mem://`, it is **not** auto-registered — an
> environment declares it here with the modes it implements. Docker implements
> `file:` **push** (outputs) only; it has no `pullasset_filestore`, so it does not
> declare `load`.

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
`launcher_config.env` → built-ins (`GB_DOCKER_LAUNCH_ID`, `GB_DOCKER_CONTAINER_NAME`). These
launcher vars are standardized on the `GB_` prefix; for backwards compatibility the legacy
`LLMB_DOCKER_*` twins are injected alongside them with the same values.

## Secrets

Declare the space secrets a step needs under `config.docker.secrets`; each is resolved in the build
server and passed to the container as an environment variable. Only declared secrets are injected,
as on LSF, SkyPilot and K8s (unlike [bash](bash.md), which exports every space secret), and a
declared secret missing from the space fails the launch. `secret_name` defaults to `env_name`
verbatim. A `config.docker.env` or launcher `env` entry of the same name wins over the secret.

```yaml
config:
  docker:
    secrets:
      secret_names_to_use_as_env_variable:
        - env_name: GRANITE_EVALS_JUDGE_API_KEY   # secret_name defaults to this
```

The value is visible to anyone who can `docker inspect` the container.

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
  cache symlinks resolve), and the in-container path is supplied as the binding. `env://` inputs (a
  path another target left on this host) are bind-mounted read-only under `/gb-env-inputs/<n>` (a
  file's parent directory, or the directory itself), and the binding is the in-container path. The
  mount goes only into the containers of the target that takes the input. The mount source is
  always the input's host path, including an `env://` output an earlier Docker target wrote in its
  own workspace. Because a file input mounts its **whole parent directory** (so the step can read
  the file's siblings), the mount source (with symlinks resolved; that resolved directory is what
  gets mounted) is checked first. The Docker daemon mounts as root, so the source is refused with an
  error, naming the rule, when:
  - the gbserver user cannot read it;
  - it contains the gbserver user's home directory (`/`, `/home`, `~`);
  - it is, contains or lies in a system or container-state directory (`/etc`, `/proc`, `/sys`,
    `/dev`, `/boot`, `/run`, `/var/run`, `/root`, `/var/lib/docker`, `/var/lib/containers`,
    `/var/log`; so all of `/var` is refused too);
  - it is, contains or lies in the GB home directory, or lies in a hidden directory of the home
    directory (`~/.ssh`, `~/.aws`, ...).

  The last rule has an exception for directories gbserver wrote step outputs to: a subdirectory of
  `<GB home>/workdir` (where a [bash](bash.md) target writes by default, e.g.
  `~/.granite.build/workdir/llm-build-<id>/...`), or a directory in the workspace of an earlier
  Docker launch of the same build. So a Docker target can read the `env://` output of a bash or
  Docker target, while the rest of the GB home (secrets, the database), `workdir` itself, and the
  build watcher's PR clones stay out of reach. Put the input in its own directory (e.g.
  `~/run1/generation.json`, not `~/generation.json`).
- **Workspace.** Each launch's own asset directory is bind-mounted at `/gb-workspace` (read-write)
  in that launch's container only; write outputs there.
- **Outputs.** Register artifacts with the standard `GB_ARTIFACT_ID:<id> GB_ARTIFACT_PATH:<path>`
  log line (or the `GB_ARTIFACT_STATE:<value>` form for `mem://` outputs; the legacy `LLMB_` prefix is
  still accepted). Docker steps reference the shared `space://monitors/docker` monitor, which carries
  both rules; they are **unanchored** (docker streams raw stdout).
  See [Monitoring and artifact events](../steps/monitoring-and-artifact-events.md).
  On push, container paths under `/gb-workspace` are translated back to the host workspace of the
  launch that produced them, so an `env://{{ binding.path }}` output written there is registered at
  its host path, and `file://` and `hf://` outputs are copied or uploaded from it. An `env://`
  output that re-emits an input (a path under `/gb-env-inputs` or `/gb-hf-models`) is registered at
  that input's host path. A literal `env://` URI naming a container path (e.g.
  `env:///gb-workspace/outputs/x`) is refused; use `env://{{ binding.path }}`.
  An `env://` output stays resolvable only while that workspace exists. It lives in a temporary
  directory under the build runner's workspace directory (`--workspace-dir`, the build watcher's
  `workspace_dir`, or `GBSERVER_DEFAULT_ROOT_WORKSPACE_DIR`), which gbserver **deletes when it
  exits**, or under the system temp dir (e.g. `/tmp`) for a build started without one, such as
  `gbserver build`. Copy an output to a `file://` URI to keep it past a restart.
- **Docker → Docker hand-offs.** Use `env://` for an output another Docker target reads. A
  `file://` output written in `/gb-workspace` is copied to its URI, but a Docker target cannot pull
  it back (there is no Docker `file://` pull), so that hand-off fails with an error.

## See also

- [Environments overview](README.md)
- [bash environment](bash.md) — the non-containerized local backend
- [Bring your own image](../steps/bring-your-own-image.md)
- [build.yaml reference](../builds/build-yaml-reference.md)
