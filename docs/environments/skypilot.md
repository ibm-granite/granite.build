# SkyPilot (`Skypilot`) environment

> **Audience:** operators configuring a `Skypilot` environment and step authors targeting it.
> For the common schema and `event_configs` see [Environment overview](README.md). This page covers the compute
> model and the config **common to all clouds**; each cloud's specifics live on its own page.

## Compute environment

The **Skypilot** environment fronts several compute backends through the
[SkyPilot](https://docs.skypilot.co/) SDK. For each step it provisions a fresh SkyPilot cluster via
`sky.launch()`, runs the step's `run:` command on it, downloads the job log, then tears the cluster
down on cleanup. One environment definition can therefore target very different clusters — a
Kubernetes namespace, an AWS account, a SLURM partition, or an LSF queue — with a uniform launcher
shape.

The implementation is [`Skypilot`](../../src/gbserver/environment/skypilot.py). A managed-jobs variant,
`Skypilot_managed` ([skypilot_managed.py](../../src/gbserver/environment/skypilot_managed.py)), runs the
job under SkyPilot's managed-jobs controller; it inherits this configuration.

### Clouds

Select the backend with `config.default_cloud` (a step may override it per launch). Each cloud has its
own provisioning, credentials, and resource story:

| `default_cloud` | Backend | Page |
|-----------------|---------|------|
| `slurm` | SLURM cluster (SSH-provisioned) | [skypilot-slurm.md](skypilot-slurm.md) |
| `lsf` | IBM LSF cluster (SSH-provisioned) | [skypilot-lsf.md](skypilot-lsf.md) |
| `kubernetes` (alias `k8s`) | Existing Kubernetes cluster | [skypilot-kubernetes.md](skypilot-kubernetes.md) |
| `aws` | AWS (EC2 provisioning) | [skypilot-aws.md](skypilot-aws.md) |

SkyPilot supports further clouds (GCP, Azure, Lambda, RunPod, …) that gbserver passes straight through
as the `infra` argument; only the four above are documented here. To try SLURM locally, see
[skypilot-slurm-setup.md](setup/skypilot-slurm-setup.md).

## `environment.yaml` — config common to all clouds

The `config:` block is intentionally small — most per-launch knobs live on the step launcher (below).

```yaml
name: my-skypilot-env
type: Skypilot
config:
  default_cloud: k8s                # SkyPilot infra to provision on when a step doesn't override it.
                                    # Forwarded as the `infra` arg to sky.Resources. Default: "k8s".

  idle_minutes_to_autostop: 10      # Stop the cluster after N idle minutes. Default: 10. 0 = ASAP,
                                    # null = disable. Per-step cleanup already runs `sky down` after
                                    # each step, so this is a safety net for crashed processes.
                                    # SLURM and LSF do not support autostop — gbserver ignores this
                                    # value when the resolved cloud is `slurm` or `lsf`.

  cluster: <name>                   # Optional. SLURM convenience field composed into
                                    # infra=slurm/<cluster>. Other clouds: use resources.infra instead.

  zone: <zone>                      # Optional. Forwarded to sky.Resources(zone=...). Overloaded
                                    # per-cloud: for LSF it maps to the queue name (normal,
                                    # preemptable, ...).

  shared_workdir: <path>            # Optional. A filesystem mounted on every worker the env launches
                                    # against, used as the base dir for gbserver-managed cross-step
                                    # caches (HF cache) and exported as GB_SHARED_WORKDIR. See below.

  # Inline SkyPilot config — three optional, mostly cloud-specific blocks (see "Inline config"):
  cluster_ssh_configs: { ... }      # SSH reachability for slurm/lsf. See skypilot-slurm/-lsf pages.
  cloud_config: { ... }             # Behavioral SkyPilot config, deep-merged into ~/.sky/config.yaml.
  aws_credentials: [ ... ]          # AWS credential profiles. See skypilot-aws page.

assetstores:
  - store_uri: space://assetstores/hf
    pull:
      - mode: default              # Dispatch is by store type (hf) — queues the builtin hfpull step.
        config:
          cache_path: /tmp/hf_cache  # Optional. Defaults to {shared_workdir}/hf_cache when set,
                                     # else ~/.cache/gbserver/hf on the worker.
    push:
      - mode: default
```

> Outside k8s, `mode` must be unset or `default` — dispatch is by store type, not `mode`. See
> [asset stores](../asset-stores/README.md#load-and-push-modes).

> `env://` (shared-filesystem, no-op push/pull) is registered implicitly for every environment, so it
> needs no `assetstores` entry.

### `shared_workdir`

Each `sky launch` is a fresh allocation, so cross-step state needs a shared filesystem the cluster
admin provisions — gbserver does not create or mount it. When set, `shared_workdir`:

- is the default base for gbserver-managed caches (currently the HF asset cache);
- is exported to every step's `run` as `GB_SHARED_WORKDIR`;
- gets a per-target-run subdir `${shared_workdir}/builds/<build_id>/runs/<targetrun_id>/` that is the
  **initial CWD** of both the `setup` and `run` scripts (with no `shared_workdir` both phases instead
  share SkyPilot's default `~/sky_workdir` — either way the two phases start in the same directory). It
  is created lazily before the first step and `rm -rf`'d at target-run teardown; retries get a fresh
  dir.

When unset, gbserver-managed caches fall back to `~/.cache/gbserver/<store>` on the worker, which only
works when consecutive steps land on the same machine. Example paths per backend: `slurm: /shared`
(NFS/Lustre/GPFS), `k8s: /mnt/shared` (RWX PVC), `aws: /mnt/efs` (EFS/FSx).

## `step.yaml` — launcher and monitor types

| `type` | Method | Notes |
|--------|--------|-------|
| `skypilot` (launcher) | `launch_skypilot` | The only launcher. Builds a `sky.Task` + `sky.Resources` and calls `sky.launch()`. |
| `skypilot_monitor` | `monitor_skypilot_monitor` | The only monitor. Polls `sky.job_status()` and, on a terminal state, downloads the job log and applies `event_configs`. |

The launcher maps directly onto SkyPilot's
[`sky.Resources`](https://docs.skypilot.co/en/latest/reference/api.html#sky.Resources) and
[`sky.Task`](https://docs.skypilot.co/en/latest/reference/api.html#sky.Task); only the fields below are
passed through.

```yaml
environment_configs:
  Skypilot:
    default_launcher: <launcher_name>
    launchers:
      <launcher_name>:
        type: skypilot
        monitors:
          - skypilot_monitor
        config:                   # This whole block is the launcher_config (see "Field reference").
          # ---- sky.Resources ----
          image_id: docker:python:3.11-slim
                                  # Optional. Container image (note the `docker:` prefix). On SLURM this
                                  # REQUIRES the Pyxis SPANK plugin; omit on bare-host SLURM or launch fails.
                                  # Empty string ("") == no image → runs on the bare launcher node.
          resources:
            cloud: <cloud>        # Optional. Per-step override of the env's default_cloud.
            cpus: "2+"            # SkyPilot resource string. "2+" = 2 or more vCPUs.
            memory: "4+"          # "4+" = 4 GiB or more.
            accelerators: A100:1  # Optional. e.g. "A100:8", "H100:1".
            disk_size: 50         # Optional. GB.
            instance_type: <t>    # Optional. Cloud-specific VM/instance type (e.g. AWS "g5.xlarge").
            use_spot: true        # Optional. Provision a spot/preemptible instance (cloud-dependent).
            infra: <infra-string> # Optional. Full infra spec, e.g. "slurm/cluster/partition".
                                  # If unset and `cluster` is set, gbserver builds "<cloud>/<cluster>[/<zone>]".
            cluster: <name>       # Optional. Combined with cloud to produce infra.
            zone: <zone>          # Optional. Cloud zone (LSF: queue name).

          # ---- sky config overrides (SkyPilot's task-level `config:`) ----
          docker:                 # Optional. Deep-merged into sky.Resources._cluster_config_overrides.
            run_options:          # Only the `docker` section is passed through per-step; other SkyPilot
              - "--shm-size=8g"   # config sections belong in the env-level `cloud_config` block.

          # ---- sky.Task ----
          setup: |                # Optional. Run once at cluster bring-up (cached across reuse).
            pip install foo bar
          run: |                  # Required. The actual job each launch. CWD is the per-run workdir
            echo "GB_ARTIFACT_ID:my_out GB_ARTIFACT_PATH:/tmp/out.json"   # (or $HOME).
          envs:                   # Optional. Extra env vars. Merged AFTER env-level secrets and
            FOO: bar              # BEFORE config.launcher_config.envs. GB_* vars are auto-injected.
          file_mounts:            # Optional. Two forms (see "file_mounts" below):
            /remote/path: /local/path          # String → local-to-remote copy (set_file_mounts).
            /remote/bucket-path:               # Dict → SkyPilot Storage mount (set_storage_mounts).
              source: s3://bucket/prefix
              mode: MOUNT                       # MOUNT (default) or COPY.

          post_launch_task:       # Optional. After the job starts, SSH to the host and run these
            run: |                # commands out-of-band (e.g. start an evaluator sidecar). A failure
              ./start-sidecar.sh  # is logged + surfaced as a MESSAGE_EVENT but does not fail the step.

          idle_minutes_to_autostop: 10         # Optional. Per-step override of the env-level value.
                                               # Ignored on slurm/lsf (they don't support autostop).
    monitors:
      skypilot_monitor:
        ref: space://monitors/skypilot   # shared monitor (GB_ARTIFACT_* rules, 300s default poll)
        config:
          # Optional overlay. Templated so a build.yaml step `config:` can override it.
          poll_interval_seconds: "{{ config.poll_interval_seconds | default(900) }}"
          log_retrieval:
            mode: "{{ config.log_retrieval_mode | default('on_completion') }}"  # see "Log retrieval"
```

### Field reference

All fields below live under a launcher's `config:` block (referred to as the **`launcher_config`** in
code). Every key here can be **overridden per-build** by the same key under a `build.yaml` step's
`config.launcher_config.*` — that layer wins over the step.yaml default. This is why steps template
values off `{{ config.* }}`: it lets a build supply resources/env without editing the step.

#### `image_id`

The container image, written with SkyPilot's `docker:` scheme prefix (e.g.
`docker:python:3.11-slim`, `docker:vllm/vllm-openai:latest`). Resolution order is
`config.launcher_config.image_id` (from build.yaml) → the launcher's own `image_id`. An **empty string
renders to "no image"** (`None`), so the `run` script executes on the bare launcher node — the pattern
the builtin `command` step uses:

```yaml
image_id: '{{ ("docker:" ~ config.command_config.image) if config.command_config.image else "" }}'
```

Cloud caveats: on **SLURM** a container image needs the Pyxis SPANK plugin installed on the cluster
(bare-host SLURM must omit `image_id`); on **Kubernetes** the image runs as the pod image; on **AWS**
it runs via Docker on the provisioned EC2 host.

#### `run` (required) and `setup`

`run` is the job body executed on every launch; `setup` runs once at cluster bring-up and is cached
across cluster reuse (put `pip install`/downloads here, not in `run`). Both are shell scripts and are
Jinja-templated against the step `config` — reference build inputs with `{{ config.<key> }}`. Both
bodies run under `set -eu` (see **Failure semantics** below).

- **Failure semantics (`set -eu`):** the launcher prepends `set -eu` to every `run` and `setup` body,
  so a non-zero exit *anywhere* aborts the step and referencing an unset variable errors out — a body
  need not add its own `set -eu`. This also governs the raw command a step interpolates: the
  [`command`](../steps/command.md) step runs `{{ config.command_config.command }}` verbatim, so
  `python train.py; echo done` never reaches `echo done` if `train.py` exits non-zero, and
  `deploy.sh $OPTIONAL_FLAG` fails under `set -u` when `OPTIONAL_FLAG` is unset. Guard optional
  variables with `${VAR:-}` and append `|| true` where a non-zero exit is acceptable. `pipefail` is
  *not* enabled by default (it would change pipeline semantics for every step); add `set -o pipefail`
  yourself if a failing command *within a pipeline* should fail the step.
- **Artifacts:** emit a line matching `GB_ARTIFACT_ID:<id> GB_ARTIFACT_PATH:<path>` (or
  `... GB_ARTIFACT_STATE:<value>` for an in-memory value; the legacy `LLMB_` prefix is still accepted)
  and the monitor registers it as the step's output binding. The marker need not be at column 0 —
  retrieved SkyPilot logs prefix stdout lines.
- **Working directory:** `setup` and `run` always start in the **same** directory, so a relative path
  one phase writes (e.g. a repo `setup` clones) is read at the same place by the other. When the env
  defines `shared_workdir`, that directory is a per-target-run workdir, so relative output paths are
  isolated per run; without `shared_workdir` both phases run in SkyPilot's default `~/sky_workdir`
  (where relative `file_mounts` also land — see [file_mounts](#file_mounts)). The shared CWD is
  guaranteed by SkyPilot, which wraps both scripts with the same prologue that `cd`s into it. Prefer
  relative paths — steps need not know where they run.

#### `resources`

Maps onto [`sky.Resources`](https://docs.skypilot.co/en/latest/reference/api.html#sky.Resources).
gbserver passes through only `cloud`, `cpus`, `memory`, `accelerators`, `disk_size`, `instance_type`,
`use_spot`, `infra`, `cluster`, and `zone`. `infra` is assembled from `cluster`/`zone` when not given
explicitly (`<cloud>/<cluster>[/<zone>]`). Prefer numeric/`"N+"` strings for `cpus`/`memory`; note
`cpus: "1+"` **breaks on the LSF cloud** (it parses the value as a float without stripping `+`), so
cloud-agnostic steps leave `resources` empty and let the build.yaml supply them.

> `compute_config` is **not** read by this launcher (unlike K8s/LSF) — see the dedicated note below.

#### `config` overrides (`docker`)

SkyPilot tasks accept a top-level `config:` block that overrides `~/.sky/config.yaml` per request; on
`sky.Resources` this is `_cluster_config_overrides`. gbserver exposes **only the `docker` section** of
it, as a launcher-level `docker:` key — e.g. `docker.run_options` to pass extra `docker run` flags
(`--shm-size`, `--gpus`, `--ipc=host`). It merges `launcher_config.docker` with
`config.launcher_config.docker` (build.yaml wins). Broader SkyPilot config (kubernetes, aws, nvidia,
etc.) is not per-step — set it once at the env level via `cloud_config` (see "Inline config").

#### `file_mounts`

Read from `launcher_config.file_mounts` or the step's top-level `config.file_mounts`. Two value forms,
which gbserver routes to different SkyPilot APIs:

- **String** (`/remote: <source>`) → `Task.set_file_mounts` — copies a local path (or SkyPilot-supported
  URI) to the remote path at bring-up. Works for files *and directories*.
- **Dict** (`{source, mode}`) → `Task.set_storage_mounts` with a `sky.Storage`. `mode` is `MOUNT`
  (default) or `COPY`. For a bucket URI with a sub-path (`s3://bucket/prefix`), gbserver splits it into
  the bucket source plus a bucket sub-path automatically so `MOUNT` mode (which requires a bucket-only
  source) works.

**Source path resolution.** A **relative** local source is resolved against the **`step.yaml`
directory** (the per-run asset dir gbserver renders the step into), so you can mount files that ship
alongside your `step.yaml`. Absolute paths and remote URIs (`s3://`, `gs://`, `file://`, …) are used
unchanged. A **`~`/`~/`-prefixed source is rejected**: `~` is not expanded for sources (it would
resolve to a literal `~` directory under the `step.yaml` dir), so use an absolute or step-relative
source instead. A **relative source that uses `..` to climb out of the `step.yaml` dir** (e.g.
`../other`) is likewise rejected, keeping sources confined to the step's own assets.

**Destination path resolution — the destination *shape* decides where the payload lands.** When the
environment defines `shared_workdir` (so a per-run workdir exists), the destination key is routed by
its shape:

| Destination | Lands at | Scope / lifetime |
|-------------|----------|------------------|
| **relative** (`payload`, `./payload`, `sub/payload`) | `<per-run-workdir>/<dst>` — the run script's CWD | per-target-run, persistent, shared across the target's steps |
| **absolute** (`/proj/…`, `/tmp/…`) | that literal path (author's responsibility) | as-is |

A **`~`/`~/…` destination is rejected** (like a `~` source): `~` is not expanded by the launcher, so
`file_mounts` has a single destination model — relative, or absolute for a fixed location.

**Relative in, relative out.** Because the `run` script's CWD is the per-run workdir, a **relative**
destination puts the payload at exactly `./<dst>` — the same path, relative to the step, that the
source occupies next to your `step.yaml`. On shared-FS backends (bluevela `/proj`) that path is also
visible **inside** the step container. This is the simplest option and gives implicit per-target
isolation:

```yaml
config:
  file_mounts:
    payload: payload        # <step.yaml dir>/payload  →  <per-run-workdir>/payload
  run: |
    ./payload/run-eval.sh   # CWD is the per-run workdir, so the mount is right here
```

Use an **absolute** path only when you must hit a fixed location. When `shared_workdir` is *not* set,
relative destinations fall back to SkyPilot's default (`~/sky_workdir/…`).

#### `envs`, `post_launch_task`, `idle_minutes_to_autostop`

- `envs` — extra environment variables for the job, merged after env-level secrets and before
  `config.launcher_config.envs`; the auto-injected `GB_*` vars (below) always win.
- `post_launch_task.run` — commands run on the host over SSH *after* the job starts (e.g. launching an
  evaluator sidecar). A failure is logged and emitted as a `MESSAGE_EVENT` but does not fail the step.
- `idle_minutes_to_autostop` — per-step override of the env-level autostop; ignored on `slurm`/`lsf`.

### Auto-injected environment variables

Added on top of (and overriding) anything in `envs`:

| Env var | Source |
|---------|--------|
| `GB_SKYPILOT_LAUNCH_ID` | The targetsteprun launch id (UUID). |
| `GB_SKYPILOT_CLUSTER_NAME` | The actual SkyPilot cluster name (`gb-<launch_id_prefix>`). |
| `GB_TARGETRUN_ID` | The enclosing target run id, when present. |
| `GB_BUILD_ID` | The build id, when present. |
| `GB_SHARED_WORKDIR` | The env-level `shared_workdir` path, when set. |
| `<env secrets>` | All secrets resolved from the env's `secret_refs`, merged before launcher `envs`. |

### `skypilot_monitor` config

The monitor polls `sky.job_status()` and applies `event_configs` (the `GB_ARTIFACT_*` rules, which
dual-accept the legacy `LLMB_` prefix) to the job log. Two config keys shape its behavior:

- **`poll_interval_seconds`** — status-poll cadence. **This gates completion detection:** the monitor
  only notices a job finished on its next poll (it sleeps the interval between polls; success does not
  wake it early), so a large interval delays detecting a *quick* job by up to that interval. The shared
  `space://monitors/skypilot` monitor defaults to **300s**, trading detection latency for lower polling
  load. Long-running steps (evals, training, services) override it *up* (e.g. `900`); build-test
  fixtures override it *down* (e.g. `5`). Written as `{{ config.poll_interval_seconds | default(300) }}`,
  so a `build.yaml` step `config:` sets it without touching the monitor.
- **`log_retrieval.mode`** — when the job log is pulled and parsed for artifact events (below).

#### Log retrieval modes

Set `config.log_retrieval.mode` on the monitor (the shared monitor templates it off
`{{ config.log_retrieval_mode | default('on_completion') }}`):

| `mode` | Behavior | Cost |
|--------|----------|------|
| `on_completion` (default) | Download the full log **once**, at terminal status. | Lightest. |
| `periodic` | Pull incrementally every `interval_seconds` (default = poll interval) while RUNNING, plus a final pull. | Medium. |
| `startup_window` | Pull periodically only for the first `startup_window_seconds` (default 120) after RUNNING, then stop (still pulls once at terminal). | Low; for early bind-address/URL scraping. |
| `stream` | Real-time `sky.tail_logs` follow stream. | Heaviest; opt-in. |

Each pull resumes past the lines already parsed, so events are not re-emitted. Additional keys under
`log_retrieval`: `interval_seconds` (periodic/startup cadence) and `startup_window_seconds`.

**Implications for artifact timing:** in the default `on_completion` mode, artifact lines
(`GB_ARTIFACT_ID:...`) are captured reliably even if they scrolled past a poll interval (download is
offline, not tail-based), but they are not **registered until the job completes** — a long step's
artifact events are batched at the end. Choose `periodic`/`stream` when a downstream step must consume
an artifact mid-run.

If a step exits with a non-`SUCCEEDED` JobStatus, the monitor emits a `WORKLOAD_STATUS_EVENT` with
status `FAILED` so the build fails even when the workload wrote no status line.

## Inline SkyPilot config (`cluster_ssh_configs` / `cloud_config` / `aws_credentials`)

These three optional blocks make a `Skypilot` environment **self-contained**: instead of an operator
pre-provisioning SkyPilot config files on the gbserver host, gbserver materializes them at build time
(in `setup_skypilot` and again just before `sky.launch`, via
[skypilot_config.py](../../src/gbserver/environment/skypilot_config.py)).

- **Where each lands.** `cluster_ssh_configs` writes the slurm/lsf reachability files SkyPilot reads
  (`~/.<cloud>/config`); `cloud_config` is deep-merged into `~/.sky/config.yaml`; `aws_credentials`
  writes `~/.aws/credentials` (mode 0600).
- **Secret resolution.** Every `cluster_ssh_configs` directive value (except the `Host` alias) and
  every `aws_credentials` value is looked up by exact name in the environment's secrets; a match is
  substituted, otherwise the literal is used. Keep credentials and sensitive hostnames as secret
  *names* so a git-tracked asset carries no secret material.
- **Content-aware merge, refuse-on-conflict.** Different clusters (distinct `Host` aliases) and AWS
  profiles coexist. An *identical* pre-existing entry is a no-op; a genuinely different one for the same
  alias/profile/leaf key raises `SkypilotConfigCollisionError`. Unrelated/foreign entries are preserved.
- **No teardown.** Materialized config is left in place (safe and idempotent); not removed on completion.
- **Concurrency.** Host-shared files are guarded by a cross-process file lock plus an in-process thread
  lock, so materialization is safe for any `GBSERVER_DEFAULT_BUILDRUNNER_TYPE` (thread/process/job).

The exact contents of each block are cloud-specific — see the per-cloud pages:
[SLURM](skypilot-slurm.md) and [LSF](skypilot-lsf.md) use `cluster_ssh_configs` (and LSF often
`cloud_config`); [AWS](skypilot-aws.md) uses `aws_credentials`; [Kubernetes](skypilot-kubernetes.md)
uses neither.

## `compute_config` is not honored by the Skypilot launcher

K8s and Lsf translate `compute_config.num_gpus_per_node` / `total_memory_per_node` into resource specs.
SkyPilot reads `resources` directly from the launcher config. If a step needs GPU/memory, set
`resources.accelerators` and `resources.memory` in the step.yaml (you may template them off
`{{ config.compute_config.* }}` for a single source of truth). The K8s-only `gb.step_contents_in_env`,
`k8s.*`, and `lsf.*` blocks are likewise ignored — step-asset code is not copied into the pod; if the
`run:` script needs files, use `file_mounts` or fetch them in `setup:` / `run:`.

## See also

- Cloud pages: [SLURM](skypilot-slurm.md) · [LSF](skypilot-lsf.md) · [Kubernetes](skypilot-kubernetes.md) · [AWS](skypilot-aws.md)
- [Local SLURM setup](setup/skypilot-slurm-setup.md) — Docker SLURM + MinIO for local testing
- [Environments overview](README.md) and the shared [event_configs schema](README.md#event_configs--log-line-parsing-rules)
