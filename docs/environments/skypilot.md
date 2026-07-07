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
    load:
      - mode: hf_pull               # Queues the builtin hfpull step on its own SkyPilot cluster.
        config:
          cache_path: /tmp/hf_cache  # Optional. Defaults to {shared_workdir}/hf_cache when set,
                                     # else ~/.cache/gbserver/hf on the worker.
    push:
      - mode: hf_push
  - store_uri: space://assetstores/env-local
    load:
      - mode: env_local             # No-op: output already on the shared FS (bare-metal HPC).
        config: {}
    push:
      - mode: env_local
        config: {}
```

### `shared_workdir`

Each `sky launch` is a fresh allocation, so cross-step state needs a shared filesystem the cluster
admin provisions — gbserver does not create or mount it. When set, `shared_workdir`:

- is the default base for gbserver-managed caches (currently the HF asset cache);
- is exported to every step's `run` as `GB_SHARED_WORKDIR`;
- gets a per-target-run subdir `${shared_workdir}/builds/<build_id>/runs/<targetrun_id>/`, exported as
  `GB_BUILD_WORKDIR` and set as the **initial CWD** of the `run` command. It is created lazily before
  the first step and `rm -rf`'d at target-run teardown; retries get a fresh dir.

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
        config:
          # ---- sky.Resources ----
          image_id: docker:python:3.11-slim
                                  # Optional. Container image. On SLURM this REQUIRES the Pyxis SPANK
                                  # plugin; omit on bare-host SLURM or the launch fails.
          resources:
            cloud: <cloud>        # Optional. Per-step override of the env's default_cloud.
            cpus: "2+"            # SkyPilot resource string. "2+" = 2 or more vCPUs.
            memory: "4+"          # "4+" = 4 GiB or more.
            accelerators: A100:1  # Optional. e.g. "A100:8", "H100:1".
            disk_size: 50         # Optional. GB.
            infra: <infra-string> # Optional. Full infra spec, e.g. "slurm/cluster/partition".
                                  # If unset and `cluster` is set, gbserver builds "<cloud>/<cluster>[/<zone>]".
            cluster: <name>       # Optional. Combined with cloud to produce infra.
            zone: <zone>          # Optional. Cloud zone (LSF: queue name).

          # ---- sky.Task ----
          setup: |                # Optional. Run once at cluster bring-up (cached across reuse).
            pip install foo bar
          run: |                  # Required. The actual job each launch.
            echo "LLMB_ARTIFACT_ID:my_out LLMB_ARTIFACT_PATH:/tmp/out.json"
          envs:                   # Optional. Extra env vars. Merged AFTER env-level secrets and
            FOO: bar              # BEFORE config.launcher_config.envs. GB_* vars are auto-injected.
          file_mounts:            # Optional. Two forms:
            /remote/path: /local/path          # String → local-to-remote copy.
            /remote/bucket-path:               # Dict → SkyPilot Storage mount.
              source: s3://bucket/prefix
              mode: MOUNT                       # MOUNT (default) or COPY.

          idle_minutes_to_autostop: 10         # Optional. Per-step override of the env-level value.
    monitors:
      skypilot_monitor:
        type: skypilot_monitor
        config:
          poll_interval_seconds: 15
          event_configs: [ ... ]  # Same schema as README.md#event_configs--log-line-parsing-rules.
```

### Auto-injected environment variables

Added on top of (and overriding) anything in `envs`:

| Env var | Source |
|---------|--------|
| `GB_SKYPILOT_LAUNCH_ID` | The targetsteprun launch id (UUID). |
| `GB_SKYPILOT_CLUSTER_NAME` | The actual SkyPilot cluster name (`gb-<launch_id_prefix>`). |
| `GB_TARGETRUN_ID` | The enclosing target run id, when present. |
| `GB_BUILD_ID` | The build id, when present. |
| `GB_SHARED_WORKDIR` | The env-level `shared_workdir` path, when set. |
| `GB_BUILD_WORKDIR` | Per-target-run subdir under `shared_workdir`, when set (also the run's initial CWD). |
| `<env secrets>` | All secrets resolved from the env's `secret_refs`, merged before launcher `envs`. |

### Monitoring & artifact-event timing caveat

`skypilot_monitor` does **not** stream logs in real time. It polls `sky.job_status()` on
`poll_interval_seconds`, and only **after** the job reaches a terminal status does it download the full
log and walk every line through `event_configs`. Consequences:

1. Artifact lines (`LLMB_ARTIFACT_ID:...`) are captured even if they scroll past a poll interval — log
   download is offline, not tail-based.
2. Artifacts are not registered until the job *completes*. There is no live event-monitor mode (unlike
   the K8s `sidecar_monitor`), so a long step's artifact events are batched at the end.

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
