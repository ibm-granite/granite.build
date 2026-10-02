# SkyPilot on LSF

> **Audience:** operators configuring a `Skypilot` environment whose `default_cloud` is `lsf`.
> Read [skypilot.md](skypilot.md) first for the compute model and config common to all clouds; this
> page covers only what is LSF-specific. For the *native* LSF backend (gbserver submits `bsub`
> itself), see [lsf.md](lsf.md) instead.

## SkyPilot fork

gbserver installs SkyPilot from the granite-build fork
([cmadam/skypilot](https://github.com/cmadam/skypilot)), pinned in `pyproject.toml` to the tag
`gb-sky-v1-stable`. Upstream SkyPilot has no LSF cloud. The fork adds the LSF cloud driver and
LSF multi-node support, which includes the `sky.skylet.executor.lsf` task executor. gbserver
refuses a multi-node LSF launch when that executor is missing. The fork's own history lists
everything else it carries.

`gb-sky-v1-stable` is a moving tag: it is re-pointed to the recommended v1-line commit as the
fork advances, and a breaking change gets a new tag. Fresh installs pick up the current target.
An existing clone keeps the old one until you run `git fetch --tags --force`.

## Compute environment

With `default_cloud: lsf`, SkyPilot provisions onto an existing **LSF** cluster. It reaches the
cluster over **SSH** (login node) and submits jobs to an LSF queue via SkyPilot's LSF provisioner.
gbserver materializes both the SSH reachability config and any behavioral LSF tuning from the
environment.yaml at launch time, so the environment asset describes how to reach *and* how to run on
the cluster.

This is the path recipes use to drive SFT training and large eval suites on an on-prem LSF cluster.

## LSF-specific configuration

An LSF environment typically carries two inline blocks: `cluster_ssh_configs.lsf` (reachability) and
`cloud_config.lsf` (behavioral tuning). See [skypilot.md](skypilot.md#inline-skypilot-config-cluster_ssh_configs--cloud_config--aws_credentials)
for the shared merge/secret/collision rules.

### `cluster_ssh_configs.lsf` — reachability

SkyPilot's LSF provisioner reads `~/.lsf/config` (OpenSSH format) and derives the available cluster
names from its `Host` entries. Inline the host entries; gbserver materializes the file at launch:

```yaml
config:
  default_cloud: lsf
  cluster_ssh_configs:
    lsf:
      - Host: lsf-cluster           # Cluster alias (always literal); LSF derives the cluster name here.
        HostName: LSF_HOSTNAME      # Secret name (or literal). Keep sensitive values as secret names.
        User: LSF_USER
        Port: 22
        IdentityKey: LSF_SSH_KEY    # Key *contents* via a secret — gbserver writes a 0600 file and
        IdentitiesOnly: "yes"       # points IdentityFile at it. Use IdentityFile instead for an
                                    # on-host key path. Specifying both is an error.
```

gbserver merges this block into `~/.lsf/config` with last-writer-wins semantics: a differing
gbserver-managed block for the same alias is **overwritten** (a stale or re-keyed entry self-heals —
no manual `rm ~/.lsf/config`); a *foreign* (non-gbserver) entry for the same alias is refused
(`SkypilotConfigCollisionError`). An LSF and a SLURM env run concurrently (separate files). See
[Inline SkyPilot config](skypilot.md#inline-skypilot-config-cluster_ssh_configs--cloud_config--aws_credentials).

#### Multiple login nodes (`HostName` list)

`HostName` may be a single value (above) **or** a list of interchangeable candidate login hostnames
under one `Host` block, for a cluster fronted by several equivalent login nodes:

```yaml
  cluster_ssh_configs:
    lsf:
      - Host: bluevela
        HostName:                     # Candidate login nodes for this one cluster.
          - login1.bluevela.rmf.ibm.com
          - login2.bluevela.rmf.ibm.com
          - login3.bluevela.rmf.ibm.com
          - login4.bluevela.rmf.ibm.com
        User: granitebuild
        IdentityFile: ~/.ssh/ibm-bluevela.key
        IdentitiesOnly: "yes"
```

At launch gbserver picks one candidate and writes it as a scalar `HostName`. The pick is **sticky** —
a candidate already written for this cluster (e.g. by a parallel launch that just failed over) is kept
rather than re-randomized onto a wedged node; a random candidate is chosen only when nothing is written
yet (spreading load). The `Host` alias stays fixed (LSF derives the cluster name from it), so all
candidates share this block's credentials.

The candidates also form a **failover pool**: if provisioning fails with a transient SSH
control-plane error (a late banner, a wedged session, a key-exchange reset), gbserver rewrites
`~/.lsf/config` to the next candidate before retrying the launch, so a single wedged login node is
skipped rather than failing the build. Capacity failures and SSH *auth* rejections do not trigger
failover. Failover reuses the ordinary provision retry budget
(`GBSERVER_SKYPILOT_PROVISION_MAX_ATTEMPTS`); when the candidates are exhausted (or there is only one)
the genuine error surfaces. Failover is **launch-time only** — once a cluster is provisioned SkyPilot
pins the chosen login node, so status polling, log streaming, and teardown stay on it (unlike the
native-LSF SSH tunnel in [lsf.md](lsf.md), which re-picks per operation). This bluevela example has no
`cluster:` under `cloud_config` on the launcher; because the env declares exactly one host for `lsf`,
that host is the unambiguous launch target, so its candidates still fail over even on a bare `lsf`
infra. This is otherwise identical to SLURM — see
[Multiple login nodes on the SLURM page](skypilot-slurm.md#multiple-login-nodes-hostname-list).

> **Re-keying caveat (test-only `GBTEST_SKY_SSH_RESET`).** Even after `~/.lsf/config` self-heals,
> SkyPilot reuses a persisted SSH ControlMaster socket keyed on `(host, port, user)` — **not** the key
> — so a changed `IdentityFile`/`IdentityKey` can be masked by a live connection until its
> `ControlPersist` window expires (300s, or up to 1 day on the interactive-auth path). To validate a
> credential change against a freshly edited key, set `GBTEST_SKY_SSH_RESET=true` in gbserver's
> environment: on each HPC launch gbserver then clears the persisted control sockets first, forcing
> re-authentication with the current key. This is a **test-only** toggle (manually set, unconditional
> — not idle-gated); production never clears sockets, since the socket root is shared by all of the OS
> user's SkyPilot SSH connections. It is not an environment-config key.

### `cloud_config.lsf` — behavioral tuning

Structured LSF settings that can't live in the SSH file are deep-merged into `~/.sky/config.yaml`:

```yaml
config:
  cloud_config:
    lsf:
      allowed_clusters:
        - lsf-cluster
      cluster_configs:
        lsf-cluster:
          workdir: /shared/gbserver/skypilot
          tmpdir: /local/nvme/$USER/skypilot-tmp
          enroot:                          # Container runtime on the LSF nodes.
            enabled: true
            share_path: /shared/gbserver
            use_local_nvme: true
            squash_options: "-comp lz4 -Xhc -no-xattrs"
          nccl_tuning_file: /shared/gbserver/nccl-tuning.sh
          queue: normal
          bsub_options:
            G: my-lsf-group
            M: 64G
            W: 240            # Job wall-clock runlimit in MINUTES (bsub -W).
```

> **`sbatch_options` is a no-op on LSF.** The per-step `sbatch_options` field
> ([skypilot.md](skypilot.md#config-overrides-docker-sbatch_options)) is a
> **SLURM-only** knob; SkyPilot's fork exposes no per-task override for LSF, so a
> value set on a step/launcher/env is ignored (gbserver logs a WARNING). Set a
> job wall-clock runlimit at the **environment level** via
> `cloud_config.lsf...bsub_options.W` (minutes, `bsub -W`) as shown above, or
> rely on the queue's own `RUNLIMIT`.

### `zone` → LSF queue

SkyPilot's `zone` is overloaded per-cloud; for LSF it maps to the **queue** name (e.g. `normal`,
`preemptable`). Recipes that expose a `QUEUE` build parameter typically plumb it through
`resources.zone` on the step launcher (`zone: "$${QUEUE}"`).

LSF is an HPC cloud, so `cluster` and `zone` (queue) resolve with the same layered precedence as
SLURM — resources override > step/build `config` > this `environment.yaml` `config` — so the queue
can be pinned at env level instead of on every launcher. See
[skypilot-slurm.md](skypilot-slurm.md#cluster--zone) for the full precedence. As there, a `zone`
(queue) set **without** a `cluster` is rejected: SkyPilot cannot express a queue without a cluster,
so set a `cluster` alongside it.

### Autostop is ignored

LSF does not support cluster autostop; gbserver forces `idle_minutes_to_autostop=None` on the `lsf`
cloud, so any value set on the env or launcher has no effect. Omit it.

### `env_local` asset store

LSF jobs write outputs directly to the shared filesystem (e.g. GPFS), so outputs are registered with
the no-op `env://` pull/push rather than transferred. Output URIs use the `env://` scheme.

The `env://` store is registered implicitly for **every** environment, so no `assetstores` entry is
needed for it — `env://` inputs/outputs work out of the box. Add an `assetstores` block only to
configure other schemes (e.g. `hf`) or to pin a specific `env://` `load`/`push` `mode`. See
[Asset stores](../asset-stores/README.md#store-types-and-uri-schemes).

## Multi-node

Set the node count in the step's `compute_config`; `launcher_config.num_nodes` (in the step or in
build.yaml) overrides it:

```yaml
compute_config:
  num_nodes: 2
launcher_config:
  resources:
    accelerators: "H100:8"   # PER NODE -> 16 GPUs across 2 nodes
```

`num_nodes` must not go under `resources`: it is a `sky.Task` field, not a `sky.Resources` one, so
SkyPilot drops it there silently. gbserver logs a warning and the run proceeds single-node.

**How it maps to LSF.** One `bsub` requests `num_nodes * num_cpus_per_node` slots with
`-R "span[ptile=<cpus per node>]"` so the slots are distributed one group per host — `-n` alone counts
slots, not hosts, and without the span term LSF may satisfy the count from fewer machines. Multi-node
jobs also get `-hl` (host-level limits), and the GPU request becomes per host rather than per task
once there is more than one slot per host.

**What the job sees.** The provisioner exports these into each node's container, derived from
`$LSB_HOSTS`:

| Variable | Meaning |
|---|---|
| `RANK` / `TOTAL_NODES` | This node's index, and the node count. Rank 0 is the first host LSF listed. |
| `MASTER_ADDR` / `MASTER_PORT` | Rendezvous address; the port is derived from the LSF job id so concurrent jobs do not collide. |
| `NUM_GPUS_PER_NODE` | GPUs detected on the node. |
| `LSB_HOSTS` / `LSB_JOBID` | Passed through from LSF. |

SkyPilot's own `SKYPILOT_NODE_RANK`, `SKYPILOT_NUM_NODES` and `SKYPILOT_NODE_IPS` are also set per
node. A step that drives `torchrun`/`accelerate` should read `RANK`/`TOTAL_NODES`/`MASTER_ADDR` and
pass them as CLI flags, then `unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT` before
launching, since those launchers set their own per-process values.

**Version requirement.** LSF multi-node needs a SkyPilot build with the driver-side LSF task executor
(`sky.skylet.executor.lsf`). An older build accepts `num_nodes`, allocates every node, and then runs
the task once with `SKYPILOT_NUM_NODES=1` — reporting success. gbserver checks for the executor and
fails at launch rather than letting that happen, so a `num_nodes > 1` LSF build requires the pinned
SkyPilot to be at least `gb-sky-v2-multinode`.

## Example `environment.yaml` (LSF)

```yaml
name: sky-lsf
type: Skypilot
config:
  default_cloud: lsf
  # autostop is intentionally omitted — gbserver forces autostop=None for the lsf cloud.
```

## Example target (`build.yaml`) on LSF

A recipe selects the LSF queue and cluster via `resources` on the launcher and writes its checkpoint to
the shared filesystem. The SFT target emits a `NEWARTIFACT_IN_ENVIRONMENT_EVENT` that resolves a
`checkpoint` binding consumed by downstream eval targets:

```yaml
targets:
  sft-training:
    environment_uri: space://environments/skypilot/lsf/my-lsf
    outputs:
      checkpoint:
        uri: "env://{{ binding.path }}"   # env_local: the run-specific dir the step wrote.
        type: model
    steps:
      - step_uri: space://steps/openinstruct-sft
        config:
          sft_config: { ... }
          launcher_config:
            resources:
              accelerators: "H100:1"
              cluster: "lsf-cluster"  # Combined with default_cloud → infra=lsf/lsf-cluster.
              zone: "normal"          # LSF queue.
              memory: "1580"

  olmes-gsm8k:
    environment_uri: space://environments/skypilot/lsf/my-lsf
    inputs:
      model_checkpoint:
        binding: sft-training.checkpoint
    outputs:
      sage_eval_results:
        type: dataset
        uri: "env:///shared/gbserver/eval/gsm8k"
    steps:
      - step_uri: space://steps/sage-eval
        config:
          sage_eval_config:
            model_path: "{{ bindings.model_checkpoint.binding.path }}"
            image_id: "docker:your-favorite-registry/sage-py311-olmes:0.025"
            # ...
          launcher_config:
            resources:
              accelerators: "H100:1"
              cluster: "lsf-cluster"
              zone: "preemptable"     # Eval targets run on the preemptable queue.
              memory: "256"
```

> Container images (`image_id` / `image_id` in the step config) require enroot on the LSF nodes — see
> the `cloud_config.lsf.cluster_configs.<cluster>.enroot` block above.

> **Container images must be Debian/Ubuntu-based (apt).** When running in a container, SkyPilot
> bootstraps its in-container SSH shim with `apt-get`, so only Debian-based images are supported (see
> the SkyPilot [Docker containers docs](https://docs.skypilot.ai/en/latest/examples/docker-containers.html)).
> A non-Debian image (e.g. a Fedora/RPM image) pulls fine but fails during job setup — enroot launches
> it, the `apt-get` step exits non-zero, and the failure surfaces only as a generic
> `ResourcesUnavailableError`. Confirm with `sacct -j <job_id> --format=JobID,State,ExitCode,Reason`:
> the container-setup sub-steps show `FAILED 1:0` while the host-side steps complete. The image must
> also grant passwordless `sudo` (or run as root).

### `file_mounts` inside enroot containers

With an image, the step's `run` executes inside an enroot container on the compute node, which has its
own HOME (`HOME=/`) and `/tmp` — neither shared with the login node where `file_mounts` are rsynced.
The only paths visible to **both** the login node and the containerized job are the shared network
filesystem roots (e.g. `/proj`), which are bind-mounted **identity** into the container. gbserver
therefore delivers container-bound mounts by writing them straight to the shared filesystem, routing by
destination shape:

- **relative** destinations are remapped to `<workdir>/<dst>` — an absolute path under the
  per-target-run workdir on `/proj`. The payload is rsynced there directly on the (sudo-less) login node
  and, because `/proj` is identity-mounted, the job (whose CWD is that workdir) reads it at exactly
  `./<dst>`. No container staging or copy-back is involved. Persistent and per-target-run on the shared
  workdir; requires `shared_workdir` on the environment.
- **absolute** destinations under a shared, identity-mounted root (e.g. `/proj/…`) are likewise written
  directly — the author's explicit shared-FS location, reachable at the same path in the container.
- **`~/…`** destinations are **rejected** by the launcher (`~` is not expanded). They would land under
  the login-node cluster home and be **invisible inside the container**, so `file_mounts` forbids them —
  use a relative destination instead.

Prefer a **relative** destination (see [file_mounts](skypilot.md#file_mounts)) — it is the simplest and
gives per-target isolation, with the payload written onto the shared workdir for the job to read.

> **Contrast with SLURM.** Because the LSF backend identity-mounts the shared-FS roots (`/proj`,
> `/opt/share`) into every container, a `shared_workdir` under one of them is automatically visible to
> containerized steps — no extra configuration. The SkyPilot **SLURM** backend does *not* do this; there
> a containerized step needs the SkyPilot `workdir` set to an ancestor of `shared_workdir` to get the
> per-run workdir mounted into the container (see
> [skypilot-slurm.md](skypilot-slurm.md#workdir-containerized-steps)). On LSF the `workdir` under
> `cloud_config.lsf.cluster_configs.<cluster>` need not be an ancestor of `shared_workdir` for this
> reason.

> **Implementation note.** SkyPilot's backend normally sudo-symlink-wraps every absolute,
> non-`~/`/non-`/tmp/` destination, which fails on the sudo-less login node and would redirect the
> payload away from the identity-mounted path. The team SkyPilot fork exempts the shared-FS roots from
> that wrap: `LsfContainerCommandRunner` (`sky/provision/lsf/command_runner.py`) exposes them via
> `get_unwrapped_mount_prefixes()` (wired with `shared_fs_roots` in `instance.py`), and
> `_execute_file_mounts` in `cloud_vm_ray_backend.py` skips the wrap for destinations under those roots.
> gbserver's launcher (`_remap_relative_dest` in `environment/skypilot.py`) does the relative→
> per-run-workdir remap for every backend; on shared-FS backends the remapped absolute path is what
> makes the payload land in the per-run workdir.

## See also

- [SkyPilot overview](skypilot.md) — compute model, launcher fields, inline-config rules
- [Native LSF environment](lsf.md) — gbserver submits `bsub` directly (no SkyPilot)
- [SkyPilot on SLURM](skypilot-slurm.md) — the other SSH-provisioned HPC backend
