# SkyPilot on SLURM

> **Audience:** operators configuring a `Skypilot` environment whose `default_cloud` is `slurm`.
> Read [skypilot.md](skypilot.md) first for the compute model and config common to all clouds; this
> page covers only what is SLURM-specific.

## Compute environment

With `default_cloud: slurm`, SkyPilot provisions onto an existing **SLURM** cluster. It reaches the
cluster over **SSH** (login node), submits the job to a partition, and runs the step on the allocated
compute node. gbserver materializes the SSH reachability config from the environment.yaml at launch
time, so the environment asset fully describes how to reach the cluster.

To stand up a local SLURM cluster for development and integration testing, see
[skypilot-slurm-setup.md](setup/skypilot-slurm-setup.md).

## SLURM-specific configuration

### `cluster_ssh_configs.slurm` — reachability

SkyPilot's SLURM provisioner reads `~/.slurm/config` (OpenSSH format). Inline the host entries and
gbserver materializes that file at launch:

```yaml
config:
  default_cloud: slurm
  cluster_ssh_configs:
    slurm:
      - Host: slurm-docker          # Cluster alias SkyPilot references (always literal).
        HostName: 127.0.0.1         # Each non-Host directive value is secret-name-or-literal.
        User: root
        Port: 2222
        IdentityFile: ~/.ssh/slurm_docker_key   # Path to a key already on the host.
        StrictHostKeyChecking: "no"
        UserKnownHostsFile: /dev/null
```

Keys are the **exact OpenSSH directive names**, so the env mirrors `~/.slurm/config` 1:1. Use either
`IdentityFile` (a path to a key already on the host) **or** `IdentityKey` (the key *contents*, typically
via a secret — gbserver writes a `0600` file and points `IdentityFile` at it); specifying both is an
error. The SSH private key and the cluster itself stay out-of-band — gbserver does not provision them.

gbserver merges this block into `~/.slurm/config` with last-writer-wins semantics: a differing
gbserver-managed block for the same alias is **overwritten** (so a stale or re-keyed entry self-heals
— no manual `rm ~/.slurm/config`); a *foreign* (non-gbserver) entry for the same alias is refused
(`SkypilotConfigCollisionError`) — gbserver never clobbers user-owned entries.
See [Inline SkyPilot config](skypilot.md#inline-skypilot-config-cluster_ssh_configs--cloud_config--aws_credentials).

> **Re-keying caveat (test-only `GBTEST_SKY_SSH_RESET`).** Even after `~/.slurm/config` self-heals,
> SkyPilot reuses a persisted SSH ControlMaster socket keyed on `(host, port, user)` — **not** the key
> — so a changed `IdentityFile`/`IdentityKey` can be masked by a live connection until its
> `ControlPersist` window expires (300s, or up to 1 day on the interactive-auth path). To validate a
> credential change against a freshly edited key, set `GBTEST_SKY_SSH_RESET=true` in gbserver's
> environment: on each HPC launch gbserver then clears the persisted control sockets first, forcing
> re-authentication with the current key. This is a **test-only** toggle (manually set, unconditional
> — not idle-gated); production never clears sockets, since the socket root is shared by all of the OS
> user's SkyPilot SSH connections. It is not an environment-config key.

### `cluster` / `zone`

- `cluster` is composed into `infra=slurm/<cluster>` for steps that don't set their own
  `resources.infra`.
- `zone` maps to the SLURM **partition** (submitted via `--partition`), composed as
  `infra=slurm/<cluster>/<zone>`. It is **omitted entirely when unset**, letting SLURM pick the
  cluster's default partition. A `zone` set **without** a `cluster` is rejected with a clear
  error: SkyPilot's `cloud/region/zone` grammar cannot place a partition without a cluster, so a
  bare `zone` would otherwise be silently mislabeled as the cluster — set a `cluster` too.

Both are resolved with the following precedence (highest first), so the partition can be set at
whichever layer is most convenient:

1. `resources.infra` on the step launcher — an explicit full infra string wins outright.
2. `resources.cluster` / `resources.zone` on the step launcher (from `step.yaml`, or `build.yaml`
   `config.launcher_config.resources`).
3. `cluster` / `zone` as plain top-level keys in the step/build `config` (e.g. a `zone:` in
   `build.yaml`).
4. `cluster` / `zone` in this `environment.yaml` `config`.

This precedence is implemented in `Skypilot._resolve_infra_and_zone` and applies to the HPC
clouds (`slurm` and `lsf` — see [skypilot-lsf.md](skypilot-lsf.md); non-HPC clouds consult only
the step launcher's `resources`). For a real-cluster example, see the
[`skypilot/slurm/ibm-bluevela`](../../configurations/assets/environments/skypilot/slurm/ibm-bluevela/environment.yaml)
environment (BlueVela's `gpu-mid` partition, reached at `login1`).

### Autostop is ignored

SLURM does not support cluster autostop, so gbserver forces `idle_minutes_to_autostop=None` on the
`slurm` cloud — any value you set is ignored. Per-step `cleanup_skypilot()` runs `sky down` after each
step, which releases the node allocation. If you queue more parallel steps than the cluster has nodes,
the surplus stay PENDING until earlier ones finish and free a node.

### No `image_id` on bare-host clusters

Setting `image_id` on a launcher runs the job in a container, which on SLURM **requires the Pyxis SPANK
plugin**. On a bare-host SLURM cluster (including the local Docker fixture), omit `image_id` or the
launch fails with `NotSupportedError`; the `run:` command then executes directly on the compute node.

## Example `environment.yaml` (bare-host SLURM)

This is the pattern used by the
[`skypilot_slurm` integration test](../../test/integration/standalone/buildrunner/skypilot_slurm/)
against the local Docker SLURM cluster from [skypilot-slurm-setup.md](setup/skypilot-slurm-setup.md). No
`image_id` is set because the local cluster has no Pyxis plugin.

```yaml
name: slurm-local
type: Skypilot
config:
  default_cloud: slurm
  cluster: slurm-docker
  zone: normal                  # SLURM partition.
  idle_minutes_to_autostop: 0   # Ignored on SLURM; per-step `sky down` handles teardown.
  shared_workdir: /shared       # Path shared across slurmctld/c1/c2 in the local Docker fixture.
                                # HF cache defaults to /shared/hf_cache via this declaration.
  cluster_ssh_configs:
    slurm:
      - Host: slurm-docker
        HostName: 127.0.0.1
        User: root
        Port: 2222
        IdentityFile: ~/.ssh/slurm_docker_key
        StrictHostKeyChecking: "no"
        UserKnownHostsFile: /dev/null
assetstores:
  - store_uri: space://assetstores/hf
    pull:
      - mode: default
    push:
      - mode: default
```

A `command` step on this env runs directly on the compute node when no image is
given (leave `command_config.image` empty so `image_id` resolves to empty and the
launcher runs on the bare node; set it to run inside a container instead):

```yaml
environment_configs:
  Skypilot:
    default_launcher: command
    launchers:
      command:
        type: skypilot
        monitors:
          - skypilot_monitor
        config:
          # image_id resolves to "" when command_config.image is empty — runs
          # directly on the SLURM compute node.
          image_id: '{{ ("docker:" ~ config.command_config.image) if config.command_config.image else "" }}'
          resources:
            cpus: "1+"
            memory: "1+"
          run: |
            {{ config.command_config.command }}
    monitors:
      skypilot_monitor:
        type: skypilot_monitor
        config:
          poll_interval_seconds: 5
          event_configs:
            # Markers standardized on GB_; the legacy LLMB_ prefix is dual-accepted.
            - event_type: NEWARTIFACT_IN_ENVIRONMENT_EVENT
              line_regex: "(?:GB_|LLMB_)ARTIFACT_ID:.* (?:GB_|LLMB_)ARTIFACT_PATH:.*"
              is_json: false
              event_fields:
                - field_name: binding_id
                  field_regex: "(?:(?<=GB_ARTIFACT_ID:)|(?<=LLMB_ARTIFACT_ID:))[^ ]+"
                - field_name: path
                  field_regex: "(?:(?<=GB_ARTIFACT_PATH:)|(?<=LLMB_ARTIFACT_PATH:)).*"
                  is_data: true
                - field_name: binding
                  field_value_template: '{ "path": "{{ fields.data.path }}" }'
                  is_json: true
```

## See also

- [SkyPilot overview](skypilot.md) — compute model, launcher fields, inline-config rules
- [Local SLURM setup](setup/skypilot-slurm-setup.md) — bring up a Docker SLURM cluster + MinIO
- [SkyPilot on LSF](skypilot-lsf.md) — the other SSH-provisioned HPC backend
