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

### `cluster` / `zone`

- `cluster` (env-level) is composed into `infra=slurm/<cluster>` for steps that don't set their own
  `resources.infra`. A step launcher can also set `resources.cluster`.
- `zone` maps to the SLURM **partition**.

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
    load:
      - mode: hf_pull
    push:
      - mode: hf_push
```

A bash step on this env omits `image_id` and runs directly on the compute node:

```yaml
environment_configs:
  Skypilot:
    default_launcher: bash
    launchers:
      bash:
        type: skypilot
        monitors:
          - skypilot_monitor
        config:
          # No image_id — runs directly on the SLURM compute node.
          resources:
            cpus: "1+"
            memory: "1+"
          run: |
            {{ config.bash_config.command }}
    monitors:
      skypilot_monitor:
        type: skypilot_monitor
        config:
          poll_interval_seconds: 5
          event_configs:
            - event_type: NEWARTIFACT_IN_ENVIRONMENT_EVENT
              line_regex: "LLMB_ARTIFACT_ID:.* LLMB_ARTIFACT_PATH:.*"
              is_json: false
              event_fields:
                - field_name: binding_id
                  field_regex: "(?<=LLMB_ARTIFACT_ID:)[^ ]+"
                - field_name: path
                  field_regex: "(?<=LLMB_ARTIFACT_PATH:).*"
                  is_data: true
                - field_name: binding
                  field_value_template: '{ "path": "{{ fields.data.path }}" }'
                  is_json: true
```

## See also

- [SkyPilot overview](skypilot.md) — compute model, launcher fields, inline-config rules
- [Local SLURM setup](setup/skypilot-slurm-setup.md) — bring up a Docker SLURM cluster + MinIO
- [SkyPilot on LSF](skypilot-lsf.md) — the other SSH-provisioned HPC backend
