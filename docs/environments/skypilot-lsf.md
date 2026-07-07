# SkyPilot on LSF

> **Audience:** operators configuring a `Skypilot` environment whose `default_cloud` is `lsf`.
> Read [skypilot.md](skypilot.md) first for the compute model and config common to all clouds; this
> page covers only what is LSF-specific. For the *native* LSF backend (gbserver submits `bsub`
> itself), see [lsf.md](lsf.md) instead.

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
```

### `zone` → LSF queue

SkyPilot's `zone` is overloaded per-cloud; for LSF it maps to the **queue** name (e.g. `normal`,
`preemptable`). Recipes that expose a `QUEUE` build parameter typically plumb it through
`resources.zone` on the step launcher (`zone: "$${QUEUE}"`).

### Autostop is ignored

LSF does not support cluster autostop; gbserver forces `idle_minutes_to_autostop=None` on the `lsf`
cloud, so any value set on the env or launcher has no effect. Omit it.

### `env_local` asset store

LSF jobs write outputs directly to the shared filesystem (e.g. GPFS), so outputs are registered with
the `env_local` no-op pull/push rather than transferred. Output URIs use the `env://` scheme.

## Example `environment.yaml` (LSF)

```yaml
name: sky-lsf
type: Skypilot
config:
  default_cloud: lsf
  # autostop is intentionally omitted — gbserver forces autostop=None for the lsf cloud.
assetstores:
  - store_uri: space://assetstores/env-local
    load:
      - mode: env_local
        config: {}
    push:
      - mode: env_local
        config: {}
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

## See also

- [SkyPilot overview](skypilot.md) — compute model, launcher fields, inline-config rules
- [Native LSF environment](lsf.md) — gbserver submits `bsub` directly (no SkyPilot)
- [SkyPilot on SLURM](skypilot-slurm.md) — the other SSH-provisioned HPC backend
