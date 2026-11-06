# Sizing a SkyPilot step

> **Audience:** build authors choosing CPUs, memory, GPUs and nodes for a step on a `Skypilot`
> environment. Environment setup is in [skypilot.md](skypilot.md) and the per-cloud pages.

## Rules for every cloud

Set these in the build.yaml step's `config`. gbserver applies `compute_config` underneath the
merged `resources` (step.yaml `resources`, then build.yaml `config.launcher_config.resources`; last
wins):

| What you want | Set this |
|---|---|
| CPUs | `compute_config.num_cpus_per_node` (or `launcher_config.resources.cpus`) |
| GPUs | `launcher_config.resources.accelerators`, e.g. `"H100:8"` (per node) |
| Memory | `compute_config.total_memory_per_node`, or `launcher_config.resources.memory` (GiB); not SLURM, see below |
| Nodes | `compute_config.num_nodes` (or `launcher_config.num_nodes`) |

- **One CPU setting is enough.** `num_cpus_per_node` becomes `sky.Resources.cpus`.
  `resources.cpus` sets the same value; if both are set, `resources.cpus` wins. Don't add it
  "to be sure".
- **A step's `resources: {}` does not mean "no CPU request".** Steps such as
  `command` and `custom_code_skypilot` leave `resources` empty to stay cloud-agnostic. gbserver
  still applies `compute_config` underneath. Only a step.yaml that sets its own `resources.cpus`
  outranks your `compute_config`; override that one in `launcher_config.resources`.
- **`num_gpus_per_node` is not read.** GPUs come only from `accelerators` (or an AWS
  `instance_type` that has them). gbserver warns when `num_gpus_per_node > 0` has neither. The
  step then runs without a GPU and can still report success.
- **`envs` does not size anything.** A step's `envs` (e.g. `GB_NUM_GPUS`, `GB_NUM_NODES`) only
  tells the workload what it got.
- **`num_nodes` does not go under `resources`.** It is a `sky.Task` field, so SkyPilot drops it
  there; gbserver warns and the run proceeds single-node.

`resources` values pass to SkyPilot verbatim; `compute_config` values are converted per cloud.
The sections below cover each cloud's conversion and its defaults.

## SLURM

What each key becomes in `#SBATCH` (the allocation is per node):

| Set this | What SLURM gets |
|---|---|
| `compute_config.num_cpus_per_node: 64` (or `resources.cpus: "64"`) | `--cpus-per-task=64` |
| `resources.accelerators: "H100:8"` | 8 GPUs via `--gres` (`gpu:<type>:8`; `sacct` shows the cluster's own GRES type name) |
| `resources.memory: "512"` (bare GiB, quoted) | `--mem=524288M` (512 GiB) |
| `compute_config.num_nodes: 2` | `--nodes=2` |

- **`total_memory_per_node` is not read on SLURM.** Many SLURM clusters don't track memory, so
  gbserver skips it; memory comes only from `resources.memory`.
- **`cpus` is an exact count** here, so a bare number is fine.
- **Not set at all?** SkyPilot's SLURM defaults apply: 2 CPUs with no GPU, 4 per GPU otherwise
  (32 for `H100:8`). Memory then follows the CPU count: 1 GiB per CPU without GPUs, 4 GiB per CPU
  with them (a cluster that doesn't track memory gets no `--mem`). A CPU-only job with no
  settings therefore gets 2 GiB, too small for a model load, so set `resources.memory` for real
  workloads.
- **The run step gets every CPU of the allocation.** Every rank of a multi-GPU task can use all
  `--cpus-per-task` cores, not one.

A full-node, 8-GPU training step, with nothing redundant:

```yaml
config:
  compute_config:
    num_nodes: 1
    num_cpus_per_node: 64
  launcher_config:
    resources:
      accelerators: "H100:8"
      memory: "512"
    sbatch_options:
      time: "6:00:00"
```

Common mistakes:

- `total_memory_per_node: 512Gi` alone does nothing here; memory falls back to the per-CPU
  default (64 CPUs with `H100:8` gives 256 GiB). Use `resources.memory: "512"`.
- `num_gpus_per_node: 8` alone gives no GPUs. Use `accelerators: "H100:8"`.
- A redundant `resources.cpus` next to `num_cpus_per_node`, justified by "the step has
  `resources: {}`". It's harmless, but the two can drift, and `resources.cpus` silently wins.
- A slow run blamed on the CPU request without checking `sacct`. Confirm the run step's
  `AllocCPUS` first; if it matches, the cause is elsewhere (data loading, dataloader workers,
  I/O on the shared filesystem).

### Check what a SLURM job actually got

Use `sacct`, and read the *step* rows, not just the job row:

```bash
sacct -j <jobid> --format=JobID,JobName,AllocCPUS,ReqMem,AllocTRES%60
```

`batch` is the allocation; `sky-<n>` is your run step and should show the same `AllocCPUS`. Inside
the workload, `python -c 'import os; print(len(os.sched_getaffinity(0)))'` prints the cores the
process may use. That is the number to compare against a slow run, not `nproc` on the login node.

Wall-clock limits, partitions and other scheduler directives are in
[SkyPilot on SLURM](skypilot-slurm.md) (`sbatch_options`, `zone`, multi-node and NCCL).

## Kubernetes

`cpus`, `memory` and `accelerators` become the pod's resource requests
([SkyPilot on Kubernetes](skypilot-kubernetes.md)). gbserver passes `num_cpus_per_node` and
`total_memory_per_node` as minimums (`"64+"`); SkyPilot drops the `+` and requests exactly
that number.

- **Not set at all?** SkyPilot's Kubernetes defaults match SLURM's: 2 CPUs with no GPU, 4 per GPU
  otherwise, and 1 GiB per CPU without GPUs, 4 GiB per CPU with them.

## AWS

SkyPilot picks the cheapest EC2 instance type that satisfies the request
([SkyPilot on AWS](skypilot-aws.md)). gbserver passes `num_cpus_per_node` and
`total_memory_per_node` as minimums (`"64+"`), so any instance with at least that much matches.

- **Write `resources.cpus` / `resources.memory` as minimums too** (`cpus: "3+"`). They pass to
  SkyPilot verbatim, and a bare number is an exact match that the catalog may not have
  ("Catalog does not contain any instances satisfying the request").
- **`instance_type` pins the machine.** Its vCPUs, memory and any GPUs come with it; don't also
  set `cpus`/`memory`.
- **Not set at all?** With no `accelerators` either, SkyPilot asks for at least 8 vCPUs and
  4 GiB per vCPU; with `accelerators`, it picks the cheapest instance type that has them.

## See also

- [How `compute_config` maps to SkyPilot](skypilot.md#how-compute_config-maps-to-skypilot)
- [SkyPilot on SLURM](skypilot-slurm.md), [on Kubernetes](skypilot-kubernetes.md), [on AWS](skypilot-aws.md)
