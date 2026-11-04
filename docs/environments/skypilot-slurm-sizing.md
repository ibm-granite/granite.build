# Sizing a SkyPilot SLURM step

> **Audience:** build authors choosing CPUs, memory, GPUs and nodes for a step on a `Skypilot`
> environment with `default_cloud: slurm`. Environment setup is in [skypilot-slurm.md](skypilot-slurm.md).

## What to set

Set these in the build.yaml step's `config`. The allocation SLURM grants is per node:

| What you want | Set this | What SLURM gets |
|---|---|---|
| CPUs | `compute_config.num_cpus_per_node: 64` (or `launcher_config.resources.cpus: "64"`) | `#SBATCH --cpus-per-task=64` |
| GPUs | `launcher_config.resources.accelerators: "H100:8"` | `--gres=gpu:H100:8` |
| Memory | `launcher_config.resources.memory: "512"` (bare GiB, quoted) | `--mem=524288M` (512 GiB) |
| Nodes | `compute_config.num_nodes: 2` | `--nodes=2` |

- **One CPU setting is enough.** `num_cpus_per_node` becomes `sky.Resources.cpus`, which
  becomes `--cpus-per-task`. `resources.cpus` sets the same value; if both are set,
  `resources.cpus` wins. Don't add it "to be sure".
- **A step's `resources: {}` does not mean "no CPU request".** Steps such as
  `command` and `custom_code_skypilot` leave `resources` empty to stay cloud-agnostic. gbserver
  still applies `compute_config.num_cpus_per_node` underneath. Only a step.yaml that sets its own
  `resources.cpus` outranks your `compute_config`; override that one in
  `launcher_config.resources`.
- **`envs` does not size anything.** A step's `envs` (e.g. `GB_NUM_GPUS`, `GB_NUM_NODES`) only
  tells the workload what it got.
- **Not set at all?** SkyPilot's SLURM defaults apply: 2 CPUs with no GPU, 4 per GPU otherwise
  (32 for `H100:8`). Without `resources.memory`, the job gets the partition's default memory,
  often a few GB and too small for a model load, so set it for real workloads.
- **`num_gpus_per_node` and `total_memory_per_node` are not read on SLURM.** GPUs come only from
  `accelerators`; memory only from `resources.memory`. gbserver warns when `num_gpus_per_node > 0`
  has no `accelerators`. The step then runs without a GPU and can still report success.
- **The run step gets every CPU of the allocation.** Every rank of a multi-GPU task can use all
  `--cpus-per-task` cores, not one.

## Example

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

## Common mistakes


- `total_memory_per_node: 512Gi` alone does nothing here, and the job gets the partition's
  default memory. Use `resources.memory: "512"`.
- `num_gpus_per_node: 8` alone gives no GPUs. Use `accelerators: "H100:8"`.
- A redundant `resources.cpus` next to `num_cpus_per_node`, justified by "the step has
  `resources: {}`". It's harmless, but the two can drift, and `resources.cpus` silently wins.
- A slow run blamed on the CPU request without checking `sacct`. Confirm the run step's
  `AllocCPUS` first; if it matches, the cause is elsewhere (data loading, dataloader workers,
  I/O on the shared filesystem).

## Check what a job actually got

Use `sacct`, and read the *step* rows, not just the job row:

```bash
sacct -j <jobid> --format=JobID,JobName,AllocCPUS,ReqMem,AllocTRES%60
```

`batch` is the allocation; `sky-<n>` is your run step and should show the same `AllocCPUS`. Inside
the workload, `python -c 'import os; print(len(os.sched_getaffinity(0)))'` prints the cores the
process may use. That is the number to compare against a slow run, not `nproc` on the login node.

## See also

- [SkyPilot on SLURM](skypilot-slurm.md): `sbatch_options`, partitions (`zone`), multi-node and NCCL
- [How `compute_config` maps to SkyPilot](skypilot.md#how-compute_config-maps-to-skypilot): the same mapping on other clouds
