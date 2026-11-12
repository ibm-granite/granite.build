# Recipes

Ready-to-run `build.yaml` files, grouped by model or workflow. Each group's
README lists its recipes and any prerequisites.

## Placeholders

`<shared-fs>` in a recipe's README or in `parameters.yaml` comments stands for a
directory on the cluster's shared filesystem, one that every compute node and
container sees at the same path.

- **Not one root.** Different `<shared-fs>` paths may be on different
  shared-filesystem roots, even in the same file. Two paths that both start
  with `<shared-fs>/` are not necessarily siblings. The matching
  `parameters.yaml` default gives the real path.
- **Not `shared_workdir`.** The environment's `shared_workdir` holds gbserver's
  per-run directories (`${shared_workdir}/builds/<build_id>/runs/<targetrun_id>/`).
  The `<shared-fs>` paths hold models, data and outputs you manage yourself.
- **Must be visible in the container.** On LSF, SkyPilot mounts the
  shared-filesystem roots into every container; see
  [`file_mounts` inside enroot containers](../docs/environments/skypilot-lsf.md#file_mounts-inside-enroot-containers).
  On SLURM, a container sees only the directories the backend mounts into it,
  so a `<shared-fs>` path must be under one of them; see
  [`workdir` (containerized steps)](../docs/environments/skypilot-slurm.md#workdir-containerized-steps).
