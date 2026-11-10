# LSF (`Lsf`) environment — retired

The native `Lsf` environment is retired, and these docs no longer cover it. Builds that target an
LSF environment should move to **SkyPilot on SLURM**:

- Set `environment_uri` to a SkyPilot SLURM environment in your space, e.g.
  `space://environments/skypilot/slurm/<cluster>`.
- Move `config.lsf.*` settings to their SkyPilot equivalents: CPUs, GPUs, memory and nodes as in
  [Sizing a SkyPilot step](skypilot-sizing.md#slurm); wall-clock limits and other scheduler
  directives via [`sbatch_options`](skypilot-slurm.md#sbatch_options--slurm-sbatch-directives);
  secrets under `config.skypilot.secrets`.

See [SkyPilot on SLURM](skypilot-slurm.md) for environment setup.
