# `file_mounts` on sudo-less SLURM (symlink wrap)

**Status:** fixed by [`sky_patches.py`](../../src/gbserver/utils/sky_patches.py) until the fix
lands in SkyPilot. Covered by `test/integration/ibm/buildrunner/skypilot/slurm_bluevela/test_dpk_pii.py`
(green on BlueVela SLURM 2026-09-28).

**Affects:** a **bare (non-containerized)** step with an absolute or relative `file_mounts`
destination on a `skypilot`/`slurm` environment that sets a SLURM `workdir` and whose SSH user
has no passwordless sudo (e.g. `dpk` and `byoc`, both `src: src`, on BlueVela).

## Cause

1. gbserver remaps a relative `file_mounts` destination onto the shared per-run workdir
   (`_remap_relative_dest`), an absolute `/proj/...` path on BlueVela.
2. SkyPilot's `_execute_file_mounts` symlink-wraps every absolute destination not under
   `~/` or `/tmp/`, which runs `sudo mkdir -p` on the node.
3. On SLURM every runner command is `srun`'d to the compute node. Bare, it runs as the login
   user, which has no sudo there:

   ```
   sudo: a password is required
   Failed to create symlinks. The target destination may already exist.
   ```

   In a container it runs as root with `sudo` aliased away, so containerized steps pass.

## Fix

Destinations under the configured SLURM `workdir` (SkyPilot's `sky_base_dir`) are left
un-wrapped. That is safe in both modes: SkyPilot creates the cluster home there, so the login
user can write it, and it is identity-mounted into the pyxis container, so the rsync lands at the
path the job reads. A cluster without an explicit `workdir` keeps the wrap.

Trade-off: un-wrapped roots skip `make_safe_symlink_command`'s clobber guard, as `~/` and
`/tmp/` already do; a mount onto an existing directory rsyncs into it. `$GB_BUILD_WORKDIR` is
per-run, so this is low risk.
