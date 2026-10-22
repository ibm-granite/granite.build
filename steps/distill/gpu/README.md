# distill/gpu — shared GPU image for the distillation steps

Shared, source-controlled GPU image for the distillation steps that load
models and/or train: `gold`, `sft`, `logit-precompute`, `eval`,
`vllm-server`, `gen-smoke`, `probe`. Replaces the opaque
`kd-sandbox-distill:0.1.0-uv` image (no accessible Dockerfile/build source
anywhere) with one built here, from a public base.

See `Dockerfile`'s own header for why this is ONE image rather than one per
step — checked directly against the real `uv.lock` files in
`github.com/laminair/gb-steps-distillation`, not assumed.

## Building and publishing

```shell
cd steps/distill   # build context is the PARENT of this directory --
                    # the Dockerfile COPYs the shared src/gb_steps_post_training/
                    # package from there, so `.` must be steps/distill/, not
                    # steps/distill/gpu/ itself.
podman build --platform linux/amd64 -f gpu/Dockerfile . -t <registry>/gb-step-distillation-gpu:<tag>
podman push <registry>/gb-step-distillation-gpu:<tag>
```

## Two things this image does NOT solve by itself

### 1. `pip`/`uv` can shadow the toolkit

If a step's own dependency resolution pulls in `nvidia-*-cu12` runtime
wheels (it does here, via torch/vllm), those pip-installed runtime
libraries can end up ahead of this image's `CUDA_HOME`/`PATH`/
`LD_LIBRARY_PATH` for anything invoked via `python -m` rather than bare
`nvcc` — silently defeating the point of the devel toolkit for anything
that needs it directly (e.g. `deepspeed`'s sdist build, which this image's
own `uv sync --locked` already exercises at build time). After `uv sync`,
verify:

```bash
which nvcc                                                        # /usr/local/cuda/bin/nvcc
/stage/.venv/bin/python -c "import torch; print(torch.version.cuda)"  # the expected version
```

### 2. The pyxis/enroot `ldconfig` gap on BlueVela SLURM is NOT fixed here

On BlueVela's pyxis/enroot SLURM container runtime, the NVIDIA driver
libraries (`libcuda.so.1`) are injected into the container **at launch
time**, after the image is already built — no image can bake in a cache
entry for a library that doesn't exist yet at build time. Triton's own
driver discovery (`libcuda_dirs()`) checks `ldconfig -p` first, falling
back to `LD_LIBRARY_PATH` only if that's empty — so a stale/empty
`/etc/ld.so.cache` at container launch breaks Triton driver discovery
regardless of toolkit presence.

This has two parts, split across two different places on purpose:

- **Driver injection itself** is a SLURM-environment-level concern, already
  handled centrally — not something each step repeats. gbserver generates
  the `sbatch_options.export` directive
  (`NVIDIA_VISIBLE_DEVICES=all,NVIDIA_DRIVER_CAPABILITIES=all,...`)
  automatically for any SLURM step that requests both an accelerator and a
  container image (`Skypilot._pyxis_gpu_sbatch_directives` in
  `src/gbserver/environment/skypilot.py`; see issue #510) — without this,
  pyxis's own NVIDIA hook sees it unset, defaults to `"void"`, and skips
  driver injection entirely (`libcuda.so.1` is then simply absent from the
  container, no amount of `ldconfig` fixes that). A SLURM `environment.yaml`
  can opt out via `slurm_gpu_sbatch_directives: false` if a cluster's own
  Pyxis/driver setup ever needs something different from BlueVela's. Not
  confirmed whether LSF's enroot/pyxis setup needs the same fix at all (LSF
  has its own, separate container bootstrap).
- **The `ldconfig` cache refresh**, once the driver *is* injected, stays a
  **runtime, per-consumer concern** a step's own launch script must do —
  and this is a hard platform constraint, not something this image (or any
  image) can absorb: pyxis/enroot on SLURM execs its own bootstrap command
  directly inside the unpacked container rootfs, bypassing the image's own
  `ENTRYPOINT`/`CMD` entirely, so a Dockerfile-level fix is silently inert
  on this backend. Add this one line as the first thing a step's `run:`
  script does, before launching anything that touches Triton's driver
  discovery:

  ```bash
  ldconfig
  ```

  (as root; see `gold`'s own `step-template.yaml` for the in-repo
  convention of where this lives in the vLLM-serving branch specifically).
  Not confirmed whether LSF's enroot/pyxis setup has the same gap — treat
  as unverified, not assumed-same.

Both are specific to **this cluster's pyxis/enroot setup**, not something
this image (or any image) can fix once for every deployment target — a
different SLURM cluster or container runtime may not need either.

### Note: enroot's image cache is per-node, confirmed

Each compute node caches imported images under its own local NVMe, not a
shared/network filesystem. The first job that lands on a given node after
this image changes pays a real cold-import cost (downloading layers,
converting whiteouts, building the squashfs — potentially tens of minutes
for a multi-GB image like this one); a job that happens to land on a node
that already imported this exact tag is fast. This is expected behavior,
not a bug — don't mistake a slow cold-import for a hang. (A real SLURM
smoke run hit an OOM during exactly this import step with an earlier,
per-step image design — see git history on `steps/distill/gold/` for that
investigation. The fix there was reducing the number of distinct images,
not this cache behavior itself, which is unrelated to image count.)
