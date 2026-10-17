# gb-cuda-devel-base

A shared, **GPU-only** base image for granite.build steps that need real CUDA
JIT compilation at runtime: `torch.compile`/Inductor, raw Triton kernels, or
FlashInfer's JIT-compiled sampler. Built `FROM nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu22.04`
(NVIDIA's NGC CUDA **devel** tier), so `nvcc`, headers, and static libs are
present — unlike a plain `python:slim`/`ubuntu` base, where pip's
`nvidia-*-cu12` wheels provide only CUDA *runtime* libraries, with no
compiler.

**Do not `FROM` this for a CPU-only step.** It is multi-GB and brings nothing
a CPU workload needs.

## Why this exists

We hit this exact gap running granite-evals' benchmark suite on BlueVela:
its image (`python:3.12-slim-bookworm`) has no CUDA toolkit, so vLLM's eager
sampler (a raw Triton kernel) and FlashInfer's JIT path both failed with
"libcuda.so cannot found!" / "Could not find nvcc". We patched around it
in that image's own code (see below); this image exists so a step author
doesn't have to.

## How to use it

In your step's own Dockerfile:

```dockerfile
FROM us.icr.io/cil15-shared-registry/gb-cuda-devel-base:<tag>
# ... your step's own uv sync / pip install, source, entrypoint ...
```

Your step's own `make image` / `make publish-image` (via its `common.mk`-based
Makefile) are otherwise unaffected — only the `FROM` line changes.

## Two things this image does NOT solve by itself

### 1. `pip`/`uv` can shadow the toolkit

If your step's own dependency install resolves `nvidia-*-cu12` runtime
wheels (most torch/vllm dependency trees will), those pip-installed runtime
libraries can end up ahead of this image's `CUDA_HOME`/`PATH`/`LD_LIBRARY_PATH`
for anything invoked via `python -m` rather than bare `nvcc` — silently
defeating the point of using this base. (`steps/bfcl-eval/skypilot/Dockerfile`
documents hitting exactly this with an NGC base: its torch/triton/vllm got
uninstalled and replaced by pip's own wheels regardless of the base image.)

This image sets `CUDA_HOME`/`PATH`/`LD_LIBRARY_PATH` via `ENV` so they're
visible to every later layer, but it cannot guarantee a downstream
`pip install` won't shadow them at the package level. After your step's own
`uv sync`, verify:

```bash
which nvcc                                              # /usr/local/cuda/bin/nvcc
python -c "import torch; print(torch.version.cuda)"     # the expected version
```

If your dependency resolver insists on its own conflicting CUDA runtime
wheels, this base may not actually help your step — say so in your own
Dockerfile's comments (same honesty bfcl-eval's Dockerfile already models).

### 2. The pyxis/enroot `ldconfig` gap on BlueVela SLURM is NOT fixed here

On BlueVela's pyxis/enroot SLURM container runtime, the NVIDIA driver
libraries (`libcuda.so.1`) are injected into the container **at launch time**,
after the image is already built — no image can bake in a cache entry for a
library that doesn't exist yet at build time. Triton's own driver discovery
(`libcuda_dirs()`) checks `ldconfig -p` first, falling back to
`LD_LIBRARY_PATH` only if that's empty — so a stale/empty `/etc/ld.so.cache`
at container launch breaks Triton driver discovery regardless of toolkit
presence.

This has two parts, split across two different places on purpose:

- **Driver injection itself** is a SLURM-environment-level concern, already
  handled centrally — not something each step repeats. gbserver generates the
  `sbatch_options.export` directive (`NVIDIA_VISIBLE_DEVICES=all,NVIDIA_DRIVER_CAPABILITIES=all,...`)
  automatically for any SLURM step that requests both an accelerator and a
  container image (`Skypilot._pyxis_gpu_sbatch_directives` in
  `src/gbserver/environment/skypilot.py`; see issue #510) — without this,
  pyxis's own NVIDIA hook sees it unset, defaults to `"void"`, and skips driver
  injection entirely (`libcuda.so.1` is then simply absent from the container,
  no amount of `ldconfig` fixes that). A SLURM `environment.yaml` can opt out
  via `slurm_gpu_sbatch_directives: false` if a cluster's own Pyxis/driver
  setup ever needs something different from BlueVela's. Not confirmed whether
  LSF's enroot/pyxis setup needs the same fix at all (LSF has its own,
  separate container bootstrap).
- **The `ldconfig` cache refresh**, once the driver *is* injected, stays a
  **runtime, per-consumer concern** your step's own launch script must do —
  and this is a hard platform constraint, not something this image (or any
  image) can absorb: pyxis/enroot on SLURM execs its own bootstrap command
  directly inside the unpacked container rootfs, bypassing the image's own
  `ENTRYPOINT`/`CMD` entirely, so a Dockerfile-level fix is silently inert on
  this backend. Add this one line as the first thing your step's `run:` script
  does, before launching anything that touches Triton's driver discovery:

  ```bash
  ldconfig
  ```

  (as root; see granite-evals' `serving.py` `_refresh_libcuda_cache()` for the
  pattern already applied there, and `steps/eval/*/skypilot/step-template.yaml`'s
  `run:` block for the in-repo convention once a GPU step adopts this image
  directly). Not confirmed whether LSF's enroot/pyxis setup has the same gap —
  treat as unverified, not assumed-same.

Both are specific to **this cluster's pyxis/enroot setup**, not something
this image (or any image) can fix once for every deployment target — a
different SLURM cluster or container runtime may not need either.

### Keep cluster-specific env out of the image

Resist baking cluster-specific values (`NCCL_*`, `TORCH_CUDA_ARCH_LIST`, or
anything else tied to one cluster's topology/hardware) into this image's
`/etc/environment` or `ENV` layer, even though some other teams' custom-image
pipelines do exactly that as a deliberate portability trade-off. This image
and any `.sqsh` built from it are meant to run unmodified on any cluster: keep
cluster-specific values as **runtime** `sbatch_options.export` /
`--container-env`, set by the environment or by gbserver (as driver injection
already is, above), not baked into the image at build time.

### Note: enroot's image cache is per-node, confirmed

Each compute node caches imported images under its own local NVMe
(`/opt/nvme/enroot/enroot-cache/user-$(id -u)`, confirmed as node-local XFS
on a local block device, not a shared/network filesystem). The first job that
lands on a given node after this image changes pays a real cold-import cost
(downloading layers, converting whiteouts, building the squashfs — tens of
minutes for a multi-GB CUDA devel image); a job that happens to land on a
node that already imported this exact tag is fast. This is expected
behavior, not a bug — don't mistake a slow cold-import for a hang.

## Version

Pinned to CUDA 12.9.1 to match vLLM's current `+cu129` prebuilt wheels (the
version granite-evals uses today). Bump deliberately — and validate
end-to-end — before moving to a newer NGC devel tag; do not bump "because
it's newer" without re-testing the JIT-compilation paths this image exists
for.
