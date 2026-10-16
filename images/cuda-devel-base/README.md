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

This stays a **runtime, per-consumer concern**: your step's own launch script
must run `ldconfig` as root after the container starts, before launching
anything that touches Triton's driver discovery. See granite-evals'
`serving.py` (`_refresh_libcuda_cache()`) for the pattern already applied
there. Not confirmed whether LSF's enroot/pyxis setup has the same gap —
treat as unverified, not assumed-same.

## Version

Pinned to CUDA 12.9.1 to match vLLM's current `+cu129` prebuilt wheels (the
version granite-evals uses today). Bump deliberately — and validate
end-to-end — before moving to a newer NGC devel tag; do not bump "because
it's newer" without re-testing the JIT-compilation paths this image exists
for.
