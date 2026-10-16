# images/

Shared base Docker images for granite.build steps. Each subdirectory is a
self-contained image: a `Dockerfile`, a `Makefile` (`make image` /
`make publish-image`), and a `README.md` with usage and caveats.

These are **not** steps (no `step.yaml`, no `space://` URI) — a step's own
Dockerfile builds `FROM` one of these, rather than referencing it as a
granite.build step or environment.

## Images

- [`cuda-devel-base/`](cuda-devel-base/README.md) — CUDA 12.9.1 devel
  toolkit base for **GPU-only** steps that need runtime CUDA JIT compilation
  (torch.compile/Inductor, Triton kernels, FlashInfer JIT).
