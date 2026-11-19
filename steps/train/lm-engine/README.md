# The lm-engine family image

`Dockerfile.gpu` here builds the one image that every step under
`steps/train/lm-engine/<step>/<env>/` runs in — `pretrain/skypilot/` and `export/skypilot/`. It
carries [open-lm-engine/lm-engine](https://github.com/open-lm-engine/lm-engine) at the
sha pinned in [`UPSTREAM`](UPSTREAM), with the four patches in `patches/` applied, plus its CUDA
dependencies and a prebuilt flash-attn.

No step builds an image. Each step's `step-template.yaml` takes the ref from its own
`config.image`, which a recipe sets through its `LME_IMAGE` parameter. That is the same
arrangement as `steps/eval/`, where the family images are built one level above the steps that
use them.

```
make -C steps/train/lm-engine image check-image    # build, then smoke-check it
make -C steps/train/lm-engine publish-image        # push (needs a quay login)
make -C steps/train/lm-engine test                 # the pin + patch tests, no build
```

`IMAGE_VERSION` defaults to the repo's git short SHA; pass a release tag
(`IMAGE_VERSION=lme-a9c6e2b8-1`) for anything a recipe will reference. `publish-image` refuses to
replace a tag that already exists, because a tag has to keep pointing at one image — SLURM's image
cache keys its `.sqsh` by tag, so a re-pushed tag never reaches it.

## What is pinned, and why that matters

| Input | Pin |
| --- | --- |
| base image | `ubuntu:22.04@sha256:9abcaffc…` (plain ubuntu — **no** CUDA base, no nvcc) |
| CPython | `3.12.16`, installed by uv |
| uv | `0.13.8`, copied from `ghcr.io/astral-sh/uv` |
| lm-engine | the `UPSTREAM` sha, cloned and checked out |
| its dependencies | lm-engine's **own committed `uv.lock`**, via `uv sync --locked --extra cuda` |
| flash-attn | an upstream release wheel, by URL **and sha256** |

CUDA comes entirely from the torch wheel. Do not switch to a CUDA base image to "fix" something:
the flash-attn wheel is built against this torch, and nothing in the image compiles CUDA.

Dependencies are resolved from lm-engine's committed `uv.lock` rather than re-resolved, which is
why the clone is a clone and not upstream's `COPY .` (upstream's `.dockerignore` drops `uv.lock`).
`--locked` fails the build rather than silently updating the lock.

### The flash-attn wheel is the most fragile pin

Its file name encodes the ABI it was compiled for: `cu12torch2.10cxx11abiTRUE-cp312-cp312`. It is
rigidly tied to **torch 2.10** (from the `cuda` extra in lm-engine's `pyproject.toml`) and
**CPython 3.12**. Bump either and this wheel stops being importable — usually as an undefined C++
symbol at `import flash_attn`, not as an install failure. If a pin bump moves torch or Python, find
the matching wheel in the [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases)
first; if there is none, stop rather than compiling one (repo policy, PR 565: pinned deps are built
from pinned sources in-repo and published to `quay.io/granitebuild`, and personal prebuilt wheels
are rejected — an upstream release artifact verified by sha256 is what makes this one acceptable).

### Why `.git` is kept

Cloning *is* how the sha gets pinned, so `/app/src/.git` comes along for free — and it is worth
keeping rather than stripping:

* The image describes itself. `check-image.sh` prints `git -C /app/src rev-parse HEAD`, so the
  running container can always be traced back to a commit, and so can a container someone pulled
  without the tag's provenance.
* `train.py` records a `commit_id` in its checkpoint metadata through GitPython. Pretraining does
  not hit that line (it is on the non-`full_finetuning` path, and as written it would raise anyway
  — it passes `Path(__file__).parents[1]`, which is the `lm_engine` package, not the repo root).
  But upstream's own Dockerfile `COPY .`s the tree under a `.dockerignore` that drops `.git`, which
  is why the reference image had to carry a guard around that read. Keeping `.git` means we need no
  such guard, now or on a tuning path later (`UPSTREAM`, "not carried").

### Why lm-engine is not pip-installed

`uv sync` runs with `--no-install-project`: only the dependencies are installed, and `lm_engine`
itself resolves through `PYTHONPATH=/app/src`. The export step imports
`tools.model_conversion.export_to_huggingface`, and `tools/` is **not** part of the wheel
(`[tool.hatch.build.targets.wheel] packages = ["lm_engine"]`) — it exists only in the source tree.
`scripts/check-image.sh` imports it for exactly that reason.

## Bumping the pin

1. Edit `UPSTREAM`: the new `sha`, and its notes.
2. Re-check the patches against it — `make check-patches`, or regenerate any that no longer apply
   (`git format-patch`-style single-file patches; keep the `0001..000N` order and keep every patch
   explained in `UPSTREAM`).
3. Edit `Dockerfile.gpu`'s `ARG LME_SHA` to the same sha. It cannot drift silently: the build
   compares it against the `UPSTREAM` it copies in, and `tests/test_image_pins.py` checks it
   offline.
4. Check whether the new sha moves `torch` or `requires-python` in `pyproject.toml`. If it does,
   see the flash-attn note above before anything else.
5. `make image check-image`, then `make publish-image IMAGE_VERSION=<new tag>`.
6. Point the recipes' `LME_IMAGE` at the new tag.

`make test` runs both test files: the offline pin assertions, and the patch-application test (which
clones lm-engine and skips without network). `check-image.sh` runs *inside* the built image and is
CPU-safe, so it passes on a GPU-less build host: it imports the patched dataloader, `GPTBaseConfig`
and `tools.model_conversion`, prints the checked-out sha and the torch / flash-attn / transformers
versions, and checks that `gcc`, `rsync` and `ibv_devinfo` exist. It does not *run* `ibv_devinfo` —
the image's job is to carry the RDMA userspace, not to have a NIC.

## Divergence from the reference image

This image replaces `us.icr.io/cil15-shared-registry/lm-engine:0.1.0-cuda-fa2cc`, which trained the
existing models. It is deliberately **not** bit-equivalent to it:

* **Patches.** `0001` and `0002` reproduce the reference image's two source changes byte for byte.
  `0003` and `0004` do not exist there — gbansible `sed`-patched those files at job launch instead.
  Baking them is what lets this recipe run without gbansible.
* **Dependency versions.** Using lm-engine's committed `uv.lock` differs from the reference image's
  resolved environment on 70 of 176 packages, because that build re-resolved from scratch (its
  `uv.lock` had been `.dockerignore`d away). `torch` (2.10.0+cu128) and `transformers` (4.57.3) are
  identical either way. The notable drift is **pyarrow 24.0.0 (the committed lock) vs 26.0.1 (the
  reference image)**, and pyarrow is on the live data path — the 350M dataset is `hf_parquet`.
* **System packages.** We add `rdma-core`, `ibverbs-providers` and `ibverbs-utils`; the reference
  image has no RDMA packages at all, yet ran NCCL over IB on Vela, so the host supplied them.
  Carrying them is additive, and untested against that precedent. (On jammy the `ibv_devinfo` /
  `ibv_devices` diagnostics are in `ibverbs-utils`, **not** in `rdma-core` — without it the image
  has the `libibverbs.so.1` NCCL needs but no way to ask what rails the host exposes.) We drop
  upstream's google-cloud-cli + gcsfuse layer (~1.05 GB, unused). We also omit
  `python3-dev`, which the reference image installed: Triton's JIT compiles against the headers of
  the interpreter it runs under, and that is uv's standalone CPython 3.12, which ships its own —
  the distro's python3.10 headers were redundant there.
