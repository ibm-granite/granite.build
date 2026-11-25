# SPDX-License-Identifier: Apache-2.0

"""Pin and pre-register the hub kernels a hybrid (granitemoehybrid) model needs.

HOW TRANSFORMERS FINDS THE MAMBA2 KERNELS (5.16 and later; this image pins 5.20.2).
The granitemoehybrid modeling module decorates its mixer functions with
``use_kernel_func_from_hub_with_fallback(<func>, <package>)``. When the module is
IMPORTED, each decorator tries ``importlib.import_module("causal_conv1d")`` /
``importlib.import_module("mamba_ssm")`` and binds the function it finds there;
if the package is absent it binds the reference PyTorch implementation, which is
much slower and raises ``ValueError`` on packed sequences (``seq_idx`` /
``cu_seqlens``). Hub kernels are only used without that package when the model is
loaded with ``use_kernels=True`` (kernelize), which also swaps RMSNorm, rotary
and other layers and so changes numerics.

transformers 5.8.0 worked differently: the modeling code called
``lazy_load_kernel("mamba-ssm")`` / ``("causal-conv1d")`` through
``_HUB_KERNEL_MAPPING``, and this module used to rewrite that mapping onto pinned
revisions. 5.16+ has no such entries, so that rewrite is gone.

WHAT THIS MODULE DOES NOW. ``prepare_hub_kernels`` loads both kernels from the
Hub with ``kernels.get_kernel`` at the pinned revisions and registers them in
``sys.modules`` as ``causal_conv1d`` and ``mamba_ssm`` BEFORE the modeling module
is imported, so transformers' decorators bind the hub kernels as if the packages
were installed. Only the mixer functions change; nothing else is kernelized. It
refuses to run after the modeling module is imported, because the decorators
have already bound the PyTorch fallback by then and a late registration would be
silently ignored.

THE REPOSITORIES (#763). Hugging Face is removing the "model" type kernel
repositories (their model cards: "Starting from September 13, 2026"). The pins
below are in the "kernel" type repositories of the same names, which have their
own history, so no SHA carries over. ``kernels`` reads kernel-type repositories
from 0.14 on; this image pins 0.17.3, the release transformers 5.20.2 accepts
(``kernels>=0.17.2,<0.18``).

- ``kernels-community/mamba-ssm`` at ``20b2508a`` (v3 branch, 2026-09-08): the
  oldest v3 commit with a ``torch214-cxx11-cu130-x86_64-linux`` build. v3 is the
  major version transformers 5.20.2's own kernel mapping asks for. The build is
  self-contained (it ships its own ``_causal_conv1d``) and imports transformers
  5.x's ``GenerateDecoderOnlyOutput``; its ``decode()`` still names the removed
  ``GreedySearchDecoderOnlyOutput`` at call time, which nothing here calls (and
  gold.py's generation-output aliases cover).
- ``kernels-community/causal-conv1d`` at ``adc40e41`` (v1 branch, 2026-08-27):
  the oldest v1 commit with a ``torch214-cxx11-cu130-x86_64-linux`` build. Its
  top level exposes ``causal_conv1d_fn`` and ``causal_conv1d_update``, the two
  names transformers' decorators look up.

Pinned by commit SHA, since a branch moves. The torch 2.8 / cu128 image cannot use
these repositories: their torch28 builds still import
``GreedySearchDecoderOnlyOutput`` and were later removed as incompatible with
kernels >= 0.14, and transformers 5.8.0 rejects kernels >= 0.13.

LOAD ORDER. causal-conv1d is registered before mamba-ssm is loaded. The model-type
mamba-ssm builds imported ``causal_conv1d`` at module load, and loading one before
the legacy names existed bound ``causal_conv1d_cuda = None`` for good (student-sft
build test, image 173af080, 2026-11-20). The v3 build no longer imports it, but the
order costs nothing and keeps a future build from rediscovering that.

THE CACHE GATE. gold-kernels.sh probes and prefetches with ``kernel_spec(repo)``,
the same arguments this module loads with, and calls ``make_offline_readable`` on the
result, so an offline (``HF_HUB_OFFLINE=1``) load is an exact cache hit under kernels
0.17.3 and huggingface_hub 2.x (see that function for the two gaps it closes). Repos that are not pinned here resolve the way
transformers itself asks for them (``flash-attn2``: the major version from
``FLASH_ATTN_KERNEL_VERSIONS``), so the prefetch and the model load agree on those
too.
"""

from __future__ import annotations

import sys

# repo_id -> commit. See the module docstring for why each revision.
PINNED_KERNEL_REVISIONS: dict[str, str] = {
    "kernels-community/causal-conv1d": "adc40e4163fbac573adb44ea47b2c8d9080f13bc",
    "kernels-community/mamba-ssm": "20b2508ad12ae40260291539bf45183000451850",
}

# The import names transformers' fallback decorators look for, in load order.
_REGISTERED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("causal_conv1d", "kernels-community/causal-conv1d"),
    ("mamba_ssm", "kernels-community/mamba-ssm"),
)

# The names transformers 5.20.2's granitemoehybrid decorators look up in each package
# (use_kernel_func_from_hub_with_fallback; checked in the dev image: all five bind).
_REQUIRED_NAMES: dict[str, tuple[str, ...]] = {
    "causal_conv1d": ("causal_conv1d_fn", "causal_conv1d_update"),
    "mamba_ssm": (
        "mamba_chunk_scan_combined",
        "mamba_split_conv1d_scan_combined",
        "selective_state_update",
    ),
}

# Imported by AutoModel*.from_pretrained for a granitemoehybrid config. Once it is in
# sys.modules, its decorators have bound whatever was importable at that moment.
_HYBRID_MODELING_MODULE = (
    "transformers.models.granitemoehybrid.modeling_granitemoehybrid"
)


def _transformers_kernel_version(repo_id: str) -> int | None:
    """The major version transformers loads ``repo_id`` at, when it says so itself."""
    try:
        from transformers.modeling_flash_attention_utils import (
            FLASH_ATTN_KERNEL_VERSIONS,
        )
    except ImportError:
        return None
    version = FLASH_ATTN_KERNEL_VERSIONS.get(repo_id)
    return int(version) if version is not None else None


def kernel_spec(repo_id: str) -> dict[str, str | int]:
    """The ``revision=`` / ``version=`` keyword to fetch or load ``repo_id`` with.

    The pin when there is one; otherwise the major version transformers asks for;
    otherwise ``main``. gold-kernels.sh's cache probe and prefetch call this, so the
    out-of-band download lands exactly what the model load will read.
    """
    if repo_id in PINNED_KERNEL_REVISIONS:
        return {"revision": PINNED_KERNEL_REVISIONS[repo_id]}
    version = _transformers_kernel_version(repo_id)
    if version is not None:
        return {"version": version}
    return {"revision": "main"}


def make_offline_readable(repo_id: str, variant_path=None, spec=None) -> None:
    """Make the cached kernel loadable with ``HF_HUB_OFFLINE=1`` under kernels 0.17.3.

    Two gaps between kernels 0.17.3 (the release transformers 5.20.2 accepts) and
    huggingface_hub 2.x, which kernels 0.18 closes itself (it tolerates the partial
    snapshot and records refs):

    1. huggingface_hub 2.x writes the repo's full file listing (``trees/<commit>.json``)
       on download and, offline, refuses a snapshot that lacks any listed file
       (``IncompleteSnapshotError``, ``_snapshot_download._raise_if_incomplete_snapshot``).
       kernels downloads one build variant, so every offline load of a prefetched kernel
       failed. The listings of this repo are removed, and hub then accepts what is
       there ("If the tree listing is not cached we cannot tell, so we do nothing").
    2. A ``version=N`` fetch resolves vN to a commit and downloads by commit, so no
       ``refs/vN`` is written; offline, kernels resolves vN from ``refs/`` only. Given the
       fetched ``variant_path`` and its ``spec``, the ref is written.

    Measured in the #676 dev image (CPU host): a prefetch of the pinned causal-conv1d, then
    an offline get_kernel, raised IncompleteSnapshotError; flash-attn2 at version 4 raised
    "not available in the local cache". Touches only this repo's cache directory.
    """
    import os
    from pathlib import Path

    from huggingface_hub import constants
    from huggingface_hub.file_download import repo_folder_name

    cache = os.environ.get("KERNELS_CACHE") or constants.HF_HUB_CACHE
    repo_dir = Path(cache) / repo_folder_name(repo_id=repo_id, repo_type="kernel")
    trees = repo_dir / "trees"
    if trees.is_dir():
        for listing in trees.glob("*.json"):
            listing.unlink(missing_ok=True)
    if variant_path is not None and spec and "version" in spec:
        commit = Path(
            variant_path
        ).parent.parent.name  # snapshots/<commit>/build/<variant>
        refs = repo_dir / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / f"v{spec['version']}").write_text(commit)


def pinned_revision(repo_id: str) -> str:
    """The pinned commit for ``repo_id``, or "main" for unpinned repos."""
    return PINNED_KERNEL_REVISIONS.get(repo_id, "main")


def _shim_legacy_causal_conv1d_cuda(cc1d) -> None:
    """Expose the legacy ``causal_conv1d_cuda`` object on the hub build's interface.

    Older modeling code (and the model-type mamba-ssm builds) asserted on
    ``causal_conv1d_interface.causal_conv1d_cuda``; the hub build only has the
    cpp_functions API. Harmless when nothing reads it.
    """
    iface = getattr(cc1d, "causal_conv1d_interface", None)
    if iface is None:
        return
    fwd = getattr(iface, "causal_conv1d_fwd_function", None)
    bwd = getattr(iface, "causal_conv1d_bwd_function", None)
    if (
        fwd is not None
        and bwd is not None
        and getattr(iface, "causal_conv1d_cuda", None) is None
    ):

        class _CausalConv1dCudaShim:
            causal_conv1d_fwd = staticmethod(fwd)
            causal_conv1d_bwd = staticmethod(bwd)

        iface.causal_conv1d_cuda = _CausalConv1dCudaShim()
    sys.modules.setdefault("causal_conv1d.causal_conv1d_interface", iface)


def prepare_hub_kernels(*, force: bool = False) -> None:
    """Register the pinned hub kernels as ``causal_conv1d`` and ``mamba_ssm``.

    Call it once, at the top of every hybrid-model entrypoint, before the first
    model load. A package that is already importable under that name (a real pip
    install, or an earlier call) is left alone. A no-op when ``kernels`` is not
    installed or there is no CUDA device (the builds are CUDA kernels; a CPU process
    imports these entrypoints only for tests and preflights), unless ``force`` asks
    for a load anyway, as the kernel probe does on a CPU host.

    Raises ``RuntimeError`` if the granitemoehybrid modeling module was imported
    first: its decorators would already have bound the PyTorch fallback.
    """
    import importlib.util

    def _importable(package):
        if package in sys.modules:
            return True
        try:
            return importlib.util.find_spec(package) is not None
        except (ImportError, ValueError):
            return False

    missing = [
        (pkg, repo) for pkg, repo in _REGISTERED_PACKAGES if not _importable(pkg)
    ]
    if not missing:
        return  # already registered (an earlier call), or real pip packages installed

    try:
        from kernels import get_kernel
    except ImportError:
        return

    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if not (has_cuda or force):
        return
    check_arch = has_cuda

    if _HYBRID_MODELING_MODULE in sys.modules:
        raise RuntimeError(
            f"{_HYBRID_MODELING_MODULE} was imported before prepare_hub_kernels(): its "
            "mamba/causal-conv1d decorators have already bound the PyTorch fallback, so "
            "registering the hub kernels now would change nothing. Call "
            "prepare_hub_kernels() before anything loads a hybrid model."
        )

    try:
        from huggingface_hub import constants

        offline = bool(constants.HF_HUB_OFFLINE)
    except ImportError:
        offline = False

    for package, repo_id in missing:
        if offline:
            make_offline_readable(repo_id)
        module = get_kernel(repo_id, check_arch=check_arch, **kernel_spec(repo_id))
        # transformers' decorators fall back to PyTorch, with only a warning, for any name
        # they cannot resolve: a build that lacks one must fail here instead.
        absent = [n for n in _REQUIRED_NAMES[package] if not hasattr(module, n)]
        if absent:
            raise RuntimeError(
                f"{repo_id} at {kernel_spec(repo_id)} has no {absent}: transformers would "
                "silently use its PyTorch path for them."
            )
        sys.modules[package] = module
        if package == "causal_conv1d":
            _shim_legacy_causal_conv1d_cuda(module)
