# SPDX-License-Identifier: Apache-2.0

"""Pin hub-resolved kernels to revisions compatible with this image's transformers.

WHY THIS EXISTS (measured 2026-11-20, BlueVela, image 91cb624c, teacher-divergence
build test). transformers 5.8.0's ``_HUB_KERNEL_MAPPING`` resolves ``mamba-ssm`` with
``version=1``, the kernels-community repo's v1 branch. Every revision up to and
including v0.0.3 ships a build whose ``mamba_ssm/utils/generation.py`` hard-imports
``GreedySearchDecoderOnlyOutput`` from ``transformers.generation`` -- a symbol
transformers 5.x renamed to ``GenerateDecoderOnlyOutput`` -- so every fresh load of
a hybrid (granitemoehybrid) model dies with an ImportError after the model load has
started. The teacher-divergence build test hit it directly; the e2e smokes did not,
because they used dense students that never reach the mamba path.

``v0.0.4`` (8ce5a492, "Fix for Falcon"; pinned by that commit's SHA, since a tag
can move) restructured the build: the top level
is now the actual module (verified in the GPU image -- ``_import_from_path``
loads it, ``MambaLMHeadModel`` is available, no transformers.generation import),
and ``mamba_ssm/`` inside the build is a compatibility shim. This is the newest
revision whose cu128 build loads cleanly under transformers 5.8.0. It also ships
``metadata.json`` at the variant root, which v0.0.3 lacked.

causal-conv1d is pinned for a different reason: revision AGREEMENT, not a broken
build (measured 2026-11-20, image 939685f2, student-sft build test). transformers
resolves it with ``version=1`` -- the repo's v1 branch (6e9a5827) -- while the
preflight's fallback for unpinned repos is ``main`` (f2651e77). Under
``HF_HUB_OFFLINE=1`` the trainer cannot resolve an uncached ref, so sft.py died
with ``OfflineModeIsEnabled`` on causal-conv1d/refs seconds after the preflight
had downloaded the repo -- at the wrong revision. The pin points both sides at
the immutable v1-branch tip (verified in the GPU image: ``causal_conv1d_fn``
available, torch28-cu128 build), so the offline lookup is an exact snapshot hit.

This is a pin, not a fix. When kernels-community/mamba-ssm ships a build that
imports cleanly under transformers 5.x on the default branch, or the image moves
past the rename, delete that entry here; delete causal-conv1d's entry when the
preflight and transformers agree on a revision by construction. Either way,
remove the call sites and the matching probe in gold-kernels.sh (which reads
``pinned_revision`` from this module -- keep the two in sync by construction,
not by discipline).
"""

from __future__ import annotations

# repo_id -> revision. Unlisted repos resolve their default (transformers passes
# version=1, the kernels package resolves the repo's v1 branch) -- which the
# preflight cannot satisfy offline, so a hybrid model's kernels belong here.
PINNED_KERNEL_REVISIONS: dict[str, str] = {
    # The v0.0.4 tag's commit. A tag can be moved; a commit SHA cannot.
    "kernels-community/mamba-ssm": "8ce5a4921b3a09bdae509d7143b88eab9e4f21f3",
    "kernels-community/causal-conv1d": "6e9a58272220a363dc50e59cba737abcedd7b59f",
}

# The transformers mapping keys whose repos are pinned above. "falcon_mamba-ssm"
# shares mamba-ssm's repo and the same broken default; pinning it costs nothing
# and keeps a Falcon run from rediscovering this.
_PINNED_KERNEL_NAMES = ("mamba-ssm", "falcon_mamba-ssm", "causal-conv1d")


def pinned_revision(repo_id: str) -> str:
    """The revision to fetch ``repo_id`` at: the pin, or "main" for unpinned repos.

    gold-kernels.sh's cache probe and prefetch call this so the out-of-band download
    lands the same revision the trainer will resolve here.
    """
    return PINNED_KERNEL_REVISIONS.get(repo_id, "main")


def pin_hub_kernels() -> None:
    """Rewrite transformers' hub-kernel mapping onto the pinned revisions.

    Idempotent, offline (it only mutates the in-process dict), and a no-op when
    transformers is unavailable or no longer maps the kernel -- so calling it from
    an entrypoint that might run without hybrid models is safe.

    ``version`` is popped, not just overridden: ``get_kernel`` refuses a call that
    passes both ``revision`` and ``version``, and the default entry carries
    ``version=1``.
    """
    try:
        from transformers.integrations.hub_kernels import _HUB_KERNEL_MAPPING
    except ImportError:
        return
    for name in _PINNED_KERNEL_NAMES:
        entry = _HUB_KERNEL_MAPPING.get(name)
        if entry is None:
            continue
        repo_id = entry.get("repo_id", "")
        if repo_id not in PINNED_KERNEL_REVISIONS:
            continue
        entry.pop("version", None)
        # setdefault, not assignment: a future transformers that pins its own
        # revision has done the work, and its pin deserves the benefit of the doubt.
        entry.setdefault("revision", PINNED_KERNEL_REVISIONS[repo_id])


def prepare_hub_kernels() -> None:
    """Everything a hybrid-model entrypoint needs before its first model load.

    Two stages, each a no-op where its dependencies are absent:

    1. ``pin_hub_kernels`` -- rewrite transformers' hub-kernel mapping onto the
       pinned revisions (see the module docstring for why each repo is pinned).
    2. Register the hub causal-conv1d build under its legacy import name and
       shim the legacy ``causal_conv1d_cuda`` object onto it: the modeling code
       asserts on that object (``causal_conv1d_cuda is not available. Please
       install causal-conv1d.``), but the hub build only exposes the new
       cpp_functions API and the ``kernels`` package registers modules with
       hash-based names, so ``causal_conv1d`` is never directly importable.

    Extracted from gold.py's preamble -- which was the only entrypoint carrying
    it while sft.py, run_divergence.py and distill_probe.py died on the
    legacy-name assert (measured 2026-11-20, image 2990684b, student-sft build
    test). Requires transformers 5.x hub_kernels; skips on older versions.

    NO CACHE PRE-LOAD, on purpose. An earlier version also loaded both kernels from
    the local cache straight into transformers' ``_KERNEL_MODULE_MAPPING`` before
    stage 2. That never ran on any validated image (it read ``refs/<pin>``, which
    huggingface_hub does not write for a commit pin), and the first image where it
    did run (173af080) broke every hybrid load: the mamba-ssm build imports
    ``causal_conv1d`` at module load, so loading it before stage 2 registers that
    name binds ``causal_conv1d_cuda = None`` for good (student-sft build test,
    BlueVela, 2026-11-20: ``AssertionError: causal_conv1d_cuda is not available``).
    The kernels now load only through ``lazy_load_kernel``, which resolves the pinned
    revision from the warm cache that gold-kernels.sh's preflight fills. Anything
    that loads mamba-ssm must come after stage 2.
    """
    import sys

    pin_hub_kernels()

    try:
        from transformers.integrations.hub_kernels import (
            lazy_load_kernel as _lazy_load_kernel,
        )
    except (ImportError, ModuleNotFoundError):
        _lazy_load_kernel = None

    # Stage 2: causal-conv1d first and alone -- see the docstring's NO CACHE PRE-LOAD.
    if _lazy_load_kernel is not None:
        _cc1d = _lazy_load_kernel("causal-conv1d")
        if _cc1d is not None and "causal_conv1d" not in sys.modules:
            sys.modules["causal_conv1d"] = _cc1d

            _cc1d_iface = getattr(_cc1d, "causal_conv1d_interface", None)
            if _cc1d_iface is not None:
                _fwd = getattr(_cc1d_iface, "causal_conv1d_fwd_function", None)
                _bwd = getattr(_cc1d_iface, "causal_conv1d_bwd_function", None)

                if _fwd is not None and _bwd is not None:

                    class _CausalConv1dCudaShim:
                        causal_conv1d_fwd = staticmethod(_fwd)
                        causal_conv1d_bwd = staticmethod(_bwd)

                    _cc1d_iface.causal_conv1d_cuda = _CausalConv1dCudaShim()

                sys.modules["causal_conv1d.causal_conv1d_interface"] = _cc1d_iface
