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


# The kernels the preamble pre-loads from the local cache: (transformers mapping key, repo).
_PRELOAD_KERNELS = (
    ("causal-conv1d", "kernels-community/causal-conv1d"),
    ("mamba-ssm", "kernels-community/mamba-ssm"),
)


def _validated_import(import_from_path, package_name, variant_path):
    """``import_from_path``, after parsing the variant's metadata.json when it has one.

    A structurally broken metadata.json (a doubled-JSON write from a parallel-download
    race) then aborts with a JSONDecodeError here, instead of surfacing later as a
    confusing ``ImportError: FlashAttention3 has been toggled on``. v0.0.3-era builds
    ship no metadata.json, so its absence is not an error.
    """
    import json
    from pathlib import Path

    meta_path = Path(variant_path) / "metadata.json"
    if meta_path.exists():
        json.loads(meta_path.read_text())
    return import_from_path(package_name, variant_path)


def preload_cached_kernels(
    *,
    install_kernel,
    import_from_path,
    hub_mapping,
    module_mapping,
    kernels=_PRELOAD_KERNELS,
    log=print,
) -> dict:
    """Load each kernel from the local HF cache at its PINNED revision, if it is there.

    Putting the module into transformers' ``_KERNEL_MODULE_MAPPING`` makes
    ``lazy_load_kernel`` short-circuit (transformers/integrations/hub_kernels.py:368-369)
    without ``HfApi().list_repo_refs`` (needed for ``version=``-style entries) or a
    ``snapshot_download`` that races across ranks on NFS. A cache read with
    ``local_files_only=True`` only follows existing symlinks, so it is rank-safe.

    The revision goes to ``install_kernel`` as pinned. kernels 0.12.3 hands it straight
    to ``snapshot_download(local_files_only=True)``, which resolves a tag through
    ``refs/<tag>`` and a full commit SHA through ``snapshots/<sha>`` -- the same lookup
    gold-kernels.sh's probe makes. (The earlier version read ``refs/<pin>`` itself;
    huggingface_hub never writes a refs file for a SHA, so a SHA pin made that read fail
    every time.) The pin is never "main", so this cannot bypass it.

    Each kernel has its own ``try``: a cold cache for one must not skip the other.
    OSError (which includes huggingface_hub's LocalEntryNotFoundError) and ImportError
    mean "not usable from the cache" and fall through to the online path. ValueError,
    including JSONDecodeError, is deliberately NOT caught: a corrupt cache must fail
    hard so the launcher's pre-download cleanup runs.

    Returns ``{kernel name: outcome}`` for logging and tests.
    """
    from types import ModuleType

    outcomes = {}
    for name, repo_id in kernels:
        if name not in hub_mapping:
            outcomes[name] = "not mapped"
            continue
        if isinstance(module_mapping.get(name), ModuleType):
            outcomes[name] = "already loaded"
            continue
        revision = pinned_revision(repo_id)
        try:
            package_name, variant_path = install_kernel(
                repo_id, revision=revision, local_files_only=True
            )
            module_mapping[name] = _validated_import(
                import_from_path, package_name, variant_path
            )
        except (OSError, ImportError) as err:
            outcomes[name] = f"skipped: {type(err).__name__}: {err}"
            log(
                f"[kernel preamble] {name}@{revision} not pre-loaded from the cache "
                f"({type(err).__name__}: {err}); the online lazy-load path takes over"
            )
        else:
            outcomes[name] = "loaded"
    return outcomes


def prepare_hub_kernels() -> None:
    """Everything a hybrid-model entrypoint needs before its first model load.

    Three stages, each a no-op where its dependencies are absent:

    1. ``pin_hub_kernels`` -- rewrite transformers' hub-kernel mapping onto the
       pinned revisions (see the module docstring for why each repo is pinned).
    2. Pre-populate transformers' kernel module mapping from the warm local
       cache, so ``lazy_load_kernel`` short-circuits without hub calls.
    3. Register the hub causal-conv1d build under its legacy import name and
       shim the legacy ``causal_conv1d_cuda`` object onto it: the modeling code
       asserts on that object (``causal_conv1d_cuda is not available. Please
       install causal-conv1d.``), but the hub build only exposes the new
       cpp_functions API and the ``kernels`` package registers modules with
       hash-based names, so ``causal_conv1d`` is never directly importable.

    Extracted from gold.py's preamble -- which was the only entrypoint carrying
    it while sft.py, run_divergence.py and distill_probe.py died on the
    legacy-name assert (measured 2026-11-20, image 2990684b, student-sft build
    test). Requires transformers 5.x hub_kernels; skips on older versions.
    """
    import sys

    pin_hub_kernels()

    try:
        from transformers.integrations.hub_kernels import (
            lazy_load_kernel as _lazy_load_kernel,
        )
    except (ImportError, ModuleNotFoundError):
        _lazy_load_kernel = None

    # Stage 2: see preload_cached_kernels for why and how.
    try:
        from kernels.utils import _import_from_path
        from kernels.utils import install_kernel as _install_kernel
        from transformers.integrations.hub_kernels import (
            _HUB_KERNEL_MAPPING,
            _KERNEL_MODULE_MAPPING,
        )
    except ImportError as _kernel_import_err:
        # kernels or transformers' hub_kernels absent or renamed: nothing to pre-load.
        # The online `lazy_load_kernel` path below (if importable) takes over.
        print(
            "[kernel preamble] kernel pre-population skipped: "
            f"{type(_kernel_import_err).__name__}: {_kernel_import_err}"
        )
    else:
        preload_cached_kernels(
            install_kernel=_install_kernel,
            import_from_path=_import_from_path,
            hub_mapping=_HUB_KERNEL_MAPPING,
            module_mapping=_KERNEL_MODULE_MAPPING,
        )

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
