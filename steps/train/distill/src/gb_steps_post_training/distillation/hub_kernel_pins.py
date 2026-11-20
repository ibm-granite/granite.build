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

``v0.0.4`` (8ce5a492, "Fix for Falcon") restructured the build: the top level
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
    "kernels-community/mamba-ssm": "v0.0.4",
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

    # Pre-populate transformers' kernel module mapping from the warm local cache so that
    # `lazy_load_kernel` short-circuits without ever calling `HfApi().list_repo_refs`
    # (which would happen for `version=`-style entries in `_HUB_KERNEL_MAPPING`) or
    # `snapshot_download` (which races across ranks on NFS). This is the supported
    # short-circuit at transformers/integrations/hub_kernels.py:368-369. Pure cache
    # reads via `install_kernel(..., revision=<sha>, local_files_only=True)` only
    # perform `os.readlink` on existing symlinks and are rank-safe.
    try:
        from pathlib import Path as _Path

        from huggingface_hub.constants import HF_HUB_CACHE as _HF_HUB_CACHE
        from kernels.utils import _import_from_path
        from kernels.utils import install_kernel as _install_kernel
        from transformers.integrations.hub_kernels import (
            _HUB_KERNEL_MAPPING,
            _KERNEL_MODULE_MAPPING,
        )

        def _read_cached_sha(_repo_id):
            # The current `kernels` package writes the standard HF hub layout
            # (`models--<repo>/...`) via `snapshot_download`, not the legacy
            # `kernels--<repo>/...` prefix. See kernels/utils.py:194-203.
            #
            # The ref is the PINNED revision (hub_kernel_pins.py), not "main":
            # reading "main" here would install the main build directly into
            # _KERNEL_MODULE_MAPPING, bypassing the pin the rest of this
            # collection goes through. On a cache warmed by the preflight or
            # the prefetch (which both download at the pinned revision) the
            # ref file exists; on a cold cache it does not, the FileNotFoundError
            # is caught below, and the online lazy-load path -- which does
            # respect the pin -- takes over.
            _ref_path = (
                _Path(_HF_HUB_CACHE)
                / f"models--{_repo_id.replace('/', '--')}"
                / "refs"
                / pinned_revision(_repo_id)
            )
            return _ref_path.read_text().strip()

        def _validated_import_from_path(_package_name, _variant_path):
            # `_import_from_path` will raise on a structurally-broken metadata.json
            # only when the loader actually parses it; pre-validate so a corrupt
            # cache aborts the preamble loudly with a JSONDecodeError instead of
            # later showing up as a hard-to-diagnose ImportError under the FA3
            # dispatcher.
            import json as _json

            _meta_path = _Path(_variant_path) / "metadata.json"
            if _meta_path.exists():
                _json.loads(_meta_path.read_text())
            return _import_from_path(_package_name, _variant_path)

        def _preload_kernel_into_mapping(_kernel_name, _repo_id):
            from types import ModuleType as _ModuleType

            _existing = _KERNEL_MODULE_MAPPING.get(_kernel_name)
            if isinstance(_existing, _ModuleType):
                return
            _sha = _read_cached_sha(_repo_id)
            _package_name, _variant_path = _install_kernel(
                _repo_id, revision=_sha, local_files_only=True
            )
            _module = _validated_import_from_path(_package_name, _variant_path)
            _KERNEL_MODULE_MAPPING[_kernel_name] = _module
            return _module

        for _kname, _kspec in (
            ("causal-conv1d", "kernels-community/causal-conv1d"),
            ("mamba-ssm", "kernels-community/mamba-ssm"),
        ):
            if _kname in _HUB_KERNEL_MAPPING:
                _preload_kernel_into_mapping(_kname, _kspec)

    except (
        FileNotFoundError,
        OSError,
        ImportError,
        ModuleNotFoundError,
    ) as _kernel_preload_err:
        # Fresh node / kernels API change / missing cache. Fall through to the
        # online `lazy_load_kernel` path below; preflight will repopulate the cache.
        # Note: ValueError (incl. JSONDecodeError) is intentionally NOT caught here.
        # A corrupt cache (e.g. doubled-JSON metadata.json from a parallel-write race)
        # must surface as a hard failure so the launcher's pre-download cleanup runs,
        # rather than fall through to the racing kernel-hub fallback path that would
        # produce a confusing downstream `ImportError: FlashAttention3 has been toggled on`.
        print(
            f"[kernel preamble] kernel pre-population skipped: "
            f"{type(_kernel_preload_err).__name__}: {_kernel_preload_err}"
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
