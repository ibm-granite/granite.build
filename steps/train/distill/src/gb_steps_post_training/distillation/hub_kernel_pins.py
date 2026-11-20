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

This is a pin, not a fix. When kernels-community/mamba-ssm ships a build that
imports cleanly under transformers 5.x on the default branch, or the image moves
past the rename, delete the entry here, the call sites, and the matching probe
in gold-kernels.sh (which reads ``pinned_revision`` from this module -- keep the
two in sync by construction, not by discipline).
"""

from __future__ import annotations

# repo_id -> revision. Unlisted repos resolve their default (transformers passes
# version=1, the kernels package resolves the repo's v1 branch).
PINNED_KERNEL_REVISIONS: dict[str, str] = {
    "kernels-community/mamba-ssm": "v0.0.4",
}

# The transformers mapping keys whose repo is mamba-ssm. "falcon_mamba-ssm" shares
# the repo and the same broken default; pinning it costs nothing and keeps a Falcon
# run from rediscovering this.
_PINNED_KERNEL_NAMES = ("mamba-ssm", "falcon_mamba-ssm")


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
