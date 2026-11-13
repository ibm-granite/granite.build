"""Pin hub-resolved kernels to revisions compatible with this image's transformers.

WHY THIS EXISTS (measured 2026-11-13, BlueVela, image ec82c1d9). transformers 5.8.0's
``_HUB_KERNEL_MAPPING`` resolves ``mamba-ssm`` with ``version=1``, which is the
kernels-community repo's ``v1`` branch (768416ab, 2026-05-05). That build's
``utils/generation.py`` hard-imports ``GreedySearchDecoderOnlyOutput`` from
``transformers.generation`` -- a symbol transformers 5.x renamed to
``GenerateDecoderOnlyOutput`` -- so every fresh load of a hybrid (granitemoehybrid)
model dies with an ImportError after the model load has started. A build-test run of
teacher-divergence hit it directly, and preflight-check's load-student probe reported
``PROBE VERDICT loaded=NONE`` for the same reason.

``v0.0.3`` (916c2f7aeccb, 2025-10-14) is the newest revision whose
``torch28-cxx11-cu128-x86_64-linux`` build predates that import. ``v0.0.4``
(2025-12-11, "Fix for Falcon") has it, as does ``main``.

This is a pin, not a fix. When kernels-community/mamba-ssm ships a build that imports
cleanly under transformers 5.x again, or the image moves past the rename, delete the
entry here, the call sites, and the matching probe in gold-kernels.sh (which reads
``pinned_revision`` from this module -- keep the two in sync by construction, not by
discipline).
"""

from __future__ import annotations

# repo_id -> revision. Unlisted repos resolve their default (transformers passes
# version=1, the kernels package resolves the repo's v1 branch).
PINNED_KERNEL_REVISIONS: dict[str, str] = {
    "kernels-community/mamba-ssm": "v0.0.3",
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
        from transformers.integrations import hub_kernels
    except ImportError:  # transformers absent (CPU-only entrypoints): nothing to pin
        return
    for name in _PINNED_KERNEL_NAMES:
        entry = hub_kernels._HUB_KERNEL_MAPPING.get(name)
        if not isinstance(entry, dict):
            continue
        revision = PINNED_KERNEL_REVISIONS.get(entry.get("repo_id", ""))
        if revision is None:
            continue
        entry.pop("version", None)
        # setdefault, not assignment: a future transformers that pins its own
        # revision has done the work, and its pin deserves the benefit of the doubt.
        entry.setdefault("revision", revision)
