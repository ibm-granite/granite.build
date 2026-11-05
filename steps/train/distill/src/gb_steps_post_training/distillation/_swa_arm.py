"""Is the vendored GraniteSWA arm in play?

`gold.py`, `sft.py` and `run_vllm_serve.py` in an earlier exploratory checkout all import
`granite_swa` and `_fa3_preamble` UNCONDITIONALLY, at module scope, before any
argument parsing. Neither is needed by the granite 4.1/4.2 pairing this
collection targets, and the FA3 preamble is not merely unused there -- it raises:

    RuntimeError: [gold preamble] FA3 kernel pre-population failed:
    FileNotFoundError: .../models--kernels-community--vllm-flash-attn3/refs/main.
    Refusing to start training to prevent silent eager-attention fallback.

(measured directly, rank 0, before a single batch). The preamble's refusal
is CORRECT for the model it was written for: `GraniteSWAAttention` compares
`config._attn_implementation` to the literal `"flash_attention_3"`, and if the
kernel-fallback rewrites that string the model drops to eager attention, which at
128K context materialises a [B, H, T, T] score tensor and OOMs. So the guard here
must not weaken that check -- it must decide, before importing anything, whether
the SWA arm is being used at all.

Default is off, matching the triage decision that `gold/granite_swa/` is a
DELETE rather than a deferral: it applies to no model in this pairing. Turning it
on requires the vendored package, which is deliberately NOT ported -- so opting in
gives a message that says where it lives instead of an ImportError on a name.
"""

import os

_TRUTHY = {"1", "true", "yes", "on"}

SWA_ARM = os.environ.get("GOLD_SWA_ARM", "0").strip().lower() in _TRUTHY

_NOT_PORTED = (
    "GOLD_SWA_ARM=1 asks for the vendored GraniteSWA arm, which is not part of "
    "this collection. Its model code (granite_swa/), the FA3 dispatch preamble "
    "(_fa3_preamble.py) and the Liger tiled-MLP patch (_liger_granite_swa_patch.py) "
    "lived in that earlier checkout's gold/ directory and were classified DELETE during triage "
    "because no model in the granite 4.1/4.2 pairing uses sliding-window attention. "
    "If the SWA arm is being revived, port those three files first and record why in "
    "docs/planning/distillation-steps-plan.md -- do not re-vendor them silently."
)


def activate_swa_arm() -> bool:
    """Import and register the SWA model classes. Returns whether it did anything.

    Called at module scope by the entrypoints, in place of their unconditional
    imports. When the arm is off this is a no-op and nothing about SWA, FA3 or the
    Hub kernel cache is touched -- which is the whole point: a run of the 4.1/4.2
    pairing must not depend on a network fetch for a kernel it will never call.
    """
    if not SWA_ARM:
        return False
    raise RuntimeError(_NOT_PORTED)


def activate_swa_vllm_model() -> bool:
    """Register the SWA vLLM model class. Same contract as activate_swa_arm()."""
    if not SWA_ARM:
        return False
    raise RuntimeError(_NOT_PORTED)
