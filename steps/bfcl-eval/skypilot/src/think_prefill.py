"""Pre-fill a CLOSED reasoning block onto BFCL's generation prompt, to separate
"the arm cannot call tools" from "the arm's own reasoning block displaces the call".

WHY THIS EXISTS. Every distilled arm of the unified100 round scores 0.00% on
BFCL multi_turn, and the raw generations say why: over 3,336 model segments,
GOLD lambda0.3 opens AND closes `<think>` on 100% of them, then writes prose on
100% of them, and produces exactly ONE parseable tool call. The obvious reading
-- "the harness forces a reasoning block, like lm-eval's chat pass does" -- is
WRONG here, and that matters for what a probe should do:

  * `Granite4FCHandler._format_prompt` (granite_4.py:27-248) builds the prompt in
    hardcoded Python. The checkpoint's chat template is never consulted for prompt
    construction; only its tokenizer is loaded.
  * That method's generation prompt is exactly
    `<|start_of_role|>assistant<|end_of_role|>` (granite_4.py:246). No `<think>`
    is opened.
  * Its system message explicitly asks for the other dialect: "For each tool call,
    return a json object with function name and arguments within
    `<tool_call></tool_call>` XML tags" (granite_4.py:150).
  * `grep -n think` over granite_4.py, base_oss_handler.py, bfcl-shim and
    run-bfcl.sh matches NOTHING. There is no reasoning scaffold in this path.

So the reasoning block is SELF-INITIATED. There is nothing to suppress, and a
"turn thinking off" flag would have nothing to turn off. What can be done instead
is to hand the arm a reasoning block that is already finished, so that generation
resumes at exactly the byte position where its training data emits a call:

    ...<|start_of_role|>assistant<|end_of_role|><think>\\n\\n</think>\\n
                                                                      ^ resumes here

`reason-if-tools-v1-32K-render` teaches precisely that continuation -- its
reasoning+tool rows read `<think>\\n...\\n</think>\\n<tool_call>\\n{"name": ...,
"arguments": ...}\\n</tool_call>` -- so the prefill puts the model one token away
from the format it was trained on.

HOW TO READ THE RESULT. If `<tool_call>` returns, tool-calling capability survived
distillation and the arm's reasoning mode was displacing it; the 0.00% is then a
generation-format result with a known remedy. If prose still follows a closed block
the arm did not even have to write, the arms genuinely lost the dialect and part of
the 0.00% is a capability result. Either answer is worth 2 GPUs and an hour; the
current 0.00% is worth nothing in either direction.

WHAT THIS CHANGES ABOUT THE MEASUREMENT, said plainly: a prefilled run is NOT
comparable to the unprefilled arms as a score. The prompt differs, the recorded
response no longer contains a reasoning block, and multi-turn history therefore
carries assistant turns without one. It is a diagnostic, and its number belongs
next to the diagnosis, not in the arm ranking.

DEFAULT IS OFF, deliberately and by more than convention: three BFCL jobs for the
lambda1.0 and both ULD arms are queued behind `ended()` dependencies at the time
this module was written, and they will import this file when they start. Unless
BFCL_THINK_PREFILL is set to something non-empty, `patch_granite_prefill` returns
without touching the class and those runs are byte-identical to the ones already
scored.

USAGE
    BFCL_THINK_PREFILL=empty   -> "<think>\\n\\n</think>\\n"  (the probe above)
    BFCL_THINK_PREFILL=close   -> "</think>\\n"   (close only, no open tag: the
                                  shape lm-eval's template produces, for contrast)
    BFCL_THINK_PREFILL='<lit>' -> that literal string, with \\n and \\t honoured,
                                  for anything these two presets do not cover
"""

from __future__ import annotations

import os

# The corpus's own bytes. `empty` opens a block and closes it immediately, which is
# the only preset that leaves the prompt at the exact position the trained pattern
# continues from. `close` exists because it is the OTHER plausible prefill -- the one
# a template-side fix would produce -- and a probe that cannot distinguish the two
# would leave the same ambiguity it is meant to remove.
_PRESETS = {
    "empty": "<think>\n\n</think>\n",
    "close": "</think>\n",
}


def resolve_prefill(spec: str) -> str:
    """Turn a BFCL_THINK_PREFILL value into the literal string to append.

    A preset name wins over a literal, so `empty` can never be mistaken for a
    zero-length prefill (which would silently make the probe a no-op that still
    reports itself as armed -- the exact failure mode this repo keeps finding in
    guards that pass by doing nothing).
    """
    if spec in _PRESETS:
        return _PRESETS[spec]
    return spec.replace("\\n", "\n").replace("\\t", "\t")


def patch_granite_prefill(handler_cls, spec: str | None = None) -> str | None:
    """Append a closed reasoning block to `handler_cls._format_prompt`'s output.

    Returns the prefill actually installed, or None if the patch was declined --
    so the caller can say which of the two happened instead of leaving it to be
    inferred from the absence of a message.

    `_format_prompt` is an ordinary instance method (`@override`, granite_4.py:27)
    called once per turn from `base_oss_handler.py:322`, and it rebuilds the whole
    prompt every turn. Appending to its return value therefore prefills EVERY turn
    of a multi_turn episode, which is what the probe wants: the failure being
    investigated recurs on every segment, not only the first.

    The token cost is ~4 tokens of input against a 131,072-token context and only
    reduces `min(4096, max_context_length - input - 2)` (base_oss_handler.py:334) by
    the same, so the output budget the arms are currently exhausting is unchanged
    for practical purposes.
    """
    if spec is None:
        spec = os.environ.get("BFCL_THINK_PREFILL", "")
    if not spec:
        return None

    prefill = resolve_prefill(spec)
    if not prefill:
        # An explicitly-set-but-empty value is a mistake, not a request for a no-op.
        raise ValueError(
            f"BFCL_THINK_PREFILL={spec!r} resolves to an empty prefill. "
            f"Use one of {sorted(_PRESETS)} or a non-empty literal."
        )

    original_format_prompt = handler_cls._format_prompt

    def _format_prompt_with_prefill(self, messages, function):
        return original_format_prompt(self, messages, function) + prefill

    handler_cls._format_prompt = _format_prompt_with_prefill
    return prefill
