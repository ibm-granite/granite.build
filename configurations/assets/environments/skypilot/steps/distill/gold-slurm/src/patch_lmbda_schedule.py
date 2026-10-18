#!/usr/bin/env python3
# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Idempotently fix the broken lmbda LINEAR ramp in the pinned GOLD trainer.

WHY THIS EXISTS. In gb-steps-distillation @ a5d59bc4, custom_gold_trainer.py has a
botched edit: ``_get_current_lmbda()`` only implements the ``"constant"`` schedule
and FALLS OFF THE END for ``"linear"`` (implicitly returning ``None``); the actual
linear-interpolation body was misplaced into ``_draw_on_policy()`` BELOW its early
``return`` (dead code). With ``lmbda_schedule=linear`` the trainer then does
``current_lmbda = _get_current_lmbda()  # -> None`` and crashes at the first step:

    on_policy_draw.py: if current_lmbda <= 0.0:
    TypeError: '<=' not supported between instances of 'NoneType' and 'float'

This script rewrites ``_get_current_lmbda`` to compute the linear ramp inline
(lmbda_init + progress*(lmbda - lmbda_init), progress = global_step/max_steps),
matching the intent of the misplaced dead code. It is a stop-gap until the pin is
bumped to a trainer that implements the ramp correctly; run from the gold-slurm
step's ``code_config.setup_command`` against the freshly-cloned checkout.

Idempotent: a no-op (exit 0) if the linear branch is already present, so it is safe
to run on every rank and across retries. Never fails the build for a patch problem
it cannot apply cleanly — it reports and exits non-zero only if the file exists but
the exact broken signature is absent (meaning the pin changed and this patch is
stale and should be re-reviewed, NOT silently skipped).
"""

import pathlib
import sys

# The exact broken body: constant-only, then straight into the next method's def.
_BROKEN = (
    '        if self.args.lmbda_schedule == "constant":\n'
    "            return self.lmbda\n"
    "\n"
    "    def _draw_on_policy"
)

# Fixed: add the linear branch (and a constant fallthrough) before the next method.
_FIXED = (
    '        if self.args.lmbda_schedule == "constant":\n'
    "            return self.lmbda\n"
    "        # --- gb patch: linear ramp (upstream a5d59bc4 omitted this branch;\n"
    "        #     its body was dead code below the return in _draw_on_policy) ---\n"
    '        if self.args.lmbda_schedule == "linear":\n'
    "            max_steps = self.state.max_steps\n"
    "            if max_steps is None or max_steps <= 0:\n"
    "                return self.args.lmbda_init\n"
    "            progress = min(1.0, self.state.global_step / max_steps)\n"
    "            return self.args.lmbda_init + progress * (\n"
    "                self.args.lmbda - self.args.lmbda_init\n"
    "            )\n"
    "        return self.lmbda\n"
    "\n"
    "    def _draw_on_policy"
)

# Marker proving the fix is already applied (idempotency check).
_ALREADY = "# --- gb patch: linear ramp"


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: patch_lmbda_schedule.py <path-to-custom_gold_trainer.py>",
            file=sys.stderr,
        )
        return 2
    path = pathlib.Path(sys.argv[1])
    if not path.is_file():
        print(
            f"patch_lmbda_schedule: {path} not found; nothing to patch", file=sys.stderr
        )
        return 1
    text = path.read_text(encoding="utf-8")

    if _ALREADY in text:
        print("patch_lmbda_schedule: linear-ramp branch already present; no-op")
        return 0

    if _BROKEN not in text:
        print(
            "patch_lmbda_schedule: FATAL: the known-broken _get_current_lmbda "
            "signature was not found. The trainer pin likely changed — re-review "
            "this patch against the new source rather than training on a possibly "
            "fixed (or differently-broken) ramp.",
            file=sys.stderr,
        )
        return 1

    path.write_text(text.replace(_BROKEN, _FIXED, 1), encoding="utf-8")
    print(f"patch_lmbda_schedule: applied linear-ramp fix to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
