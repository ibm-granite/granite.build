# Vendored: gb_steps_post_training

The `gb_steps_post_training/distillation/` package under this directory is
vendored verbatim from
[`github.com/laminair/gb-steps-distillation`](https://github.com/laminair/gb-steps-distillation),
pinned at commit `a5d59bc45524a8d75706e20d44ae1a254f273f23`, vendored on
2026-10-21.

This is a one-time import, not a git subtree or submodule, and there is no
sync mechanism to this external repo. A trainer fix lands as a direct edit
to the files here, the same as any other granite.build code — it does not
get re-pulled from `gb-steps-distillation`. If that external repo advances
independently, this directory simply does not track it; re-vendoring a
later commit (if ever needed) is a deliberate, separate action, not
something any build or CI step does automatically.

This package is shared across both the GPU image
(`steps/distill/gpu/Dockerfile`) and, for the step(s) whose trainer source
is fully baked in rather than cloned at job-launch time, imported via
`PYTHONPATH`. See `steps/distill/gpu/Dockerfile` for how it's consumed.

## Local patch folded in: CE anchor + entropy guard

`custom_gold_config.py` and `custom_gold_trainer.py` carry a local patch
(formerly `steps/distill/gold/skypilot/patches/ce_anchor_and_entropy_guard.diff`,
applied here on 2026-10-21 and removed as a separate file) adding `ce_coef`,
`log_student_entropy`, `entropy_guard_*` fields to `CustomGOLDConfig` and the
matching CE-anchor/entropy-collapse-guard logic to `CustomGOLDTrainer`. All
new fields default to off/0.0, so an unpatched config computes exactly the
same loss as before.

This used to be a recipe-applied patch against a `code_config.code_dir`
checkout (see `recipes/granite4-350m/lsf/distill-stage1-v2`), which only
worked for `gold` because `gold` was the one step still cloning source at
job-launch time with a recipe-overridable checkout path. Since `gold` now
uses this vendored copy baked into the image instead (no more
`code_config`), folding the patch in here — rather than losing the
capability — keeps it available to every recipe unconditionally, with no
separate checkout to point at. The patch's own header already recommended
this exact move ("offer this upstream to gb-steps-distillation... point the
recipes' CODE_DIR back at \"\" and delete this file").

If `gb-steps-distillation` upstream ever lands an equivalent of this patch
and this vendor pin is bumped past that commit, these two files' local
patch becomes redundant — reconcile by hand against the new upstream
content at that time, the same as any other vendored-file update.
