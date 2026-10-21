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
