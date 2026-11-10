# Vendored: gb_steps_post_training

The `gb_steps_post_training/distillation/` package under this directory is copied from
[`github.com/laminair/gb-steps-distillation`](https://github.com/laminair/gb-steps-distillation)
at commit `a5d59bc45524a8d75706e20d44ae1a254f273f23`, imported on 2026-10-21. It differs
from that commit in these ways, and in no other:

- **Formatting.** Every file is reformatted with `black` and `isort`, which CI enforces
  repo-wide. To compare against upstream, format both sides the same way first.
- **Local patches** in `custom_gold_config.py` and `custom_gold_trainer.py`, described
  below: the two on-policy study patches.

This is a one-time import, not a git subtree or submodule, and there is no sync mechanism
to the external repo. A trainer fix lands as a direct edit here, like any other
granite.build code; it is not re-pulled from `gb-steps-distillation`. Re-importing a later
upstream commit, if ever needed, is a deliberate, separate change.

## How the steps get it

Both shared images (`steps/train/distill/images/{gpu,cpu}/Dockerfile`, built by
`steps/train/distill/Makefile`) COPY this package to `$DISTILL_HOME/src`
(`/opt/distill/src`) and put it on `PYTHONPATH`. Nothing is cloned or mounted at job
launch: the trainer a build ran is exactly what the step's pinned image tag contains.
Changing anything here therefore means rebuilding the images and repinning their tags in
the step templates.

## History: CE anchor + entropy guard (folded in, then removed)

PR 540 folded a local patch into `custom_gold_config.py` and `custom_gold_trainer.py` that
added `ce_coef` (an additive cross-entropy anchor), `log_student_entropy` and the
`entropy_guard_*` fields (an entropy-collapse guard). #664 removed it again: a CE sweep at
four coefficients was monotone-bad on capability, the entropy collapse the guard watched
for turned out to be harmless (the real defect was the export dropping `rope_theta`), and
every recipe already ran both at zero. With them gone the trainer computes the upstream
loss, as it did with the fields at their defaults.

## Local patches folded in: linear lmbda schedule, on-policy study

Two more patches, folded in on 2026-11-09 from the on-policy study's branch
(`epic-61-onpolicy-followup`, `steps/distill/gold/skypilot/patches/`), where a recipe
applied them to its own checkout of the trainer. Applied in order on top of a5d59bc4 they
produce commit `00edb0db1978c385e1469361f8bcb9293b396222` in that patched checkout; the
files here match it apart from formatting.

- **`lmbda_linear_schedule.diff`** (2026-10-05). At a5d59bc4 the `linear` branch of
  `_get_current_lmbda` sat after the `return` of `_draw_on_policy`, so it was unreachable:
  with `lmbda_schedule: linear` the method returned `None` and `draw_on_policy` raised
  `TypeError` at the first training step. The branch is moved back, unchanged; the
  `constant` path is untouched. Upstream HEAD `59c36a2` has the same defect.
- **`onpolicy_study.diff`** (2026-10-11). Four `CustomGOLDConfig` fields, each defaulting
  to the behaviour without them:
  - `lmbda_ramp_start_step`, `lmbda_ramp_steps` -- the linear schedule ramps from
    `lmbda_ramp_start_step` for `lmbda_ramp_steps` steps (0 = until `max_steps`), so a run
    can ramp over part of its length, or a resumed run over its own steps.
  - `min_completion_length` -- a generation floor, sent as `{"min_tokens": N}` in every
    vLLM server request's `generation_kwargs` (and as `SamplingParams.min_tokens` in
    colocate mode). 0 sends nothing.
  - `stop_at_step` -- save and stop at a fixed step without changing `max_steps`, and with
    it the LR schedule's shape.

`steps/train/distill/logit-distill/skypilot/test/test_vendored_lmbda.py` runs these methods
from the source text against stubs, since the module's torch/trl imports are not in the
granite.build venv.

## A second copy elsewhere

`steps/misc/hf-export/skypilot/src/hf_export_state/` carries its own copies of
`step_state.py`, `export_state.py` and `tokenizer_identity.py`, so that generic step depends
on nothing in this collection. Both copies must keep writing and reading the same
`.step-done.json` marker: change the marker format in both places, together.
`steps/misc/hf-export/skypilot/test/test_step_template.py::TestResumeCompatibility` fails if
the schema or marker name drift apart.
