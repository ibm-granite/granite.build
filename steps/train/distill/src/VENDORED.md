# Vendored: gb_steps_post_training

The `gb_steps_post_training/distillation/` package under this directory is copied from
[`github.com/laminair/gb-steps-distillation`](https://github.com/laminair/gb-steps-distillation)
at commit `a5d59bc45524a8d75706e20d44ae1a254f273f23`, imported on 2026-10-21. It differs
from that commit in two ways, and in no other:

- **Formatting.** Every file is reformatted with `black` and `isort`, which CI enforces
  repo-wide. To compare against upstream, format both sides the same way first.
- **A local patch** in two files, described below.

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

## Local patch: CE anchor + entropy guard

`custom_gold_config.py` and `custom_gold_trainer.py` carry a patch that adds the
`ce_coef`, `log_student_entropy` and `entropy_guard_*` fields to `CustomGOLDConfig`, and the
matching CE-anchor and entropy-collapse-guard logic to `CustomGOLDTrainer`. Every new field
defaults to off (`0.0` / `false`), so a config that sets none of them computes exactly the
upstream loss. A recipe turns them on with parameters alone (`CE_COEF`, `LOG_STUDENT_ENTROPY`,
`ENTROPY_GUARD_*`).

The patch used to ship as a separate `.diff` that a recipe applied to its own checkout of
the trainer. It was folded in here on 2026-10-21 when the steps stopped cloning the trainer
at launch, so the capability stays available without a checkout to point at. If upstream
lands an equivalent and the import is bumped past that commit, reconcile these two files by
hand at that time.

## A second copy elsewhere

`steps/misc/hf-export/skypilot/src/hf_export_state/` carries its own copies of
`step_state.py`, `export_state.py` and `tokenizer_identity.py`, so that generic step depends
on nothing in this collection. Both copies must keep writing and reading the same
`.step-done.json` marker: change the marker format in both places, together.
`steps/misc/hf-export/skypilot/test/test_step_template.py::TestResumeCompatibility` fails if
the schema or marker name drift apart.
