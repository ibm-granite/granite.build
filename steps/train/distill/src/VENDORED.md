# Vendored: gb_steps_post_training

The `gb_steps_post_training/distillation/` package under this directory is copied from
[`github.com/laminair/gb-steps-distillation`](https://github.com/laminair/gb-steps-distillation)
at commit `a5d59bc45524a8d75706e20d44ae1a254f273f23`, imported on 2026-10-21. It differs
from that commit in these ways, and in no other:

- **Formatting.** Every file is reformatted with `black` and `isort`, which CI enforces
  repo-wide. To compare against upstream, format both sides the same way first.
- **Local patches** in `custom_gold_config.py`, `custom_gold_trainer.py` and `masking.py`,
  described below: the two on-policy study patches, the ULD base-path fix, the crop for a
  teacher that reuses the student's input, the ULD alignment guard, and check-config's
  `last_message_only` rule.

## Licence

The copyright holder of `gb-steps-distillation` contributes the code imported at
`a5d59bc4` to this repository under Apache-2.0, this repository's licence, so the copy
here is properly licensed. Files that carry another project's notice keep it: the
trainer files ported from the HuggingFace Team carry their notice plus a
`PORTED, not authored here` note naming this import, and `liger_losses.py` notes its
use of Liger-Kernel (BSD-2-Clause) in its header.

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

## Local port: trl 0.26.2 -> 1.18.2 (#676)

The trainer was written against trl 0.26.2's `GOLDTrainer`; the image now runs trl 1.18.2,
vLLM 0.33.0 and transformers 5.20.2. The port keeps this trainer's behaviour where trl's
changed, and uses trl's machinery where it does the same job:

- Data path (decision D-1): `_prepare_inputs`, `get_train_dataloader`,
  `_get_train_sampler` and `get_batch_samples` skip `GOLDTrainer`'s (which buffer one
  generation per gradient-accumulation window behind a `RepeatSampler`) and return to the
  Trainer's, so lambda is still drawn per micro-step and `group_by_length` and
  `overlap_generation` keep working.
- vLLM: generation and weight sync go through trl's `VLLMGeneration` (`generate`,
  `sync_weights`), with this trainer's sampling settings set on it once; weight sync runs
  from a callback, as trl 0.26.2's `GOLDVLLMSyncCallback` did. `MultiVLLMClient` fans both
  out over several servers.
- Prompts go to vLLM as the trainer's token IDs (D-5); logprobs are read as trl reads them,
  the sampled token's at index 0 of the top-k structure (D-2/D-4); a NaN logprob (None from
  trl's client) masks the token out of importance sampling (D-3).
- The non-hybrid ULD path (`CustomULDLoss._compute_base_distillation_loss`) and the
  probability merge (`_merge_probabilities_with_alignment_groups`) are trl 0.26.2's: the
  merge vendored unchanged (D-7: bit-identical numerics), the base path with the local fix
  below. trl 1.x's base path needs byte offsets this trainer does not pass, and its
  "observed" merge is unnormalised and needs `token_ids`.
- `CustomULDLoss._compute_distillation_loss` accepts trl 1.x's byte-offset keywords and
  ignores them (it aligns from token IDs).

`steps/train/distill/logit-distill/skypilot/test/test_trl1_port.py` pins each of these.

## Local fix: the ULD base path

`CustomULDLoss._compute_base_distillation_loss`, the path `loss_arm: uld` runs, is trl
0.26.2's non-hybrid `ULDLoss._compute_distillation_loss` with two fixes, both described in
its docstring and pinned in
`steps/train/distill/logit-distill/skypilot/test/test_loss_numerics.py`:

- TRL read the logits AT each completion position, one late: logits[t] predicts token
  t + 1, and nothing upstream shifts them. It now reads start-1, as the hybrid path and
  the JSD path do. trl 1.18.2 has the same default, `uld_token_merge_strategy="observed"`
  (its opt-in `"bayesian"` strategy shifts).
- A row whose alignment yields no groups reached the merge, which then returned the
  probabilities unmerged, so the L1 compared positions that were never paired: a shape
  `RuntimeError` when the completions differ in length, except when one side has a single
  position, which `F.l1_loss` broadcasts against every position of the other with only a
  `UserWarning`; and a distance between unaligned positions when they have the same
  length. Such a row now contributes no loss, as on the hybrid path, and is counted.

One branch is removed rather than fixed: TRL's loop also called the hybrid loss per row
when `use_hybrid_loss` and a vocabulary mapping were set. `_compute_distillation_loss`
dispatches exactly that case to the hybrid path, so the branch was unreachable here.

Because it reads start-1, a completion that starts at position 0 would have no predictor
and would slice from the end of the row. Both ULD paths now skip such a row and count it
(`CustomULDLoss._unpredictable_start`; on the hybrid path before every start-1 read), and
`ULDAlignmentGuard` refuses it at the optimizer step on every rank (below). A raise in the
loss would be one rank's alone, and the other ranks would wait in backward's collective
until the NCCL timeout.
The layouts the logit-distill step builds never put a completion there: the collator's spans
start after the response template or the prompt, and the single-region teacher crop keeps
`teacher_prompt_length - 1`.

## Local fix: the crop for a teacher that reuses the student's input

With `uld_reuse_student_input` (a debugging mode the step cannot reach), the teacher's
`input_ids` and labels are the student's, and `teacher_prompt_length` was
`inputs["prompts"].shape[1]`, the batch's longest prompt. The collator left-pads, so the rows
end together and, at a batch larger than 1, a row with a shorter prompt and a longer
completion starts its completion before that column: the crop cut into it, the teacher lost
its first tokens, and its labels started at position 0. The mis-crop came with the import.
`teacher_prompt_length` is now the first column any row labels, and at least 1
(`earliest_completion_column`), so every row's completion is kept whole with the logit
before it. The crop is exactly as wide as a per-row crop padded to one width: for rows that
end together, both are the longest completion plus one.

## Local addition: refusing a ULD run that trains on nothing

Skipping an unaligned row is right for the rare row and silent when the two renders
disagree systematically: every row is skipped, the loss and the gradient are exactly zero,
and the run checkpoints and reports success. So both ULD paths count, rank-locally, every
row (region, on the hybrid path) that reaches alignment (`CustomULDLoss._alignment_rows`)
and every one that yields no groups (`_alignment_unaligned`; the hybrid path's skip now
increments it too), and `ULDAlignmentGuard` in `custom_gold_trainer.py`, run by
`_ULDAlignmentGuardCallback` at `on_pre_optimizer_step` on every rank, drains them at each
optimizer step and sums them over ranks. It raises `RuntimeError`:

- when any row of a step, on any rank, was skipped for a completion at position 0
  (`_alignment_unpredictable`, reduced in the same all-reduce as the other two counts);
  always on, whatever the threshold. An evaluated row counts towards the next step;
- when every row of a step, across all ranks, was unaligned; always on;
- when more than `uld_max_unaligned_fraction` (default 0.5; 1.0 turns it off) of the rows
  in a window of `uld_unaligned_window_steps` optimizer steps (default 0 = `logging_steps`)
  were unaligned; the final, partial window is judged at train end (`on_train_end`). Both
  are new `CustomGOLDConfig` fields.

The guard is the runtime backstop. The known systematic cause, `last_message_only=false`
under the single-region ULD teacher, is refused before training: the logit-distill step's
renderer requires `last_message_only: true` for `loss_arm: uld` and, for that arm alone,
emits it off-policy (`render_gold_config.py`, outside this package). `CustomGOLDConfig.__post_init__`
refuses `use_uld_loss` without `last_message_only=true` too, for a config that bypasses the
renderer, as it already did for `use_sampled_opd_loss`. It exempts the three modes that do not
pair the collator's labels with the last-turn teacher: `uld_per_turn_regions` (which requires
false), `uld_reuse_student_input`, and `uld_rebuild_student_input_with_teacher_template`, which
rebuilds the student's input and labels from the teacher's (prompt, last-turn completion) texts.

Every log line carries three new keys, the counts since the previous line summed over
ranks: `train/alignment_rows`, `train/alignment_unaligned_rows` and
`train/alignment_unaligned_rate`. They are identical on every rank.

## Local fix: check-config's `last_message_only` rule

`masking.check_config` (`masking.py check-config`) reported `last_message_only: true` as a
problem on every gold config, because it bypasses the response_template scan that
masking.json describes. For the two arms whose `render_gold_config.py` LOSS_ARMS entry
requires it, `uld` and `sampled_opd`, that inverted the renderer's rule. It now follows that
rule: for those arms, true is the contract and false or unset is the problem. The arms are
recognised from `loss_arm` or from the switch they render (`use_uld_loss`,
`use_sampled_opd_loss`). The `use_uld_loss` switch, which is what reaches `CustomGOLDConfig`,
takes the same three ULD exemptions; a named `loss_arm: uld` takes none, because the renderer
renders none of those modes and refuses the arm without `true` in every case. In an exempt
mode either value is sound, and `true` is not reported as the scan bypass, since
`CustomGOLDConfig` accepts it; `uld_per_turn_regions` with `true` is a problem in either
form, since `CustomGOLDConfig` refuses it. Only the boolean `true` satisfies the rule. Every
other arm keeps the old report.
`steps/train/distill/logit-distill/skypilot/test/test_masking_check_config.py` pins both
directions against the renderer's table, and every ULD form, mode and `last_message_only`
value against the renderer's rule and `CustomGOLDConfig`'s verdict.

## A second copy elsewhere

`steps/aux/hf-export/skypilot/src/hf_export_state/` carries its own copies of
`step_state.py`, `export_state.py` and `tokenizer_identity.py`, so that generic step depends
on nothing in this collection. Both copies must keep writing and reading the same
`.step-done.json` marker: change the marker format in both places, together.
`steps/aux/hf-export/skypilot/test/test_step_template.py::TestResumeCompatibility` fails if
the schema or marker name drift apart.
