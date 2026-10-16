# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""HF TrainerCallback that suppresses intermediate train.report() calls.

When BLDS is paired with an early-stopping scheduler (ASHA), non-top-rung
trials must NOT emit per-epoch metrics — otherwise the scheduler would compare
losses across fidelity rungs that are not directly comparable (e.g. a 10%-data
trial's epoch-1 loss vs. a 100%-data trial's epoch-1 loss). See plan
silky-cantering-lovelace.md (Phase 3) for the full rationale.

`FinalSaveOnlyReportCallback` extends `RayTrainReportCallback` and only forwards
the final save (when training has stopped) to Ray Train. This preserves the
final-loss feedback path that BLDS needs (via `train_result.metrics`) while
denying ASHA the per-epoch signal it would otherwise act on.
"""

from ray.train.huggingface.transformers import RayTrainReportCallback


class FinalSaveOnlyReportCallback(RayTrainReportCallback):
    """Subclass of RayTrainReportCallback that reports only on the final save.

    Use this in place of RayTrainReportCallback when the trial's fidelity rung
    is below the top rung and the run pairs BLDS with an early-stopping
    scheduler. Falls back silently to per-save reporting if HF's TrainerControl
    does not signal training stop on the relevant save (e.g. truncated runs).
    """

    def on_save(self, args, state, control, **kwargs):
        # `control.should_training_stop` is set True after the trainer's
        # final save in HF Transformers. Defer to the parent's reporting
        # logic only on that final save; otherwise no-op.
        if not getattr(control, "should_training_stop", False):
            return
        super().on_save(args, state, control, **kwargs)
