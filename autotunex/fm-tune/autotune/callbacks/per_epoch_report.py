# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""HF TrainerCallback that emits a tune.report() after each epoch.

Single-GPU drivers (driver_single.py, driver_single_trl.py) run multi-epoch
HPO trials but only emit a terminal tune.report(). Without per-epoch reports,
Ray Tune's early-stopping schedulers (ASHA) cannot compare trials at matching
time_attr rungs and become no-ops.

This callback bridges the gap by forwarding the eval metrics from each
evaluation_strategy="epoch" event to Ray Tune via tune.report(). The
terminal driver-level tune.report(result) at end-of-train is preserved.

`done` is set to False on every per-epoch report so that BLDS's defensive
guard in on_trial_complete continues to treat partial-training losses as
non-final results.
"""

from ray import tune
from transformers import TrainerCallback


class PerEpochTuneReportCallback(TrainerCallback):
    """Report eval metrics to Ray Tune after each epoch.

    Hooks `on_evaluate` (which fires once per epoch when
    `evaluation_strategy="epoch"`, with the just-computed metrics in
    `kwargs["metrics"]`). The terminal driver-level `tune.report(result)`
    after `trainer.train()` returns is preserved separately and carries
    the final aggregate result (with `done: True`).
    """

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss", float("nan"))
        train_loss = metrics.get("loss")
        if train_loss is None:
            train_loss = metrics.get("train_loss", float("nan"))
        tune.report(
            {
                "loss": eval_loss,
                "eval_loss": eval_loss,
                "train_loss": train_loss,
                "epoch": float(state.epoch) if state.epoch is not None else 0.0,
                "global_step": int(state.global_step),
                "done": False,
            }
        )
