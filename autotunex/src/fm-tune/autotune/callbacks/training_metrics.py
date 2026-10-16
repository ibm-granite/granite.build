# coding=utf-8
# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""HF TrainerCallback that persists per-step training metrics.

On every HF Trainer logging step, `on_log` receives the metrics dict
(`{'loss', 'grad_norm', 'learning_rate', 'epoch'}`). This callback turns it
into one structured row and hands it to `metrics_emitter.emit_metric_row`,
which POSTs it to the api-bridge (llmb) or prints one `@@FMTUNE_METRIC@@`
marker line (local / standalone). The callback never raises into the training
loop.
"""

from __future__ import annotations

import logging
import os

from transformers import TrainerCallback

from autotune.callbacks.logging_service import BufferedLogHandler
from autotune.callbacks.metrics_emitter import (
    METRIC_MARKER,
    close_metrics_handler,
    emit_metric_row,
    finite,
    make_metrics_handler,
)

# METRIC_MARKER is re-exported: existing code and tests import it from this module.
__all__ = ["METRIC_MARKER", "TrainingMetricsCallback"]

logger = logging.getLogger(__name__)

_KNOWN_KEYS = {"loss", "grad_norm", "learning_rate", "epoch"}


class TrainingMetricsCallback(TrainerCallback):
    """Emit one `training_metrics` row per HF logging step.

    Args:
        trial_id: The Ray Tune trial id for the current run (or None for a run
            with no trial context). Passed by the driver at construction so the
            callback need not reach into the tune context itself.
    """

    def __init__(self, trial_id: str | None = None) -> None:
        self._trial_id = trial_id
        self._job_id = os.environ.get("AUTOTUNE_JOB_ID")
        self._handler: BufferedLogHandler | None = make_metrics_handler(self._job_id)

    def _build_row(self, logs: dict, state) -> dict:
        # Every metric value goes through finite: NaN/Infinity are unstorable
        # downstream, and `extra` is passed through recursively since a trainer can
        # report anything there.
        return {
            "job_id": self._job_id,
            "trial_id": self._trial_id,
            "global_step": int(state.global_step),
            "epoch": finite(float(state.epoch)) if state.epoch is not None else None,
            "loss": finite(logs.get("loss")),
            "grad_norm": finite(logs.get("grad_norm")),
            "learning_rate": finite(logs.get("learning_rate")),
            "split": "eval" if "eval_loss" in logs else "train",
            "extra": {k: finite(v) for k, v in logs.items() if k not in _KNOWN_KEYS},
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Best-effort: a metrics emit must never crash training.
        try:
            if not logs or not getattr(state, "is_world_process_zero", True):
                return
            emit_metric_row(self._build_row(logs, state), self._handler)
        except Exception:
            logger.debug("TrainingMetricsCallback.on_log suppressed an error", exc_info=True)
            return

    def on_train_end(self, args, state, control, **kwargs):
        """Release the HTTP handler at the end of the run (see `close_metrics_handler`)."""
        handler, self._handler = self._handler, None
        close_metrics_handler(handler)
