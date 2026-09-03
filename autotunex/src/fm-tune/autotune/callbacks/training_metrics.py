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
into one structured row and emits it toward AutotuneX by whichever channel the
current backend provides:

* llmb (remote): if `AUTOTUNE_ENDPOINT_URL` is set, POST the row to the
  api-bridge via `BufferedLogHandler.record_data(row, RECORD_METRICS)`.
* local / standalone: otherwise print one marked line,
  `@@FMTUNE_METRIC@@ {json}`, which AutotuneX's local `_SinkStream` parses.

`@@FMTUNE_METRIC@@` is a cross-repo contract shared with AutotuneX's
`FMTUNE_METRIC_MARKER`. The callback never raises into the training loop.
"""

from __future__ import annotations

import json
import logging
import math
import os
from contextlib import suppress

from transformers import TrainerCallback

from autotune.callbacks.logging_service import BufferedLogHandler, RecordType

logger = logging.getLogger(__name__)

METRIC_MARKER = "@@FMTUNE_METRIC@@"

_KNOWN_KEYS = {"loss", "grad_norm", "learning_rate", "epoch"}


def _finite(value):
    """Return `value` unless it is a non-finite float, in which case None.

    A diverging run reports `loss=nan` (or `inf`), and `json.dumps` serializes
    those as the non-standard `NaN`/`Infinity` tokens. Both consumers parse them
    back to Python floats, and storing one then fails: AutotuneX's
    `training_metrics.loss` is a MySQL DOUBLE, whose bind formats NaN as the bare
    token `nan` and errors. Because both write paths swallow that error, the row —
    and on the remote path the whole batch — would be lost silently, at exactly
    the step worth charting. Emitting null instead keeps the rest of the row.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


class TrainingMetricsCallback(TrainerCallback):
    """Emit one `training_metrics` row per HF logging step.

    Args:
        trial_id: The Ray Tune trial id for the current run (or None for a run
            with no trial context). Passed by the driver at construction so the
            callback need not reach into the tune context itself.
    """

    def __init__(self, trial_id: str | None = None) -> None:
        self._trial_id = trial_id
        self._endpoint = os.environ.get("AUTOTUNE_ENDPOINT_URL")
        self._job_id = os.environ.get("AUTOTUNE_JOB_ID")
        self._handler: BufferedLogHandler | None = None
        if self._endpoint:
            # No flush_interval => no background timer; record_data POSTs directly.
            self._handler = BufferedLogHandler(job_id=self._job_id, endpoint_url=self._endpoint)

    def _build_row(self, logs: dict, state) -> dict:
        # Every metric value goes through _finite: NaN/Infinity are unstorable
        # downstream, and `extra` is passed through recursively since a trainer can
        # report anything there.
        return {
            "job_id": self._job_id,
            "trial_id": self._trial_id,
            "global_step": int(state.global_step),
            "epoch": _finite(float(state.epoch)) if state.epoch is not None else None,
            "loss": _finite(logs.get("loss")),
            "grad_norm": _finite(logs.get("grad_norm")),
            "learning_rate": _finite(logs.get("learning_rate")),
            "split": "eval" if "eval_loss" in logs else "train",
            "extra": {k: _finite(v) for k, v in logs.items() if k not in _KNOWN_KEYS},
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Best-effort: a metrics emit must never crash training.
        try:
            if not logs or not getattr(state, "is_world_process_zero", True):
                return
            row = self._build_row(logs, state)
            if self._handler is not None:
                self._handler.record_data(row, RecordType.RECORD_METRICS)
            else:
                print(f"{METRIC_MARKER} {json.dumps(row, default=str)}", flush=True)
        except Exception:
            logger.debug("TrainingMetricsCallback.on_log suppressed an error", exc_info=True)
            return

    def on_train_end(self, args, state, control, **kwargs):
        """Release the HTTP handler at the end of the run.

        `BufferedLogHandler` is a `logging.Handler`, and `logging.Handler.__init__`
        registers every instance in the module-global `logging._handlerList`. One
        callback is constructed per trial, so without this the handlers accumulate
        for the lifetime of the driver process — a slow leak across a long sweep.
        """
        handler, self._handler = self._handler, None
        if handler is not None:
            with suppress(Exception):
                handler.close()
