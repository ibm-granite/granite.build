# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Delivery of per-step training-metric rows toward AutotuneX.

Each row reaches AutotuneX's `training_metrics` table by whichever channel the
current backend provides:

* llmb (remote): if `AUTOTUNE_ENDPOINT_URL` is set, POST the row to the
  api-bridge via `BufferedLogHandler.record_data(row, RECORD_METRICS)`.
* local / standalone: otherwise print one marked line,
  `@@FMTUNE_METRIC@@ {json}`, which AutotuneX's local `_SinkStream` parses.

Shared by the HF `TrainingMetricsCallback` and the verl step reporter, so the
channel choice and the NaN rule live in one place. `@@FMTUNE_METRIC@@` is a
cross-repo contract shared with AutotuneX's `FMTUNE_METRIC_MARKER`.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import suppress

from autotune.callbacks.logging_service import BufferedLogHandler, RecordType

METRIC_MARKER = "@@FMTUNE_METRIC@@"


def finite(value):
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
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    return value


def make_metrics_handler(job_id: str | None) -> BufferedLogHandler | None:
    """Build the api-bridge handler when `AUTOTUNE_ENDPOINT_URL` is set, else None.

    No `flush_interval`, so no background timer: `record_data` POSTs each row
    directly rather than through the log buffer.
    """
    endpoint = os.environ.get("AUTOTUNE_ENDPOINT_URL")
    if not endpoint:
        return None
    return BufferedLogHandler(job_id=job_id, endpoint_url=endpoint)


def emit_metric_row(row: dict, handler: BufferedLogHandler | None) -> None:
    """Send one metrics row: POST via `handler`, or print a marker line without one."""
    if handler is not None:
        handler.record_data(row, RecordType.RECORD_METRICS)
    else:
        print(f"{METRIC_MARKER} {json.dumps(row, default=str)}", flush=True)


def close_metrics_handler(handler: BufferedLogHandler | None) -> None:
    """Close a handler from `make_metrics_handler`, never raising.

    `BufferedLogHandler` is a `logging.Handler`, and `logging.Handler.__init__`
    registers every instance in the module-global `logging._handlerList`. One
    handler is built per trial, so without this they accumulate for the lifetime
    of the process — a slow leak across a long sweep.
    """
    if handler is None:
        return
    with suppress(Exception):
        handler.close()
