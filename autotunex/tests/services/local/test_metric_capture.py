# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Marked per-step metric lines are routed to ``sink.training_metric``.

Drives :class:`_SinkStream` directly (no Ray, no DB) to prove the routing
contract in :meth:`_SinkStream._forward`: a line carrying
``FMTUNE_METRIC_MARKER`` is parsed as JSON and forwarded as a
:class:`TrainingMetricRecord` — never double-written as a log line — while an
unmarked or malformed-marker line falls back to ``sink.log`` unchanged.
"""

from __future__ import annotations

import io
import json

from autotunex.services.local.protocols import LogRecord, TrainingMetricRecord
from autotunex.services.local.trainer import FMTUNE_METRIC_MARKER, _SinkStream, _TrialContext


class _FakeSink:
    """A ``TrialSink`` that records logged and metric rows; lifecycle is a no-op.

    Only ``training_metric``/``log`` are exercised by these tests; the
    trial-lifecycle methods are present to satisfy the ``TrialSink`` Protocol
    structurally (mirroring ``test_trainer.py``'s ``_RecordingSink``).
    """

    def __init__(self) -> None:
        self.metrics: list[TrainingMetricRecord] = []
        self.logs: list[LogRecord] = []

    def trial_started(self, trial_id: str, config: dict[str, object] | None) -> None:
        pass

    def trial_result(self, trial_id: str, metric: str, metrics: dict[str, object] | None) -> None:
        pass

    def trial_completed(self, trial_id: str) -> None:
        pass

    def trial_error(self, trial_id: str) -> None:
        pass

    def training_metric(self, record: TrainingMetricRecord) -> None:
        self.metrics.append(record)

    def log(self, record: LogRecord) -> None:
        self.logs.append(record)


def test_marked_line_becomes_a_training_metric() -> None:
    sink = _FakeSink()
    ctx = _TrialContext()
    ctx.current_trial_id = "ctx-trial"
    stream = _SinkStream(sink, io.StringIO(), context=ctx)
    row = {
        "trial_id": "t1",
        "global_step": 10,
        "epoch": 0.04,
        "loss": 1.5,
        "grad_norm": 2.0,
        "learning_rate": 1e-6,
        "split": "train",
        "extra": {},
    }

    stream.write(f"{FMTUNE_METRIC_MARKER} {json.dumps(row)}\n")

    assert len(sink.metrics) == 1
    assert sink.metrics[0].trial_id == "t1"  # payload wins over ctx
    assert sink.metrics[0].loss == 1.5
    assert sink.logs == []  # not double-written as a log


def test_unmarked_line_stays_a_log() -> None:
    sink = _FakeSink()
    stream = _SinkStream(sink, io.StringIO(), context=_TrialContext())

    stream.write("plain training output\n")

    assert sink.metrics == []
    assert len(sink.logs) == 1


def test_malformed_marker_falls_back_to_log() -> None:
    sink = _FakeSink()
    stream = _SinkStream(sink, io.StringIO(), context=_TrialContext())

    stream.write(f"{FMTUNE_METRIC_MARKER} not-json\n")

    assert sink.metrics == []
    assert len(sink.logs) == 1


def test_non_dict_payload_falls_back_to_log() -> None:
    """Valid JSON that isn't an object (e.g. a list) must still be logged, not dropped."""
    sink = _FakeSink()
    stream = _SinkStream(sink, io.StringIO(), context=_TrialContext())

    stream.write(f"{FMTUNE_METRIC_MARKER} [1,2]\n")

    assert sink.metrics == []
    assert len(sink.logs) == 1


def test_bad_global_step_falls_back_to_log() -> None:
    """A dict payload with a non-coercible ``global_step`` must fall back, not drop."""
    sink = _FakeSink()
    stream = _SinkStream(sink, io.StringIO(), context=_TrialContext())
    row = {"trial_id": "t1", "global_step": None, "loss": 1.0}

    stream.write(f"{FMTUNE_METRIC_MARKER} {json.dumps(row)}\n")

    assert sink.metrics == []
    assert len(sink.logs) == 1


def test_missing_global_step_falls_back_to_log() -> None:
    """A payload with no ``global_step`` key is not a metric row — recording step 0 would lie."""
    sink = _FakeSink()
    stream = _SinkStream(sink, io.StringIO(), context=_TrialContext())

    stream.write(f'{FMTUNE_METRIC_MARKER} {{"trial_id": "t1", "loss": 1.0}}\n')

    assert sink.metrics == []
    assert len(sink.logs) == 1


def test_nan_loss_is_recorded_as_none_rather_than_lost() -> None:
    """A diverging loss is exactly when the chart matters, and NaN breaks a MySQL DOUBLE bind."""
    sink = _FakeSink()
    stream = _SinkStream(sink, io.StringIO(), context=_TrialContext())
    # json.dumps emits the non-standard `NaN` token, which json.loads round-trips —
    # this is byte-for-byte what fm-tune's callback prints for a diverged step.
    row = {"trial_id": "t1", "global_step": 5, "loss": float("nan"), "epoch": 1.0}

    stream.write(f"{FMTUNE_METRIC_MARKER} {json.dumps(row)}\n")

    assert len(sink.metrics) == 1
    assert sink.metrics[0].loss is None
    assert sink.metrics[0].epoch == 1.0  # the rest of the row survives
    assert sink.metrics[0].global_step == 5


def test_infinite_grad_norm_is_recorded_as_none() -> None:
    sink = _FakeSink()
    stream = _SinkStream(sink, io.StringIO(), context=_TrialContext())
    row = {"global_step": 6, "grad_norm": float("inf"), "loss": 2.0}

    stream.write(f"{FMTUNE_METRIC_MARKER} {json.dumps(row)}\n")

    assert len(sink.metrics) == 1
    assert sink.metrics[0].grad_norm is None
    assert sink.metrics[0].loss == 2.0


def test_non_dict_extra_is_dropped_rather_than_stored() -> None:
    """``extra`` reads back through ``MetricPointRead``, whose dict annotation would 500."""
    sink = _FakeSink()
    stream = _SinkStream(sink, io.StringIO(), context=_TrialContext())
    row = {"global_step": 7, "extra": ["not", "a", "dict"]}

    stream.write(f"{FMTUNE_METRIC_MARKER} {json.dumps(row)}\n")

    assert len(sink.metrics) == 1
    assert sink.metrics[0].extra is None


def test_overlong_split_is_truncated_to_the_column_width() -> None:
    """``training_metrics.split`` is VARCHAR(16); MySQL errors on a longer value."""
    sink = _FakeSink()
    stream = _SinkStream(sink, io.StringIO(), context=_TrialContext())
    row = {"global_step": 8, "split": "x" * 40}

    stream.write(f"{FMTUNE_METRIC_MARKER} {json.dumps(row)}\n")

    assert len(sink.metrics) == 1
    assert sink.metrics[0].split == "x" * 16
