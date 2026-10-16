import json
import types

from autotune.callbacks.logging_service import BufferedLogHandler, RecordType
from autotune.callbacks.metrics_emitter import (
    METRIC_MARKER,
    close_metrics_handler,
    emit_metric_row,
    finite,
    make_metrics_handler,
)


def test_finite_replaces_non_finite_floats_with_none():
    assert finite(float("nan")) is None
    assert finite(float("inf")) is None
    assert finite(float("-inf")) is None


def test_finite_keeps_finite_numbers_and_non_numbers():
    assert finite(1.5) == 1.5
    assert finite(3) == 3
    assert finite("x") == "x"
    assert finite(None) is None


def test_finite_recurses_into_dicts_and_lists():
    assert finite({"a": float("nan"), "b": [1.0, float("inf")]}) == {"a": None, "b": [1.0, None]}


def test_make_metrics_handler_is_none_without_an_endpoint(monkeypatch):
    monkeypatch.delenv("AUTOTUNE_ENDPOINT_URL", raising=False)

    assert make_metrics_handler("job-1") is None


def test_make_metrics_handler_builds_a_bridge_handler_with_an_endpoint(monkeypatch):
    monkeypatch.setenv("AUTOTUNE_ENDPOINT_URL", "http://x/fmtune/api")

    handler = make_metrics_handler("job-1")

    try:
        assert isinstance(handler, BufferedLogHandler)
        assert handler.get_job_id() == "job-1"
    finally:
        close_metrics_handler(handler)


def test_emit_metric_row_prints_one_marker_line_without_a_handler(capsys):
    emit_metric_row({"global_step": 3, "loss": 1.0}, None)

    lines = [ln for ln in capsys.readouterr().out.splitlines() if METRIC_MARKER in ln]
    assert len(lines) == 1
    assert json.loads(lines[0].split(METRIC_MARKER, 1)[1]) == {"global_step": 3, "loss": 1.0}


def test_emit_metric_row_posts_through_the_handler_when_given_one():
    calls = []
    handler = types.SimpleNamespace(record_data=lambda data, record_type: calls.append((data, record_type)))

    emit_metric_row({"global_step": 3}, handler)

    assert calls == [({"global_step": 3}, RecordType.RECORD_METRICS)]


def test_close_metrics_handler_ignores_none_and_a_failing_close():
    def _boom():
        raise RuntimeError("endpoint gone")

    close_metrics_handler(None)
    close_metrics_handler(types.SimpleNamespace(close=_boom))
