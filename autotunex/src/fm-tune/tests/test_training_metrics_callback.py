import json
import types

from autotune.callbacks.logging_service import RecordType
from autotune.callbacks.training_metrics import METRIC_MARKER, TrainingMetricsCallback


def _state(global_step=10, epoch=0.04, is_zero=True):
    return types.SimpleNamespace(global_step=global_step, epoch=epoch, is_world_process_zero=is_zero)


def test_record_type_has_record_metrics():
    assert RecordType.RECORD_METRICS.value == "record_metrics"


def test_build_row_maps_known_fields_and_collects_extra():
    cb = TrainingMetricsCallback(trial_id="t1")
    logs = {"loss": 15.3477, "grad_norm": 2.85, "learning_rate": 6.5e-07, "epoch": 0.04}
    row = cb._build_row(logs, _state())
    assert row["trial_id"] == "t1"
    assert row["global_step"] == 10
    assert row["loss"] == 15.3477
    assert row["grad_norm"] == 2.85
    assert row["learning_rate"] == 6.5e-07
    assert row["split"] == "train"
    assert row["extra"] == {}


def test_build_row_marks_eval_split_and_keeps_eval_loss_in_extra():
    cb = TrainingMetricsCallback(trial_id="t1")
    row = cb._build_row({"eval_loss": 1.2, "eval_runtime": 3.0}, _state())
    assert row["split"] == "eval"
    assert row["extra"]["eval_loss"] == 1.2
    assert row["extra"]["eval_runtime"] == 3.0


def test_on_log_prints_marker_when_no_endpoint(monkeypatch, capsys):
    monkeypatch.delenv("AUTOTUNE_ENDPOINT_URL", raising=False)
    cb = TrainingMetricsCallback(trial_id="t1")
    cb.on_log(None, _state(), None, logs={"loss": 1.0, "epoch": 0.5})
    out = capsys.readouterr().out.strip().splitlines()
    marker_lines = [ln for ln in out if METRIC_MARKER in ln]
    assert len(marker_lines) == 1
    payload = json.loads(marker_lines[0].split(METRIC_MARKER, 1)[1].strip())
    assert payload["loss"] == 1.0
    assert payload["trial_id"] == "t1"


def test_on_log_posts_when_endpoint_set(monkeypatch):
    monkeypatch.setenv("AUTOTUNE_ENDPOINT_URL", "http://x/fmtune/api")
    monkeypatch.setenv("AUTOTUNE_JOB_ID", "job-1")
    cb = TrainingMetricsCallback(trial_id="t1")
    calls = []
    cb._handler = types.SimpleNamespace(record_data=lambda data, record_type: calls.append((data, record_type)))
    cb.on_log(None, _state(), None, logs={"loss": 1.0})
    assert len(calls) == 1
    data, record_type = calls[0]
    assert record_type is RecordType.RECORD_METRICS
    assert data["loss"] == 1.0
    assert data["job_id"] == "job-1"


def test_on_log_noop_off_main_process(monkeypatch, capsys):
    monkeypatch.delenv("AUTOTUNE_ENDPOINT_URL", raising=False)
    cb = TrainingMetricsCallback(trial_id="t1")
    cb.on_log(None, _state(is_zero=False), None, logs={"loss": 1.0})
    assert METRIC_MARKER not in capsys.readouterr().out


def test_on_log_swallows_build_row_error(monkeypatch):
    monkeypatch.delenv("AUTOTUNE_ENDPOINT_URL", raising=False)
    cb = TrainingMetricsCallback(trial_id="t1")

    def _boom(logs, state):
        raise RuntimeError("boom")

    cb._build_row = _boom
    result = cb.on_log(None, _state(), None, logs={"loss": 1.0})
    assert result is None


def test_on_log_swallows_record_data_error(monkeypatch):
    monkeypatch.setenv("AUTOTUNE_ENDPOINT_URL", "http://x/fmtune/api")
    monkeypatch.setenv("AUTOTUNE_JOB_ID", "job-1")
    cb = TrainingMetricsCallback(trial_id="t1")

    def _boom(data, record_type):
        raise RuntimeError("boom")

    cb._handler = types.SimpleNamespace(record_data=_boom)
    result = cb.on_log(None, _state(), None, logs={"loss": 1.0})
    assert result is None


def test_build_row_replaces_a_diverged_nan_loss_with_none():
    """NaN is unstorable downstream (MySQL DOUBLE), and both writers swallow the error."""
    cb = TrainingMetricsCallback(trial_id="t1")

    row = cb._build_row({"loss": float("nan"), "grad_norm": 2.0}, _state())

    assert row["loss"] is None
    assert row["grad_norm"] == 2.0
    assert row["global_step"] == 10


def test_build_row_replaces_infinite_values_with_none():
    cb = TrainingMetricsCallback(trial_id="t1")

    row = cb._build_row({"loss": 1.0, "grad_norm": float("inf")}, _state())

    assert row["grad_norm"] is None
    assert row["loss"] == 1.0


def test_build_row_sanitizes_nested_extra_values():
    cb = TrainingMetricsCallback(trial_id="t1")

    row = cb._build_row({"custom": {"inner": float("nan")}, "seq": [float("inf"), 2.0]}, _state())

    assert row["extra"]["custom"] == {"inner": None}
    assert row["extra"]["seq"] == [None, 2.0]


def test_emitted_marker_line_carries_no_nan_token(monkeypatch, capsys):
    """`json.dumps` writes a bare `NaN` token, which is what breaks the consumers."""
    monkeypatch.delenv("AUTOTUNE_ENDPOINT_URL", raising=False)
    cb = TrainingMetricsCallback(trial_id="t1")

    cb.on_log(None, _state(), None, logs={"loss": float("nan"), "epoch": 0.5})

    line = [ln for ln in capsys.readouterr().out.splitlines() if METRIC_MARKER in ln][0]
    assert "NaN" not in line
    assert json.loads(line.split(METRIC_MARKER, 1)[1].strip())["loss"] is None


def test_on_train_end_closes_and_drops_the_handler(monkeypatch):
    """logging.Handler registers itself globally, so one per trial must be closed."""
    monkeypatch.setenv("AUTOTUNE_ENDPOINT_URL", "http://x/fmtune/api")
    cb = TrainingMetricsCallback(trial_id="t1")
    closed = []
    cb._handler = types.SimpleNamespace(close=lambda: closed.append(True))

    cb.on_train_end(None, _state(), None)

    assert closed == [True]
    assert cb._handler is None


def test_on_train_end_is_a_noop_without_a_handler(monkeypatch):
    monkeypatch.delenv("AUTOTUNE_ENDPOINT_URL", raising=False)
    cb = TrainingMetricsCallback(trial_id="t1")

    cb.on_train_end(None, _state(), None)

    assert cb._handler is None


def test_on_train_end_survives_a_failing_close(monkeypatch):
    """Teardown must not raise into the training loop any more than on_log does."""
    monkeypatch.setenv("AUTOTUNE_ENDPOINT_URL", "http://x/fmtune/api")
    cb = TrainingMetricsCallback(trial_id="t1")

    def _boom():
        raise RuntimeError("endpoint gone")

    cb._handler = types.SimpleNamespace(close=_boom)

    cb.on_train_end(None, _state(), None)

    assert cb._handler is None
