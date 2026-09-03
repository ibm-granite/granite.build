"""Tests for :class:`AutotuneLocalTrainer`.

Ray and ``autotune`` are installed in this environment, so the lazy import guard
cannot be exercised by uninstalling them. Instead ``builtins.__import__`` is
monkeypatched to fail for ``ray``/``autotune`` while delegating every other import
to the real machinery, proving that :meth:`AutotuneLocalTrainer.run` translates a
missing training core into :class:`AutotuneCoreUnavailableError`.
"""

from __future__ import annotations

import builtins
import io
import logging
import os
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from uuid import UUID

import pytest

from autotunex.core.exceptions import AutotuneCoreUnavailableError
from autotunex.services.local.protocols import LocalRunContext, LogRecord, TrainingMetricRecord
from autotunex.services.local.trainer import (
    AutotuneLocalTrainer,
    _capture_run_logs,
    _capture_std_streams,
    _export_run_env,
    _run_cancel_watcher,
    _SinkStream,
    _trial_event_log,
    _TrialContext,
)
from autotunex.services.local.trainer import logger as trainer_logger


def _context() -> LocalRunContext:
    """Return a minimal, constructible run context for the guard test."""
    return LocalRunContext(
        job_id=UUID(int=1),
        model="m",
        model_source="huggingface",
        experiment_name="e",
        tuning_algo="lora",
        rl_algo="none",
        config_name="c",
        config_data={},
        train_file=Path("/t"),
        validation_file=Path("/v"),
        output_dir=Path("/o"),
        seed=1,
        autotune=True,
        cleanup=True,
        save_history=True,
        reward_function_code=None,
        reward_function_name=None,
    )


def test_run_raises_autotune_core_unavailable_when_ray_or_autotune_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globs: Mapping[str, object] | None = None,
        locs: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "ray" or name.startswith("autotune"):
            raise ImportError(name)
        return real_import(name, globs, locs, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    trainer = AutotuneLocalTrainer(ray_address=None)
    ctx = _context()

    with pytest.raises(AutotuneCoreUnavailableError):
        trainer.run(ctx, sink=object())  # type: ignore[arg-type]


class _RecordingSink:
    """A ``TrialSink`` that records the log lines flushed to it.

    Only ``log`` is exercised by these tests; the trial-lifecycle methods are
    present to satisfy the ``TrialSink`` Protocol structurally.
    """

    def __init__(self) -> None:
        self.records: list[LogRecord] = []

    def trial_started(self, trial_id: str, config: dict[str, object] | None) -> None:
        pass

    def trial_result(self, trial_id: str, metric: str, metrics: dict[str, object] | None) -> None:
        pass

    def trial_completed(self, trial_id: str) -> None:
        pass

    def trial_error(self, trial_id: str) -> None:
        pass

    def log(self, record: LogRecord) -> None:
        self.records.append(record)

    def training_metric(self, record: TrainingMetricRecord) -> None:
        pass


def test_capture_run_logs_captures_trainer_and_autotune_lines_as_job_level() -> None:
    sink = _RecordingSink()

    with _capture_run_logs(sink):
        trainer_logger.info("driver line")
        logging.getLogger("autotune.pipeline").info("autotune line")

    messages = [record.message for record in sink.records]
    assert "driver line" in messages
    assert "autotune line" in messages
    assert all(record.trial_id is None for record in sink.records)


def test_capture_run_logs_detaches_the_handler_on_exit() -> None:
    sink = _RecordingSink()

    with _capture_run_logs(sink):
        pass
    trainer_logger.info("after the run")

    assert sink.records == []


def test_capture_run_logs_flushes_buffered_lines_when_the_body_raises() -> None:
    sink = _RecordingSink()

    with pytest.raises(RuntimeError), _capture_run_logs(sink):
        trainer_logger.info("before boom")
        raise RuntimeError("boom")

    assert any(record.message == "before boom" for record in sink.records)


def test_trial_event_log_tags_the_trial_and_defaults_to_info() -> None:
    record = _trial_event_log("ray_00001", "Trial ray_00001 started")

    assert record.trial_id == "ray_00001"
    assert record.message == "Trial ray_00001 started"
    assert record.level == "INFO"


def test_trial_event_log_supports_an_error_level() -> None:
    record = _trial_event_log("ray_00001", "Trial ray_00001 errored", level="ERROR")

    assert record.level == "ERROR"


def test_sink_stream_forwards_complete_lines_tagged_with_the_current_trial() -> None:
    sink = _RecordingSink()
    context = _TrialContext()
    original = io.StringIO()
    stream = _SinkStream(sink, original, context=context)
    context.current_trial_id = "t7"

    stream.write("hello from a ray worker\n")

    assert any(
        record.message == "hello from a ray worker" and record.trial_id == "t7"
        for record in sink.records
    )
    assert "hello from a ray worker" in original.getvalue()  # passed through to the console


def test_sink_stream_buffers_partial_lines_until_a_newline() -> None:
    sink = _RecordingSink()
    stream = _SinkStream(sink, io.StringIO(), context=_TrialContext())

    stream.write("partial ")
    assert sink.records == []  # nothing forwarded without a line boundary

    stream.write("line complete\n")
    assert any(record.message == "partial line complete" for record in sink.records)


def test_sink_stream_flush_emits_the_residual_buffer() -> None:
    sink = _RecordingSink()
    stream = _SinkStream(sink, io.StringIO(), context=_TrialContext())

    stream.write("no trailing newline")
    stream.flush()

    assert any(record.message == "no trailing newline" for record in sink.records)


async def test_sink_stream_passes_through_without_sinking_on_the_loop_thread() -> None:
    sink = _RecordingSink()
    original = io.StringIO()
    stream = _SinkStream(sink, original, context=_TrialContext())

    # This test runs inside the event loop, so a running loop is present — the
    # sink write would deadlock, so it must be skipped and only passed through.
    stream.write("emitted on the loop thread\n")

    assert sink.records == []
    assert "emitted on the loop thread" in original.getvalue()


def test_capture_std_streams_captures_prints_and_restores_the_streams() -> None:
    sink = _RecordingSink()
    saved_out, saved_err = sys.stdout, sys.stderr

    with _capture_std_streams(sink, _TrialContext()):
        assert sys.stdout is not saved_out
        print("captured stdout line")

    assert sys.stdout is saved_out
    assert sys.stderr is saved_err
    assert any(record.message == "captured stdout line" for record in sink.records)


def test_capture_std_streams_restores_the_streams_when_the_body_raises() -> None:
    saved_out, saved_err = sys.stdout, sys.stderr

    with pytest.raises(RuntimeError), _capture_std_streams(_RecordingSink(), _TrialContext()):
        raise RuntimeError("boom")

    assert sys.stdout is saved_out
    assert sys.stderr is saved_err


def test_watcher_calls_shutdown_when_cancel_fires() -> None:
    cancel_evt, done_evt = threading.Event(), threading.Event()
    calls: list[bool] = []
    t = threading.Thread(
        target=_run_cancel_watcher, args=(cancel_evt, done_evt, lambda: calls.append(True), 0.01)
    )
    t.start()
    cancel_evt.set()
    t.join(timeout=2)

    assert calls == [True]


def test_watcher_exits_without_shutdown_on_normal_completion() -> None:
    cancel_evt, done_evt = threading.Event(), threading.Event()
    calls: list[bool] = []
    t = threading.Thread(
        target=_run_cancel_watcher, args=(cancel_evt, done_evt, lambda: calls.append(True), 0.01)
    )
    t.start()
    done_evt.set()
    t.join(timeout=2)

    assert calls == []


def test_export_run_env_stamps_the_job_id_for_fm_tunes_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUTOTUNE_JOB_ID", raising=False)

    _export_run_env("11111111-2222-3333-4444-555555555555")

    assert os.environ["AUTOTUNE_JOB_ID"] == "11111111-2222-3333-4444-555555555555"


def test_export_run_env_disables_ray_log_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ray's dedup collapses same-pattern worker lines, and every metric marker line
    # differs only in its numbers — leaving it on silently thins the series.
    monkeypatch.setenv("RAY_DEDUP_LOGS", "1")

    _export_run_env("job")

    assert os.environ["RAY_DEDUP_LOGS"] == "0"


def test_export_run_env_keeps_the_2025_ray_and_tokenizers_knobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAY_CHDIR_TO_TRIAL_DIR", raising=False)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)

    _export_run_env("job")

    assert os.environ["RAY_CHDIR_TO_TRIAL_DIR"] == "0"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
