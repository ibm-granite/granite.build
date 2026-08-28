# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The only :class:`~autotunex.services.local.protocols.LocalTrainer` implementation.

:class:`AutotuneLocalTrainer` drives the ``autotune`` HPO pipeline (Ray Tune) end
to end in-process and forwards every trial lifecycle event to the injected
:class:`~autotunex.services.local.protocols.TrialSink`. It is a faithful port of
the 2025 ``LocalRunner.run``/``CustomLoggerCallback`` pair, adapted to this repo's
seams, ``pathlib`` paths, and the committed packaging helpers.

Ray and ``autotune`` are the only heavy, optional dependencies here, so — exactly
like ``services/autotune.py`` — they are imported **lazily inside** :meth:`run`
and their absence is surfaced as a runtime
:class:`~autotunex.core.exceptions.AutotuneCoreUnavailableError`, never an
import-time crash. This keeps the module importable (and the rest of the test
suite runnable) on a credential-free install that never selects ``job_backend=local``.
The Ray-callback class is likewise defined inside :meth:`run`, because its base
class (``ray.tune.Callback``) only exists after that lazy import.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import threading
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

from autotunex.core.exceptions import AutotuneCoreUnavailableError
from autotunex.core.logging import get_logger
from autotunex.services.local.cancellation import cancel_event, is_cancelled
from autotunex.services.local.packaging import (
    generate_bash_script,
    generate_install_bash_script,
    generate_readme,
    parse_result,
    write_inference_script,
    zip_folder,
)
from autotunex.services.local.protocols import (
    LocalRunContext,
    LogRecord,
    TrialSink,
    inject_reward_function,
)
from autotunex.services.local.sink import SinkLogHandler

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from typing import TextIO

logger = get_logger(__name__)


class AutotuneLocalTrainer:
    """Runs the ``autotune`` HPO pipeline in-process, reporting to a ``TrialSink``.

    Satisfies :class:`~autotunex.services.local.protocols.LocalTrainer`
    structurally. The Ray cluster address is owned here (injected from settings in
    ``api/deps.py``) rather than by the runner or the run context: ``None`` means a
    local ``ray.init()``; any other value connects to that address.
    """

    def __init__(self, *, ray_address: str | None) -> None:
        """Store the Ray cluster address used by :meth:`run`.

        Args:
            ray_address: Address passed to ``ray.init(address=...)``. ``None`` starts
                (or attaches to) a local Ray instance via a bare ``ray.init()``.
        """
        self._ray_address = ray_address

    def run(self, ctx: LocalRunContext, sink: TrialSink) -> None:
        """Run the pipeline described by ``ctx``, forwarding events to ``sink``.

        Imports Ray and ``autotune`` lazily; builds the config/pipeline/optimizer,
        runs the search and the best-config training, then packages the tuned
        artifacts. Ray is always shut down in a ``finally``.

        ``AutotuneOptimizer.fit()`` exposes no ``Stopper`` hook, so a cooperative
        cancel is enforced by tearing Ray down instead: a daemon watcher thread
        (:func:`_run_cancel_watcher`) calls ``ray.shutdown()`` when the run's
        cancel event (``services.local.cancellation``) fires, which makes
        ``fit()`` raise. The method then returns early rather than raising, so a
        cancelled run is distinguishable (via ``is_cancelled``) from a genuine
        failure by its caller.

        Args:
            ctx: The frozen run context assembled from the job and its snapshot.
            sink: Receives trial lifecycle events from the Ray callback threads.

        Raises:
            AutotuneCoreUnavailableError: Ray or ``autotune`` is not installed.
            RuntimeError: A trial in the search grid finished with an error.
        """
        try:
            import ray
            from autotune.config import AutotuneConfig
            from autotune.optimizer import AutotuneOptimizer
            from autotune.pipeline import AutotunePipeline
            from autotune.utils import (
                cleanup,
                generate_unique_id,
                save_hpo_history,
                set_seed,
            )
            from autotune.validation import validate_config_for_pipeline
            from ray.tune import Callback
            from ray.tune.experiment.trial import Trial
        except ImportError as exc:
            raise AutotuneCoreUnavailableError() from exc

        # ``Callback`` resolves to ``Any`` under mypy (ray is follow_imports=skip in
        # pyproject.toml), and strict mode forbids subclassing ``Any``.
        class _SinkCallback(Callback):  # type: ignore[misc]
            """Ray Tune callback forwarding trial lifecycle events to ``sink``.

            The local analogue of the 2025 ``CustomLoggerCallback``: instead of
            writing to a sync DB it calls the injected ``TrialSink``, which bridges
            each call back to the async repositories on the event loop.

            Ray Tune pickles ``RunConfig.callbacks`` into the experiment's
            ``tuner.pkl`` at ``Tuner`` construction, so this must stay picklable.
            It holds nothing unpicklable of its own (``_TrialContext`` is just a
            string); ``DbTrialSink.__getstate__`` handles the loop/engine it would
            otherwise drag in. Do not add unpicklable state (a live client, a lock)
            here without the same care.

            Besides forwarding lifecycle events, it drives ``context`` — setting
            the current trial id on start and clearing it on complete/error — so
            :class:`_SinkStream` can tag Ray's forwarded worker output with the
            trial that produced it.
            """

            def __init__(self, target: TrialSink, context: _TrialContext) -> None:
                super().__init__()
                self._sink = target
                self._context = context

            def on_trial_start(
                self,
                iteration: int,
                trials: list[Trial],
                trial: Trial,
                **info: object,
            ) -> None:
                self._context.current_trial_id = trial.trial_id
                self._sink.trial_started(trial.trial_id, trial.config)
                self._sink.log(_trial_event_log(trial.trial_id, f"Trial {trial.trial_id} started"))

            def on_trial_result(
                self,
                iteration: int,
                trials: list[Trial],
                trial: Trial,
                result: dict[str, Any],
                **info: object,
            ) -> None:
                metrics = parse_result({**result, "metric": "loss"})
                self._sink.trial_result(trial.trial_id, "loss", metrics)
                self._sink.log(
                    _trial_event_log(
                        trial.trial_id, f"Trial {trial.trial_id} reported result: {metrics}"
                    )
                )

            def on_trial_complete(
                self,
                iteration: int,
                trials: list[Trial],
                trial: Trial,
                **info: object,
            ) -> None:
                self._sink.trial_completed(trial.trial_id)
                self._sink.log(
                    _trial_event_log(trial.trial_id, f"Trial {trial.trial_id} completed")
                )
                self._context.current_trial_id = None

            def on_trial_error(
                self,
                iteration: int,
                trials: list[Trial],
                trial: Trial,
                **info: object,
            ) -> None:
                self._sink.trial_error(trial.trial_id)
                self._sink.log(
                    _trial_event_log(
                        trial.trial_id, f"Trial {trial.trial_id} errored", level="ERROR"
                    )
                )
                self._context.current_trial_id = None

        # Faithful to the 2025 runner: keep Ray from chdir-ing into each trial dir
        # (so relative output paths resolve against output_dir) and silence the
        # tokenizers fork warning. These are Ray/tokenizers runtime knobs, not
        # AutoTuneX settings, so writing them here does not violate the settings rule.
        os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Online RL: persist the reward function and point the snapshot at it.
        if ctx.reward_function_code is not None:
            inject_reward_function(
                ctx.config_data,
                output_dir=ctx.output_dir,
                code=ctx.reward_function_code,
                name=ctx.reward_function_name,
            )

        set_seed(ctx.seed)

        # Capture everything this run emits as `log_entries` rows for the whole
        # run, on this worker thread so the flush drives the worker-thread-only
        # sink safely:
        #   * `_capture_run_logs` takes the driver-side Python-logger lines (this
        #     trainer's own and the `autotune` pipeline's) as job-level rows;
        #   * `_capture_std_streams` redirects stdout/stderr so Ray's
        #     `log_to_driver` output — every trial worker's stdout/stderr, forwarded
        #     to this driver — is persisted too, tagged with the running trial via
        #     `context` (the piece the 2025 runner's `PrintLogger` provided).
        context = _TrialContext()
        with _capture_run_logs(sink), _capture_std_streams(sink, context):
            done_evt = threading.Event()
            watcher: threading.Thread | None = None
            try:
                if self._ray_address is not None:
                    if not ray.is_initialized():
                        ray.init(address=self._ray_address)
                elif not ray.is_initialized():
                    ray.init()

                # Cooperative cancellation: a watcher tears Ray down if a cancel is
                # requested, so fit() raises and `process` records `terminated`.
                cancel_evt = cancel_event(ctx.job_id)
                if cancel_evt is not None:
                    watcher = threading.Thread(
                        target=_run_cancel_watcher,
                        args=(cancel_evt, done_evt, ray.shutdown),
                        name=f"autotunex-cancel-watch-{ctx.job_id}",
                        daemon=True,
                    )
                    watcher.start()

                if ctx.cleanup:
                    cleanup(str(ctx.output_dir / "ray_results"))

                # from_dict mutates in place and returns None, so build then populate —
                # do not chain (matching the installed autotune API used in 2025).
                config = AutotuneConfig()
                config.from_dict(config=ctx.config_data)

                pipeline = AutotunePipeline(
                    tuning_algo=ctx.tuning_algo,
                    rl_algo=ctx.rl_algo,
                    model_name_or_path=ctx.model,
                )
                validate_config_for_pipeline(
                    config,
                    tuning_algo=pipeline.get_tuning_algo(),
                    rl_algo=pipeline.get_rl_algo(),
                )

                metric = config.get_metric()
                mode = config.get_mode()
                run_id = generate_unique_id()
                logger.info("Starting local HPO run %s for job %s", run_id, ctx.job_id)

                optimizer = AutotuneOptimizer(
                    pipeline=pipeline,
                    config=config,
                    train_file=str(ctx.train_file),
                    validation_file=str(ctx.validation_file),
                    output_dir=str(ctx.output_dir),
                    output_model_name=ctx.experiment_name,
                    resume_from_checkpoint=False,
                    keep_checkpoints=False,
                    cluster_resources=ray.cluster_resources(),
                    run_id=run_id,
                    tuner_callbacks=[_SinkCallback(sink, context)],
                )

                result_grid = optimizer.fit()
                if is_cancelled(ctx.job_id):
                    logger.info(
                        "Job %s cancelled during search; skipping best-config run.", ctx.job_id
                    )
                    return
                for result in result_grid:
                    if result.error:
                        raise RuntimeError(result.error)

                if ctx.save_history:
                    save_hpo_history(
                        result_grid=result_grid,
                        metric=metric,
                        mode=mode,
                        output_dir=str(ctx.output_dir),
                        run_name=ctx.experiment_name,
                    )

                best_grid = optimizer.fit_best_config()
                best_result = best_grid.get_best_result(metric=metric, mode=mode)
                logger.info("Trained best config with result: %s", best_result.metrics)

                _package_artifacts(ctx)

                if ctx.cleanup:
                    cleanup(str(ctx.output_dir / "ray_results"))
            finally:
                done_evt.set()  # let the watcher exit without tearing down
                if watcher is not None:
                    watcher.join(timeout=2 * _CANCEL_POLL_SECONDS)
                ray.shutdown()


_CANCEL_POLL_SECONDS = 0.5


def _run_cancel_watcher(
    cancel_evt: threading.Event,
    done_evt: threading.Event,
    shutdown_fn: Callable[[], None],
    poll: float = _CANCEL_POLL_SECONDS,
) -> None:
    """Tear Ray down when cancellation fires, else exit on normal completion.

    Runs on a daemon thread for the duration of a local run. Blocks on
    ``cancel_evt`` in ``poll``-second slices; if it fires, calls ``shutdown_fn``
    (``ray.shutdown``) so the running ``optimizer.fit()`` raises. ``done_evt`` is
    set by the run on normal completion, which lets the watcher exit *without*
    tearing anything down — the watcher never sets ``cancel_evt`` itself, so a
    clean run is never mis-read as cancelled.
    """
    while not done_evt.is_set():
        if cancel_evt.wait(timeout=poll):
            shutdown_fn()
            return


@contextmanager
def _capture_run_logs(sink: TrialSink) -> Iterator[None]:
    """Persist this run's driver-side logs as job-level ``log_entries`` rows.

    Attaches a :class:`~autotunex.services.local.sink.SinkLogHandler` (bound to
    ``trial_id=None``, so its lines are job-level) to the ``autotune`` pipeline
    logger and this module's trainer logger for the duration of the run — the
    async-DB analogue of 2025's root-attached ``BufferedLogHandler``.

    Only those two loggers are targeted, never the root: unlike the 2025
    standalone CLI, this runner shares its process with the FastAPI app, so a
    root handler would vacuum every unrelated request's log lines into this job's
    ``log_entries`` — and risk a flush on the event-loop thread, which
    :class:`~autotunex.services.local.sink.DbTrialSink` forbids. Each target's
    level is lowered to ``INFO`` while attached (restored on exit) so a run's
    lines are captured regardless of the ambient logging configuration.

    The handler is removed and closed in a ``finally`` — ``close`` flushes the
    tail of the buffer — so a run that raises still persists what it logged
    before failing. Enter this on the worker thread the trainer runs on: the
    flush drives the worker-thread-only sink, and a flush from the event-loop
    thread would deadlock.
    """
    handler = SinkLogHandler(sink, trial_id=None)
    targets = [logger, logging.getLogger("autotune")]
    previous_levels = {target: target.level for target in targets}
    for target in targets:
        if target.level == logging.NOTSET or target.level > logging.INFO:
            target.setLevel(logging.INFO)
        target.addHandler(handler)
    try:
        yield
    finally:
        for target in targets:
            target.removeHandler(handler)
            target.setLevel(previous_levels[target])
        handler.close()


def _trial_event_log(trial_id: str, message: str, *, level: str = "INFO") -> LogRecord:
    """Build a trial-tagged :class:`LogRecord` for a Ray lifecycle event.

    The trial-level analogue of the job-level lines captured by
    :func:`_capture_run_logs`: :class:`_SinkCallback` emits one of these through
    the sink at each lifecycle edge, so ``log_entries`` records the trial's
    progress tagged with its ``trial_id`` (the sink coerces the raw Ray id to the
    stored short code). ``iteration`` / ``epoch`` are ``None`` — a lifecycle edge
    carries neither.
    """
    return LogRecord(
        trial_id=trial_id,
        level=level,
        filename=None,
        message=message,
        iteration=None,
        epoch=None,
    )


def _current_thread_has_running_loop() -> bool:
    """Return whether the calling thread is running an asyncio event loop.

    :class:`_SinkStream` uses this as its deadlock guard: only the application's
    event-loop thread has a running loop, so a ``True`` result means driving the
    worker-thread-only sink from here would deadlock (and would recurse if the DB
    write itself logs — e.g. SQLAlchemy echo). The trainer's worker thread and
    Ray's log-forwarding threads have no running loop, so they forward normally.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class _TrialContext:
    """Mutable holder for the trial a captured stdout/stderr line belongs to.

    :class:`_SinkCallback` sets ``current_trial_id`` at ``on_trial_start`` and
    clears it at ``on_trial_complete`` / ``on_trial_error``; :class:`_SinkStream`
    reads it when tagging each captured line, so Ray's forwarded worker output is
    attributed to the trial that is currently running (or job-level, ``None``,
    between trials). A single attribute set/read is atomic under the GIL, so no
    lock is needed for the cross-thread hand-off; under parallel trials a line may
    be tagged with whichever trial was most recently active, which mirrors the
    2025 handler's single ``set_trial_id``.

    Ray pickles :class:`_SinkCallback` — and therefore this — into ``tuner.pkl``;
    it holds only a string, so it stays picklable.
    """

    def __init__(self) -> None:
        self.current_trial_id: str | None = None


class _SinkStream:
    """A text stream that persists complete lines to a ``TrialSink``.

    Installed as ``sys.stdout`` / ``sys.stderr`` for a local run by
    :func:`_capture_std_streams`, so Ray's ``log_to_driver`` output — every trial
    worker's stdout/stderr, forwarded to this driver process — is captured into
    ``log_entries`` the way the 2025 runner's ``PrintLogger`` was. Writes are
    always passed through to the original stream so console output is preserved,
    and buffered until a newline so partial writes do not split a line across rows.

    A line produced on the asyncio event-loop thread is passed through only, never
    sent to the sink (see :func:`_current_thread_has_running_loop`): the sink
    blocks the caller on a loop round-trip, which on the loop thread deadlocks.
    """

    def __init__(
        self,
        sink: TrialSink,
        original: TextIO,
        *,
        context: _TrialContext,
        level: str = "INFO",
    ) -> None:
        self._sink = sink
        self._original = original
        self._context = context
        self._level = level
        self._buffer = ""
        self._lock = threading.Lock()

    def write(self, message: str) -> int:
        """Pass ``message`` through to the console and forward completed lines."""
        with suppress(Exception):  # a broken console stream must not break the run
            self._original.write(message)
        if _current_thread_has_running_loop():
            return len(message)
        lines: list[str] = []
        with self._lock:
            self._buffer += message
            if "\n" in self._buffer:
                parts = self._buffer.split("\n")
                self._buffer = parts.pop()
                lines = [part for part in parts if part.strip()]
        for line in lines:
            self._forward(line)
        return len(message)

    def _forward(self, line: str) -> None:
        # capturing logs must never break the run
        with suppress(Exception):
            self._sink.log(
                LogRecord(
                    trial_id=self._context.current_trial_id,
                    level=self._level,
                    filename=None,
                    message=line.rstrip(),
                    iteration=None,
                    epoch=None,
                )
            )

    def flush(self) -> None:
        """Flush the console and forward any residual (newline-less) buffer."""
        with suppress(Exception):
            self._original.flush()
        if _current_thread_has_running_loop():
            return
        residual: str | None = None
        with self._lock:
            if self._buffer.strip():
                residual = self._buffer.strip()
                self._buffer = ""
        if residual is not None:
            self._forward(residual)

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return self._original.fileno()


@contextmanager
def _capture_std_streams(sink: TrialSink, context: _TrialContext) -> Iterator[None]:
    """Redirect ``sys.stdout`` / ``sys.stderr`` into ``sink`` for the run.

    This is what captures Ray's ``log_to_driver`` output: Ray forwards each trial
    worker's stdout/stderr to this driver process's streams, so wrapping them in
    :class:`_SinkStream` persists the full training logs — not just the driver's
    Python-logger lines. The original streams are restored (and the wrappers
    flushed) in a ``finally``, so a run that raises still restores the process's
    streams and persists what was buffered.

    ``sys.stdout`` / ``sys.stderr`` are process-global, so this hijacks the whole
    process's console output for the duration — acceptable because a local Ray
    cluster is a process-global singleton, so the runner already assumes one local
    run at a time. Enter it on the trainer's worker thread.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    # _SinkStream implements the slice of TextIO that stdout/stderr consumers use
    # (write/flush/isatty/fileno).
    sys.stdout = _SinkStream(sink, original_stdout, context=context)
    sys.stderr = _SinkStream(sink, original_stderr, context=context)
    try:
        yield
    finally:
        installed_stdout, sys.stdout = sys.stdout, original_stdout
        installed_stderr, sys.stderr = sys.stderr, original_stderr
        installed_stdout.flush()
        installed_stderr.flush()


def _package_artifacts(ctx: LocalRunContext) -> None:
    """Turn the best-trial checkpoint into a self-contained artifact bundle.

    Moves the trained weights into ``<output_dir>/<experiment_name>_weights/``, adds
    the README + inference/run/install scripts (aLoRA-aware when ``tuning_algo`` is
    ``"alora"``), then zips both the Ray results and the weights into
    ``<output_dir>/results/``. A faithful port of the 2025 post-HPO packaging.

    Args:
        ctx: The run context; supplies ``output_dir``, ``experiment_name``,
            ``model``, and ``tuning_algo``.
    """
    ray_result_folder: Path = ctx.output_dir / "ray_results"
    trained_folder: Path = ctx.output_dir / "models" / ctx.experiment_name
    weights_folder: Path = ctx.output_dir / f"{ctx.experiment_name}_weights"
    results_dir: Path = ctx.output_dir / "results"

    weights_folder.mkdir(parents=True, exist_ok=True)
    shutil.move(str(trained_folder), str(weights_folder))

    generate_readme(weights_folder)
    generate_bash_script(weights_folder)
    write_inference_script(
        ctx.model,
        ctx.experiment_name,
        weights_folder,
        for_alora=ctx.tuning_algo == "alora",
    )
    generate_install_bash_script(weights_folder)

    zip_folder(ray_result_folder, f"{ctx.experiment_name}_ray_results.zip", results_dir)
    zip_folder(weights_folder, f"{ctx.experiment_name}_weights.zip", results_dir)
