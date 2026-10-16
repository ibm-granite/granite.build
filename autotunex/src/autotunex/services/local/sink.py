# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The persisting ``TrialSink`` and its log handler for the ``local`` runner.

:class:`DbTrialSink` bridges the **worker-thread** calls a Ray callback (or the
captured stdout/stderr stream) makes (``trial_started`` / ``trial_result`` /
``trial_completed`` / ``trial_error`` / ``log`` / ``training_metric``) to the
async write repositories running on the application's event loop, using
:func:`asyncio.run_coroutine_threadsafe`. :class:`SinkLogHandler` is a
``logging.Handler`` that buffers captured log lines and flushes them to a
:class:`~autotunex.services.local.protocols.TrialSink` in batches, so log
capture does not cost one DB round-trip per line.

**Load-bearing invariant:** :class:`DbTrialSink` is used **only from worker
threads**. Each public method blocks on ``run_coroutine_threadsafe(...).result()``
until the DB write commits; calling it on the event loop's own thread would
schedule the coroutine behind the very ``result()`` that is blocking that thread,
and deadlock. Job-level writes that happen on the loop (inside the runner's
``process()``) must call the async repositories directly, never through the sink.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Coroutine
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from autotunex.db.repositories.sqlalchemy import (
    SqlAlchemyJobRepository,
    SqlAlchemyResultRepository,
    SqlAlchemyTrainingMetricsRepository,
    SqlAlchemyTrialRepository,
)
from autotunex.models.status import RunStatus
from autotunex.services.local.protocols import LogRecord, TrainingMetricRecord, TrialSink

_MAX_TRIAL_ID_LEN = 16
"""Width of ``trials.id`` / ``results.trial_id`` (``VARCHAR(16)``)."""

_DEFAULT_FLUSH_THRESHOLD = 20
"""Buffered log lines that trigger an automatic flush in :class:`SinkLogHandler`."""


def _json_safe(value: Any) -> Any:  # noqa: ANN401 — normalizes genuinely-arbitrary input
    """Reduce ``value`` to a form the ``trials.config`` JSON column can store.

    Ray hands each trial's ``config`` back carrying the ``autotune`` machinery the
    optimizer stuffed into the *shared* ``param_space`` — including live
    ``search_alg`` / ``scheduler`` **instances** (``get_tune_config`` swaps the
    string ``"lds"`` for a ``LimitedDiscrepancySearch`` object), which JSON cannot
    encode. Persisting that config raw is what crashed a real run with
    ``Object of type LimitedDiscrepancySearch is not JSON serializable``.

    The coercion is structure-preserving so the trial's sampled hyperparameters
    survive intact:

    * JSON primitives (``None`` / ``bool`` / ``int`` / ``float`` / ``str``, and
      thus ``numpy.float64``, a ``float`` subclass) pass through unchanged;
    * ``dict`` and ``list`` / ``tuple`` are walked recursively;
    * a NumPy-style scalar (anything exposing a zero-arg ``.item()`` returning a
      primitive — e.g. ``numpy.int64``, which is *not* an ``int`` subclass) is
      unwrapped to that primitive, so an integer hyperparameter keeps its value
      rather than being stringified away;
    * anything still unserializable (the searcher / scheduler instances) collapses
      to its stable ``module.qualname`` — a deterministic marker that records
      *what* ran without dragging the object into the row.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except Exception:
            scalar = object()  # sentinel: fall through to the qualname marker
        if scalar is None or isinstance(scalar, (bool, int, float, str)):
            return scalar
    return f"{type(value).__module__}.{type(value).__qualname__}"


class DbTrialSink:
    """Persist trial lifecycle events, bridging worker threads to async repos.

    Satisfies :class:`~autotunex.services.local.protocols.TrialSink`. Each public
    method builds a coroutine that opens its **own** short-lived session from
    ``session_factory`` (never sharing a session across threads) and calls the
    committed write repository, then runs it on ``loop`` from the calling worker
    thread and blocks until it commits.

    See the module docstring for the worker-thread-only invariant.
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        loop: asyncio.AbstractEventLoop,
        job_id: UUID,
    ) -> None:
        """Store the session factory, the target loop, and the owning job id.

        Args:
            session_factory: Opens a fresh :class:`AsyncSession` per DB write.
            loop: The event loop the async repositories run on — the loop that is
                free while the trainer runs under ``asyncio.to_thread``.
            job_id: The job every trial, result, and log row is attributed to.
        """
        self._session_factory = session_factory
        self._loop = loop
        self._job_id = job_id

    def __getstate__(self) -> dict[str, Any]:
        """Return a picklable state carrying only ``job_id``.

        Ray Tune serializes ``RunConfig.callbacks`` into the experiment's
        ``tuner.pkl`` when the ``Tuner`` is constructed, and each callback the
        local trainer installs wraps this sink — so Ray pickles the sink
        transitively. The event loop (``_loop``) and the async engine behind
        ``_session_factory`` both hold unpicklable resources (thread locks,
        weakrefs), which is what made ``Tuner`` construction crash with
        ``cannot pickle 'weakref.ReferenceType'``. Ray needs the callbacks only
        to *serialize*, never to function after unpickling, so both are dropped
        here; pickle's default ``__setstate__`` then rehydrates a hollow shell
        (no loop, no factory) whose sole live field is ``job_id``.

        This is safe because ``__getstate__`` is invoked only when something
        pickles the sink — never on the trial write path, where the live object
        keeps every field — and the pickled shell is read back only by
        ``Tuner.restore``, which the local runner never calls
        (``resume_from_checkpoint=False``; a crashed HPO sweep is rerun, not
        resumed).
        """
        return {"_job_id": self._job_id}

    @staticmethod
    def coerce_trial_id(raw: str) -> str:
        """Return a stable trial id that fits ``trials.id`` (``VARCHAR(16)``).

        A Ray trial id short enough to fit is returned unchanged; a longer one is
        replaced by a deterministic 16-hex-char BLAKE2b digest of the raw id. The
        derivation is stable, so every lifecycle call for the same raw id maps to
        the same row.

        Args:
            raw: The trial id reported by the trainer / Ray.

        Returns:
            ``raw`` when ``len(raw) <= 16``, else its 16-char BLAKE2b hex digest.
        """
        if len(raw) <= _MAX_TRIAL_ID_LEN:
            return raw
        return hashlib.blake2b(raw.encode(), digest_size=8).hexdigest()

    def _run(self, coro: Coroutine[Any, Any, None]) -> None:
        """Run ``coro`` on the event loop from a worker thread and block on it.

        Worker-thread only: calling this on the loop's own thread deadlocks (see
        the module docstring).
        """
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        fut.result()  # block the worker thread until the DB write commits

    async def _upsert_trial(
        self, trial_id: str, *, status: RunStatus, config: dict[str, Any] | None
    ) -> None:
        async with self._session_factory() as session:
            await SqlAlchemyTrialRepository(session).upsert(
                self._job_id, trial_id, status=status, config=config
            )

    async def _set_trial_status(self, trial_id: str, status: RunStatus) -> None:
        async with self._session_factory() as session:
            await SqlAlchemyTrialRepository(session).set_status(trial_id, status)

    async def _upsert_result(
        self, trial_id: str, *, metric: str, metrics: dict[str, Any] | None
    ) -> None:
        async with self._session_factory() as session:
            await SqlAlchemyResultRepository(session).upsert(
                self._job_id, trial_id, metric=metric, metrics=metrics
            )

    async def _insert_metric(self, record: TrainingMetricRecord) -> None:
        trial_id = self.coerce_trial_id(record.trial_id) if record.trial_id is not None else None
        async with self._session_factory() as session:
            await SqlAlchemyTrainingMetricsRepository(session).insert(
                self._job_id,
                trial_id=trial_id,
                global_step=record.global_step,
                epoch=record.epoch,
                loss=record.loss,
                grad_norm=record.grad_norm,
                learning_rate=record.learning_rate,
                split=record.split,
                extra=record.extra,
            )

    async def _append_log(self, record: LogRecord) -> None:
        trial_id = self.coerce_trial_id(record.trial_id) if record.trial_id is not None else None
        async with self._session_factory() as session:
            await SqlAlchemyJobRepository(session).append_log(
                self._job_id,
                trial_id=trial_id,
                level=record.level,
                filename=record.filename,
                message=record.message,
                iteration=record.iteration,
                epoch=record.epoch,
            )

    def trial_started(self, trial_id: str, config: dict[str, Any] | None) -> None:
        """Upsert the trial as ``running`` with its concrete parameter ``config``.

        The config Ray reports carries the ``autotune`` machinery embedded in the
        shared ``param_space`` — including live searcher / scheduler instances the
        JSON column cannot store — so it is reduced to a JSON-safe form (see
        :func:`_json_safe`) before it is persisted.
        """
        self._run(
            self._upsert_trial(
                self.coerce_trial_id(trial_id),
                status=RunStatus.RUNNING,
                config=_json_safe(config),
            )
        )

    def trial_result(self, trial_id: str, metric: str, metrics: dict[str, Any] | None) -> None:
        """Upsert the trial's one-to-one result under the objective ``metric``."""
        self._run(
            self._upsert_result(self.coerce_trial_id(trial_id), metric=metric, metrics=metrics)
        )

    def trial_completed(self, trial_id: str) -> None:
        """Set the trial's status to ``completed``."""
        self._run(self._set_trial_status(self.coerce_trial_id(trial_id), RunStatus.COMPLETED))

    def trial_error(self, trial_id: str) -> None:
        """Set the trial's status to ``error``."""
        self._run(self._set_trial_status(self.coerce_trial_id(trial_id), RunStatus.ERROR))

    def log(self, record: LogRecord) -> None:
        """Persist one captured log line as a ``log_entries`` row."""
        self._run(self._append_log(record))

    def training_metric(self, record: TrainingMetricRecord) -> None:
        """Persist one per-step training-metrics row (worker-thread only)."""
        self._run(self._insert_metric(record))


class SinkLogHandler(logging.Handler):
    """A ``logging.Handler`` that buffers log lines and flushes them to a sink.

    Attached to the runner / ``autotune`` loggers during a local run. ``emit``
    maps each ``logging.LogRecord`` to the domain
    :class:`~autotunex.services.local.protocols.LogRecord` and buffers it;
    the buffer is flushed automatically once it reaches ``flush_threshold`` and
    on every explicit :meth:`flush` / :meth:`close`. Each flush forwards the whole
    buffered batch to ``sink.log`` in a single pass, so log capture is not one DB
    round-trip per line (the async-DB analogue of the 2025 ``BufferedLogHandler``).

    Because flushing drives the sink, and :class:`DbTrialSink` is worker-thread
    only, this handler must be flushed from a worker thread — which it is, since it
    is attached only for the duration of the thread the trainer runs on.
    """

    def __init__(self, sink: TrialSink, *, trial_id: str | None = None) -> None:
        """Bind the handler to ``sink``, tagging every line with ``trial_id``.

        Args:
            sink: Where buffered log lines are flushed.
            trial_id: The trial context stamped on every captured line, or ``None``
                for job-level logs.
        """
        super().__init__()
        self._sink = sink
        self._trial_id = trial_id
        self._buffer: list[LogRecord] = []
        self._flush_threshold = _DEFAULT_FLUSH_THRESHOLD

    def emit(self, record: logging.LogRecord) -> None:
        """Buffer ``record`` as a :class:`LogRecord`, flushing at the threshold.

        ``iteration`` and ``epoch`` are left ``None``: a standard log line carries
        neither. Any failure is routed through ``handleError`` so logging never
        raises into the caller.
        """
        try:
            entry = LogRecord(
                trial_id=self._trial_id,
                level=record.levelname,
                filename=record.filename,
                message=record.getMessage(),
                iteration=None,
                epoch=None,
            )
            self.acquire()
            try:
                self._buffer.append(entry)
                should_flush = len(self._buffer) >= self._flush_threshold
            finally:
                self.release()
            if should_flush:
                self.flush()
        except Exception:  # logging must not raise into the caller
            self.handleError(record)

    def flush(self) -> None:
        """Forward every buffered log line to the sink, then clear the buffer.

        The buffer is snapshotted and cleared under the handler lock, then drained
        outside it so the (blocking) sink writes never hold the lock.
        """
        self.acquire()
        try:
            records = self._buffer
            self._buffer = []
        finally:
            self.release()
        for record in records:
            self._sink.log(record)

    def close(self) -> None:
        """Flush any remaining buffered lines, then close the handler."""
        self.flush()
        super().close()
