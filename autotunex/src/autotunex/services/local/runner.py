# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The in-process ``local`` job runner.

:class:`LocalJobRunner` is the ``local`` analogue of
:class:`~autotunex.services.runner.InProcessJobRunner`: ``submit`` schedules
``process`` as a background task and returns immediately, and ``process`` opens
its own sessions (the request session is long closed) and is directly awaitable
so tests are deterministic.

Unlike the granite.build path, this runner executes the HPO pipeline itself, on
this process, via the :class:`~autotunex.services.local.protocols.LocalTrainer`
seam. The trainer is CPU/GPU-bound and synchronous, so ``process`` runs it under
:func:`asyncio.to_thread` — which frees the event loop to service the DB writes
the trainer drives through :class:`~autotunex.services.local.sink.DbTrialSink`
from that worker thread. A run therefore does move the job's status: ``pending
→ running`` before the trainer starts, then ``running → completed`` on success
or ``running → error`` on any failure, with still-``running`` trials swept to
``error`` so none is left dangling.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from autotunex.core.exceptions import JobCancellationInProgressError
from autotunex.core.logging import get_logger
from autotunex.db.repositories.sqlalchemy import (
    SqlAlchemyDatasetRepository,
    SqlAlchemyJobRepository,
    SqlAlchemyTrialRepository,
)
from autotunex.db.tables import DatasetTable, JobTable
from autotunex.models.job import TERMINAL_JOB_STATUSES
from autotunex.models.status import GbTaskType, RunStatus
from autotunex.services.jobs import check_transition
from autotunex.services.local.cancellation import (
    is_active,
    is_cancelled,
    register,
    request_cancel,
    unregister,
)
from autotunex.services.local.protocols import LocalRunContext, LocalTrainer
from autotunex.services.local.sink import DbTrialSink

logger = get_logger(__name__)

_SUPPORTED_MODEL_SOURCES: frozenset[str] = frozenset({"huggingface", "custom_path"})
"""Model sources the local runner can resolve; external model catalogues are not supported here."""

_POLL_INTERVAL_SECONDS = 0.1
"""How often :meth:`LocalJobRunner.cancel` polls the registry while waiting for a stop."""


class _LocalRunError(Exception):
    """A job cannot be run locally (bad model source, or missing data on disk).

    Raised only inside :meth:`LocalJobRunner.process` and caught by its own
    broad ``except`` there, which lands the job in ``error``. It never escapes
    the runner, so it is intentionally not a domain exception from
    ``core/exceptions.py`` — nothing translates it to an HTTP response.
    """


class LocalJobRunner:
    """Runs an accepted job's HPO pipeline in-process, persisting its progress.

    Satisfies :class:`autotunex.services.protocols.JobRunner`.

    Args:
        session_factory: Opens fresh sessions (:class:`AsyncSession`) — one for
            the prepare phase, one each for the success/failure write-backs, and
            one per DB write inside :class:`DbTrialSink`.
        trainer: The pipeline seam ``process`` runs under
            :func:`asyncio.to_thread`.
        output_root: ``settings.local_output_dir``; a run's ``output_dir`` is
            ``output_root / <job_id>``, resolved to an absolute path (Ray Tune's
            ``storage_path`` cannot be relative).
        dataset_root: ``settings.dataset_storage_dir``; a dataset's files live at
            ``dataset_root / <dataset_id> / <train_file|validation_file>``.
        cancel_timeout: Seconds :meth:`cancel` waits for a registered run to stop
            before raising ``JobCancellationInProgressError``. The cancel is
            latched regardless of the wait outcome — see :meth:`cancel`.
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        trainer: LocalTrainer,
        output_root: Path,
        dataset_root: Path,
        cancel_timeout: float,
    ) -> None:
        self._session_factory = session_factory
        self._trainer = trainer
        self._output_root = output_root
        self._dataset_root = dataset_root
        self._cancel_timeout = cancel_timeout
        self._tasks: set[asyncio.Task[None]] = set()

    async def submit(self, job_id: UUID) -> None:
        """Fire ``process`` as a background task and return immediately.

        A strong reference is held in ``self._tasks`` until the task finishes,
        so it is never garbage-collected mid-run (per the ``asyncio`` docs);
        the done-callback discards it.
        """
        task = asyncio.create_task(self.process(job_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def process(self, job_id: UUID) -> None:
        """Run the job's pipeline to completion, recording its terminal state.

        Directly awaitable so tests need no background-task plumbing. Registers a
        cancel token for the run's lifetime (see
        :mod:`autotunex.services.local.cancellation`) so a concurrent
        :meth:`cancel` can signal it, and unregisters only after the terminal
        status is written — so a delete that waits on ``is_active`` never races
        the sink's last writes.

        Loads and validates the job, flips it to ``running``, runs the trainer on
        a worker thread, then writes its terminal state. A run that was cancelled
        lands in ``terminated`` (not ``completed``/``error``), whether the trainer
        raised because of the cancellation or simply returned after noticing it,
        with still-``running`` trials swept accordingly. Any other failure — a
        bad model source, missing data, or an exception from the trainer — lands
        the job in ``error`` and sweeps its still-``running`` trials to ``error``
        too, so a run never leaves the job stuck ``running``.
        """
        register(job_id)
        try:
            loop = asyncio.get_running_loop()
            try:
                context = await self._prepare(job_id)
                if context is None:
                    return  # Job vanished before the run started; nothing to do.
                sink = DbTrialSink(session_factory=self._session_factory, loop=loop, job_id=job_id)
                # Runs on a worker thread so the loop stays free to service the DB
                # writes the sink schedules back onto it from that thread.
                await asyncio.to_thread(self._trainer.run, context, sink)
            # Broad except is intentional: the trainer may raise anything, and a
            # non-cancellation failure must land the job in `error` rather than
            # leaving it `running`.
            except Exception:
                if is_cancelled(job_id):
                    logger.info("Local run for job %s stopped by cancellation.", job_id)
                    await self._terminate(job_id)
                    return
                logger.exception("Local run failed for job %s", job_id)
                await self._fail(job_id)
                return
            if is_cancelled(job_id):
                logger.info("Local run for job %s ended after a cancellation request.", job_id)
                await self._terminate(job_id)
                return
            await self._succeed(job_id)
        finally:
            unregister(job_id)

    async def _prepare(self, job_id: UUID) -> LocalRunContext | None:
        """Load and validate the job, flip it to ``running``, return its context.

        Returns ``None`` (after logging) when the job has vanished — a benign
        "nothing to run" case, distinct from a validation failure, which raises
        :class:`_LocalRunError` for :meth:`process` to turn into ``error``.
        """
        async with self._session_factory() as session:
            jobs = SqlAlchemyJobRepository(session)
            job = await jobs.get(job_id)
            if job is None:
                logger.error("Job %s vanished before the local run; nothing to do.", job_id)
                return None
            # Unscoped by id on purpose: the job's ownership was validated at
            # submission, and this must still resolve shared system-owned
            # datasets. Do NOT add owner_id here.
            dataset = await SqlAlchemyDatasetRepository(session).get(job.dataset_id)
            context = self._context_from(job, dataset)
            check_transition(job.status, RunStatus.RUNNING)
            await jobs.set_status(job_id, RunStatus.RUNNING)
            await jobs.upsert_task(job_id, GbTaskType.TUNING, status=RunStatus.RUNNING)
        context.output_dir.mkdir(parents=True, exist_ok=True)
        return context

    def _context_from(self, job: JobTable, dataset: DatasetTable | None) -> LocalRunContext:
        """Assemble a :class:`LocalRunContext` from the job, its snapshot and dataset.

        Validates what a local run needs before it starts: the model source must
        be one the runner can resolve, the dataset must exist, and its training
        file must be present on disk. The validation file is passed through even
        if absent — many runs have no validation split.

        Raises:
            _LocalRunError: the model source is unsupported, the dataset is
                missing, or the training file is not on disk.
        """
        if job.model_source not in _SUPPORTED_MODEL_SOURCES:
            raise _LocalRunError(
                f"Job {job.id} has unsupported model_source {job.model_source!r}; "
                f"the local runner supports only {sorted(_SUPPORTED_MODEL_SOURCES)}."
            )
        if dataset is None:
            raise _LocalRunError(
                f"Job {job.id} references dataset {job.dataset_id}, which is missing."
            )

        # The on-disk filenames carry the format extension, matching where
        # ``LocalStorageBackend`` actually writes them (``<name>_<split>.<ext>``).
        # The ``datasets.train_file``/``validation_file`` generated columns are
        # ``CONCAT(name, '_train')`` with NO extension, so they must not be used
        # to build the real path — they would resolve to a non-existent file.
        dataset_dir = self._dataset_root / str(dataset.id)
        train_file = dataset_dir / f"{dataset.name}_train.{dataset.data_format}"
        validation_file = dataset_dir / f"{dataset.name}_validation.{dataset.data_format}"
        if not train_file.exists():
            raise _LocalRunError(
                f"Training file {train_file} for job {job.id} does not exist on disk."
            )

        snapshot = job.config_snapshot or {}
        rl_algo = snapshot.get("rl_tuner_type") or "none"
        config_name = snapshot.get("name") or "config"
        config_data = snapshot.get("config_data") or {}
        return LocalRunContext(
            job_id=job.id,
            model=job.model,
            model_source=job.model_source,
            experiment_name=job.experiment_name,
            tuning_algo=job.tuning_type or "none",
            rl_algo=rl_algo if isinstance(rl_algo, str) else "none",
            config_name=config_name if isinstance(config_name, str) else "config",
            config_data=config_data if isinstance(config_data, dict) else {},
            train_file=train_file,
            validation_file=validation_file,
            # Absolute, always: the trainer forwards this to Ray Tune's
            # ``storage_path``, and pyarrow's ``FileSystem.from_uri`` rejects a
            # relative path with "URI has empty scheme". ``local_output_dir``
            # defaults to the relative ``artifact_dir / "local"``, so resolve here
            # rather than trusting the operator to pin an absolute path.
            output_dir=(self._output_root / str(job.id)).resolve(),
            seed=job.seed,
            autotune=bool(job.autotune),
            cleanup=True,
            save_history=True,
            reward_function_code=job.reward_function_code,
            reward_function_name=job.reward_function_name,
        )

    async def cancel(self, job_id: UUID) -> None:
        """Signal the in-process run to stop and wait until it has.

        A no-op when no run is registered in this process (a run that never
        started, or one orphaned by an API restart — whose in-memory token died
        with it). Waits up to ``cancel_timeout`` for the run to finalize and
        unregister; on timeout raises ``JobCancellationInProgressError`` (the
        signal stays latched, so the run still stops and a retry succeeds).
        """
        if not request_cancel(job_id):
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._cancel_timeout
        while is_active(job_id):
            if loop.time() >= deadline:
                raise JobCancellationInProgressError(job_id)
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    async def _terminate(self, job_id: UUID) -> None:
        """Record a cancelled run: job + ``TUNING`` task ``terminated``; sweep trials."""
        async with self._session_factory() as session:
            jobs = SqlAlchemyJobRepository(session)
            await jobs.set_status(job_id, RunStatus.TERMINATED)
            await jobs.upsert_task(job_id, GbTaskType.TUNING, status=RunStatus.TERMINATED)
            await SqlAlchemyTrialRepository(session).terminate_running(job_id)

    async def _succeed(self, job_id: UUID) -> None:
        """Record a successful run: job and its ``TUNING`` task both ``completed``.

        Guarded against overwriting a terminal status already written by another
        path (e.g. :meth:`_terminate` on a cancellation that raced the trainer's
        own normal-completion return) — the check-then-write is not atomic across
        the two sessions, so this narrows but does not eliminate that race; the
        job's status only ever moves *into* a terminal state, never out, so the
        guard cannot itself corrupt a clean run.
        """
        async with self._session_factory() as session:
            jobs = SqlAlchemyJobRepository(session)
            existing = await jobs.get(job_id)
            if existing is not None and existing.status in TERMINAL_JOB_STATUSES:
                return  # already terminal (e.g. cancelled) — do not overwrite
            await jobs.set_status(job_id, RunStatus.COMPLETED)
            await jobs.upsert_task(job_id, GbTaskType.TUNING, status=RunStatus.COMPLETED)

    async def _fail(self, job_id: UUID) -> None:
        """Record a failed run: job and ``TUNING`` task ``error``; sweep live trials.

        Sweeping still-``running`` trials to ``error`` (via
        :meth:`SqlAlchemyTrialRepository.fail_running`) is the whole reason this
        is separate from :meth:`_succeed`: a trainer that dies mid-trial cannot
        report the trial's terminal state itself, so the runner does it here.
        Guarded against overwriting an already-terminal status; see :meth:`_succeed`.
        """
        async with self._session_factory() as session:
            jobs = SqlAlchemyJobRepository(session)
            existing = await jobs.get(job_id)
            if existing is not None and existing.status in TERMINAL_JOB_STATUSES:
                return  # already terminal (e.g. cancelled) — do not overwrite
            await jobs.set_status(job_id, RunStatus.ERROR)
            await jobs.upsert_task(job_id, GbTaskType.TUNING, status=RunStatus.ERROR)
            await SqlAlchemyTrialRepository(session).fail_running(job_id)
