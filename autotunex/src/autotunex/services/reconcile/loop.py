# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The periodic job-status reconcile sweep.

Off the read path entirely: a background task started in ``lifespan`` that
advances ``jobs.status`` from what gbserver reports for each job's ``TUNING``
build. Restart-safe by construction — its whole working set is one query per
sweep (``JobRepository.list_reconcilable``); nothing is held in memory between
sweeps, so a process restart resumes exactly where it left off.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from autotunex.core.exceptions import InvalidStateTransitionError
from autotunex.core.logging import get_logger
from autotunex.db.repositories.protocols import ReconcilableJob
from autotunex.db.repositories.sqlalchemy import SqlAlchemyJobRepository
from autotunex.models.status import TERMINAL_RUN_STATUSES, GbTaskType
from autotunex.services.jobs import check_transition
from autotunex.services.reconcile.build_detail import build_status_detail, output_artifact_ref
from autotunex.services.reconcile.mapping import to_run_status
from autotunex.services.reconcile.protocols import (
    BuildNotFoundError,
    BuildState,
    BuildStatusAuthError,
    BuildStatusError,
    BuildStatusReader,
    BuildStatusUnavailableError,
    MalformedBuildStatusError,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class _TerminalDetail:
    """What the terminal write records beyond status: build detail + the model id.

    ``build_status`` is the transformed ``{details, targets, build_history}`` the
    Status tab renders; ``artifact_id``/``artifact_uri`` identify the produced
    model, extracted from that detail (``None`` when the build registered none).
    """

    build_status: dict[str, Any]
    artifact_id: UUID | None
    artifact_uri: str | None


class ReconcileLoop:
    """Periodically reconciles non-terminal jobs against gbserver.

    The only writer to ``jobs.status`` for cluster-driven transitions. Every
    write is gated by ``check_transition``; a racing ``InvalidStateTransitionError``
    means another sweep or replica already advanced the job and is swallowed.
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        reader: BuildStatusReader,
        interval_seconds: int = 30,
        concurrency: int = 5,
    ) -> None:
        self._session_factory = session_factory
        self._reader = reader
        self._interval = interval_seconds
        self._semaphore = asyncio.Semaphore(concurrency)

    async def run(self) -> None:
        """Sweep forever, sleeping ``interval_seconds`` between passes.

        A failed sweep is logged and the loop continues; only cancellation
        (shutdown) stops it.
        """
        while True:
            try:
                await self.sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Reconcile sweep failed; continuing.")
            await asyncio.sleep(self._interval)

    async def sweep_once(self) -> None:
        """Run one reconcile pass. Public and directly awaitable for tests."""
        async with self._session_factory() as session:
            batch = await SqlAlchemyJobRepository(session).list_reconcilable()
        if not batch:
            return

        async def _guarded(item: ReconcilableJob) -> bool:
            async with self._semaphore:
                return await self._reconcile_one(item)

        results = await asyncio.gather(*(_guarded(item) for item in batch))
        if any(results):
            # 401/403 logged once per sweep, not per job: it is almost always one
            # expired GB_TOKEN affecting every read, and per-job logging floods.
            logger.error(
                "gbserver rejected status reads with 401/403 this sweep; "
                "the GB token is likely expired."
            )

    async def _reconcile_one(self, item: ReconcilableJob) -> bool:
        """Read one build's status and apply it. Returns ``True`` on an auth error.

        The bool is how ``sweep_once`` logs 401/403 once per sweep rather than
        once per job. Every other failure is handled here per the spec's taxonomy
        and never escalates: the loop's broad handler is a backstop, not the
        primary path.
        """
        try:
            state = await self._reader.read(item.build_id)
        except BuildStatusAuthError:
            return True
        except BuildNotFoundError:
            # A 404 is likelier a bad id on our side than a real failure, and
            # ERROR is terminal and irreversible — so log, do not mark ERROR.
            logger.error(
                "Build %s for job %s not found; leaving the job untouched.",
                item.build_id,
                item.job_id,
            )
            return False
        except BuildStatusUnavailableError as exc:
            logger.warning(
                "Could not read build %s for job %s: %s; retrying next sweep.",
                item.build_id,
                item.job_id,
                exc,
            )
            return False
        except MalformedBuildStatusError as exc:
            logger.error(
                "Malformed status for build %s (job %s): %s; no write.",
                item.build_id,
                item.job_id,
                exc,
            )
            return False
        await self._apply(item, state)
        return False

    async def _apply(self, item: ReconcilableJob, state: BuildState) -> None:
        """Advance the job + its TUNING task to the mapped status, if legal.

        Opens its OWN session: an ``AsyncSession`` is not concurrency-safe and
        the sweep runs jobs concurrently. Never touches ``trials`` — this poller
        is authoritative for job-level status only.
        """
        mapped = to_run_status(state.status)
        if mapped is None:
            return
        # Build the terminal-only detail BEFORE opening the session, so the extra
        # events round trip is not held inside a DB transaction. Non-terminal
        # sweeps skip it entirely — the events fetch is paid once, at the end.
        detail = (
            await self._terminal_detail(item, state) if mapped in TERMINAL_RUN_STATUSES else None
        )
        async with self._session_factory() as session:
            jobs = SqlAlchemyJobRepository(session)
            try:
                check_transition(item.status, mapped)
            except InvalidStateTransitionError:
                logger.debug(
                    "Job %s already past %s -> %s; skipping.",
                    item.job_id,
                    item.status,
                    mapped,
                )
                return
            # Task detail before status, in both branches: on a terminal transition,
            # writing the TUNING task first means a crash between the two writes
            # leaves the job non-terminal, so the next sweep re-applies it
            # (self-healing) rather than leaving the job terminal with the build
            # detail never recorded. check_transition already ran above, so
            # legality of the eventual status write is unaffected by the reorder.
            if detail is not None:
                await jobs.upsert_task(
                    item.job_id,
                    GbTaskType.TUNING,
                    status=mapped,
                    build_status=detail.build_status,
                    artifact_id=detail.artifact_id,
                    artifact_uri=detail.artifact_uri,
                    started_at=state.created_at,
                    updated_at=state.updated_at,
                )
            else:
                await jobs.upsert_task(item.job_id, GbTaskType.TUNING, status=mapped)
            await jobs.set_status(item.job_id, mapped)

    async def _terminal_detail(self, item: ReconcilableJob, state: BuildState) -> _TerminalDetail:
        """Assemble the terminal ``build_status`` + output-artifact ids for a job.

        Fetches the build's event log to build ``build_history`` and extracts the
        produced model's id/uri from the transformed targets — what the Status tab
        and the model-download flow read. ``state.raw`` is the status body already
        fetched this sweep by :meth:`read`, so only the events endpoint is a fresh
        round trip. An events read failure is non-fatal (``build_history`` is left
        empty) so a transient gbserver blip never blocks the terminal status write.
        ``item.build_id`` is guaranteed non-null: ``list_reconcilable`` only returns
        jobs whose TUNING task has one.
        """
        build_id: UUID = item.build_id
        try:
            events = await self._reader.read_events(build_id)
        except BuildStatusError as exc:
            logger.warning(
                "Could not read build events for %s (job %s): %s; build_history empty.",
                build_id,
                item.job_id,
                exc,
            )
            events = {}
        detail = build_status_detail(state.raw, events)
        artifact_id: UUID | None = None
        artifact_uri: str | None = None
        ref = output_artifact_ref(detail)
        if ref is not None:
            try:
                artifact_id = UUID(ref["artifact_id"])
                artifact_uri = ref["uri"] or None
            except ValueError:
                logger.warning(
                    "Build %s output artifact id %r is not a UUID; leaving artifact unset.",
                    build_id,
                    ref["artifact_id"],
                )
        return _TerminalDetail(
            build_status=detail, artifact_id=artifact_id, artifact_uri=artifact_uri
        )
