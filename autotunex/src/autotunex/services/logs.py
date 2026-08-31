# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Log retrieval business logic.

Reads a job's structured (DB ``log_entries``) and live (gb) logs, scoped to the
calling principal. Knows nothing about HTTP: raises ``core/exceptions`` classes.
Scope is resolved through the shared ``services/scoping.py`` helpers, exactly as
``JobService`` does (the per-service ``_owner_id`` / ``_can_see_anything`` were
removed by the admin-scope-default change).
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from autotunex.core.exceptions import GbLogsUnavailableError, JobNotFoundError
from autotunex.db.repositories.protocols import JobRepository
from autotunex.db.tables import LogEntryTable
from autotunex.models.auth import Principal
from autotunex.models.common import DataScope
from autotunex.models.log import LogPage
from autotunex.models.status import GbTaskType
from autotunex.services.gb_logs.protocols import GbLogReader
from autotunex.services.mappers import log_entry_to_read
from autotunex.services.scoping import resolve_owner_filter, sees_nothing


class LogService:
    """Reads a job's structured (DB) and live (gb) logs, scoped to the principal."""

    def __init__(
        self,
        repository: JobRepository,
        principal: Principal,
        gb_log_reader: GbLogReader,
    ) -> None:
        self._repository = repository
        self._principal = principal
        self._gb_log_reader = gb_log_reader

    async def get_job_logs(
        self, job_id: UUID, *, before_id: int, limit: int, scope: DataScope = DataScope.OWN
    ) -> LogPage:
        """Return one keyset page of the job's job-level log lines, newest first.

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
            JobNotFoundError: no such job, or it is not visible under ``scope``.
        """
        await self._require_visible(job_id, scope)
        rows, has_more = await self._repository.logs_page(
            job_id, trial_id=None, before_id=before_id, limit=limit
        )
        return self._to_page(rows, has_more)

    async def get_trial_logs(
        self,
        job_id: UUID,
        trial_id: str,
        *,
        before_id: int,
        limit: int,
        scope: DataScope = DataScope.OWN,
    ) -> LogPage:
        """Return one keyset page of the trial's log lines, newest first.

        A trial with no matching rows is an empty page, not a 404 (see the design
        spec — ``log_entries.trial_id`` cannot be verified against ``trials.id``).

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
            JobNotFoundError: no such job, or it is not visible under ``scope``.
        """
        await self._require_visible(job_id, scope)
        rows, has_more = await self._repository.logs_page(
            job_id, trial_id=trial_id, before_id=before_id, limit=limit
        )
        return self._to_page(rows, has_more)

    async def get_gb_logs(
        self, job_id: UUID, *, fetch_all: bool, scope: DataScope = DataScope.OWN
    ) -> list[str]:
        """Return the job's live gb container logs (oldest-first).

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
            JobNotFoundError: no such job, or it is not visible under ``scope``.
            GbLogsUnavailableError: the job has no build handle, or gb is unconfigured.
            GbLogsUpstreamError: the gb log server was unreachable or errored.
        """
        await self._require_visible(job_id, scope)
        task = await self._repository.get_task(job_id, GbTaskType.TUNING)
        build_ref = (task.build_id or task.pr_url) if task is not None else None
        if build_ref is None:
            raise GbLogsUnavailableError()
        return await self._gb_log_reader.fetch(str(build_ref), fetch_all=fetch_all)

    async def _require_visible(self, job_id: UUID, scope: DataScope) -> UUID | None:
        """Resolve scope and 404 unless the caller may see ``job_id``.

        Order matches ``JobService``: the non-admin ``all`` 403 fires first
        (``resolve_owner_filter``), then the ``own``-no-identity short-circuit
        (``sees_nothing``), then the existence probe. Returns the resolved
        ``owner_id`` (``None`` only for an admin's ``all``).
        """
        owner_id = resolve_owner_filter(self._principal, scope)
        if sees_nothing(self._principal, scope):
            raise JobNotFoundError(job_id)
        if not await self._repository.is_visible(job_id, owner_id=owner_id):
            raise JobNotFoundError(job_id)
        return owner_id

    def _to_page(self, rows: Sequence[LogEntryTable], has_more: bool) -> LogPage:
        logs = [log_entry_to_read(row) for row in rows]
        next_before_id = logs[-1].id if has_more and logs else None
        return LogPage(logs=logs, has_more=has_more, next_before_id=next_before_id)
