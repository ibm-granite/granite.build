# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Lists a caller's completed tuning outputs that can seed a new job.

The start-tuning wizard offers these as "My tuned models", so a user can run,
say, GRPO on top of their own full SFT. Only full-weight, HF-hosted outputs in
an allowlisted namespace are listed: fm-tune cannot use a PEFT adapter as a
base, and a private base model loads only through the ``inputs.base_model``
binding, which is only emitted for allowlisted owners
(``services/launch/_shared.base_model_uri``). See
``docs/superpowers/specs/2026-09-26-tuned-model-search-design.md``.

Separate from :class:`~autotunex.services.jobs.JobService` because it reads
through its own narrow repository Protocol.
"""

from __future__ import annotations

from collections.abc import Sequence

from autotunex.db.repositories.protocols import TunedModelRepository
from autotunex.models.auth import Principal
from autotunex.models.common import DataScope, Page
from autotunex.models.job import TunedModelSummary
from autotunex.services.mappers import tuned_model_to_summary
from autotunex.services.scoping import resolve_owner_filter, sees_nothing
from autotunex.services.storage.hf_hub import is_allowlisted


class TunedModelService:
    """Searches the principal's tuned models (every owner's, for an admin with ``scope=all``)."""

    def __init__(
        self,
        repository: TunedModelRepository,
        principal: Principal,
        namespaces: Sequence[str],
    ) -> None:
        self._repository = repository
        self._principal = principal
        self._namespaces = list(namespaces)

    async def search(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        scope: DataScope = DataScope.OWN,
        q: str | None = None,
    ) -> Page[TunedModelSummary]:
        """Return one page of tuned models, newest first — own rows by default.

        An unresolvable caller or an empty allowlist gets an empty page without a
        query: the first sees nothing under ``own``, and with no allowlisted
        namespace no output could be loaded by a new job.

        Each row is re-checked with the launch path's own owner rule
        (:func:`~autotunex.services.storage.hf_hub.is_allowlisted`), which the
        repository's ``LIKE`` only approximates — it folds case on SQLite and
        MySQL. A row that fails would be offered yet passed by name at launch, so
        it is dropped; ``total`` still counts it, so a page can come up short, but
        only when ``job_output_uri_root`` and ``hf_import_namespaces`` spell an
        owner in different case.

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
        """
        owner_id = resolve_owner_filter(self._principal, scope)
        if sees_nothing(self._principal, scope) or not self._namespaces:
            return Page[TunedModelSummary](items=[], total=0, limit=limit, offset=offset)
        rows, total = await self._repository.tuned_models(
            namespaces=self._namespaces, limit=limit, offset=offset, owner_id=owner_id, q=q
        )
        items = [tuned_model_to_summary(job, finished_at, uri) for job, finished_at, uri in rows]
        return Page[TunedModelSummary](
            items=[item for item in items if is_allowlisted(item.repo_id, self._namespaces)],
            total=total,
            limit=limit,
            offset=offset,
        )
