# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace model search and model cards for the tuning wizard.

The browser used to call huggingface.co directly, which can never see a private
repo — including every model AutoTuneX itself pushes. This service makes those
reads server-side and sends ``HF_TOKEN`` only where ``hf_import_namespaces``
allows: a search pinned to an allowlisted ``author``, or a card whose repo is in
one (:func:`~autotunex.services.storage.hf_hub.token_for`).
"""

from __future__ import annotations

import asyncio
import os

import httpx

from autotunex.core.config import Settings
from autotunex.core.exceptions import HfHubUnreachableError, HfModelNotFoundError
from autotunex.core.logging import get_logger
from autotunex.services.storage import hf_hub

logger = get_logger(__name__)

_PINNED_SEARCH_LIMIT = 100
"""Page size for each tokened, ``author=``-pinned search.

That page holds the namespace's public repos too, which are dropped only after it
arrives, so asking for just ``limit`` let public matches crowd out every private
one. The merged result is still cut to ``limit``.
"""


class HfModelService:
    """Search the Hub for models and read their cards, token-scoped by namespace."""

    def __init__(self, *, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._token = os.environ.get(settings.hf_token_env)

    async def search(self, *, query: str, limit: int) -> list[str]:
        """Model ids matching ``query``: private allowlisted hits first, then public.

        One anonymous search, plus one tokened search per allowlisted namespace
        pinned with ``author=`` (none without a token). What bounds where the
        token is *sent* is that each tokened call is built only for one
        allowlisted namespace; the owner filter below is hygiene on the response,
        not the security boundary. Each pinned call keeps only its private hits
        (``private_only=True``), so a namespace's public repos arrive in normal
        relevance order via the anonymous search instead of crowding it out. A
        failed pinned search degrades to public results; a failed public search
        is an outage.

        Raises:
            HfHubUnreachableError: the anonymous search could not be completed.
        """
        base = self._settings.hf_hub_base_url
        namespaces = list(self._settings.hf_import_namespaces) if self._token else []
        public, *scoped = await asyncio.gather(
            hf_hub.search_models(self._client, base_url=base, query=query, limit=limit, token=None),
            *(
                hf_hub.search_models(
                    self._client,
                    base_url=base,
                    query=query,
                    limit=max(limit, _PINNED_SEARCH_LIMIT),
                    token=self._token,
                    author=namespace,
                    private_only=True,
                )
                for namespace in namespaces
            ),
            return_exceptions=True,
        )
        if isinstance(public, hf_hub.HfHubUnavailable):
            raise HfHubUnreachableError() from public
        if isinstance(public, BaseException):
            raise public
        ids: list[str] = []
        for namespace, result in zip(namespaces, scoped, strict=True):
            if isinstance(result, hf_hub.HfHubUnavailable):
                logger.warning("Scoped HF model search for %s failed: %s", namespace, result)
                continue
            if isinstance(result, BaseException):
                raise result
            ids.extend(i for i in result if hf_hub.is_allowlisted(i, [namespace]))
        ids.extend(public)
        return list(dict.fromkeys(ids))[:limit]

    async def card(self, *, repo_id: str) -> str:
        """``repo_id``'s README, read with the token only if its namespace is allowlisted.

        Raises:
            HfModelNotFoundError: absent, or private and not readable by this server.
            HfHubUnreachableError: the Hub could not be reached.
        """
        token = hf_hub.token_for(
            repo_id, token=self._token, namespaces=self._settings.hf_import_namespaces
        )
        try:
            return await hf_hub.fetch_model_card(
                self._client,
                base_url=self._settings.hf_hub_base_url,
                repo_id=repo_id,
                token=token,
            )
        except hf_hub.HfModelNotFound as exc:
            raise HfModelNotFoundError(repo_id) from exc
        except hf_hub.HfHubUnavailable as exc:
            raise HfHubUnreachableError() from exc
