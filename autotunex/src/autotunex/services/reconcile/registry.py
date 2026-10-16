# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""BuildStatusReader selection from settings.

Two-arg like ``build_llm_client(settings, http_client)`` rather than one-arg
like ``launch/registry.py``: the reader needs an ``httpx.AsyncClient``, opened
with its own timeout in ``lifespan``.
"""

from __future__ import annotations

import httpx

from autotunex.core.config import Settings
from autotunex.services.reconcile.gbserver import GbServerStatusReader
from autotunex.services.reconcile.protocols import BuildStatusReader


def get_build_status_reader(
    settings: Settings, http_client: httpx.AsyncClient
) -> BuildStatusReader:
    """Return the status reader for the configured backend.

    Only ``job_backend="llmb"`` has one. ``gb_server_url`` is guaranteed present
    by ``Settings._validate_job_backend`` when llmb; the ``None`` check narrows
    the type and fails loudly if that invariant is ever bypassed.
    """
    if settings.job_backend != "llmb":
        raise ValueError(f"No status reader for job_backend={settings.job_backend!r}.")
    if settings.gb_server_url is None:
        raise ValueError('job_backend="llmb" requires gb_server_url for reconcile.')
    return GbServerStatusReader(
        http_client=http_client,
        base_url=settings.gb_server_url,
        token_env=settings.gb_token_env,
    )
