# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

# api/services/datasets/tus_app.py
"""Embedded tus server router for resumable dataset uploads.

Mounts tuspyserver's router and wires its on_upload_complete hook to the
AutoTuneX finalize tail (decode metadata -> rendezvous -> last-completes
finalize, reusing the Dataset upload seam). This is the ONLY module that imports
tuspyserver; everything testable lives in tus_metadata/tus_rendezvous/tus_finalize.
"""

from __future__ import annotations

import logging
import os

import models as api
from auth import get_current_user
from fastapi import Depends
from services import db_service, user_service
from services.file import CONFIG
from tuspyserver import create_tus_router

from .service import Dataset
from .tus_finalize import handle_completed_file
from .tus_metadata import parse_upload_metadata
from .tus_rendezvous import UploadRendezvous

logger = logging.getLogger(__name__)

# Process-wide rendezvous shared across all tus completions (see single-process
# note in tus_rendezvous).
_rendezvous = UploadRendezvous()

TUS_STAGING_DIR = os.path.join(CONFIG["UPLOAD_DIR"], ".tus")
# Sub-prefix for the tus router itself; mounted under the /fmtune/api prefix in
# server.py, giving the effective path /fmtune/api/datasets/tus.
TUS_PREFIX = "datasets/tus"


def _ttl_days() -> int:
    minutes = int(os.getenv("UPLOAD_STAGING_TTL_MINUTES", "360"))
    # tuspyserver retention granularity is days; ceil to >= 1.
    return max(1, (minutes + 1439) // 1440)


def _make_complete_dep():
    """Build the per-request FastAPI dependency that produces the completion hook.

    tuspyserver only invokes ``upload_complete_dep`` when ``on_upload_complete is
    None`` (routes/core.py). The dependency itself depends on ``get_current_user``,
    so identity is resolved from the VALIDATED session cookie — never from
    client-supplied Upload-Metadata (which is why ``auth_email`` was removed from
    the contract). The route awaits the returned hook on the last chunk.
    """

    async def _dep(user: api.AuthUser = Depends(get_current_user)):
        async def _on_complete(file_path: str, metadata: dict):
            # Short-lived services (no FastAPI Depends() inside the hook — it runs
            # in the route body, mirroring mcp_server._get_services()).
            db = db_service.Database()
            dataset = Dataset(db)
            user_svc = user_service.User(db)
            try:
                intent = parse_upload_metadata(metadata)
            except ValueError as e:
                logger.error("tus completion: bad Upload-Metadata: %s", e)
                raise
            # Identity from the validated session, NOT from metadata.
            user_id = (await user_svc.get_user(user.email))["id"]
            return await handle_completed_file(
                file_path, intent, dataset, _rendezvous, user_id=user_id
            )

        return _on_complete

    return _dep


def create_dataset_tus_router():
    """Build the tus APIRouter (prefix 'datasets/tus'); mount under /fmtune/api in server.py."""
    os.makedirs(TUS_STAGING_DIR, exist_ok=True)

    # CONFIRMED against installed tuspyserver 4.x: every route depends on
    # Depends(options.auth), so passing auth=get_current_user requires a valid
    # session (401 before any staging). The completion hook is injected via
    # upload_complete_dep (resolved per-request, can depend on get_current_user);
    # on_upload_complete MUST be None for the DI variant to fire (core.py ~L280).
    # The hook is async and is awaited on the last chunk.
    return create_tus_router(
        prefix=TUS_PREFIX,
        files_dir=TUS_STAGING_DIR,
        days_to_keep=_ttl_days(),
        auth=get_current_user,
        on_upload_complete=None,
        upload_complete_dep=_make_complete_dep(),
    )
