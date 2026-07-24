# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Internal helpers shared across storage backends."""

from fastapi import HTTPException

from .base import StorageError, StorageNotFound


def translate_http_exc(exc: HTTPException, context: str) -> StorageError:
    """Map a file_service HTTPException onto the storage contract.

    Backends call file_service helpers that raise FastAPI HTTPException; this
    keeps FastAPI types from leaking across the storage abstraction boundary.
    """
    if exc.status_code == 404:
        return StorageNotFound(context)
    return StorageError(f"{context}: {exc.detail}")
