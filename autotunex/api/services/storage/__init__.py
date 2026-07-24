# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Pluggable dataset storage.

`get_storage_backend()` returns the active backend. Selection mirrors the
historical behavior: Granite Build when enabled, otherwise local disk. To add a
backend (e.g. object storage), implement StorageBackend in a new module and add
a branch below.

The GB backend is imported LAZILY inside the factory: its import chain
(gb_service -> db_service -> pymysql/gbcli, plus configureGBWorkingEnv()) is
heavy and only needed when GB is enabled. A local/default deployment never
loads it, and this module stays importable without the GB dependencies.
"""

from utils import is_gb_enabled

from .base import (  # noqa: F401
    DatasetFiles,
    DatasetRef,
    StorageBackend,
    StorageError,
    StorageLocator,
    StorageNotFound,
    StorageValidationError,
)
from .local_backend import LocalStorageBackend


def get_storage_backend(gb_enabled: bool | None = None) -> StorageBackend:
    """Select the active storage backend.

    gb_enabled defaults to utils.is_gb_enabled(); pass an explicit bool in tests.
    """
    if gb_enabled is None:
        gb_enabled = is_gb_enabled()
    if gb_enabled:
        # Lazy import: only load the heavy GB/gbcli chain when actually needed.
        from .gb_backend import GBStorageBackend

        return GBStorageBackend()
    return LocalStorageBackend()
