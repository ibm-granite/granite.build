# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The ``AutotuneCore`` seam — read-only access to the optional ``autotune`` core.

The UI configuration *template* and the dataset-*type* catalog are both produced
by the torch-free ``autotune.catalog`` module (``get_autotune_config`` /
``get_autotune_dataset_types``). ``autotune`` is vendored in-tree at
``src/fm-tune``, and its catalog is torch-free (``autotune.catalog`` imports
only the standard library and PyYAML), so the slim ``autotune`` package IS
installed by ``make install`` and the wizard works out of the box. Only
*training* (the ``local`` runner) needs the heavy stack (Ray, torch, verl and
numpy), installed via ``make install-training``
(``pip install -e "./src/fm-tune[full,mlx]"``). The import stays *lazy* anyway,
and its absence is a request-time
:class:`~autotunex.core.exceptions.AutotuneCoreUnavailableError` (503), never an
import-time crash of the whole app — covering the case where ``autotune`` is
not installed at all. The seam's shape itself is unchanged; only the reason for
keeping it optional did.

Both payloads are static for the process lifetime, so the two module-level
loaders are memoized with :func:`functools.lru_cache`. ``lru_cache`` does not
cache raised exceptions, so an ``ImportError`` is retried on a later call (a
subsequent install/redeploy is picked up). The loaders expose ``.cache_clear()``
for test isolation. Both loaders do a blocking import; the config-template
loader also does a small blocking YAML read, while the dataset-type catalog is
returned as an in-memory dict with no file I/O. Either way, the work runs off
the event loop via :func:`asyncio.to_thread`, per the async rule in
``CLAUDE.md``.

This is one more Protocol seam in the same shape as ``JobRunner`` and
``StorageBackend``: services depend on :class:`AutotuneCore`, tests substitute a
fake, and the concrete :class:`AutotuneCoreAdapter` is the only place the real
package is touched.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any, Protocol

from autotunex.core.exceptions import AutotuneCoreUnavailableError
from autotunex.core.logging import get_logger

logger = get_logger(__name__)


class AutotuneCore(Protocol):
    """Read-only access to the optional ``autotune`` training core."""

    async def get_config_template(self) -> dict[str, Any]:
        """Return autotune's UI configuration template.

        Raises:
            AutotuneCoreUnavailableError: the ``autotune`` package is not installed.
        """
        ...

    async def get_dataset_types(self) -> dict[str, Any]:
        """Return autotune's dataset-type catalog, JSON-serializable.

        Raises:
            AutotuneCoreUnavailableError: the ``autotune`` package is not installed.
        """
        ...


def _type_to_str(value: object) -> str:
    """Render a value as a clean string, unwrapping Python ``type`` objects.

    ``autotune``'s dataset-type descriptors carry Python ``type`` objects in
    ``columns[*]["type"]`` (e.g. ``<class 'str'>``); those are not
    JSON-serializable, so a column's type is reduced to its bare name (``str``).
    """
    if isinstance(value, type):
        return value.__name__
    return str(value)


@lru_cache(maxsize=1)
def _load_config_template() -> dict[str, Any]:
    """Import ``autotune`` and return its UI config template (memoized).

    Blocking (an import plus a small YAML read) — call via ``asyncio.to_thread``.
    """
    try:
        from autotune.catalog import get_autotune_config
    except ImportError as exc:
        raise AutotuneCoreUnavailableError() from exc
    config: dict[str, Any] = get_autotune_config()
    return config


@lru_cache(maxsize=1)
def _load_dataset_types() -> dict[str, Any]:
    """Import ``autotune`` and return its dataset-type catalog (memoized).

    Column ``type`` objects are stringified so the result is JSON-serializable.
    Blocking (an import) — call via ``asyncio.to_thread``.
    """
    try:
        from autotune.catalog import get_autotune_dataset_types
    except ImportError as exc:
        raise AutotuneCoreUnavailableError() from exc
    dataset_types: dict[str, Any] = get_autotune_dataset_types()
    for dtype_val in dataset_types.values():
        for col_val in dtype_val.get("columns", {}).values():
            if "type" in col_val:
                col_val["type"] = _type_to_str(col_val["type"])
    return dataset_types


class AutotuneCoreAdapter:
    """Satisfies :class:`AutotuneCore`, backed by the real ``autotune`` package.

    Stateless: the adapter holds nothing and defers to the module-level memoized
    loaders, so many request-scoped instances share one cache.
    """

    async def get_config_template(self) -> dict[str, Any]:
        """Return autotune's UI configuration template (off the event loop)."""
        return await asyncio.to_thread(_load_config_template)

    async def get_dataset_types(self) -> dict[str, Any]:
        """Return autotune's JSON-serializable dataset-type catalog."""
        return await asyncio.to_thread(_load_dataset_types)
