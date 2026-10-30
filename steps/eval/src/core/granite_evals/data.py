"""Dataset resolution.

Every benchmark reads its upstream dataset straight from the HF hub, pinned by
commit: each ``Benchmark`` declares ``dataset`` and ``dataset_revision``, so a
published score always names the exact bytes it was computed on. Public data
is not mirrored. ``--dataset`` / ``--dataset-revision`` override the pin with
any hub id or local path.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def load_split(
    source: str,
    *,
    revision: str | None = None,
    split: str = "test",
    name: str | None = None,
) -> list[dict]:
    """Load one split of ``source`` (hub id or local path) as a list of rows."""
    from datasets import load_dataset

    if not source:
        raise SystemExit("no dataset: the benchmark pins none, pass --dataset <hub-id-or-path>")
    log.info("loading %s split=%s revision=%s", source, split, revision or "default")
    if Path(source).exists():
        return list(load_dataset(source, name=name, split=split))
    return list(load_dataset(source, name=name, split=split, revision=revision))


def take(rows: list[dict], limit: int | None, *, key: str) -> list[dict]:
    """First ``limit`` rows in a stable order, so smoke runs are comparable."""
    rows = sorted(rows, key=lambda r: r[key])
    return rows if limit is None else rows[:limit]
