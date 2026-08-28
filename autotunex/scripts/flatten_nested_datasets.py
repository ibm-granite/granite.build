#!/usr/bin/env python3
# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Flatten legacy nested dataset directories to the current flat layout.

Older AutoTuneX versions stored a dataset's files one directory too deep::

    <root>/<id>/<name>/<name>_train.jsonl
    <root>/<id>/<name>/<name>_validation.jsonl

The current app — ``LocalStorageBackend`` (previews), the local runner, and the
launch spec — reads the *flat* layout::

    <root>/<id>/<name>_train.jsonl
    <root>/<id>/<name>_validation.jsonl

so a legacy dataset returns an empty preview and a job cannot find its files.
This one-time, idempotent migration moves each legacy dataset's split files up
one level and removes the emptied subdirectory.

It is a **dry run by default**; pass ``--apply`` to make changes. It never
overwrites an existing target and never removes a non-empty directory, so a
dataset that is already flat is skipped and re-running the script is safe.

Usage::

    python scripts/flatten_nested_datasets.py [ROOT] [--apply]

``ROOT`` defaults to ``/pvc/datasets`` (the value of ``AUTOTUNEX_DATASET_STORAGE_DIR``
in the stage deployment).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

_SPLIT_MARKERS = ("_train.", "_validation.")
"""A dataset split file is recognised by one of these substrings in its name."""

_STAGING_DIR = ".staging"
"""The in-progress upload directory under the storage root; never a dataset."""


@dataclass(frozen=True)
class Move:
    """A single planned relocation of a split file from a subdir up to the id dir."""

    src: Path
    dst: Path


@dataclass
class Plan:
    """Everything the migration would do, computed without touching disk."""

    moves: list[Move] = field(default_factory=list)
    conflicts: list[Move] = field(default_factory=list)
    prune_dirs: list[Path] = field(default_factory=list)


def _is_split_file(path: Path) -> bool:
    """Return whether ``path`` is a dataset split file (``*_train.*``/``*_validation.*``)."""
    return path.is_file() and any(marker in path.name for marker in _SPLIT_MARKERS)


def _has_flat_split_files(dataset_dir: Path) -> bool:
    """Return whether ``dataset_dir`` already holds split files directly (flat/migrated)."""
    return any(_is_split_file(child) for child in dataset_dir.iterdir())


def _dataset_dirs(root: Path) -> Iterator[Path]:
    """Yield the candidate dataset directories under ``root`` (skipping ``.staging``)."""
    for path in sorted(root.iterdir()):
        if path.is_dir() and path.name != _STAGING_DIR:
            yield path


def plan_migration(root: Path) -> Plan:
    """Scan ``root`` and return the moves needed to flatten nested dataset dirs.

    Pure: reads the tree but changes nothing. A dataset directory is migrated
    only when it holds no flat split files of its own *and* has a subdirectory
    that contains them — so already-flat datasets are left untouched and the
    scan is idempotent. A subdirectory is marked for removal only when every one
    of its entries is a split file that moves without a naming conflict, so a
    subdir holding anything else is never deleted.
    """
    plan = Plan()
    for dataset_dir in _dataset_dirs(root):
        if _has_flat_split_files(dataset_dir):
            continue  # already flat — nothing to do
        for sub in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            entries = sorted(sub.iterdir())
            split_files = [c for c in entries if _is_split_file(c)]
            if not split_files:
                continue
            sub_conflicts = [
                Move(c, dataset_dir / c.name)
                for c in split_files
                if (dataset_dir / c.name).exists()
            ]
            plan.conflicts.extend(sub_conflicts)
            plan.moves.extend(
                Move(c, dataset_dir / c.name)
                for c in split_files
                if not (dataset_dir / c.name).exists()
            )
            if not sub_conflicts and all(_is_split_file(c) for c in entries):
                plan.prune_dirs.append(sub)
    return plan


def apply_plan(plan: Plan, *, dry_run: bool) -> None:
    """Execute (or, when ``dry_run``, only print) the planned moves and prunes.

    Moves use :meth:`Path.replace`, which is atomic within a single filesystem
    (source and target share the ``<id>`` directory, so this always holds). A
    target that has appeared since planning is skipped rather than overwritten.
    """
    for move in plan.moves:
        if dry_run:
            print(f"[dry-run] move  {move.src}  ->  {move.dst}")
            continue
        if move.dst.exists():  # re-check: never overwrite, even on a race
            print(f"SKIP (target appeared)  {move.src}  ->  {move.dst}", file=sys.stderr)
            continue
        move.src.replace(move.dst)
        print(f"moved  {move.src}  ->  {move.dst}")
    for sub in plan.prune_dirs:
        if dry_run:
            print(f"[dry-run] rmdir {sub}")
        elif not any(sub.iterdir()):
            sub.rmdir()
            print(f"removed empty dir {sub}")
    for move in plan.conflicts:
        print(f"SKIP (target exists)  {move.src}  ->  {move.dst}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, plan the migration, and apply it (or dry-run) — return an exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "root", nargs="?", default=Path("/pvc/datasets"), type=Path, help="storage root to scan"
    )
    parser.add_argument(
        "--apply", action="store_true", help="perform changes (default: dry run only)"
    )
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"root {args.root} is not a directory", file=sys.stderr)
        return 2

    plan = plan_migration(args.root)
    apply_plan(plan, dry_run=not args.apply)

    mode = "APPLIED" if args.apply else "DRY RUN"
    print(
        f"\n{mode}: {len(plan.moves)} file(s) to move, "
        f"{len(plan.prune_dirs)} dir(s) to remove, {len(plan.conflicts)} conflict(s) skipped."
    )
    if not args.apply and (plan.moves or plan.prune_dirs):
        print("Re-run with --apply to make these changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
