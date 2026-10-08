#!/usr/bin/env python3
"""Merge per-shard BFCL `generate` output trees into one tree for a final
`evaluate` pass (see `shard_test_ids.py` / `run-bfcl.sh --num-shards`).

Each shard directory is a full BFCL_PROJECT_ROOT (what `--output-dir` pointed
`generate` at for that shard) containing a `result/<model>/**/*.json` tree of
JSONL result files, plus (for memory categories) a `memory_snapshot/` subtree
under each memory category's directory. Shards partition the full id set (and
memory scenarios) disjointly by construction (`shard_test_ids.py`), so merging
is a concatenation, not a real merge -- any collision found here means the
partitioning was wrong, and this aborts loudly rather than silently
deduplicating or overwriting.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _iter_result_files(shard_dir: Path):
    result_root = shard_dir / "result"
    if not result_root.is_dir():
        return
    for path in sorted(result_root.rglob("*.json")):
        if "memory_snapshot" in path.parts:
            continue
        yield path.relative_to(result_root)


def _iter_memory_snapshot_dirs(shard_dir: Path):
    result_root = shard_dir / "result"
    if not result_root.is_dir():
        return
    for path in sorted(result_root.rglob("memory_snapshot")):
        if path.is_dir():
            yield path.relative_to(result_root)


def merge(shard_dirs: list[Path], dest_dir: Path) -> None:
    result_dest_root = dest_dir / "result"
    # A re-run (e.g. retrying one failed/preempted shard) must start from a
    # clean destination: leftover files from a prior merge attempt are
    # indistinguishable from a genuine cross-shard collision to the
    # dest_file.exists() checks below, causing false-positive assertions.
    if result_dest_root.is_dir():
        shutil.rmtree(result_dest_root)

    rel_paths = sorted(
        {rel for shard_dir in shard_dirs for rel in _iter_result_files(shard_dir)}
    )
    for rel_path in rel_paths:
        merged_lines_by_id: dict[str, str] = {}
        for shard_dir in shard_dirs:
            src = shard_dir / "result" / rel_path
            if not src.is_file():
                continue
            for line in src.read_text().splitlines():
                if not line.strip():
                    continue
                entry_id = json.loads(line)["id"]
                if entry_id in merged_lines_by_id:
                    raise AssertionError(
                        f"id {entry_id!r} appears in more than one shard for {rel_path} -- "
                        "shards are supposed to partition ids disjointly (shard_test_ids.py bug?)"
                    )
                merged_lines_by_id[entry_id] = line

        dest = result_dest_root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            "".join(merged_lines_by_id[i] + "\n" for i in sorted(merged_lines_by_id))
        )

    snapshot_rel_dirs = sorted(
        {
            rel
            for shard_dir in shard_dirs
            for rel in _iter_memory_snapshot_dirs(shard_dir)
        }
    )
    for rel_dir in snapshot_rel_dirs:
        dest_snapshot_dir = result_dest_root / rel_dir
        for shard_dir in shard_dirs:
            src_snapshot_dir = shard_dir / "result" / rel_dir
            if not src_snapshot_dir.is_dir():
                continue
            for src_file in sorted(src_snapshot_dir.rglob("*")):
                if src_file.is_dir():
                    continue
                rel_file = src_file.relative_to(src_snapshot_dir)
                dest_file = dest_snapshot_dir / rel_file
                if dest_file.exists():
                    raise AssertionError(
                        f"memory_snapshot file {rel_dir / rel_file} appears in more than one "
                        "shard -- shards are supposed to partition memory scenarios disjointly "
                        "(shard_test_ids.py bug?)"
                    )
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    merge([Path(p) for p in args.shard_dirs], Path(args.output_dir))


if __name__ == "__main__":
    main()
