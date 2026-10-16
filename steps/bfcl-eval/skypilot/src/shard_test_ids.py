#!/usr/bin/env python3
"""Write a `test_case_ids_to_generate.json` covering one disjoint shard of
the *entire* BFCL corpus (not a sample), for splitting a full-corpus run
across N parallel jobs.

Sibling to `sample_test_ids.py` (same `--run-ids` mechanism, same
memory-scenario-grouping rationale -- see that file's module docstring for
why memory categories can't be split by individual id). The difference:
`sample_test_ids.py` draws a random fraction; this script partitions 100%
of each category's ids into N nearly-equal contiguous chunks and returns
chunk `shard_index`, so the union of all N shards' outputs is exactly the
full corpus with no overlaps and no gaps.

Chunking is done per-category (not globally over the flattened id list) so
every shard gets a proportional slice of *every* category -- this keeps
per-shard wall-clock roughly even even though multi-turn/memory entries are
much slower per-item than simple/AST ones; a global split could otherwise
dump a disproportionate share of the slow categories on one shard.

`--exclude-categories` (space-separated, same collection-alias names
`--test-categories` accepts, e.g. `web_search`) removes categories from the
partition entirely, so excluded categories never appear in any shard's
output file -- for a category with no `SERPAPI_API_KEY`-backed backend
available, generating and then discarding it would just waste GPU time.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from bfcl_eval.constants.category_mapping import (
    NON_SCORING_CATEGORY,
    TEST_COLLECTION_MAPPING,
)
from bfcl_eval.utils import is_memory, load_dataset_entry, parse_test_category_argument


def chunk_evenly(
    sorted_items: list[str], num_shards: int, shard_index: int
) -> list[str]:
    """Split `sorted_items` into `num_shards` nearly-equal contiguous chunks and return chunk `shard_index`.

    The first `len(sorted_items) % num_shards` chunks get one extra item, so every
    item is assigned to exactly one chunk and chunk sizes differ by at most 1.
    """
    total = len(sorted_items)
    base_size, remainder = divmod(total, num_shards)

    start = shard_index * base_size + min(shard_index, remainder)
    size = base_size + (1 if shard_index < remainder else 0)
    return sorted_items[start : start + size]


def shard_category_ids(category: str, num_shards: int, shard_index: int) -> list[str]:
    entries = load_dataset_entry(category)

    if is_memory(category):
        groups: dict[str, list[str]] = defaultdict(list)
        for entry in entries:
            groups[entry["scenario"]].append(entry["id"])
        scenario_keys = sorted(groups)
        chosen_scenarios = chunk_evenly(scenario_keys, num_shards, shard_index)
        return [eid for key in chosen_scenarios for eid in groups[key]]

    ids = sorted(entry["id"] for entry in entries)
    return chunk_evenly(ids, num_shards, shard_index)


def build_shard(
    num_shards: int, shard_index: int, exclude_categories: list[str] | None = None
) -> dict[str, list[str]]:
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")

    categories = set(TEST_COLLECTION_MAPPING["all"]) - set(NON_SCORING_CATEGORY)
    if exclude_categories:
        categories -= set(parse_test_category_argument(exclude_categories))
    categories = sorted(categories)

    return {
        category: sorted(shard_category_ids(category, num_shards, shard_index))
        for category in categories
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write test_case_ids_to_generate.json to",
    )
    parser.add_argument(
        "--num-shards", type=int, required=True, help="Total number of shards"
    )
    parser.add_argument(
        "--shard-index", type=int, required=True, help="This shard's index, 0-based"
    )
    parser.add_argument(
        "--exclude-categories",
        nargs="*",
        default=None,
        help="Category or collection names to leave out of every shard entirely (e.g. web_search)",
    )
    args = parser.parse_args()

    shard = build_shard(args.num_shards, args.shard_index, args.exclude_categories)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(shard, indent=2) + "\n")

    total = sum(len(ids) for ids in shard.values())
    print(
        f"shard_test_ids: wrote {total} ids across {len(shard)} categories "
        f"(shard {args.shard_index}/{args.num_shards}) to {output_path}"
    )
    for category, ids in sorted(shard.items()):
        print(f"  {category}: {len(ids)}")


if __name__ == "__main__":
    main()
