#!/usr/bin/env python3
"""Write a `test_case_ids_to_generate.json` that samples a fixed fraction of
each BFCL test category's ids, for a fast directional read on eval score
before committing to a full-corpus run.

bfcl_eval's own `generate --run-ids` flag reads exactly this file (via
`TEST_IDS_TO_GENERATE_PATH = <BFCL_PROJECT_ROOT>/test_case_ids_to_generate.json`)
and, when passed, ignores `--test-category` entirely -- see
`bfcl_eval._llm_response_generation.get_involved_test_entries`. So this
script's output is run-bfcl.sh's only lever for shrinking the corpus; there
is no native `--sample`/`--limit` flag on `bfcl generate` itself.

`format_sensitivity` (5,200 of the 10,417 entries across all 23 "all"
categories) is deliberately left out: bfcl_eval already skips it for any
is_fc_model (see `_llm_response_generation.py`'s "Skip format sensitivity
test cases for FC models"), so sampling it would be wasted work.

Memory categories (memory_kv/memory_rec_sum/memory_vector) are sampled by
whole `scenario` group, not by individual id: their entries carry an
explicit `depends_on` chain (turn N depends on turns 0..N-1 of the same
scenario), and dropping an id mid-chain would silently reorder/orphan the
remaining turns via bfcl_eval's own `clean_up_memory_prereq_entries` dep-list
pruning. Sampling whole scenarios keeps each retained scenario's chain
exactly as authored.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from bfcl_eval.constants.category_mapping import (
    NON_SCORING_CATEGORY,
    TEST_COLLECTION_MAPPING,
)
from bfcl_eval.utils import is_memory, load_dataset_entry


def sample_category_ids(
    category: str, fraction: float, rng: random.Random
) -> list[str]:
    entries = load_dataset_entry(category)

    if is_memory(category):
        groups: dict[str, list[str]] = defaultdict(list)
        for entry in entries:
            groups[entry["scenario"]].append(entry["id"])
        scenario_keys = sorted(groups)
        k = max(1, round(len(scenario_keys) * fraction))
        chosen_scenarios = set(rng.sample(scenario_keys, k))
        return [eid for key in chosen_scenarios for eid in groups[key]]

    ids = sorted(entry["id"] for entry in entries)
    k = max(1, round(len(ids) * fraction))
    return rng.sample(ids, k)


def build_sample(fraction: float, seed: int) -> dict[str, list[str]]:
    # Above 1.0, k exceeds the population and rng.sample raises on the first
    # category; a mistyped percentage (25 for 0.25) is the likely way to get here.
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    categories = sorted(set(TEST_COLLECTION_MAPPING["all"]) - set(NON_SCORING_CATEGORY))
    rng = random.Random(seed)
    return {
        category: sorted(sample_category_ids(category, fraction, rng))
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
        "--fraction",
        type=float,
        default=0.25,
        help="Fraction of each category's ids to sample",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Deterministic sampling seed"
    )
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error(f"--fraction must be in (0, 1], got {args.fraction}")

    sample = build_sample(args.fraction, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(sample, indent=2) + "\n")

    total = sum(len(ids) for ids in sample.values())
    print(
        f"sample_test_ids: wrote {total} ids across {len(sample)} categories to {output_path}"
    )
    for category, ids in sorted(sample.items()):
        print(f"  {category}: {len(ids)}")


if __name__ == "__main__":
    main()
