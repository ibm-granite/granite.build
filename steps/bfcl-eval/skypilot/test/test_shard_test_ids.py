import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from bfcl_eval.constants.category_mapping import (  # noqa: E402
    NON_SCORING_CATEGORY,
    TEST_COLLECTION_MAPPING,
)
from bfcl_eval.utils import is_memory, load_dataset_entry  # noqa: E402
from shard_test_ids import build_shard, chunk_evenly  # noqa: E402

NUM_SHARDS = 8


def _all_shards(num_shards=NUM_SHARDS, exclude_categories=None):
    return [build_shard(num_shards, i, exclude_categories) for i in range(num_shards)]


def test_chunk_evenly_covers_everything_with_sizes_differing_by_at_most_one():
    items = [f"x{i}" for i in range(23)]  # not evenly divisible by 8
    chunks = [chunk_evenly(items, 8, i) for i in range(8)]

    flattened = [x for chunk in chunks for x in chunk]
    assert sorted(flattened) == sorted(items)
    assert len(flattened) == len(items)

    sizes = [len(chunk) for chunk in chunks]
    assert max(sizes) - min(sizes) <= 1


def test_chunk_evenly_rejects_out_of_range_shard_index():
    with pytest.raises(ValueError):
        build_shard(4, 4)
    with pytest.raises(ValueError):
        build_shard(4, -1)


def test_excludes_format_sensitivity():
    for shard in _all_shards():
        assert "format_sensitivity" not in shard
        assert set(NON_SCORING_CATEGORY).isdisjoint(shard)


def test_shards_cover_every_scoring_category_exactly_once_each():
    expected_categories = set(TEST_COLLECTION_MAPPING["all"]) - set(
        NON_SCORING_CATEGORY
    )
    for shard in _all_shards():
        assert set(shard) == expected_categories


def test_shards_partition_every_category_with_no_gaps_and_no_overlap():
    shards = _all_shards()
    expected_categories = set(TEST_COLLECTION_MAPPING["all"]) - set(
        NON_SCORING_CATEGORY
    )

    for category in expected_categories:
        full_ids = sorted(entry["id"] for entry in load_dataset_entry(category))
        per_shard_ids = [shard[category] for shard in shards]

        union = [eid for ids in per_shard_ids for eid in ids]
        assert sorted(union) == full_ids, f"gap/overlap in category {category}"

        seen = set()
        for ids in per_shard_ids:
            assert seen.isdisjoint(
                ids
            ), f"id shared across shards in category {category}"
            seen.update(ids)


def test_memory_categories_shard_by_whole_scenario_not_individual_turns():
    shards = _all_shards()
    for category in ("memory_kv", "memory_rec_sum", "memory_vector"):
        entries_by_id = {entry["id"]: entry for entry in load_dataset_entry(category)}
        for shard in shards:
            shard_ids = set(shard[category])
            scenarios_touched = {entries_by_id[eid]["scenario"] for eid in shard_ids}
            for scenario in scenarios_touched:
                full_scenario_ids = {
                    eid
                    for eid, entry in entries_by_id.items()
                    if entry["scenario"] == scenario
                }
                # Every id belonging to a touched scenario must land in this
                # same shard -- otherwise a depends_on chain gets split
                # across two jobs with two separate memory_snapshot dirs.
                assert full_scenario_ids <= shard_ids


def test_exclude_categories_removes_category_from_every_shard():
    shards = _all_shards(exclude_categories=["web_search"])
    for shard in shards:
        assert "web_search_base" not in shard
        assert "web_search_no_snippet" not in shard

    baseline_categories = set(TEST_COLLECTION_MAPPING["all"]) - set(
        NON_SCORING_CATEGORY
    )
    excluded_categories = baseline_categories - set(shards[0])
    assert excluded_categories == {"web_search_base", "web_search_no_snippet"}


def test_deterministic_no_randomness_involved():
    assert build_shard(NUM_SHARDS, 3) == build_shard(NUM_SHARDS, 3)


def test_single_shard_is_the_whole_corpus():
    (shard,) = _all_shards(num_shards=1)
    expected_categories = set(TEST_COLLECTION_MAPPING["all"]) - set(
        NON_SCORING_CATEGORY
    )
    for category in expected_categories:
        full_ids = sorted(entry["id"] for entry in load_dataset_entry(category))
        assert shard[category] == full_ids
