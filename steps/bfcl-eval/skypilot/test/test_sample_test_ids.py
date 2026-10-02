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
from sample_test_ids import build_sample  # noqa: E402


def test_excludes_format_sensitivity():
    sample = build_sample(fraction=0.25, seed=42)
    assert "format_sensitivity" not in sample
    assert set(NON_SCORING_CATEGORY).isdisjoint(sample)


def test_covers_every_scoring_category_at_roughly_a_quarter():
    sample = build_sample(fraction=0.25, seed=42)
    expected_categories = set(TEST_COLLECTION_MAPPING["all"]) - set(
        NON_SCORING_CATEGORY
    )
    assert set(sample) == expected_categories

    for category, ids in sample.items():
        total = len(load_dataset_entry(category))
        assert len(ids) >= 1
        # Small categories round to a wider ratio; just check we didn't
        # sample everything or (for a non-memory category) more than a
        # comfortable margin over a quarter.
        if not is_memory(category):
            assert len(ids) <= max(1, round(total * 0.25)) + 1


def test_memory_categories_sample_whole_scenarios_not_individual_turns():
    sample = build_sample(fraction=0.25, seed=42)
    for category in ("memory_kv", "memory_rec_sum", "memory_vector"):
        entries_by_id = {entry["id"]: entry for entry in load_dataset_entry(category)}
        sampled_ids = set(sample[category])
        scenarios_touched = {entries_by_id[eid]["scenario"] for eid in sampled_ids}
        for scenario in scenarios_touched:
            full_scenario_ids = {
                eid
                for eid, entry in entries_by_id.items()
                if entry["scenario"] == scenario
            }
            # Every id belonging to a touched scenario must be present --
            # otherwise a depends_on chain gets silently truncated.
            assert full_scenario_ids <= sampled_ids


def test_deterministic_given_same_seed():
    assert build_sample(0.25, 42) == build_sample(0.25, 42)


def test_different_seeds_generally_differ():
    assert build_sample(0.25, 42) != build_sample(0.25, 7)


@pytest.mark.parametrize("fraction", [0, -0.5, 1.5, 25])
def test_rejects_a_fraction_outside_zero_to_one(fraction):
    # 25 is the mistyped-percentage case; without the check rng.sample raised
    # "Sample larger than population" on the first category.
    with pytest.raises(ValueError, match="fraction"):
        build_sample(fraction=fraction, seed=42)


def test_accepts_a_fraction_of_one():
    sample = build_sample(fraction=1.0, seed=42)
    for category, ids in sample.items():
        if not is_memory(category):
            assert len(ids) == len(load_dataset_entry(category))
