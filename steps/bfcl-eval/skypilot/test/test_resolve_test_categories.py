import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from bfcl_eval.constants.category_mapping import (  # noqa: E402
    NON_SCORING_CATEGORY,
    TEST_COLLECTION_MAPPING,
)
from resolve_test_categories import resolve  # noqa: E402


def test_no_exclusion_returns_full_resolved_set():
    assert resolve("all", None) == sorted(TEST_COLLECTION_MAPPING["all"])


def test_excludes_the_named_collection():
    resolved = resolve("all", "web_search")
    assert "web_search_base" not in resolved
    assert "web_search_no_snippet" not in resolved

    all_categories = set(TEST_COLLECTION_MAPPING["all"])
    assert set(resolved) == all_categories - {
        "web_search_base",
        "web_search_no_snippet",
    }


def test_excludes_a_single_concrete_category_name():
    resolved = resolve("all", "web_search_base")
    assert "web_search_base" not in resolved
    assert "web_search_no_snippet" in resolved


def test_excludes_every_name_in_a_space_separated_list():
    # The README's --exclude-categories form. Passed whole, the two names were
    # one token that matched nothing, so multi_turn was scored though never
    # generated.
    resolved = set(resolve("all", "web_search multi_turn"))
    assert resolved == set(TEST_COLLECTION_MAPPING["all"]) - set(
        TEST_COLLECTION_MAPPING["web_search"]
    ) - set(TEST_COLLECTION_MAPPING["multi_turn"])


def test_accepts_a_comma_separated_test_category_list():
    # generate gets --test-category as one value and bfcl splits it on ",".
    assert resolve("simple_python,multiple", None) == ["multiple", "simple_python"]


def test_empty_exclude_string_is_a_no_op():
    assert resolve("all", "") == resolve("all", None)


def test_format_sensitivity_still_included_when_not_excluded():
    # resolve() has no opinion on format_sensitivity -- that's run-bfcl.sh's
    # test_categories default ("all"), not this script's job to filter.
    assert set(NON_SCORING_CATEGORY) <= set(resolve("all", None))
