#!/usr/bin/env python3
"""Print the space-separated category names for `--test-categories` minus
`--exclude-categories`, for `evaluate`'s explicit `--test-category` list
when categories were excluded from generation (see `shard_test_ids.py` /
README's `--exclude-categories`).

Passing evaluate `--test-category all` against an output tree with, e.g.,
`web_search` excluded would have it try to score a category with no
generated results at all. This resolves the same subtraction
`shard_test_ids.py` already applies to the generate side, using
`bfcl_eval`'s own `parse_test_category_argument` so alias resolution (e.g.
`web_search` -> `{web_search_base, web_search_no_snippet}`) matches exactly.
"""

from __future__ import annotations

import argparse
import re

from bfcl_eval.utils import parse_test_category_argument


def _split(categories: str) -> list[str]:
    """Split a category list into the separate tokens bfcl expects.

    `--exclude-categories` is space-separated (shard_test_ids.py receives it
    word-split, as separate argv), and `--test-categories` reaches generate as
    one value that bfcl's own CLI callback splits on ",". Passed through whole,
    `"web_search multi_turn"` is one token that names no category.
    """
    return [c for c in re.split(r"[,\s]+", categories) if c]


def resolve(test_categories: str, exclude_categories: str | None) -> list[str]:
    included = set(parse_test_category_argument(_split(test_categories)))
    if exclude_categories:
        included -= set(parse_test_category_argument(_split(exclude_categories)))
    return sorted(included)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-categories", required=True)
    parser.add_argument("--exclude-categories", default=None)
    args = parser.parse_args()

    print(" ".join(resolve(args.test_categories, args.exclude_categories)))


if __name__ == "__main__":
    main()
