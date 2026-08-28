# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Standalone test for normalization.normalized(). Run: python test_normalization.py"""

from api_bridge.normalization import normalized


def test_key_order_insensitive():
    assert normalized({"a": 1, "b": 2}) == normalized({"b": 2, "a": 1})


def test_int_float_equivalent():
    assert normalized({"lr": 1}) == normalized({"lr": 1.0})


def test_nested_dicts_and_lists():
    a = {"x": {"p": 1, "q": [1, 2, 3]}}
    b = {"x": {"q": [1, 2, 3], "p": 1.0}}
    assert normalized(a) == normalized(b)


def test_genuinely_different_not_equal():
    assert normalized({"lr": 1}) != normalized({"lr": 2})


def test_null_vs_missing_are_distinct():
    # A present null key is NOT the same as an absent key (conservative).
    assert normalized({"a": None}) != normalized({})


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
