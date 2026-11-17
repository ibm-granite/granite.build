#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0


import pytest

from gbserver.utils.utils_k8s import is_helm_v4_or_higher


class TestHelmVersion:
    def test_is_helm_v4_or_higher(self):
        test_cases = [
            {
                "input": "v3.19.0",
                "expected": False,
            },
            {
                "input": "v4.2.0",
                "expected": True,
            },
            {
                "input": "v4.1.1+g5caf004",
                "expected": True,
            },
        ]
        for test_case in test_cases:
            test_input = test_case["input"]
            expected = test_case["expected"]
            actual = is_helm_v4_or_higher(test_input)
            assert (
                actual == expected
            ), f"test_input: {test_input} actual: {actual} expected: {expected}"
