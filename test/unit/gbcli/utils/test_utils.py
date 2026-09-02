#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for gbcli utils pure functions.

Ported from gbcli-upstream test/unit_tests/utils/test_utils.py. Pure functions —
no mocks and no IBM infrastructure required.
"""

import unittest

import pytest

from gbcli.utils.utils import (
    generate_unique_id,
    remove_prefix,
    remove_suffix,
    resolve_canonical_expression_to_url,
    resolve_url_to_canonical_expression,
)

pytestmark = pytest.mark.standalone


class TestUtils(unittest.TestCase):
    def test__remove_prefix(self):
        """Test remove_prefix()"""

        x = remove_prefix("https://", "https://github.ibm.com/granite-dot-build/assets")
        assert x == "github.ibm.com/granite-dot-build/assets"

    def test__remove_suffix(self):
        """Test remove_suffix()"""

        x = remove_suffix("github.ibm.com/granite-dot-build/assets", "/assets")
        assert x == "github.ibm.com/granite-dot-build"

    def test__generate_unique_id(self):
        """Test generate_unique_id()"""

        x = generate_unique_id()
        assert len(x) == 8

    def test_resolve_canonical_expression_to_url(self):
        """test resolve_canonical_expression_to_url with multiple cases"""

        test_cases = [
            {
                "expression": "space-abc",
                "expected": "https://github.ibm.com/granite-dot-build/space-abc",
            },
            {
                "expression": "granite-dot-build/space-abc",
                "expected": "https://github.ibm.com/granite-dot-build/space-abc",
            },
            {
                "expression": "github.ibm.com/granite-dot-build/space-abc",
                "expected": "https://github.ibm.com/granite-dot-build/space-abc",
            },
            {
                "expression": "https://github.ibm.com/granite-dot-build/space-abc",
                "expected": "https://github.ibm.com/granite-dot-build/space-abc",
            },
            {
                "expression": "/space-abc",
                "expected": "https://github.ibm.com/granite-dot-build/space-abc",
            },
        ]

        for case in test_cases:
            with self.subTest(case=case):
                result = resolve_canonical_expression_to_url(
                    case["expression"], addSuffix=False
                )
                self.assertEqual(result, case["expected"])

    def test_resolve_url_to_canonical_expression(self):
        """test resolve_url_to_canonical_expression"""

        test_cases = [
            {
                "url": "https://github.ibm.com/granite-dot-build/space-abc",
                "expected": "space-abc",
            },
            {
                "url": "https://github.ibm.com/project-org/space-abc",
                "expected": "project-org/space-abc",
            },
        ]

        for case in test_cases:
            with self.subTest(case=case):
                result = resolve_url_to_canonical_expression(
                    case["url"], removeSuffix=True
                )
                self.assertEqual(result, case["expected"])
