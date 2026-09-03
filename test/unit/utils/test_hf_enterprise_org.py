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

"""Unit tests for ``is_enterprise_hf_org`` (the Enterprise/non-Enterprise split)."""

import pytest

from gbcommon.utils.hf_utils import is_enterprise_hf_org


class TestIsEnterpriseHfOrgBackCompat:
    def test_none_treats_every_org_as_enterprise(self):
        """An absent config key preserves the pre-split behavior."""
        assert is_enterprise_hf_org("ibm-research", None) is True
        assert is_enterprise_hf_org("some-random-user", None) is True

    def test_empty_list_treats_no_org_as_enterprise(self):
        """An explicit empty list is a full opt-out, distinct from None."""
        assert is_enterprise_hf_org("ibm-research", []) is False


class TestIsEnterpriseHfOrgMatching:
    @pytest.mark.parametrize("org", ["ibm-research", "ibm-granite"])
    def test_listed_orgs_are_enterprise(self, org):
        assert is_enterprise_hf_org(org, ["ibm-research", "ibm-granite"]) is True

    def test_unlisted_org_is_not_enterprise(self):
        assert is_enterprise_hf_org("my-user", ["ibm-research"]) is False

    def test_match_is_case_insensitive(self):
        """HF namespaces are case-insensitive, so matching must be too."""
        assert is_enterprise_hf_org("IBM-Research", ["ibm-research"]) is True
        assert is_enterprise_hf_org("ibm-research", ["IBM-RESEARCH"]) is True

    def test_surrounding_whitespace_is_ignored(self):
        assert is_enterprise_hf_org("  ibm-research  ", [" ibm-research "]) is True

    def test_no_substring_or_prefix_matching(self):
        """Exact match only — a prefix must not qualify."""
        assert is_enterprise_hf_org("ibm-research-team", ["ibm-research"]) is False
        assert is_enterprise_hf_org("ibm", ["ibm-research"]) is False

    def test_wildcard_is_not_special(self):
        """'*' is a literal name, not a wildcard (exact-match-only design)."""
        assert is_enterprise_hf_org("anything", ["*"]) is False

    def test_empty_and_none_org_are_not_enterprise(self):
        assert is_enterprise_hf_org("", ["ibm-research"]) is False
        assert is_enterprise_hf_org(None, ["ibm-research"]) is False

    def test_empty_entries_in_list_are_ignored(self):
        assert is_enterprise_hf_org("", ["", None, "ibm-research"]) is False
        assert is_enterprise_hf_org("ibm-research", ["", "ibm-research"]) is True
