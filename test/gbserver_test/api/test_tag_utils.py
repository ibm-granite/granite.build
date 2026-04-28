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

import pytest

from gbserver.api.utils import get_tags_to_set, split_tags
from gbserver.types.constants import SYSTEM_TAG_PREFIX


class MockTaggedItem:
    """Mock implementation of TaggedItem for testing."""

    def __init__(self, tags=None):
        self.tags = tags


class TestSplitTags:
    """Test the split_tags utility function."""

    def test_split_tags_with_system_tags(self):
        """Test splitting tags with system tags."""
        tags = ["sys-prod", "user-tag", "sys-approved"]
        sys_tags, user_tags = split_tags(tags)
        assert sys_tags == ["sys-prod", "sys-approved"]
        assert user_tags == ["user-tag"]

    def test_split_tags_only_user_tags(self):
        """Test splitting when there are only user tags."""
        tags = ["tag1", "tag2", "tag3"]
        sys_tags, user_tags = split_tags(tags)
        assert sys_tags == []
        assert user_tags == ["tag1", "tag2", "tag3"]

    def test_split_tags_only_system_tags(self):
        """Test splitting when there are only system tags."""
        tags = ["sys-prod", "sys-approved", "sys-test"]
        sys_tags, user_tags = split_tags(tags)
        assert sys_tags == ["sys-prod", "sys-approved", "sys-test"]
        assert user_tags == []

    def test_split_tags_empty_list(self):
        """Test splitting an empty list."""
        tags = []
        sys_tags, user_tags = split_tags(tags)
        assert sys_tags == []
        assert user_tags == []

    def test_split_tags_none(self):
        """Test splitting None."""
        tags = None
        sys_tags, user_tags = split_tags(tags)
        assert sys_tags == []
        assert user_tags == []


class TestGetTagsToSet:
    """Test the get_tags_to_set function."""

    # ============================================================================
    # Non-super user tests (is_super=False)
    # ============================================================================

    def test_non_super_set_empty_tags_with_system_tags(self):
        """Non-super user should be able to set tags to empty list, preserving system tags."""
        item = MockTaggedItem(tags=["sys-prod", "user-tag-1", "user-tag-2"])
        result = get_tags_to_set(
            is_super=False, tagged_item=item, tags=[], appending=False
        )
        # System tags should be preserved, user tags should be cleared
        assert "sys-prod" in result
        assert "user-tag-1" not in result
        assert "user-tag-2" not in result
        assert result == ["sys-prod"]

    def test_non_super_set_new_user_tags_with_system_tags(self):
        """Non-super user should be able to replace user tags while preserving system tags."""
        item = MockTaggedItem(tags=["sys-prod", "old-tag-1", "old-tag-2"])
        result = get_tags_to_set(
            is_super=False, tagged_item=item, tags=["new-tag"], appending=False
        )
        # System tags should be preserved, old user tags replaced with new ones
        assert "sys-prod" in result
        assert "new-tag" in result
        assert "old-tag-1" not in result
        assert "old-tag-2" not in result
        assert set(result) == {"sys-prod", "new-tag"}

    def test_non_super_set_empty_tags_without_system_tags(self):
        """Non-super user should be able to set tags to empty list when no system tags exist."""
        item = MockTaggedItem(tags=["user-tag-1", "user-tag-2"])
        result = get_tags_to_set(
            is_super=False, tagged_item=item, tags=[], appending=False
        )
        # All tags should be cleared
        assert result == []

    def test_non_super_set_multiple_user_tags_without_system_tags(self):
        """Non-super user should be able to set multiple user tags when no system tags exist."""
        item = MockTaggedItem(tags=["old-tag"])
        result = get_tags_to_set(
            is_super=False,
            tagged_item=item,
            tags=["tag1", "tag2", "tag3"],
            appending=False,
        )
        assert set(result) == {"tag1", "tag2", "tag3"}

    def test_non_super_append_user_tags_with_system_tags(self):
        """Non-super user should be able to append user tags while preserving system tags."""
        item = MockTaggedItem(tags=["sys-prod", "existing-tag"])
        result = get_tags_to_set(
            is_super=False, tagged_item=item, tags=["new-tag"], appending=True
        )
        # All tags should be present
        assert "sys-prod" in result
        assert "existing-tag" in result
        assert "new-tag" in result
        assert set(result) == {"sys-prod", "existing-tag", "new-tag"}

    def test_non_super_cannot_set_system_tags(self):
        """Non-super user should not be able to set or add system tags."""
        item = MockTaggedItem(tags=["user-tag"])
        with pytest.raises(Exception) as exc_info:
            get_tags_to_set(
                is_super=False, tagged_item=item, tags=["sys-new-tag"], appending=False
            )
        assert (
            "401" in str(exc_info.value) or "non-admin" in str(exc_info.value).lower()
        )

    def test_non_super_cannot_append_system_tags(self):
        """Non-super user should not be able to append system tags."""
        item = MockTaggedItem(tags=["user-tag"])
        with pytest.raises(Exception) as exc_info:
            get_tags_to_set(
                is_super=False, tagged_item=item, tags=["sys-new-tag"], appending=True
            )
        assert (
            "401" in str(exc_info.value) or "non-admin" in str(exc_info.value).lower()
        )

    def test_non_super_with_none_tags(self):
        """Non-super user should handle None tags correctly."""
        item = MockTaggedItem(tags=["sys-prod", "user-tag"])
        result = get_tags_to_set(
            is_super=False, tagged_item=item, tags=None, appending=False
        )
        # Only system tags should remain
        assert result == ["sys-prod"]

    def test_non_super_with_empty_item_tags(self):
        """Non-super user should handle items with no tags."""
        item = MockTaggedItem(tags=None)
        result = get_tags_to_set(
            is_super=False, tagged_item=item, tags=["tag1"], appending=False
        )
        assert result == ["tag1"]

    # ============================================================================
    # Super user tests (is_super=True)
    # ============================================================================

    def test_super_set_empty_tags_clears_all(self):
        """Super user should be able to clear all tags including system tags."""
        item = MockTaggedItem(tags=["sys-prod", "user-tag"])
        result = get_tags_to_set(
            is_super=True, tagged_item=item, tags=[], appending=False
        )
        # Super users can clear everything
        assert result == []

    def test_super_set_system_tags_only(self):
        """Super user should be able to set only system tags."""
        item = MockTaggedItem(tags=["sys-prod", "user-tag"])
        result = get_tags_to_set(
            is_super=True, tagged_item=item, tags=["sys-new"], appending=False
        )
        # Super users can set arbitrary tags
        assert result == ["sys-new"]

    def test_super_set_mixed_tags(self):
        """Super user should be able to set mixed system and user tags."""
        item = MockTaggedItem(tags=["sys-prod", "user-tag"])
        result = get_tags_to_set(
            is_super=True,
            tagged_item=item,
            tags=["sys-prod", "sys-new", "user-tag-new"],
            appending=False,
        )
        assert set(result) == {"sys-prod", "sys-new", "user-tag-new"}

    def test_super_append_tags(self):
        """Super user should be able to append tags."""
        item = MockTaggedItem(tags=["sys-prod", "user-tag"])
        result = get_tags_to_set(
            is_super=True,
            tagged_item=item,
            tags=["sys-new", "user-tag-new"],
            appending=True,
        )
        assert set(result) == {"sys-prod", "user-tag", "sys-new", "user-tag-new"}

    def test_super_with_none_tags(self):
        """Super user should handle None tags correctly."""
        item = MockTaggedItem(tags=["sys-prod", "user-tag"])
        result = get_tags_to_set(
            is_super=True, tagged_item=item, tags=None, appending=False
        )
        # When setting None, system tags are not preserved for super users
        assert result == []

    def test_super_with_empty_item_tags(self):
        """Super user should handle items with no tags."""
        item = MockTaggedItem(tags=None)
        result = get_tags_to_set(
            is_super=True,
            tagged_item=item,
            tags=["sys-prod", "user-tag"],
            appending=False,
        )
        assert set(result) == {"sys-prod", "user-tag"}

    # ============================================================================
    # Edge cases
    # ============================================================================

    def test_system_tag_prefix_consistency(self):
        """Ensure SYSTEM_TAG_PREFIX is used consistently."""
        item = MockTaggedItem(tags=[f"{SYSTEM_TAG_PREFIX}prod"])
        result = get_tags_to_set(
            is_super=False, tagged_item=item, tags=[], appending=False
        )
        assert f"{SYSTEM_TAG_PREFIX}prod" in result

    def test_large_number_of_tags(self):
        """Test with a large number of tags."""
        existing_tags = [f"sys-tag{i}" for i in range(5)] + [
            f"user-tag{i}" for i in range(95)
        ]
        new_tags = [f"new-tag{i}" for i in range(50)]
        item = MockTaggedItem(tags=existing_tags)
        result = get_tags_to_set(
            is_super=False, tagged_item=item, tags=new_tags, appending=False
        )
        # System tags should be preserved
        for i in range(5):
            assert f"sys-tag{i}" in result
        # New tags should be added
        for i in range(50):
            assert f"new-tag{i}" in result
        # Old user tags should not be present
        for i in range(95):
            assert f"user-tag{i}" not in result

    def test_duplicate_tags_preserved(self):
        """Test that duplicate tags in input are preserved in output."""
        item = MockTaggedItem(tags=["sys-prod", "tag1"])
        result = get_tags_to_set(
            is_super=False, tagged_item=item, tags=["tag1", "tag1"], appending=False
        )
        # Both duplicates should be present
        assert result.count("tag1") == 2

    def test_mixed_set_with_multiple_system_tags(self):
        """Non-super user setting tags with multiple existing system tags."""
        item = MockTaggedItem(tags=["sys-prod", "sys-approved", "sys-test", "user-tag"])
        result = get_tags_to_set(
            is_super=False, tagged_item=item, tags=["new-user-tag"], appending=False
        )
        # All system tags should be preserved
        assert "sys-prod" in result
        assert "sys-approved" in result
        assert "sys-test" in result
        assert "new-user-tag" in result
        assert "user-tag" not in result
