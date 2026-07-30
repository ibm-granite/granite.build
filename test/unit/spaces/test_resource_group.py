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

"""Unit tests for ``resolve_space_resource_group_id`` (table-first + HF fallback).

Storage and the HF API are mocked; no live Hub calls are made.
"""

from unittest.mock import MagicMock, patch

import pytest

from gbserver.spaces.resource_group import resolve_space_resource_group_id
from gbserver.storage.stored_space import StoredSpace


def _make_space(name="public", hf_default_resource_group_id=None):
    return StoredSpace(
        name=name,
        git_repo_uri="http://example/repo",
        lakehouse_namespace="lh",
        hf_default_resource_group_id=hf_default_resource_group_id,
    )


class TestResolveSpaceResourceGroupId:
    def test_returns_cached_id_without_hf_call(self):
        """A space row with a cached id short-circuits the HF lookup."""
        space = _make_space(hf_default_resource_group_id="cached-id")
        storage = MagicMock()
        storage.space_storage.get_by_name.return_value = space
        admin = MagicMock(return_value=storage)

        with (
            patch("gbserver.spaces.resource_group.get_admin_storage", admin),
            patch(
                "gbserver.spaces.resource_group.HfURI.resolve_resource_group_id_for_org"
            ) as mock_hf,
        ):
            result = resolve_space_resource_group_id(
                space_name="public",
                organization="ibm-research",
                token="tok",
            )

        assert result == "cached-id"
        mock_hf.assert_not_called()
        storage.space_storage.update.assert_not_called()

    def test_falls_back_to_hf_and_writes_back(self):
        """A row with no cached id triggers the HF lookup and a write-back."""
        space = _make_space(hf_default_resource_group_id=None)
        storage = MagicMock()
        storage.space_storage.get_by_name.return_value = space

        with (
            patch(
                "gbserver.spaces.resource_group.get_admin_storage",
                return_value=storage,
            ),
            patch(
                "gbserver.spaces.resource_group.HfURI.resolve_resource_group_id_for_org",
                return_value="resolved-id",
            ) as mock_hf,
        ):
            result = resolve_space_resource_group_id(
                space_name="public",
                organization="ibm-research",
                token="tok",
            )

        assert result == "resolved-id"
        mock_hf.assert_called_once()
        # The resolved id is written back onto the (same) space object.
        assert space.hf_default_resource_group_id == "resolved-id"
        storage.space_storage.update.assert_called_once_with(space)

    def test_no_row_falls_back_without_write_back(self):
        """When no space row exists, resolve via HF but do not persist."""
        storage = MagicMock()
        storage.space_storage.get_by_name.return_value = None

        with (
            patch(
                "gbserver.spaces.resource_group.get_admin_storage",
                return_value=storage,
            ),
            patch(
                "gbserver.spaces.resource_group.HfURI.resolve_resource_group_id_for_org",
                return_value="resolved-id",
            ),
        ):
            result = resolve_space_resource_group_id(
                space_name="unknown-space",
                organization="ibm-research",
                token="tok",
            )

        assert result == "resolved-id"
        storage.space_storage.update.assert_not_called()

    def test_unresolved_returns_none_no_write_back(self):
        """A failed HF lookup returns None and does not write back."""
        space = _make_space(hf_default_resource_group_id=None)
        storage = MagicMock()
        storage.space_storage.get_by_name.return_value = space

        with (
            patch(
                "gbserver.spaces.resource_group.get_admin_storage",
                return_value=storage,
            ),
            patch(
                "gbserver.spaces.resource_group.HfURI.resolve_resource_group_id_for_org",
                return_value=None,
            ),
        ):
            result = resolve_space_resource_group_id(
                space_name="public",
                organization="ibm-research",
                token="tok",
            )

        assert result is None
        storage.space_storage.update.assert_not_called()

    def test_explicit_non_default_name_bypasses_cache(self):
        """An explicit non-default resource_group_name ignores the cached default id.

        The cache holds ONLY the space's default group. When a caller requests a
        different group by name, the helper must NOT return (or overwrite) the
        cached default id: it resolves via the HF API and does not write back.
        """
        # Row has a cached DEFAULT id, but the request names a different group.
        space = _make_space(hf_default_resource_group_id="default-cached-id")
        storage = MagicMock()
        storage.space_storage.get_by_name.return_value = space

        with (
            patch(
                "gbserver.spaces.resource_group.get_admin_storage",
                return_value=storage,
            ),
            patch(
                "gbserver.spaces.resource_group.HfURI.space_name_to_resource_group_name",
                return_value="gbspace-public",
            ),
            patch(
                "gbserver.spaces.resource_group.HfURI.resolve_resource_group_id_for_org",
                return_value="non-default-id",
            ) as mock_hf,
        ):
            result = resolve_space_resource_group_id(
                space_name="public",
                organization="ibm-research",
                token="tok",
                resource_group_name="some-other-group",
            )

        # Resolved via HF (the explicit group), NOT the cached default id.
        assert result == "non-default-id"
        mock_hf.assert_called_once()
        # The cached default id is untouched (no poisoning).
        assert space.hf_default_resource_group_id == "default-cached-id"
        storage.space_storage.update.assert_not_called()

    def test_explicit_default_name_uses_cache(self):
        """An explicit name equal to the derived default still hits the cache."""
        space = _make_space(hf_default_resource_group_id="default-cached-id")
        storage = MagicMock()
        storage.space_storage.get_by_name.return_value = space

        with (
            patch(
                "gbserver.spaces.resource_group.get_admin_storage",
                return_value=storage,
            ),
            patch(
                "gbserver.spaces.resource_group.HfURI.space_name_to_resource_group_name",
                return_value="gbspace-public",
            ),
            patch(
                "gbserver.spaces.resource_group.HfURI.resolve_resource_group_id_for_org"
            ) as mock_hf,
        ):
            result = resolve_space_resource_group_id(
                space_name="public",
                organization="ibm-research",
                token="tok",
                resource_group_name="gbspace-public",
            )

        assert result == "default-cached-id"
        mock_hf.assert_not_called()
        storage.space_storage.update.assert_not_called()
