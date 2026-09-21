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

"""``get_public_repo_tags`` pagination and endpoint behavior.

The version check reads tags from the ``/repos/{org}/{repo}/tags`` list endpoint (which
peels annotated tags to their commit SHA) and follows ``Link`` pagination so repos with
more than one page of tags are fully covered.
"""

from unittest.mock import MagicMock, patch

from gbcli.utils import gh_clone


def _resp(payload, next_url=None):
    r = MagicMock()
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    r.links = {"next": {"url": next_url}} if next_url else {}
    return r


class TestGetPublicRepoTags:
    def test_follows_link_pagination(self):
        """All pages are concatenated by following the `next` link until it's absent."""
        page1 = _resp(
            [{"name": "v1.0.0", "commit": {"sha": "a"}}],
            next_url="https://api.github.com/next-page",
        )
        page2 = _resp([{"name": "v2.0.0", "commit": {"sha": "b"}}])

        with patch.object(gh_clone.requests, "get", side_effect=[page1, page2]) as get:
            tags = gh_clone.get_public_repo_tags("ibm-granite", "granite.build")

        assert [t["name"] for t in tags] == ["v1.0.0", "v2.0.0"]
        # Second request targets the `next` URL from the first response's Link header.
        assert get.call_count == 2
        assert get.call_args_list[1].args[0] == "https://api.github.com/next-page"

    def test_single_page_no_next(self):
        with patch.object(
            gh_clone.requests,
            "get",
            return_value=_resp([{"name": "v1.0.0", "commit": {"sha": "a"}}]),
        ) as get:
            tags = gh_clone.get_public_repo_tags("ibm-granite", "granite.build")
        assert len(tags) == 1
        assert get.call_count == 1

    def test_uses_tags_list_endpoint(self):
        """Targets /repos/.../tags (peeled commit SHAs), not /git/refs/tags."""
        with patch.object(gh_clone.requests, "get", return_value=_resp([])) as get:
            gh_clone.get_public_repo_tags("ibm-granite", "granite.build")
        url = get.call_args_list[0].args[0]
        assert url == "https://api.github.com/repos/ibm-granite/granite.build/tags"

    def test_timeout_is_bounded_connect_read_tuple(self):
        """Each request caps connect and read phases at the budget, so a hung page can't
        stall the CLI indefinitely."""
        with patch.object(gh_clone.requests, "get", return_value=_resp([])) as get:
            gh_clone.get_public_repo_tags("ibm-granite", "granite.build")
        timeout = get.call_args_list[0].kwargs["timeout"]
        assert isinstance(timeout, tuple) and len(timeout) == 2
        connect, read = timeout
        assert 0 < connect <= gh_clone.PUBLIC_REPO_TAGS_TIMEOUT_S
        assert 0 < read <= gh_clone.PUBLIC_REPO_TAGS_TIMEOUT_S
