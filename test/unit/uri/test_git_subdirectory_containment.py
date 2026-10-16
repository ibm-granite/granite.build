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

"""A git URI's ``#subdirectory=`` must stay inside the clone.

``repo / "/abs"`` is just ``/abs``, ``parse_qs`` decodes ``%2E%2E`` back to
``..``, and a repo may contain a symlink pointing anywhere, so without a check a
step/environment ``git+https://...#subdirectory=/home/gbserver/.kube`` would copy
server files into the step (and on to compute).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from gbcommon.uri.git import GitURI
from gbcommon.uri.uri import URI


@pytest.fixture
def clone(tmp_path: Path):
    repo = tmp_path / "clone"
    (repo / "steps" / "cat").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "link").symlink_to(outside)
    with patch.object(GitURI, "get_repo_from_cache", return_value=repo):
        yield repo, outside


def _git(fragment: str) -> GitURI:
    uri = URI.get_uri(f"git+https://github.com/org/repo.git#{fragment}")
    assert isinstance(uri, GitURI)
    return uri


def test_subdirectory_inside_clone_resolves(clone):
    repo, _ = clone
    assert _git("subdirectory=steps/cat").get_path_in_repo_from_cache() == (
        repo / "steps" / "cat"
    )


@pytest.mark.parametrize(
    "fragment",
    [
        "subdirectory=/etc",  # absolute: repo / "/etc" == "/etc"
        "subdirectory=steps/../../outside",
        "subdirectory=%2E%2E/outside",  # percent-encoded "..", decoded by parse_qs
        "subdirectory=link",  # in-repo symlink to a directory outside the clone
    ],
)
def test_subdirectory_escaping_clone_rejected(clone, fragment):
    with pytest.raises(ValueError, match="outside the repository"):
        _git(fragment).get_path_in_repo_from_cache()


def test_pull_of_escaping_subdirectory_fails_without_copying(clone, tmp_path):
    dest = tmp_path / "dest"
    with patch("gbcommon.uri.git.sync_or_copy") as copy:
        assert _git("subdirectory=/etc").pull(dest) is False
    copy.assert_not_called()


def test_append_path_cannot_escape(clone):
    uri = _git("subdirectory=steps")
    uri.append_path("../../outside")
    with pytest.raises(ValueError, match="outside the repository"):
        uri.get_path_in_repo_from_cache()
