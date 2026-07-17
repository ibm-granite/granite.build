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

"""Unit tests for validation's scheme-aware space-repo comparison.

Guards the regression where ``validate_stored_build`` re-derived the space config
URI with a GitHub token that was frozen empty at import time (git.py imported
before the token env var is set), dropping the ``@<config-branch>`` suffix and
mismatching the runner's ``space.uristr``. ``_same_space_repo`` compares identity
per scheme: ``(host, owner, repo)`` for remote git (ignoring scheme, userinfo,
port, ``.git``, ``@<ref>``, ``#subdirectory``, case) and the normalized path for
local ``file://`` spaces.
"""

import pytest

from gbserver.buildrunner.validation import _same_space_repo


class TestSameSpaceRepo:
    """`_same_space_repo` matches repo identity per scheme, crash-free."""

    STORED = "https://github.ibm.com/granite-dot-build/gb-test"

    @pytest.mark.parametrize(
        "other",
        [
            # The exact pair from the failing build-setup validation.
            "git+ssh://github.ibm.com/granite-dot-build/gb-test.git@gbspace-config",
            "git+ssh://github.ibm.com/granite-dot-build/gb-test.git",  # token empty: no branch
            "git+ssh://github.ibm.com/granite-dot-build/gb-test.git@gbspace-config#subdirectory=steps/x",
            "https://github.ibm.com/granite-dot-build/gb-test",
            # SSH userinfo + port must not defeat the match (uses hostname).
            "git+ssh://git@github.ibm.com/granite-dot-build/gb-test.git@gbspace-config",
            "git+ssh://git@github.ibm.com:22/granite-dot-build/gb-test.git",
            # Case-insensitive host/owner/repo (GitHub treats these equal).
            "https://GitHub.IBM.com/Granite-Dot-Build/GB-Test",
            # scp-form git URL (the form git.py emits: git@host:org/repo.git).
            "git@github.ibm.com:granite-dot-build/gb-test.git",
            "git@github.ibm.com:granite-dot-build/gb-test",
            # Uppercase .GIT suffix must also be stripped (case-insensitive).
            "git+ssh://github.ibm.com/granite-dot-build/gb-test.GIT",
        ],
    )
    def test_same_git_repo_variants_match(self, other: str) -> None:
        """Scheme, userinfo/port, .git, @<ref>, #subdirectory, case, and scp form all match."""
        assert _same_space_repo(self.STORED, other)

    def test_scp_form_different_repo_does_not_match(self) -> None:
        """An scp-form URL for a different repo must not match, and must not be
        mistaken for a local path (regression: urlparse read scp as a file path)."""
        assert not _same_space_repo(
            self.STORED, "git@github.ibm.com:granite-dot-build/other.git"
        )
        # scp form vs a local path for a same-looking string are not equal.
        assert not _same_space_repo(
            "git@github.ibm.com:granite-dot-build/gb-test.git",
            "file:///granite-dot-build/gb-test",
        )

    @pytest.mark.parametrize(
        "other",
        [
            "git+ssh://github.ibm.com/granite-dot-build/other-repo.git@gbspace-config",
            "git+ssh://github.ibm.com/someone-else/gb-test.git",
            "git+ssh://example.com/granite-dot-build/gb-test.git",
        ],
    )
    def test_different_git_repo_does_not_match(self, other: str) -> None:
        """A different host, owner, or repo must not compare equal."""
        assert not _same_space_repo(self.STORED, other)

    def test_local_spaces_sharing_prefix_do_not_match(self) -> None:
        """Two distinct local spaces must NOT collapse to the same identity.

        Regression: the old get_uri_parts-based helper reduced
        file:///home/user/spaceA and .../spaceB both to ('', 'home', 'user').
        """
        a = "file:///home/user/spaceA"
        b = "file:///home/user/spaceB"
        assert not _same_space_repo(a, b)
        assert _same_space_repo(a, a)
        # Trailing-slash / redundant-segment normalization still matches.
        assert _same_space_repo(a, "file:///home/user/spaceA/")
        assert _same_space_repo(a, "file:///home/user/./spaceA")

    def test_local_and_git_do_not_match(self) -> None:
        """A local space and a git space are never the same space."""
        assert not _same_space_repo("file:///home/user/gb-test", self.STORED)

    @pytest.mark.parametrize(
        "pair",
        [
            ("file:///space", "file:///other"),  # short local URIs: no IndexError
            ("", "https://github.ibm.com/o/repo"),  # empty stored URI: graceful
            # missing repo segment (only owner) must not IndexError:
            (
                "https://github.ibm.com/onlyowner",
                "https://github.ibm.com/granite-dot-build/gb-test",
            ),
        ],
    )
    def test_short_or_empty_uris_do_not_crash(self, pair: tuple) -> None:
        """Short/empty URIs return a graceful mismatch, never raise."""
        a, b = pair
        assert _same_space_repo(a, b) is False
