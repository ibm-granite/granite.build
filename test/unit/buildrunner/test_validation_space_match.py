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

"""Unit tests for validation's token-free space-repo comparison.

Guards the regression where ``validate_stored_build`` re-derived the space
config URI with a GitHub token that was frozen empty at import time (git.py
imported before the token env var is set), causing the derived URI to drop the
``@<config-branch>`` suffix and mismatch the runner's ``space.uristr``.
``_same_git_repo`` compares only host/owner/repo, so scheme, ``.git``,
``@<ref>`` and ``#subdirectory`` differences don't matter.
"""

import pytest

from gbserver.buildrunner.validation import _same_git_repo


class TestSameGitRepo:
    """`_same_git_repo` matches on repo identity, ignoring scheme/ref/subdir."""

    # The exact pair from the failing build-setup validation.
    STORED = "https://github.ibm.com/granite-dot-build/gb-test"

    @pytest.mark.parametrize(
        "other",
        [
            "git+ssh://github.ibm.com/granite-dot-build/gb-test.git@gbspace-config",
            "git+ssh://github.ibm.com/granite-dot-build/gb-test.git",  # token empty: no branch
            "git+ssh://github.ibm.com/granite-dot-build/gb-test.git@gbspace-config#subdirectory=steps/x",
            "https://github.ibm.com/granite-dot-build/gb-test",
        ],
    )
    def test_same_repo_variants_match(self, other: str) -> None:
        """Scheme, .git, @<ref>, and #subdirectory differences still match."""
        assert _same_git_repo(self.STORED, other)

    @pytest.mark.parametrize(
        "other",
        [
            "git+ssh://github.ibm.com/granite-dot-build/other-repo.git@gbspace-config",
            "git+ssh://github.ibm.com/someone-else/gb-test.git",
            "git+ssh://example.com/granite-dot-build/gb-test.git",
        ],
    )
    def test_different_repo_does_not_match(self, other: str) -> None:
        """A different host, owner, or repo must not compare equal."""
        assert not _same_git_repo(self.STORED, other)
