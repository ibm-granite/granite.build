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

"""Prwatcherconfig module."""

from pathlib import Path
from typing import Optional, Self, Tuple
from urllib.parse import urlparse

from gbserver.storage.stored_space import StoredSpace
from gbserver.types.spacesconfig import CLISpacesConfig
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def get_uri_parts(uri: str) -> Tuple[str, str, str, str, str]:
    """
    >>> urlparse('git+ssh://mysourcecontrol.com/granite-dot-build/assets.git#subdirectory=steps/hello-helm-data-upload')
    ParseResult(
        scheme='git+ssh',
        netloc='mysourcecontrol.com',
        path='/granite-dot-build/assets.git',
        params='',
        query='',
        fragment='subdirectory=steps/hello-helm-data-upload',
    )
    >>> Path(u.path).parts
    ('/', 'granite-dot-build', 'assets.git')

    Example: ('git+ssh', 'mysourcecontrol.com', 'granite-dot-build', 'assets', 'steps/hello-helm-data-upload')

    Returns (scheme, domain, owner, repo, sub_directory)
    """
    u = urlparse(uri)
    scheme = u.scheme
    domain = u.netloc
    upath_parts = Path(u.path).parts
    owner = upath_parts[1]
    repo = upath_parts[2]
    if repo.endswith(".git"):
        repo = repo.removesuffix(".git")
    subdirprefix = "subdirectory="
    subdir = u.fragment.removeprefix(subdirprefix) if u.fragment.startswith(subdirprefix) else ""
    return (scheme, domain, owner, repo, subdir)


class PrWatcherConfig(CLISpacesConfig):
    """The GitHub PR watcher config."""

    lh_max_retries: int = 3
    monitoring_interval: int = 5
    validate_inputs_are_registered: bool = True

    def get_space_from_pr_url(self: Self, pr_html_url: str) -> Optional[StoredSpace]:
        """Get the space from pr url."""
        _, domain, owner, repo_name, _ = get_uri_parts(pr_html_url)
        for space in self._spaces.values():
            repo_uri = space.git_repo_uri
            if repo_uri == "":
                logger.error("the git repo URI is empty for the space: %s", space)
                continue
            _, repo_domain, repo_owner, repo_repo_name, _ = get_uri_parts(repo_uri)
            if domain == repo_domain and owner == repo_owner and repo_name == repo_repo_name:
                return space
        return None

    def get_repo_from_pr_url(self: Self, pr_html_url: str) -> Optional[str]:
        """Returns a matching repo (or empty string) from the pull request URL"""
        stored_space = self.get_space_from_pr_url(pr_html_url=pr_html_url)
        if stored_space is None:
            return None
        return stored_space.git_repo_uri
