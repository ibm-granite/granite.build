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

"""Tests for ``Build._read_env_types`` env-directory resolution.

The build-creation validator reads each target env's ``type``/``subtype`` to
scope ``space://steps/<name>`` resolution (env-class-match tier).  A ``file://``
space carries the env dir directly in the resolved env URI's ``uri.path``, but a
**git-backed** space resolves ``space://environments/...`` to a ``GitURI`` whose
``uri.path`` is the repo *URL* path — not a local clone.  These tests pin that
git-backed env URIs are materialized to their local clone (via
``get_path_in_repo_from_cache``) so the env class is still read, and that the
``file://`` path is unchanged.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from urllib.parse import urlparse

from gbserver.build.build import Build


def _write_env_yaml(env_dir: Path, env_type: str, subtype: Optional[str]) -> None:
    """Write a minimal ``environment.yaml`` with ``type`` (and optional ``subtype``)."""
    env_dir.mkdir(parents=True, exist_ok=True)
    body = f"type: {env_type}\n"
    if subtype is not None:
        body += f"subtype: {subtype}\n"
    (env_dir / "environment.yaml").write_text(body, encoding="utf-8")


class _FakeGitEnvURI:
    """Stand-in for a resolved git-backed env URI.

    Mimics a ``GitURI``: ``uri.path`` is the repo URL path (never a local dir),
    and ``get_path_in_repo_from_cache`` returns the already-materialized clone
    subdir that actually holds ``environment.yaml``.
    """

    def __init__(self, url: str, local_dir: Optional[Path]) -> None:
        self.uri = urlparse(url)
        self._local_dir = local_dir

    def get_path_in_repo_from_cache(self, force: bool = False) -> Optional[Path]:
        """Return the reused local clone subdir (or ``None`` on clone failure)."""
        return self._local_dir


def test_read_env_types_materializes_git_backed_env(tmp_path):
    """A git-backed env URI resolves its class/subtype via the local clone.

    Regression: previously ``_read_env_types`` used ``uri.path`` verbatim (a git
    URL path, not a dir), so it returned ``(None, None)`` and left builtin
    ``space://steps/<name>`` URIs unresolvable during validation.
    """
    env_dir = tmp_path / "clone" / "environments" / "skypilot" / "slurm" / "bluevela"
    _write_env_yaml(env_dir, env_type="Skypilot", subtype="slurm")
    git_uri = _FakeGitEnvURI(
        "git+ssh://github.ibm.com/granite-dot-build/gb-test.git@gbspace-config"
        "#subdirectory=environments/skypilot/slurm/bluevela",
        local_dir=env_dir,
    )

    assert Build._read_env_types(git_uri) == ("Skypilot", "slurm")


def test_read_env_types_git_clone_failure_degrades_to_none(tmp_path):
    """A git env URI whose clone can't be materialized skips the facet (no raise)."""
    git_uri = _FakeGitEnvURI(
        "git+ssh://github.ibm.com/granite-dot-build/gb-test.git@gbspace-config"
        "#subdirectory=environments/skypilot/slurm/bluevela",
        local_dir=None,
    )

    assert Build._read_env_types(git_uri) == (None, None)


def test_read_env_types_local_file_uri_unchanged(tmp_path):
    """A local ``file://`` env dir is read directly from ``uri.path`` (no regression)."""
    env_dir = tmp_path / "environments" / "skypilot" / "slurm" / "bluevela"
    _write_env_yaml(env_dir, env_type="Skypilot", subtype=None)
    file_uri = SimpleNamespace(uri=urlparse(f"file://{env_dir}"))

    assert Build._read_env_types(file_uri) == ("Skypilot", None)
