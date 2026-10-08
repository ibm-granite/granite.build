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

"""A ``space://`` URI must resolve inside the space base it matched.

The env-agnostic fallback (tier 3) appends the URI's suffix to each base_uri
verbatim, so ``space://../../home/gbserver/.kube/config`` against a ``file:``
base -- in a build input, step.yaml, environment.yaml or monitor ref -- would
otherwise read a server file. ``..`` segments (also percent-encoded) are
rejected outright, and a ``file:`` result must stay inside its base after
symlinks are resolved.
"""

from pathlib import Path

import pytest

from gbcommon.uri.space import SpaceURI
from gbcommon.uri.uri import URI


@pytest.fixture
def space(tmp_path: Path):
    base = tmp_path / "space"
    (base / "environments" / "k8s").mkdir(parents=True)
    (base / "environments" / "k8s" / "environment.yaml").write_text("name: k8s\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "config").write_text("secret")
    (base / "environments" / "link").symlink_to(outside)
    with SpaceURI._scope_thread_local(base_uris=[base.as_uri()], space_secrets={}):
        yield base


def test_uri_inside_base_resolves(space):
    uri = URI.get_uri("space://environments/k8s")
    assert Path(uri.uri.path) == space / "environments" / "k8s"


@pytest.mark.parametrize(
    "uri",
    [
        "space://../outside/config",
        "space://environments/../../outside/config",
        "space://%2E%2E/outside/config",  # percent-encoded ".."
        "gb://environments/%2e%2e/%2e%2e/outside",
    ],
)
def test_parent_segments_rejected(space, uri):
    with pytest.raises(ValueError, match="Unresolvable space uri.*'\\.\\.'"):
        URI.get_uri(uri)


def test_symlink_out_of_file_base_rejected(space):
    with pytest.raises(ValueError, match="Unresolvable space uri"):
        URI.get_uri("space://environments/link/config")
