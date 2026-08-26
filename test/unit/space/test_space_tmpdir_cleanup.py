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

"""The Space checkout tempdir must not leak (issue #300).

`Space.__init__` pulls the space repo into a `tempfile.mkdtemp()` checkout.
Nothing reads that checkout after construction, so on the long-lived rest-server
(one `Space` per request) it would otherwise accumulate and fill ephemeral
storage. These tests pin the hotfix: the whole checkout is removed before
`__init__` returns (retained only in debug mode).
"""

import tempfile
from pathlib import Path

import gbserver.build.space as space_mod
from gbserver.build.space import Space

SPACE_YAML = """\
name: test-space
secret_manager:
  type: env
  config: {}
"""


def _make_local_space(root: Path) -> str:
    """Write a minimal local space and return its file:// URI string."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "space.yaml").write_text(SPACE_YAML)
    return root.as_uri()


def _track_mkdtemp(monkeypatch):
    """Record every mkdtemp checkout Space creates, so we can assert on it."""
    created: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def _spy(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(Path(path))
        return path

    monkeypatch.setattr(space_mod.tempfile, "mkdtemp", _spy)
    return created


def test_checkout_dir_removed_after_construction(tmp_path, monkeypatch):
    monkeypatch.setattr(space_mod, "is_debug_mode", lambda: False)
    created = _track_mkdtemp(monkeypatch)
    space_uri = _make_local_space(tmp_path / "space")

    Space(space_uri)

    assert created, "Space.__init__ was expected to create a checkout tempdir"
    for checkout in created:
        assert not checkout.exists(), f"leaked checkout dir: {checkout}"


def test_repeated_construction_does_not_accumulate(tmp_path, monkeypatch):
    monkeypatch.setattr(space_mod, "is_debug_mode", lambda: False)
    created = _track_mkdtemp(monkeypatch)
    space_uri = _make_local_space(tmp_path / "space")

    for _ in range(5):
        Space(space_uri)

    assert len(created) == 5
    assert not any(c.exists() for c in created), "checkout dirs accumulated"


def test_debug_mode_retains_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(space_mod, "is_debug_mode", lambda: True)
    created = _track_mkdtemp(monkeypatch)
    space_uri = _make_local_space(tmp_path / "space")

    Space(space_uri)

    assert created and created[0].exists(), "debug mode should retain the checkout"


def test_local_space_source_files_are_not_deleted(tmp_path, monkeypatch):
    """Standalone local-space case: the user's original folder must survive.

    The checkout is a copy under /tmp; deleting it must never touch the
    on-disk space folder the file:// URI points at.
    """
    monkeypatch.setattr(space_mod, "is_debug_mode", lambda: False)
    space_dir = tmp_path / "my-space"
    space_uri = _make_local_space(space_dir)
    # An extra file next to space.yaml stands in for the user's real content.
    (space_dir / "keepme.txt").write_text("precious")

    Space(space_uri)

    assert space_dir.exists(), "local space folder was deleted"
    assert (space_dir / "space.yaml").exists(), "space.yaml was deleted"
    assert (space_dir / "keepme.txt").read_text() == "precious"
