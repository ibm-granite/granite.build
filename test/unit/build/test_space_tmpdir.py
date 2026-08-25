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

"""Tests for the Space checkout temp-dir lifecycle (Leak A fix).

``Space.__init__`` used to leak its per-construction ``mkdtemp`` checkout — it
removed only the ``experiments/`` subtree and abandoned the rest, which on the
long-lived rest-server filled ephemeral storage until the pod was evicted.

For the common case nothing reads the checkout after ``__init__`` returns
(``space.yaml`` is parsed out of it, base_uris resolve against the original space
URI, and assetstores load into config objects with no retained path), so it is
deleted before the constructor returns. The exception is a space that bundles a
local ``file://`` asset/store resolving *inside* the checkout — then the dir is
kept (``self.a_tmpdir``) and reclaimed by the cache on eviction. These tests pin
both: no temp dir survives the common case, and the reference check that decides.
"""

import tempfile
import textwrap
from pathlib import Path

import pytest

from gbserver.build.space import Space, _anything_references, _uri_path_under

SPACE_YAML = textwrap.dedent("""\
    name: test-space
    secret_manager:
      type: local
      config: {}
    variables:
      FOO: bar
    """)


@pytest.fixture
def local_space(tmp_path):
    """A minimal local file:// space with an experiments/ dir to prune."""
    space_dir = tmp_path / "myspace"
    space_dir.mkdir()
    (space_dir / "space.yaml").write_text(SPACE_YAML)
    exp = space_dir / "experiments"
    exp.mkdir()
    (exp / "big.txt").write_text("x" * 1024)
    return space_dir


@pytest.fixture(autouse=True)
def _no_secret_fetch(monkeypatch):
    """Space construction should not touch any secret backend in these tests."""
    monkeypatch.setattr(Space, "_fetch_secrets", lambda self, username=None: {})


def _tempdir_count() -> int:
    """Number of gb space-checkout temp dirs currently under the temp root."""
    root = Path(tempfile.gettempdir())
    # Space uses bare tempfile.mkdtemp() -> names like 'tmpXXXXXXXX'.
    return sum(1 for p in root.glob("tmp*") if p.is_dir())


def test_no_checkout_dir_survives_construction(local_space):
    """After __init__ completes, the throwaway checkout dir is gone."""
    before = _tempdir_count()
    space = Space(f"file://{local_space}")
    after = _tempdir_count()
    # Net temp-dir count did not grow — the checkout was deleted in __init__.
    assert after <= before
    # The parsed state the cache/validation rely on is present.
    assert space.uristr in space.base_uris
    assert space.space_config.name == "test-space"


def test_repeated_construction_does_not_accumulate(local_space):
    """The whole point: repeated Space() must not pile up temp dirs."""
    before = _tempdir_count()
    for _ in range(5):
        Space(f"file://{local_space}")
    after = _tempdir_count()
    assert after <= before


def test_debug_mode_keeps_checkout_for_inspection(local_space, monkeypatch):
    """In debug mode the checkout is intentionally retained (mirrors Step)."""
    import gbserver.build.space as space_mod

    monkeypatch.setattr(space_mod, "is_debug_mode", lambda: True)
    before = _tempdir_count()
    space = Space(f"file://{local_space}")
    after = _tempdir_count()
    # Debug keeps the dir around, recorded on a_tmpdir for cleanup.
    assert after == before + 1
    assert space.a_tmpdir is not None and space.a_tmpdir.exists()
    assert space.space_config.name == "test-space"
    # reclaim() frees it (as the cache would on eviction).
    space.reclaim()
    assert space.a_tmpdir is None


def test_common_space_does_not_retain_checkout(local_space):
    """A space with no in-checkout references deletes the dir (a_tmpdir None)."""
    space = Space(f"file://{local_space}")
    assert space.a_tmpdir is None


class TestReferenceCheck:
    """The #4 guard: keep the checkout only when something resolves inside it."""

    def test_uri_path_under_file_inside(self, tmp_path):
        inside = tmp_path / "sub" / "store"
        assert _uri_path_under(f"file://{inside}", tmp_path) is True

    def test_uri_path_under_file_outside(self, tmp_path):
        assert _uri_path_under("file:///etc/hosts", tmp_path) is False

    def test_uri_path_under_non_file_scheme(self, tmp_path):
        # Remote schemes never point at the on-disk checkout.
        assert _uri_path_under("git+ssh://h/o/r.git", tmp_path) is False
        assert _uri_path_under("space://steps/x", tmp_path) is False

    def test_anything_references_git_only_false(self, tmp_path):
        assert (
            _anything_references(tmp_path, ["git+ssh://h/o/r.git", "space://x"], {})
            is False
        )

    def test_anything_references_base_uri_inside_true(self, tmp_path):
        inside = tmp_path / "assets"
        assert _anything_references(tmp_path, [f"file://{inside}"], {}) is True

    def test_anything_references_bundled_store_inside_true(self, tmp_path):
        inside = tmp_path / "store"
        assert (
            _anything_references(tmp_path, [], {f"file://{inside}": object()}) is True
        )
