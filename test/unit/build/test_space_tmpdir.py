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
long-lived rest-server filled ephemeral storage until the pod was evicted. These
tests pin the fixed behavior: the checkout is recorded on the instance, one-shot
callers register it for atexit cleanup, the cache-owned path does not, ``reclaim``
deletes it, and ``experiments/`` is still pruned.
"""

import textwrap

import pytest

import gbserver.build.space as space_mod
import gbserver.utils.filesystem as filesystem
from gbserver.build.space import Space

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


def test_records_tmpdir_and_prunes_experiments(local_space):
    space = Space(f"file://{local_space}", manage_tmpdir=False)
    # The checkout dir is recorded and exists during use.
    assert space.a_tmpdir.exists()
    # base_uris are stored for the cache to re-apply on serving threads.
    assert space.uristr in space.base_uris
    # experiments/ was pruned from the retained checkout.
    assert not list(space.a_tmpdir.glob("**/experiments"))
    # space.yaml (the rest of the checkout) is retained.
    assert list(space.a_tmpdir.glob("**/space.yaml"))


def test_reclaim_deletes_checkout(local_space):
    space = Space(f"file://{local_space}", manage_tmpdir=False)
    tmpdir = space.a_tmpdir
    assert tmpdir.exists()
    space.reclaim()
    assert not tmpdir.exists()
    # Idempotent.
    space.reclaim()


def test_one_shot_caller_registers_atexit(local_space, monkeypatch):
    """Default (manage_tmpdir=True) registers the checkout for atexit cleanup so
    it is reclaimed on process exit rather than leaked."""
    registered = []
    monkeypatch.setattr(filesystem, "_TEMP_DIRS", registered, raising=False)
    # space_mod.create_temp_subdir is the imported reference used by __init__.
    monkeypatch.setattr(space_mod, "create_temp_subdir", filesystem.create_temp_subdir)

    space = Space(f"file://{local_space}", manage_tmpdir=True)
    assert str(space.a_tmpdir) in registered


def test_cache_owned_path_not_atexit_registered(local_space, monkeypatch):
    """The cache-owned path (manage_tmpdir=False) must NOT register for atexit —
    the cache reclaims it explicitly on eviction instead."""
    registered = []
    monkeypatch.setattr(filesystem, "_TEMP_DIRS", registered, raising=False)

    space = Space(f"file://{local_space}", manage_tmpdir=False)
    assert str(space.a_tmpdir) not in registered
