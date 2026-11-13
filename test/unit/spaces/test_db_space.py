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

"""A db-backed space (git_repo_uri: db://spaces/<name>) loads from its stored
snapshots with no git access, and keeps its source repo as its identity."""

import io
import zipfile
from pathlib import Path
from typing import Dict

import pytest

import gbserver.build.space as space_mod
import gbserver.spaces.db_space as db_space
from gbcommon.uri.space import SpaceURI
from gbcommon.uri.uri import URI
from gbserver.buildrunner.validation import _same_space_repo
from gbserver.spaces.db_space import (
    DbUri,
    create_space,
    get_space_config_uri,
    make_db_uri,
    parse_db_uri,
    resolve_db_uri,
)
from gbserver.storage.stored_space import RepoSnapshot, StoredSpace

SOURCE = "https://github.example.com/org/gbspace-x"
ASSETS_SOURCE = "https://github.example.com/org/assets"

SPACE_YAML = """\
name: x
secret_manager:
  type: env
  config: {}
base_uris:
  - db://spaces/x/assets
"""


def _zip(files: Dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return buf.getvalue()


def _snapshot(files: Dict[str, str], source, ref: str = "gbspace-config"):
    snap = RepoSnapshot(source_uri=source, source_ref=ref, source_commit="abc")
    snap.save_archive(_zip(files))
    return snap


def _row(name: str = "x", **kw) -> StoredSpace:
    return StoredSpace(
        name=name,
        git_repo_uri=kw.pop("git_repo_uri", SOURCE),
        space_uri=kw.pop("space_uri", make_db_uri(name)),
        space=kw.pop("space", _snapshot({"space.yaml": SPACE_YAML}, SOURCE)),
        assets=kw.pop(
            "assets",
            _snapshot({"steps/echo/step.yaml": "name: echo\n"}, ASSETS_SOURCE),
        ),
        **kw,
    )


@pytest.fixture
def rows(tmp_path, monkeypatch):
    """In-memory gb_spaces, and a per-test snapshot cache root."""
    table: Dict[str, StoredSpace] = {}

    def get_row(name):
        if name not in table:
            raise ValueError(f"db space URI names unknown space {name!r}")
        return table[name]

    monkeypatch.setattr(db_space, "_get_row", get_row)
    monkeypatch.setattr(db_space, "get_snapshot_cache_root", lambda: tmp_path / "c")
    return table


@pytest.mark.parametrize(
    "uri, expected",
    [
        ("db://spaces/x", DbUri("x")),
        ("db://spaces/x/space", DbUri("x")),
        ("db://spaces/x/assets", DbUri("x", "assets")),
        ("db://spaces/x/assets#subdirectory=a/b", DbUri("x", "assets", "a/b")),
        ("db://spaces/a%2Fb", DbUri("a/b")),
    ],
)
def test_parse(uri, expected):
    assert parse_db_uri(uri) == expected


@pytest.mark.parametrize(
    "uri", ["db://space", "db://spaces/", "db://spaces/x/other", "db://other/x"]
)
def test_parse_rejects(uri):
    with pytest.raises(ValueError):
        parse_db_uri(uri)


def test_make_round_trips():
    assert parse_db_uri(make_db_uri("a/b", "assets")) == DbUri("a/b", "assets")


def test_resolve_unpacks_once_and_reuses(rows, tmp_path):
    rows["x"] = _row()
    first = resolve_db_uri("db://spaces/x")
    assert Path(first.file_uri[len("file://") :], "space.yaml").is_file()
    assert first.identity_uri == (
        "git+ssh://github.example.com/org/gbspace-x.git@gbspace-config"
    )
    assert resolve_db_uri("db://spaces/x") == first
    assert len(list((tmp_path / "c").iterdir())) == 1  # no temp leftovers


def test_refreshed_snapshot_gets_new_dir(rows):
    rows["x"] = _row()
    before = resolve_db_uri("db://spaces/x").file_uri
    rows["x"] = _row(space=_snapshot({"space.yaml": SPACE_YAML + "#\n"}, SOURCE))
    assert resolve_db_uri("db://spaces/x").file_uri != before


def test_partial_extraction_is_redone(rows, tmp_path):
    rows["x"] = _row()
    path = Path(resolve_db_uri("db://spaces/x").file_uri[len("file://") :])
    (path / ".gb-snapshot-complete").unlink()
    (path / "space.yaml").unlink()
    resolve_db_uri("db://spaces/x")
    assert (path / "space.yaml").is_file()


def test_subdirectory(rows):
    rows["x"] = _row()
    uri = resolve_db_uri("db://spaces/x/assets#subdirectory=steps").file_uri
    assert Path(uri[len("file://") :], "echo", "step.yaml").is_file()


def test_missing_snapshot(rows):
    rows["x"] = _row(assets=None)
    with pytest.raises(ValueError, match="no assets snapshot"):
        resolve_db_uri("db://spaces/x/assets")


def test_alias_uses_target_row(rows):
    rows["x"] = _row()
    alias = StoredSpace(name="alias", git_repo_uri=SOURCE, space_uri=make_db_uri("x"))
    assert get_space_config_uri(alias) == resolve_db_uri("db://spaces/x").file_uri


def test_git_backed_unchanged():
    row = StoredSpace(name="g", git_repo_uri=SOURCE)
    assert row.get_space_uri() == SOURCE
    assert get_space_config_uri(row, token="") == (
        "git+ssh://github.example.com/org/gbspace-x.git"
    )


def test_create_space_loads_offline(rows, monkeypatch):
    """Space is served from the snapshots and identifies as the source repo."""
    rows["x"] = _row()
    seen_uris = []
    real = space_mod.SpaceSecretManager.get_spacesecretmanager

    def spy(*args, **kwargs):
        seen_uris.append(kwargs["uri"])
        return real(*args, **kwargs)

    monkeypatch.setattr(space_mod.SpaceSecretManager, "get_spacesecretmanager", spy)
    space = create_space(rows["x"])

    assert space.uristr.startswith("file://")
    assert space.identity_uri == (
        "git+ssh://github.example.com/org/gbspace-x.git@gbspace-config"
    )
    # Secret groups are matched on the source repo, never the local copy.
    assert seen_uris == [space.identity_uri]
    # The space-match check in validation passes for the stored row.
    assert _same_space_repo(rows["x"].git_repo_uri, space.identity_uri)
    # db://spaces/x/assets became the unpacked assets snapshot.
    assets_uri = resolve_db_uri("db://spaces/x/assets").file_uri
    assert SpaceURI._thread_local.base_uris[:2] == [space.uristr, assets_uri]
    dest = Path(space.uristr[len("file://") :]).parent / "pulled"
    assert URI.get_uri("space://steps/echo").pull(dest=dest)
    assert (dest / "echo" / "step.yaml").is_file()


def test_without_archives_keeps_metadata():
    row = _row()
    stripped = row.without_archives()
    assert stripped.space.archive is None and stripped.assets.archive is None
    assert stripped.space.source_commit == "abc"
    assert row.space.archive  # original untouched
    assert "archive" not in repr(row.space)


def test_snapshot_round_trips_through_json():
    row = _row()
    again = StoredSpace.model_validate_json(row.model_dump_json())
    assert again.space.load_archive() == row.space.load_archive()
    assert again.assets.source_uri == ASSETS_SOURCE


def test_portable_uri_round_trips(rows):
    rows["x"] = _row()
    local = resolve_db_uri("db://spaces/x").file_uri + "/environments/e"
    portable = db_space.portable_uri(local)
    assert portable == "db://spaces/x#subdirectory=environments/e"
    assert resolve_db_uri(portable).file_uri == local
    assert db_space.portable_uri("git+ssh://h/o/r.git") == "git+ssh://h/o/r.git"


def test_base_uri_may_not_name_another_space(rows):
    """A space.yaml naming another row's snapshot must not get its assets."""
    rows["other"] = _row(name="other")
    yaml_other = SPACE_YAML.replace("db://spaces/x/assets", "db://spaces/other/assets")
    rows["x"] = _row(space=_snapshot({"space.yaml": yaml_other}, SOURCE))
    with pytest.raises(ValueError, match="another space's snapshot"):
        create_space(rows["x"])


def test_concurrent_finish_is_kept(rows, tmp_path, monkeypatch):
    """If another extractor completes first, its tree is reused, not deleted."""
    rows["x"] = _row()
    real_extract = db_space.extract_archive

    def extract_then_race(data, out, **kw):
        ok = real_extract(data, out, **kw)
        # Simulate the other process finishing between our check and rename.
        digest = out.name.split("-")[0].lstrip(".")
        other = tmp_path / "c" / digest
        other.mkdir(parents=True)
        (other / "theirs").touch()
        (other / ".gb-snapshot-complete").touch()
        return ok

    monkeypatch.setattr(db_space, "extract_archive", extract_then_race)
    path = Path(resolve_db_uri("db://spaces/x").file_uri[len("file://") :])
    assert (path / "theirs").is_file()


def test_identity_is_git_repo_uri_in_any_form(rows):
    """Space reports git_repo_uri (its config form), whatever form it is stored in."""
    row = _row(
        git_repo_uri="git@github.example.com:org/gbspace-x.git",
        space=_snapshot({"space.yaml": SPACE_YAML}, source=None),
    )
    rows["x"] = row
    space = create_space(row)
    assert space.identity_uri == (
        "git+ssh://github.example.com/org/gbspace-x.git@gbspace-config"
    )
    assert _same_space_repo(row.git_repo_uri, space.identity_uri)


def test_damaged_tree_in_use_is_not_deleted(rows, tmp_path):
    """A tree that lost its marker is left alone; a sibling copy is served."""
    rows["x"] = _row()
    path = Path(resolve_db_uri("db://spaces/x").file_uri[len("file://") :])
    (path / ".gb-snapshot-complete").unlink()  # still has space.yaml: in use
    other = Path(resolve_db_uri("db://spaces/x").file_uri[len("file://") :])
    assert other != path
    assert (path / "space.yaml").is_file() and (other / "space.yaml").is_file()


def test_cache_root_must_be_private(rows, tmp_path, monkeypatch):
    rows["x"] = _row()
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o777)
    shared.chmod(0o777)
    monkeypatch.setattr(db_space, "get_snapshot_cache_root", lambda: shared)
    with pytest.raises(ValueError, match="refusing snapshot cache"):
        resolve_db_uri("db://spaces/x")
