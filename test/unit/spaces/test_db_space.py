#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

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
from gbserver.spaces import snapshot_cache
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
BASE_SOURCE = "https://github.example.com/org/assets"

SPACE_YAML = """\
name: x
secret_manager:
  type: env
  config: {}
base_uris:
  - db://spaces/x/base_uri_0
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
        base_uris=kw.pop(
            "base_uris",
            [_snapshot({"steps/echo/step.yaml": "name: echo\n"}, BASE_SOURCE)],
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
    monkeypatch.setenv("GBSERVER_DB_SPACE_CACHE_DIR", str(tmp_path / "c"))
    yield table
    snapshot_cache.release_holds()


@pytest.mark.parametrize(
    "uri, expected",
    [
        ("db://spaces/x", DbUri("x")),
        ("db://spaces/x/space", DbUri("x")),
        ("db://spaces/x/base_uri_0", DbUri("x", "base_uri_0")),
        ("db://spaces/x/base_uri_12", DbUri("x", "base_uri_12")),
        ("db://spaces/x/base_uri_0#subdirectory=a/b", DbUri("x", "base_uri_0", "a/b")),
        ("db://spaces/a%2Fb", DbUri("a/b")),
    ],
)
def test_parse(uri, expected):
    assert parse_db_uri(uri) == expected


@pytest.mark.parametrize(
    "uri",
    [
        "db://space",
        "db://spaces/",
        "db://spaces/x/other",
        "db://other/x",
        "db://spaces/x/assets",
        "db://spaces/x/base_uri_01",
        "db://spaces/x/base_uri_",
    ],
)
def test_parse_rejects(uri):
    with pytest.raises(ValueError):
        parse_db_uri(uri)


def test_make_round_trips():
    assert parse_db_uri(make_db_uri("a/b", 3)) == DbUri("a/b", "base_uri_3")
    assert parse_db_uri(make_db_uri("a/b", 3)).base_uri_index == 3
    assert parse_db_uri(make_db_uri("a/b")).base_uri_index is None


def test_subdirectory(rows):
    rows["x"] = _row()
    uri = resolve_db_uri("db://spaces/x/base_uri_0#subdirectory=steps").file_uri
    assert Path(uri[len("file://") :], "echo", "step.yaml").is_file()


def test_missing_snapshot(rows):
    rows["x"] = _row(base_uris=None)
    with pytest.raises(ValueError, match="no base_uri_0 snapshot"):
        resolve_db_uri("db://spaces/x/base_uri_0")


def test_other_rows_space_uri_is_not_supported(rows):
    """The notation could name another row; serving from it is not supported."""
    rows["x"] = _row()
    other = StoredSpace(name="alias", git_repo_uri=SOURCE, space_uri=make_db_uri("x"))
    with pytest.raises(ValueError, match="not supported"):
        get_space_config_uri(other)
    with pytest.raises(ValueError, match="not supported"):
        create_space(other)


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
    # db://spaces/x/base_uri_0 became the unpacked base_uris[0] snapshot.
    assets_uri = resolve_db_uri("db://spaces/x/base_uri_0").file_uri
    assert SpaceURI._thread_local.base_uris[:2] == [space.uristr, assets_uri]
    dest = Path(space.uristr[len("file://") :]).parent / "pulled"
    assert URI.get_uri("space://steps/echo").pull(dest=dest)
    assert (dest / "echo" / "step.yaml").is_file()


def test_without_archives_keeps_metadata():
    row = _row()
    stripped = row.without_archives()
    assert stripped.space.archive is None and stripped.base_uris[0].archive is None
    assert stripped.space.source_commit == "abc"
    assert row.space.archive  # original untouched
    assert "archive" not in repr(row.space)


def test_snapshot_round_trips_through_json():
    row = _row()
    again = StoredSpace.model_validate_json(row.model_dump_json())
    assert again.space.load_archive() == row.space.load_archive()
    assert again.base_uris[0].source_uri == BASE_SOURCE


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
    yaml_other = SPACE_YAML.replace(
        "db://spaces/x/base_uri_0", "db://spaces/other/base_uri_0"
    )
    rows["x"] = _row(space=_snapshot({"space.yaml": yaml_other}, SOURCE))
    with pytest.raises(ValueError, match="another space's snapshot"):
        create_space(rows["x"])


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


@pytest.mark.parametrize(
    "subdirectory", ["../other/steps", "steps/../../other", "/etc", "a/../../x"]
)
def test_subdirectory_may_not_escape_the_snapshot(rows, subdirectory):
    """'..' or an absolute path would reach a sibling (another space's) snapshot."""
    rows["x"] = _row()
    with pytest.raises(ValueError, match="escapes the snapshot"):
        resolve_db_uri(f"db://spaces/x/base_uri_0#subdirectory={subdirectory}")


def test_subdirectory_within_the_snapshot_is_fine(rows):
    rows["x"] = _row()
    uri = resolve_db_uri(
        "db://spaces/x/base_uri_0#subdirectory=steps/../steps"
    ).file_uri
    assert uri.endswith("/steps")


def test_slots_keep_base_uris_order(rows):
    """Each base_uris[i] is served from its slot, in its original position."""
    yaml_multi = SPACE_YAML.replace(
        "  - db://spaces/x/base_uri_0\n",
        "  - db://spaces/x/base_uri_0\n"
        "  - steps-local\n"
        "  - db://spaces/x/base_uri_2\n",
    )
    first = _snapshot({"steps/a/step.yaml": "a\n"}, BASE_SOURCE)
    third = _snapshot({"steps/c/step.yaml": "c\n"}, "https://github.example.com/o/c")
    rows["x"] = _row(
        space=_snapshot({"space.yaml": yaml_multi, "steps-local/x": ""}, SOURCE),
        base_uris=[first, None, third],
    )
    space = create_space(rows["x"])
    bases = SpaceURI._thread_local.base_uris
    assert bases[1] == resolve_db_uri("db://spaces/x/base_uri_0").file_uri
    assert bases[2] == f"{space.uristr}/steps-local"
    assert bases[3] == resolve_db_uri("db://spaces/x/base_uri_2").file_uri


def test_relative_entry_may_not_escape_the_snapshot(rows):
    """A relative file base_uri of a db space can't reach a sibling snapshot."""
    yaml_rel = SPACE_YAML.replace("db://spaces/x/base_uri_0", "file://../other")
    rows["x"] = _row(space=_snapshot({"space.yaml": yaml_rel}, SOURCE))
    with pytest.raises(ValueError, match="escapes its snapshot"):
        create_space(rows["x"])
