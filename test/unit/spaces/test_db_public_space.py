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

"""PUBLIC_SPACE_GIT_URI names the public space's git repo. When the registered
public space is served from the database, code keyed on that constant must use
the stored snapshot and never reach git."""

import io
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

import gbserver.commands.command_create_spaces as create_spaces
import gbserver.spaces.db_space as db_space
from gbcommon.uri.git import GitURI
from gbcommon.uri.uri import URI
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.storage.stored_space import RepoSnapshot, StoredSpace
from gbserver.types import constants as c

PUBLIC_GIT = "https://github.example.com/org/gbspace-public"
SPACE_YAML = "name: public\nsecret_manager:\n  type: env\n  config: {}\n"


def _snapshot() -> RepoSnapshot:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("space.yaml", SPACE_YAML)
        zf.writestr("environments/bluevela/environment.yaml", "type: lsf\n")
    snap = RepoSnapshot(source_uri=PUBLIC_GIT, source_ref="gbspace-config")
    snap.save_archive(buf.getvalue())
    return snap


@pytest.fixture
def no_git(tmp_path, monkeypatch):
    """Snapshot cache under tmp_path; any git/GitHub access fails the test."""
    monkeypatch.setattr(db_space, "get_snapshot_cache_root", lambda: tmp_path / "c")
    monkeypatch.setattr(c, "PUBLIC_SPACE_GIT_URI", PUBLIC_GIT)
    monkeypatch.setattr(create_spaces, "PUBLIC_SPACE_GIT_URI", PUBLIC_GIT)

    def reached_git(*_a, **_k):
        raise AssertionError("git was reached for a db-backed public space")

    monkeypatch.setattr(GitURI, "get_gb_space_config_uri", staticmethod(reached_git))
    monkeypatch.setattr(GitURI, "pull", reached_git)
    c._DERIVED_ENV_FOR_FILES_URI_CACHE.clear()
    yield get_admin_storage().space_storage
    c._DERIVED_ENV_FOR_FILES_URI_CACHE.clear()


def _add_db_public(storage) -> StoredSpace:
    row = StoredSpace(
        name=c.PUBLIC_SPACE_NAME,
        git_repo_uri=PUBLIC_GIT,
        space_uri=db_space.make_db_uri(c.PUBLIC_SPACE_NAME),
        lakehouse_namespace="old",
        space=_snapshot(),
    )
    storage.add(row)
    return row


def test_env_files_uri_from_snapshot(no_git):
    _add_db_public(no_git)
    uri = c.get_supported_env_for_files_uri()
    assert uri.startswith("file://") and uri.endswith("/environments/bluevela")
    assert Path(uri[len("file://") :], "environment.yaml").is_file()
    assert URI.get_uri(uri).exists()


def test_env_files_uri_unreadable_snapshot_is_503(no_git):
    no_git.add(
        StoredSpace(
            name=c.PUBLIC_SPACE_NAME,
            git_repo_uri=PUBLIC_GIT,
            space_uri=db_space.make_db_uri(c.PUBLIC_SPACE_NAME),
        )
    )  # db-backed but no snapshot
    assert c.get_supported_env_for_files_uri() == ""


def test_env_files_uri_git_backed_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "PUBLIC_SPACE_GIT_URI", PUBLIC_GIT)
    monkeypatch.setattr(
        GitURI,
        "get_gb_space_config_uri",
        staticmethod(lambda uri, *a, **k: "git+ssh://github.example.com/o/p.git"),
    )
    c._DERIVED_ENV_FOR_FILES_URI_CACHE.clear()
    get_admin_storage().space_storage.add(
        StoredSpace(name=c.PUBLIC_SPACE_NAME, git_repo_uri=PUBLIC_GIT)
    )
    try:
        assert c.get_supported_env_for_files_uri().startswith("git+ssh://")
    finally:
        c._DERIVED_ENV_FOR_FILES_URI_CACHE.clear()


@pytest.mark.parametrize("extra", [["--replace"], ["--clear"]])
def test_create_spaces_keeps_db_public(no_git, extra):
    """The predefined public spec (PUBLIC_SPACE_GIT_URI) keeps the snapshot."""
    before = _add_db_public(no_git)
    spec = StoredSpace(
        name=c.PUBLIC_SPACE_NAME, git_repo_uri=PUBLIC_GIT, lakehouse_namespace="new"
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(create_spaces, "predefined_spaces", [spec])
        result = CliRunner().invoke(create_spaces.cli, extra, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    (row,) = [s for s in no_git.get_by_uuid(None) if s.name == c.PUBLIC_SPACE_NAME]
    assert row.git_repo_uri == PUBLIC_GIT
    assert row.space_uri == before.space_uri
    assert row.space.archive == before.space.archive
    assert row.lakehouse_namespace == "new"  # other fields come from the spec


def test_create_spaces_other_repo_replaces(no_git):
    """A spec for a different repo is a real re-point, not a refresh."""
    _add_db_public(no_git)
    other = "file:///nonexistent/other-space"
    spec = StoredSpace(name=c.PUBLIC_SPACE_NAME, git_repo_uri=other)
    kept = create_spaces._keep_db_snapshots(no_git, [spec])
    assert kept == [spec]


def test_load_paths_never_reach_git(no_git, monkeypatch):
    """Every path that loads a db-served space reads the snapshot, not git."""
    import gbserver.api.secrets as secrets_api
    from gbserver.spaces.db_space import create_space, get_space_config_uri
    from gbserver.spaces.local_secrets_isolation import load_space_config

    row = _add_db_public(no_git)
    # Build runner / validation.
    space = create_space(row)
    assert space.uristr.startswith("file://")
    assert space.identity_uri.startswith("git+ssh://github.example.com/org/")
    # space.yaml readers (local-secrets isolation, standalone secrets admin).
    assert load_space_config(get_space_config_uri(row)).name == "public"
    monkeypatch.setattr("gbcommon.types.gbenvconfig.is_standalone", lambda: True)
    secrets_api._space_secret_manager_cache.clear()
    admin = secrets_api._get_space_secret_admin(
        {
            "name": row.name,
            "git_repo_uri": row.git_repo_uri,
            "space_uri": row.space_uri,
        }
    )
    assert admin is not None
    secrets_api._space_secret_manager_cache.clear()


def test_create_spaces_clear_refuses_dangling_alias(no_git):
    """--clear would remove an alias's target: refuse before deleting anything."""
    _add_db_public(no_git)
    no_git.add(
        StoredSpace(
            name="alias",
            git_repo_uri=PUBLIC_GIT,
            space_uri=db_space.make_db_uri(c.PUBLIC_SPACE_NAME),
        )
    )
    spec = StoredSpace(name="alias", git_repo_uri=PUBLIC_GIT)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(create_spaces, "predefined_spaces", [spec])
        result = CliRunner().invoke(create_spaces.cli, ["--clear"])
    assert result.exit_code != 0
    assert "would not exist" in result.output
    assert {s.name for s in no_git.get_by_uuid(None)} == {"public", "alias"}


def test_env_files_storage_error_falls_back_to_git(monkeypatch):
    """A storage error is not a db-backed space: derive from git as before."""
    monkeypatch.setattr(c, "PUBLIC_SPACE_GIT_URI", PUBLIC_GIT)
    monkeypatch.setattr(
        GitURI,
        "get_gb_space_config_uri",
        staticmethod(lambda uri, *a, **k: "git+ssh://github.example.com/o/p.git"),
    )

    def broken():
        raise RuntimeError("db down")

    monkeypatch.setattr("gbserver.storage.singleton_storage.get_admin_storage", broken)
    c._DERIVED_ENV_FOR_FILES_URI_CACHE.clear()
    try:
        assert c.get_supported_env_for_files_uri().startswith("git+ssh://")
    finally:
        c._DERIVED_ENV_FOR_FILES_URI_CACHE.clear()
