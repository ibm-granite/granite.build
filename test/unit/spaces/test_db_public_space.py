#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

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
from gbserver.storage import singleton_storage
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


@pytest.fixture(autouse=True)
def _fresh_admin_storage(monkeypatch):
    """Build the admin storage afresh, on this test's temp db.

    Every test here adds a "public" space. A test that ran earlier on this
    worker (e.g. an integration build) may have built the singleton on a db it
    shares, and test/unit/conftest.py restores that singleton after each test
    rather than clearing it, so without this the second test's add collides.
    """
    monkeypatch.setattr(singleton_storage, "__STORAGE", None)


@pytest.fixture
def no_git(tmp_path, monkeypatch):
    """Snapshot cache under tmp_path; any git/GitHub access fails the test."""
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
    """A db:// URI the reader unpacks only while it reads environment.yaml."""
    _add_db_public(no_git)
    uri = c.get_supported_env_for_files_uri()
    assert uri == "db://spaces/public#subdirectory=environments/bluevela"
    with db_space.local_uri(uri) as local:
        path = Path(local[len("file://") :])
        assert (path / "environment.yaml").is_file()
        assert URI.get_uri(local).exists()
    assert not path.exists()


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


def _create(specs, *args, env=None):
    with pytest.MonkeyPatch.context() as mp:
        if env:
            mp.setattr(create_spaces, "LAKEHOUSE_ENVIRONMENT", env)
        mp.setattr(create_spaces, "predefined_spaces", specs)
        return CliRunner().invoke(create_spaces.cli, list(args))


def test_create_spaces_refuses_another_rows_space_uri(no_git):
    """Serving a row from another row's snapshot is not supported."""
    _add_db_public(no_git)
    spec = StoredSpace(
        name="standalone",
        git_repo_uri=PUBLIC_GIT,
        space_uri=db_space.make_db_uri(c.PUBLIC_SPACE_NAME),
    )
    result = _create([spec], "--clear")
    assert result.exit_code != 0
    assert (
        "not supported" in result.output and "db://spaces/standalone" in result.output
    )
    assert {s.name for s in no_git.get_by_uuid(None)} == {"public"}  # untouched


def test_create_spaces_refuses_db_spec_without_snapshot(no_git):
    spec = StoredSpace(
        name="x", git_repo_uri=PUBLIC_GIT, space_uri=db_space.make_db_uri("x")
    )
    result = _create([spec])
    assert result.exit_code != 0
    assert "gbserver space-snapshot --space-name x" in result.output
    assert no_git.get_by_uuid(None) == []


def test_create_spaces_prod_clear_without_force_checks_kept_rows(tmp_path):
    """PROD --clear without --force keeps its rows, so a new space that would
    share a kept row's local secrets is still refused."""

    def local_space(dirname):
        d = tmp_path / dirname
        d.mkdir()
        (d / "space.yaml").write_text(
            "name: shared\nsecret_manager:\n  type: local\n  config: {}\n"
        )
        return d.as_uri()

    storage = get_admin_storage().space_storage  # per-test SQLite (conftest)
    storage.add(StoredSpace(name="kept", git_repo_uri=local_space("a")))
    spec = StoredSpace(
        name="new", git_repo_uri=local_space("b"), lakehouse_namespace=""
    )
    result = _create([spec], "--clear", env="PROD")
    assert result.exit_code != 0
    assert "kept" in str(result.exception)
    assert {s.name for s in storage.get_by_uuid(None)} == {"kept"}
