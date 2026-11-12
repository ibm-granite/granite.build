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

"""`gbserver space-snapshot` stores only the branch head (no history), rewrites
space.yaml's assets base_uri to db://, and records the normalized source."""

import io
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import pytest
import yaml
from click.testing import CliRunner

import gbserver.commands.command_space_snapshot as cmd
import gbserver.spaces.db_space as db_space
from gbserver.storage.stored_space import StoredSpace

SPACE_YAML = """\
name: x
# the assets repo
base_uris:
  - git+ssh://github.example.com/org/assets.git@gbspace-config
secret_manager:
  type: env
  config: {}
"""


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo(path: Path, files: Dict[str, str], origin: str, branch: str) -> str:
    path.mkdir()
    _git(path, "init", "-q", "-b", branch)
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    _git(path, "config", "commit.gpgsign", "false")
    _git(path, "remote", "add", "origin", origin)
    (path / "old.txt").write_text("from history\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "old")
    _git(path, "rm", "-q", "old.txt")
    for name, text in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text)
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "head")
    (path / "untracked.txt").write_text("not committed\n")
    return _git(path, "rev-parse", "HEAD")


class _FakeSpaceStorage:
    def __init__(self):
        self.rows: Dict[str, StoredSpace] = {}

    def get_by_name(self, name):
        return self.rows.get(name)

    def get_by_uuid(self, _):
        return list(self.rows.values())

    def add(self, row):
        self.rows[row.name] = row

    def update(self, row, create_if_not_exist=False):
        assert row.name in self.rows
        self.rows[row.name] = row


@pytest.fixture(scope="module")
def repos(tmp_path_factory):
    """The space and assets checkouts; read-only, so built once per module."""
    root = tmp_path_factory.mktemp("repos")
    space_sha = _repo(
        root / "space",
        {"space.yaml": SPACE_YAML, "environments/e/environment.yaml": "type: bash\n"},
        "git@github.example.com:org/gbspace-x.git",
        "gbspace-config",
    )
    assets_sha = _repo(
        root / "assets",
        {"steps/echo/step.yaml": "name: echo\n"},
        "https://github.example.com/org/assets.git",
        "gbspace-config",
    )
    _repo(
        root / "other",
        {"a.txt": "a\n"},
        "https://github.example.com/org/other.git",
        "gbspace-config",
    )
    return SimpleNamespace(tmp=root, space_sha=space_sha, assets_sha=assets_sha)


@pytest.fixture
def env(repos, tmp_path, monkeypatch):
    storage = _FakeSpaceStorage()
    monkeypatch.setattr(db_space, "get_snapshot_cache_root", lambda: tmp_path / "c")
    monkeypatch.setattr(
        cmd.singleton_storage,
        "get_admin_storage",
        lambda: SimpleNamespace(space_storage=storage),
    )
    return SimpleNamespace(storage=storage, **vars(repos))


def _names(archive: bytes):
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        return {n for n in zf.namelist() if not n.endswith("/")}


def _run(env, *extra):
    args = [
        "--space-name",
        "x",
        "--space-repo",
        str(env.tmp / "space"),
        "--space-ref",
        "gbspace-config",
        "--assets-repo",
        str(env.tmp / "assets"),
        *extra,
    ]
    return CliRunner().invoke(cmd.cli, args, catch_exceptions=False)


def test_create_stores_snapshots_and_switches(env):
    result = _run(env, "--create")
    assert result.exit_code == 0, result.output
    row = env.storage.rows["x"]
    assert row.git_repo_uri == "db://spaces/x"

    assert row.space.source_uri == "https://github.example.com/org/gbspace-x"
    assert row.space.source_ref == "gbspace-config"
    assert row.space.source_commit == env.space_sha
    assert _names(row.space.load_archive()) == {
        "space.yaml",
        "environments/e/environment.yaml",
    }  # no history (old.txt), no .git, no untracked files

    assert row.assets.source_uri == "https://github.example.com/org/assets"
    assert row.assets.source_ref == "gbspace-config"  # from the base_uri's @ref
    assert row.assets.source_commit == env.assets_sha
    assert _names(row.assets.load_archive()) == {"steps/echo/step.yaml"}

    with zipfile.ZipFile(io.BytesIO(row.space.load_archive())) as zf:
        stored_yaml = zf.read("space.yaml").decode()
    assert yaml.safe_load(stored_yaml)["base_uris"] == ["db://spaces/x/assets"]
    assert "# the assets repo" in stored_yaml  # rewritten in place


def test_missing_row_requires_create(env):
    result = CliRunner().invoke(
        cmd.cli, ["--space-name", "x", "--space-repo", str(env.tmp / "space")]
    )
    assert result.exit_code != 0
    assert "--create" in result.output


def test_no_switch_keeps_git_uri(env):
    env.storage.rows["x"] = StoredSpace(
        name="x", git_repo_uri="https://github.example.com/org/gbspace-x"
    )
    result = _run(env, "--no-switch", "--skip-secret-group-check")
    assert result.exit_code == 0, result.output
    row = env.storage.rows["x"]
    assert row.git_repo_uri == "https://github.example.com/org/gbspace-x"
    assert row.space.source_commit == env.space_sha


def test_dry_run_stores_nothing(env):
    result = _run(env, "--create", "--dry-run")
    assert result.exit_code == 0, result.output
    assert env.storage.rows == {}


def test_assets_repo_not_in_base_uris_is_refused(env):
    result = CliRunner().invoke(
        cmd.cli,
        [
            "--space-name",
            "x",
            "--create",
            "--space-repo",
            str(env.tmp / "space"),
            "--space-ref",
            "gbspace-config",
            "--assets-repo",
            str(env.tmp / "other"),
        ],
    )
    assert result.exit_code != 0
    assert "no space.yaml base_uris entry" in result.output


def test_secret_group_change_is_refused(env, monkeypatch):
    env.storage.rows["x"] = StoredSpace(
        name="x", git_repo_uri="git+ssh://github.example.com/org/old-name.git"
    )
    monkeypatch.setattr(
        cmd,
        "_secret_group_names",
        lambda _yaml, uri: frozenset({"old" if "old-name" in uri else "new"}),
    )
    result = _run(env)
    assert result.exit_code != 0
    assert "secret groups differ" in result.output
    assert env.storage.rows["x"].space is None


@pytest.mark.parametrize(
    "uri, expected",
    [
        ("git+ssh://h/o/r.git@b", ("git+ssh://h/o/r.git", "b")),
        ("git+ssh://h/o/r.git@b#subdirectory=s", ("git+ssh://h/o/r.git", "b")),
        ("https://h/o/r", ("https://h/o/r", None)),
        ("git@h:o/r.git", ("git@h:o/r.git", None)),
    ],
)
def test_split_ref(uri, expected):
    assert cmd._split_ref(uri) == expected


def test_clone_url_maps_git_ssh():
    assert cmd._clone_url("git+ssh://h/o/r.git") == "ssh://git@h/o/r.git"
    assert cmd._clone_url("https://h/o/r") == "https://h/o/r"


def test_identical_identity_skips_secret_lookup(env, monkeypatch):
    """Refreshing an https-registered space compares equal strings: no lookup."""
    env.storage.rows["x"] = StoredSpace(
        name="x", git_repo_uri="https://github.example.com/org/gbspace-x"
    )

    def boom(*_):
        raise AssertionError("secret groups looked up for identical URIs")

    monkeypatch.setattr(cmd, "_secret_group_names", boom)
    # The branch the git-backed row resolved to, as in a non-standalone env.
    monkeypatch.setattr(cmd, "SPACE_REPO_CONFIG_BRANCH_NAME", "gbspace-config")
    result = _run(env)
    assert result.exit_code == 0, result.output
    assert env.storage.rows["x"].git_repo_uri == "db://spaces/x"
