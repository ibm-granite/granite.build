#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""`gbserver space-snapshot` stores only the branch head (no history), rewrites
space.yaml's base_uris to db://, and records the normalized source."""

import io
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import click
import pytest
import yaml
from click.testing import CliRunner

import gbserver.commands.command_space_snapshot as cmd
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
        {
            "steps/echo/step.yaml": "name: echo\n",
            "templates/t/template.yaml": "name: t\n",
            "docs/guide.md": "# guide\n",
            "images/diagram.png": "png\n",
            "README.md": "assets\n",
        },
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
    monkeypatch.setenv("GBSERVER_DB_SPACE_CACHE_DIR", str(tmp_path / "c"))
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
        "--checkout",
        str(env.tmp / "assets"),
        *extra,
    ]
    return CliRunner().invoke(cmd.cli, args, catch_exceptions=False)


def test_create_stores_snapshots_and_switches(env):
    result = _run(env, "--create")
    assert result.exit_code == 0, result.output
    row = env.storage.rows["x"]
    assert row.git_repo_uri == "https://github.example.com/org/gbspace-x"
    assert row.space_uri == "db://spaces/x"

    assert row.space.source_uri == "https://github.example.com/org/gbspace-x"
    assert row.space.source_ref == "gbspace-config"
    assert row.space.source_commit == env.space_sha
    assert _names(row.space.load_archive()) == {
        "space.yaml",
        "environments/e/environment.yaml",
    }  # no history (old.txt), no .git, no untracked files

    assert row.base_uris[0].source_uri == "https://github.example.com/org/assets"
    assert row.base_uris[0].source_ref == "gbspace-config"  # from the base_uri's @ref
    assert row.base_uris[0].source_commit == env.assets_sha
    # Default --base-uri-path: steps/ and templates/ only (no docs, images, README).
    assert _names(row.base_uris[0].load_archive()) == {
        "steps/echo/step.yaml",
        "templates/t/template.yaml",
    }

    with zipfile.ZipFile(io.BytesIO(row.space.load_archive())) as zf:
        stored_yaml = zf.read("space.yaml").decode()
    assert yaml.safe_load(stored_yaml)["base_uris"] == ["db://spaces/x/base_uri_0"]
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
    result = _run(env, "--no-switch")
    assert result.exit_code == 0, result.output
    row = env.storage.rows["x"]
    assert row.git_repo_uri == "https://github.example.com/org/gbspace-x"
    assert row.space_uri is None  # stored, not served
    assert row.space.source_commit == env.space_sha


def test_dry_run_stores_nothing(env):
    result = _run(env, "--create", "--dry-run")
    assert result.exit_code == 0, result.output
    assert env.storage.rows == {}


def test_unused_checkout_is_reported(env):
    result = _run(env, "--create", "--checkout", str(env.tmp / "other"))
    assert result.exit_code == 0, result.output
    assert "matches no base_uris entry" in result.output


def test_checkout_mapped_explicitly(env, tmp_path):
    """REPO=DIR serves an entry from a checkout whose origin is elsewhere."""
    mirror = tmp_path / "mirror"
    subprocess.run(
        ["git", "clone", "-q", str(env.tmp / "assets"), str(mirror)], check=True
    )
    _git(mirror, "remote", "set-url", "origin", "https://mirror.example.com/m/assets")
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
            "--checkout",
            f"https://github.example.com/org/assets={mirror}",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    snap = env.storage.rows["x"].base_uris[0]
    assert snap.source_uri == "https://github.example.com/org/assets"  # the entry's
    assert snap.source_commit == env.assets_sha


@pytest.mark.parametrize(
    "uri, expected",
    [
        ("git+ssh://h/o/r.git@b", ("git+ssh://h/o/r.git", "b")),
        ("git+ssh://h/o/r.git@b#subdirectory=s", ("git+ssh://h/o/r.git", "b")),
        ("https://h/o/r", ("https://h/o/r", None)),
        ("git@h:o/r.git", ("git@h:o/r.git", None)),
        ("git@h:o/r.git@v2#subdirectory=s", ("git@h:o/r.git", "v2")),
        ("git+ssh://h/o/r.git@release/1.0", ("git+ssh://h/o/r.git", "release/1.0")),
        ("git@h:o/r.git@feat/x#subdirectory=s", ("git@h:o/r.git", "feat/x")),
    ],
)
def test_split_ref(uri, expected):
    assert cmd._split_ref(uri) == expected


def test_clone_url_maps_git_ssh():
    assert cmd._clone_url("git+ssh://h/o/r.git") == "ssh://git@h/o/r.git"
    assert cmd._clone_url("https://h/o/r") == "https://h/o/r"


@pytest.mark.parametrize(
    "uri",
    [
        "git+ssh://github.example.com/org/r.git",
        "ssh://git@github.example.com:22/org/r.git",
        "git@github.example.com:org/r.git",
        "https://github.example.com/org/r",
    ],
)
def test_clone_url_https_forces_https(uri):
    assert cmd._clone_url(uri, https=True) == "https://github.example.com/org/r.git"


def test_https_clones_space_and_base_uris_over_https(env, monkeypatch):
    """--https clones every remote repo via https, whatever space.yaml writes."""
    cloned = []
    real = cmd._git

    def record(args, cwd=None):
        if args[0] == "clone":
            cloned.append(args[-2])
            # Serve the clone from the local fixture repos instead of the network.
            src = env.tmp / ("space" if "gbspace-x" in args[-2] else "assets")
            args = [*args[:-2], str(src), args[-1]]
        return real(args, cwd=cwd)

    monkeypatch.setattr(cmd, "_git", record)
    result = CliRunner().invoke(
        cmd.cli,
        [
            "--space-name",
            "x",
            "--create",
            "--https",
            "--space-repo",
            "git+ssh://github.example.com/org/gbspace-x.git",
            "--space-ref",
            "gbspace-config",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert cloned == [
        "https://github.example.com/org/gbspace-x.git",
        "https://github.example.com/org/assets.git",
    ]
    assert env.storage.rows["x"].base_uris[0].source_commit == env.assets_sha


def test_create_no_switch_keeps_git(env):
    result = _run(env, "--create", "--no-switch")
    assert result.exit_code == 0, result.output
    row = env.storage.rows["x"]
    assert row.git_repo_uri == "https://github.example.com/org/gbspace-x"
    assert row.space_uri is None
    assert row.space.source_commit == env.space_sha


def test_entry_without_ref_uses_default_branch(env, tmp_path, monkeypatch):
    """No @ref in base_uris: snapshot the default branch, not the config one."""
    space_dir = tmp_path / "space-noref"
    _repo(
        space_dir,
        {"space.yaml": SPACE_YAML.replace("@gbspace-config", "")},
        "git@github.example.com:org/gbspace-x.git",
        "gbspace-config",
    )
    result = CliRunner().invoke(
        cmd.cli,
        [
            "--space-name",
            "x",
            "--create",
            "--space-repo",
            str(space_dir),
            "--space-ref",
            "gbspace-config",
            "--checkout",
            str(env.tmp / "assets"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert env.storage.rows["x"].base_uris[0].source_commit == env.assets_sha


def test_token_header_scoped_to_ghe_host(monkeypatch):
    seen = {}

    def fake_run(args, cwd=None, env=None, **_):
        seen.update(env)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(cmd, "GBSERVER_GITHUB_TOKEN", "tok")
    monkeypatch.setattr(cmd, "run_git", fake_run)
    cmd._git(["status"])
    assert seen["GIT_CONFIG_KEY_0"] == (
        f"http.https://{cmd.DEFAULT_GH_DOMAIN}/.extraHeader"
    )


def test_snapshot_from_another_repo_is_refused(env):
    """git_repo_uri is the identity; a snapshot of a different repo is refused."""
    env.storage.rows["x"] = StoredSpace(
        name="x", git_repo_uri="https://github.example.com/org/other-space"
    )
    result = _run(env)
    assert result.exit_code != 0
    assert "is not space 'x'" in result.output
    assert env.storage.rows["x"].space is None


def test_existing_row_in_another_form_is_accepted(env):
    """The same repo in git+ssh form is the same identity; it is left as stored."""
    uri = "git+ssh://github.example.com/org/gbspace-x.git@gbspace-config"
    env.storage.rows["x"] = StoredSpace(name="x", git_repo_uri=uri)
    result = _run(env)
    assert result.exit_code == 0, result.output
    assert env.storage.rows["x"].git_repo_uri == uri
    assert env.storage.rows["x"].space_uri == "db://spaces/x"


def test_base_uri_path_override(env):
    result = _run(
        env, "--create", "--base-uri-path", "steps", "--base-uri-path", "nope"
    )
    assert result.exit_code == 0, result.output
    assert "nope not found" in result.output
    assert _names(env.storage.rows["x"].base_uris[0].load_archive()) == {
        "steps/echo/step.yaml"
    }


def test_local_branch_behind_origin_uses_origin(env, tmp_path):
    """A stale local branch is not snapshotted; origin/<ref> is, with a warning."""
    clone = tmp_path / "space-clone"
    subprocess.run(
        ["git", "clone", "-q", str(env.tmp / "space"), str(clone)], check=True
    )
    _git(clone, "config", "user.email", "t@example.com")
    _git(clone, "config", "user.name", "t")
    _git(clone, "config", "commit.gpgsign", "false")
    _git(
        clone, "remote", "set-url", "origin", "git@github.example.com:org/gbspace-x.git"
    )
    _git(clone, "reset", "-q", "--hard", "HEAD~1")  # local behind origin
    result = CliRunner().invoke(
        cmd.cli,
        [
            "--space-name",
            "x",
            "--create",
            "--space-repo",
            str(clone),
            "--space-ref",
            "gbspace-config",
            "--checkout",
            str(env.tmp / "assets"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "differs from origin/gbspace-config" in result.output
    assert env.storage.rows["x"].space.source_commit == env.space_sha


def test_missing_default_space_branch_falls_back(env, tmp_path):
    """Without the deployment's config branch, the default branch is used."""
    result = CliRunner().invoke(
        cmd.cli,
        [
            "--space-name",
            "x",
            "--create",
            "--space-repo",
            str(env.tmp / "space"),
            "--checkout",
            str(env.tmp / "assets"),
        ],
        catch_exceptions=False,
    )
    # The test repo has only gbspace-config; the standalone default is main.
    assert result.exit_code == 0, result.output
    assert "using its default branch" in result.output
    assert env.storage.rows["x"].space.source_commit == env.space_sha


def test_base_uri_paths_follow_subdirectory(env, tmp_path):
    """--base-uri-path is relative to the base_uri's #subdirectory."""
    sub_assets = tmp_path / "sub-assets"
    _repo(
        sub_assets,
        {"gb/steps/echo/step.yaml": "name: echo\n", "steps/other/step.yaml": "x\n"},
        "https://github.example.com/org/assets.git",
        "gbspace-config",
    )
    space_dir = tmp_path / "space-sub"
    _repo(
        space_dir,
        {
            "space.yaml": SPACE_YAML.replace(
                "@gbspace-config", "@gbspace-config#subdirectory=gb"
            )
        },
        "git@github.example.com:org/gbspace-x.git",
        "gbspace-config",
    )
    result = CliRunner().invoke(
        cmd.cli,
        [
            "--space-name",
            "x",
            "--create",
            "--space-repo",
            str(space_dir),
            "--space-ref",
            "gbspace-config",
            "--checkout",
            str(sub_assets),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert _names(env.storage.rows["x"].base_uris[0].load_archive()) == {
        "gb/steps/echo/step.yaml"
    }


def test_several_base_uris_one_slot_each_in_order(env, tmp_path):
    """Each git entry gets base_uri_<i> in place; same repo+ref shares a slot."""
    second = tmp_path / "second"
    second_sha = _repo(
        second,
        {"steps/two/step.yaml": "two\n"},
        "https://github.example.com/org/second.git",
        "main",
    )
    yaml_multi = SPACE_YAML.replace(
        "  - git+ssh://github.example.com/org/assets.git@gbspace-config\n",
        "  - git+ssh://github.example.com/org/assets.git@gbspace-config\n"
        "  - file://./local-steps\n"
        "  - git@github.example.com:org/second.git@main\n"
        "  - https://github.example.com/org/assets@gbspace-config#subdirectory=steps\n",
    )
    space_dir = tmp_path / "space-multi"
    _repo(
        space_dir,
        {"space.yaml": yaml_multi, "local-steps/x/step.yaml": "x\n"},
        "git@github.example.com:org/gbspace-x.git",
        "gbspace-config",
    )
    result = CliRunner().invoke(
        cmd.cli,
        [
            "--space-name",
            "x",
            "--create",
            "--space-repo",
            str(space_dir),
            "--space-ref",
            "gbspace-config",
            "--checkout",
            str(env.tmp / "assets"),
            "--checkout",
            str(second),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    row = env.storage.rows["x"]
    with zipfile.ZipFile(io.BytesIO(row.space.load_archive())) as zf:
        stored = yaml.safe_load(zf.read("space.yaml"))["base_uris"]
    assert stored == [
        "db://spaces/x/base_uri_0",
        "file://./local-steps",
        "db://spaces/x/base_uri_2",
        "db://spaces/x/base_uri_0#subdirectory=steps",  # same repo+ref: slot 0
    ]
    assert [snap and snap.source_commit for snap in row.base_uris] == [
        env.assets_sha,
        None,
        second_sha,
        None,
    ]
    assert row.base_uris[2].entry == "git@github.example.com:org/second.git@main"


def test_relative_entry_outside_the_space_repo_is_refused(env, tmp_path):
    space_dir = tmp_path / "space-rel"
    _repo(
        space_dir,
        {
            "space.yaml": SPACE_YAML.replace(
                "git+ssh://github.example.com/org/assets.git@gbspace-config",
                "file://../../assets",
            )
        },
        "git@github.example.com:org/gbspace-x.git",
        "gbspace-config",
    )
    result = CliRunner().invoke(
        cmd.cli,
        [
            "--space-name",
            "x",
            "--create",
            "--space-repo",
            str(space_dir),
            "--space-ref",
            "gbspace-config",
        ],
    )
    assert result.exit_code != 0
    assert "outside the space repo" in result.output


ENTRY = "git+ssh://github.example.com/org/assets.git@gbspace-config"
DB = "db://spaces/x/base_uri_0"


@pytest.mark.parametrize(
    "before, after",
    [
        (  # the same URL in a comment and another field is left alone
            f"# mirrors {ENTRY}\nname: x\nnote: {ENTRY}\nbase_uris:\n  - {ENTRY}\n",
            f"# mirrors {ENTRY}\nname: x\nnote: {ENTRY}\nbase_uris:\n  - {DB}\n",
        ),
        (  # quoted entries keep their quotes; trailing comments survive
            f"base_uris:\n  - \"{ENTRY}\"  # assets\n  - '{ENTRY}'\n",
            f"base_uris:\n  - \"{DB}\"  # assets\n  - '{DB}'\n",
        ),
        (  # flow style
            f"base_uris: [{ENTRY}, file://./local]\n",
            f"base_uris: [{DB}, file://./local]\n",
        ),
    ],
)
def test_rewrite_touches_only_base_uris_entries(before, after):
    entries = cmd._base_uris(before)
    new = [DB if e == ENTRY else e for e in entries]
    assert cmd._rewrite_base_uris(before, entries, new) == after


@pytest.mark.parametrize(
    "text",
    [
        f"x: &a {ENTRY}\nbase_uris:\n  - *a\n",  # alias of a node elsewhere
        f"base_uris:\n  - &a {ENTRY}\ny: *a\n",  # anchored entry used elsewhere
        f"base_uris:\n  - &a {ENTRY}\n  - *a\n",  # one node twice
    ],
)
def test_rewrite_refuses_yaml_anchors_and_aliases(text):
    entries = cmd._base_uris(text)
    with pytest.raises(click.ClickException, match="anchors or aliases"):
        cmd._rewrite_base_uris(text, entries, [DB] * len(entries))


def test_git_command_is_killed_at_git_timeout(monkeypatch):
    """A git command that hangs (e.g. a stalled ssh clone) fails clearly."""
    seen = {}

    def hang(args, timeout=None, env=None, **_):
        seen.update(timeout=timeout, env=env)
        raise subprocess.TimeoutExpired(args, timeout)

    monkeypatch.setattr(cmd, "run_git", hang)
    token = cmd._git_timeout.set(42)
    try:
        with pytest.raises(click.ClickException) as exc:
            cmd._git(["clone", "https://user:s3cr3t@github.example.com/o/r.git"])
    finally:
        cmd._git_timeout.reset(token)
    assert seen["timeout"] == 42
    assert (
        seen["env"]["GIT_HTTP_LOW_SPEED_LIMIT"]
        and seen["env"]["GIT_TERMINAL_PROMPT"] == "0"
    )
    assert "--git-timeout" in exc.value.message and "s3cr3t" not in exc.value.message
