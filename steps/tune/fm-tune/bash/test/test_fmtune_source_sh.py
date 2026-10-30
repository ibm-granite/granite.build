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

"""Unit tests for src/fmtune_run.sh: the git source, its clone mirror and the
immutable tree-<sha> per commit (branches, tags and full commit SHAs).

Each test builds a local file:// repo (_git_repo). Mode 1 only.
"""

import pathlib
import shutil
import subprocess

import pytest
from _fmtune_helpers import (
    _SCRIPT,
    _clear_output,
    _clone_lock,
    _commit,
    _git,
    _git_repo,
    _git_spy,
    _head,
    _record,
    _run,
    _stub,
)


def test_git_source_is_cloned_and_cached(env, tmp_path):
    """source.repo clones once at ref and runs <tree>/<subdir>/main.py."""
    _git_repo(env, tmp_path)
    first = _run(env)
    assert first.returncode == 0, first.stderr
    assert first.stdout.count("[fm-tune] cloning") == 1
    assert _record(env)["cwd"].endswith("/sub/fm-tune")
    cache = pathlib.Path(env["FMTUNE_CACHE_DIR"])
    clones = list(cache.glob("src-*"))
    assert len(clones) == 1
    trees = list(cache.glob("tree-*"))
    assert len(trees) == 1
    _clear_output(env)
    second = _run(env)  # second run reuses the clone, and the tree: same commit
    assert second.returncode == 0, second.stderr
    assert second.stdout.count("[fm-tune] cloning") == 0
    assert list(cache.glob("src-*")) == clones
    assert list(cache.glob("tree-*")) == trees


def test_cached_clone_is_updated_to_the_current_ref(env, tmp_path):
    """The cache is shared across builds, so a moving ref must not stay pinned."""
    repo = _git_repo(env, tmp_path)
    assert _run(env).returncode == 0
    (repo / "sub" / "fm-tune" / "NEW").write_text("2\n")
    _git("add", ".", cwd=repo)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "y", cwd=repo)
    _clear_output(env)
    second = _run(env)
    assert second.returncode == 0, second.stderr
    assert "[fm-tune] updating" in second.stdout
    assert second.stdout.count("[fm-tune] cloning") == 0
    tree = pathlib.Path(env["FMTUNE_CACHE_DIR"]) / f"tree-{_head(repo)}"
    assert _record(env)["cwd"] == str(tree / "sub" / "fm-tune")
    assert (tree / "sub" / "fm-tune" / "NEW").is_file()


def test_main_py_runs_from_the_tree_of_the_repos_head(env, tmp_path):
    repo = _git_repo(env, tmp_path)
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    sha = _head(repo)
    tree = pathlib.Path(env["FMTUNE_CACHE_DIR"]) / f"tree-{sha}"
    assert _record(env)["cwd"] == str(tree / "sub" / "fm-tune")
    assert f"[fm-tune] using {env['FMTUNE_SOURCE_REPO']}@rel = {sha}" in proc.stdout


def test_a_later_commit_never_touches_an_earlier_runs_tree(env, tmp_path):
    """Two builds of one repo@ref can overlap; build 1 trains from (and pip -e
    installs) its tree while build 2 refreshes the mirror to a newer commit."""
    repo = _git_repo(env, tmp_path)
    (repo / "sub" / "fm-tune" / "GONE").write_text("1\n")
    _commit(repo, "add GONE")
    assert _run(env).returncode == 0
    tree1 = pathlib.Path(_record(env)["cwd"])
    snap = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in tree1.rglob("*")
        if p.is_file()
    }
    (repo / "sub" / "fm-tune" / "GONE").unlink()
    _commit(repo, "drop GONE")
    _clear_output(env)
    second = _run(env)
    assert second.returncode == 0, second.stderr
    tree2 = pathlib.Path(_record(env)["cwd"])
    assert tree2 != tree1 and tree2.parents[1].name == f"tree-{_head(repo)}"
    assert not (tree2 / "GONE").exists()
    assert (tree1 / "GONE").is_file()
    assert {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in tree1.rglob("*")
        if p.is_file()
    } == snap


def test_failed_update_of_a_cached_clone_dies(env, tmp_path):
    """A fetch failure on reuse must stop the run, not train on the stale copy."""
    repo = _git_repo(env, tmp_path)
    assert _run(env).returncode == 0
    pathlib.Path(env["STUB_RECORD"]).unlink()  # run 1's record; run 2 must not make one
    shutil.rmtree(repo)  # the remote is gone: the fetch fails
    proc = _run(env)
    assert proc.returncode != 0
    assert "fm-tune: " in proc.stderr and "update" in proc.stderr
    assert not pathlib.Path(env["STUB_RECORD"]).exists()  # main.py never ran
    assert "GB_ARTIFACT_ID:" not in proc.stdout
    # It died holding the clone lock: the EXIT trap must release it.
    assert not list(pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("*.lock*"))


def test_an_annotated_tag_resolves_to_one_tree_and_its_commit(env, tmp_path):
    repo = _git_repo(env, tmp_path)
    _git(
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "tag",
        "-a",
        "v1",
        "-m",
        "v1",
        cwd=repo,
    )
    env["FMTUNE_SOURCE_REF"] = "v1"
    commit = subprocess.run(
        ["git", "rev-parse", "v1^{commit}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    for _ in range(2):
        _clear_output(env)
        proc = _run(env)
        assert proc.returncode == 0, proc.stderr
        assert f"@v1 = {commit}" in proc.stdout
    trees = list(pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("tree-*"))
    assert [t.name for t in trees] == [f"tree-{commit}"]


def test_a_commit_sha_ref_is_fetched_and_runs_from_its_tree(env, tmp_path):
    repo = _git_repo(env, tmp_path)
    sha = _head(repo)
    (repo / "sub" / "fm-tune" / "LATER").write_text("1\n")
    _commit(repo, "later")  # sha is no longer the branch tip
    env["FMTUNE_SOURCE_REF"] = sha
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    cache = pathlib.Path(env["FMTUNE_CACHE_DIR"])
    assert _record(env)["cwd"] == str(cache / f"tree-{sha}" / "sub" / "fm-tune")
    assert f"@{sha} = {sha}" in proc.stdout
    assert "[fm-tune] cloning" not in proc.stdout
    # Fetched by id, not cloned: the branch tip never reached the mirror.
    (mirror,) = cache.glob("src-*[0-9]")
    tip = subprocess.run(
        ["git", "-C", str(mirror), "cat-file", "-e", f"{_head(repo)}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    assert tip.returncode != 0


def test_a_warm_sha_run_needs_no_git_and_no_remote(env, tmp_path):
    repo = _git_repo(env, tmp_path)
    env["FMTUNE_SOURCE_REF"] = sha = _head(repo)
    assert _run(env).returncode == 0
    pathlib.Path(env["STUB_RECORD"]).unlink()
    shutil.rmtree(repo)  # the remote is gone
    spy = _git_spy(env, tmp_path)
    _clear_output(env)
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert not spy.exists()  # git was never called
    tree = pathlib.Path(env["FMTUNE_CACHE_DIR"]) / f"tree-{sha}"
    assert _record(env)["cwd"] == str(tree / "sub" / "fm-tune")
    assert not list(pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("*.lock*"))


def test_one_commit_by_branch_and_by_sha_shares_one_tree(env, tmp_path):
    repo = _git_repo(env, tmp_path)
    sha = _head(repo)
    for ref in ("rel", sha):
        env["FMTUNE_SOURCE_REF"] = ref
        _clear_output(env)
        proc = _run(env)
        assert proc.returncode == 0, proc.stderr
        assert f"@{ref} = {sha}" in proc.stdout
    trees = list(pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("tree-*"))
    assert [t.name for t in trees] == [f"tree-{sha}"]


def test_two_refs_extracting_one_commit_at_once_never_leave_an_empty_tree(
    env, tmp_path
):
    """A branch and its SHA hold different locks but write the same tree-<sha>."""
    repo = _git_repo(env, tmp_path)
    sha = _head(repo)
    real_tar, gate = shutil.which("tar"), tmp_path / "gate"
    gate.mkdir()
    # The first extractor waits until the second has started, then both finish.
    _stub(
        tmp_path / "bin",
        "tar",
        f"""\
if mkdir "{gate}/first" 2>/dev/null; then
  i=0; while [ ! -e "{gate}/second" ] && [ $i -lt 100 ]; do sleep 0.1; i=$((i+1)); done
else
  : > "{gate}/second"
fi
exec "{real_tar}" "$@"
""",
    )
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    procs = []
    for ref in ("rel", sha):
        run_env = dict(
            env,
            FMTUNE_SOURCE_REF=ref,
            STUB_RECORD=str(tmp_path / f"rec-{ref}"),
            FMTUNE_OUTPUT_DIR=str(tmp_path / f"out-{ref}"),  # one model each
        )
        procs.append(
            subprocess.Popen(
                ["bash", str(_SCRIPT)],
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    for proc in procs:
        _, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, err
    cache = pathlib.Path(env["FMTUNE_CACHE_DIR"])
    assert (cache / f"tree-{sha}" / "sub" / "fm-tune" / "main.py").is_file()
    # The loser's copy is dropped, not nested inside the winner's tree or left beside it.
    assert sorted(p.name for p in (cache / f"tree-{sha}").iterdir()) == ["sub"]
    assert [t.name for t in cache.glob("tree-*")] == [f"tree-{sha}"]


def test_an_uppercase_sha_is_lowercased_to_the_same_tree_and_cache_key(env, tmp_path):
    repo = _git_repo(env, tmp_path)
    sha = _head(repo)
    env["FMTUNE_SOURCE_REF"] = sha.upper()
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert f"@{sha} = {sha}" in proc.stdout
    cache = pathlib.Path(env["FMTUNE_CACHE_DIR"])
    assert [t.name for t in cache.glob("tree-*")] == [f"tree-{sha}"]
    env["FMTUNE_SOURCE_REF"] = sha
    assert [m.name for m in cache.glob("src-*")] == [
        _clone_lock(env).name[: -len(".lock")]
    ]
    _clear_output(env)
    assert _run(env).returncode == 0
    assert [t.name for t in cache.glob("tree-*")] == [f"tree-{sha}"]


def test_a_missing_full_sha_dies_before_any_venv(env, tmp_path):
    _git_repo(env, tmp_path)
    env["FMTUNE_SOURCE_REF"] = "0123456789abcdef" * 2 + "01234567"
    env.pop("FMTUNE_SKIP_INSTALL")
    proc = _run(env)
    assert proc.returncode == 2, proc.stderr
    assert "fm-tune: could not fetch commit 0123456789abcdef" in proc.stderr
    cache = pathlib.Path(env["FMTUNE_CACHE_DIR"])
    assert not list(cache.glob("tree-*")) and not list(cache.glob("venv-*"))
    assert not pathlib.Path(env["STUB_RECORD"]).exists()
    assert "GB_ARTIFACT_ID:" not in proc.stdout
    assert not list(cache.glob("*.lock*"))


@pytest.mark.parametrize(
    "ref",
    [lambda sha: sha[:7], lambda sha: "no-such-branch"],
    ids=["short-hex", "missing-branch"],
)
def test_a_failed_branch_clone_dies_naming_the_full_sha_option(env, tmp_path, ref):
    """A short hex value is not a SHA: it goes to clone --branch like any branch name."""
    repo = _git_repo(env, tmp_path)
    env["FMTUNE_SOURCE_REF"] = ref(_head(repo))
    proc = _run(env)
    assert proc.returncode == 2, proc.stderr
    assert (
        f"fm-tune: could not clone {env['FMTUNE_SOURCE_REPO']} at {env['FMTUNE_SOURCE_REF']}"
        in proc.stderr
    )
    assert "full 40-character commit SHA" in proc.stderr
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


def test_a_token_in_the_repo_url_never_reaches_the_log(env, tmp_path):
    repo = _git_repo(env, tmp_path)
    env["FMTUNE_SOURCE_REPO"] = (
        f"https://x-access-token:s3cr3t@127.0.0.1:9/{repo.name}.git"
    )
    proc = _run(env)
    assert proc.returncode == 2
    assert "could not clone https://***@127.0.0.1:9/repo.git at rel" in proc.stderr
    assert "s3cr3t" not in proc.stdout + proc.stderr


@pytest.mark.parametrize("obj", ["v1", "HEAD^{tree}"], ids=["tag-object", "tree"])
def test_a_sha_that_is_not_a_commit_dies(env, tmp_path, obj):
    repo = _git_repo(env, tmp_path)
    _git(
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "tag",
        "-a",
        "v1",
        "-m",
        "v1",
        cwd=repo,
    )
    env["FMTUNE_SOURCE_REF"] = subprocess.run(
        ["git", "rev-parse", obj], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    proc = _run(env)
    assert proc.returncode == 2, proc.stderr
    assert "is not a commit" in proc.stderr
    assert not pathlib.Path(env["STUB_RECORD"]).exists()
    assert not list(pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("tree-*"))


def test_a_sha_whose_tree_was_deleted_is_checked_on_the_warm_mirror(env, tmp_path):
    """tree-<sha> deleted (to free space) but src-* kept: the refetch from the warm
    mirror must apply the same is-a-commit check as the cold fetch."""
    repo = _git_repo(env, tmp_path)
    _git(
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "tag",
        "-a",
        "v1",
        "-m",
        "v1",
        cwd=repo,
    )
    tag = subprocess.run(
        ["git", "rev-parse", "v1"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    env["FMTUNE_SOURCE_REF"] = _head(repo)
    assert _run(env).returncode == 0  # warms src-<key> for this ref
    cache = pathlib.Path(env["FMTUNE_CACHE_DIR"])
    (mirror,) = cache.glob("src-*[0-9]")
    for tree in cache.glob("tree-*"):
        shutil.rmtree(tree)
    # Point the same cache key at the tag object: its key dir is the warm mirror.
    env["FMTUNE_SOURCE_REF"] = tag
    mirror.rename(_clone_lock(env).with_suffix(""))
    pathlib.Path(env["STUB_RECORD"]).unlink()
    _clear_output(env)
    proc = _run(env)
    assert proc.returncode == 2, proc.stderr
    assert "is not a commit" in proc.stderr
    assert not pathlib.Path(env["STUB_RECORD"]).exists()
    assert not list(cache.glob("tree-*"))


@pytest.mark.parametrize(
    "over",
    [
        {"FMTUNE_SOURCE_REF": "-uoops"},
        {"FMTUNE_SOURCE_REPO": "--upload-pack=touch /tmp/x"},
    ],
)
def test_a_source_value_that_looks_like_an_option_dies_before_git(env, tmp_path, over):
    _git_repo(env, tmp_path)
    env.update(over)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "git").write_text(f"#!/bin/sh\ntouch {tmp_path}/git-ran\nexit 1\n")
    (bin_dir / "git").chmod(0o755)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    proc = _run(env)
    assert proc.returncode != 0
    assert (
        "fm-tune: source." in proc.stderr and "must not start with '-'" in proc.stderr
    )
    assert not (tmp_path / "git-ran").exists()
