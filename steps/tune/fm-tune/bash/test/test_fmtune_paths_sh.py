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

"""Unit tests for src/fmtune_run.sh: the cache dir, the output dir guard, path
normalisation, and the venv. The ``slow`` venv test builds a real venv and
needs network access for pip. Mode 1 only.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest
from _fmtune_helpers import (
    _SCRIPT,
    _clear_output,
    _flag,
    _git_repo,
    _git_spy,
    _record,
    _run,
    _staged_as,
)


def test_bash_builtins_default_output_cache_and_python(env, tmp_path):
    """On bash the template leaves these empty; the script derives them."""
    for k in ("FMTUNE_OUTPUT_DIR", "FMTUNE_CACHE_DIR", "FMTUNE_PYTHON"):
        env.pop(k)
    # The real bash layout (build 2ad03fab): <GB_HOME>/workdir/llm-build-<b>/
    # target-<t>/target-run-<r>/step-<s>/step-run-<sr>/launch-<l>/outputs. The cache
    # must sit beside workdir/, shared by every build, so the venv is reused.
    out = tmp_path / "gbhome" / "workdir" / "llm-build-b" / "target-t" / "target-run-r"
    out = out / "step-fm-tune" / "step-run-s" / "launch-l" / "outputs"
    env["LLMB_BASH_OUTPUT_DIR"] = str(out)
    env["LLMB_BASH_PYTHON_DIR"] = os.path.dirname(sys.executable)
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert _staged_as(_record(env)["argv"], f"{env['LLMB_BASH_OUTPUT_DIR']}/model")
    assert (tmp_path / "gbhome" / ".fm-tune-cache").is_dir()
    assert not (out.parents[1] / ".fm-tune-cache").exists()


def test_cache_cuts_at_the_last_workdir_segment(env, tmp_path):
    """A GB_HOME that itself sits under some /workdir/ must not move the cache up."""
    env.pop("FMTUNE_CACHE_DIR")
    home = tmp_path / "workdir" / "gbhome"
    out = home / "workdir" / "llm-build-b" / "target-t" / "launch-l" / "outputs"
    env["LLMB_BASH_OUTPUT_DIR"] = str(out)
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert (home / ".fm-tune-cache").is_dir()
    assert not (tmp_path / ".fm-tune-cache").exists()


def test_cache_falls_back_to_two_levels_up_outside_a_workdir(env, tmp_path):
    env.pop("FMTUNE_CACHE_DIR")
    env["LLMB_BASH_OUTPUT_DIR"] = str(tmp_path / "w" / "a" / "outputs")
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "w" / ".fm-tune-cache").is_dir()


def test_log_path_defaults_to_a_sibling_of_the_output_dir(env):
    """fm-tune's bridge logs to $LOG_PATH/logs; unset, that is cwd (the source tree).
    Not the output dir (--model_at_root clears it) nor its parent, which builds with
    sibling output dirs share."""
    assert _run(env).returncode == 0
    assert _record(env)["log_path"] == env["FMTUNE_OUTPUT_DIR"] + ".logs"


def test_builds_with_sibling_output_dirs_get_their_own_log_paths(env, tmp_path):
    paths = []
    for name in ("a", "b"):
        env["FMTUNE_OUTPUT_DIR"] = str(tmp_path / "shared" / name)
        assert _run(env).returncode == 0
        paths.append(_record(env)["log_path"])
    assert paths == [str(tmp_path / "shared" / n) + ".logs" for n in ("a", "b")]


def test_a_preset_log_path_is_kept(env, tmp_path):
    env["LOG_PATH"] = str(tmp_path / "mylogs")
    assert _run(env).returncode == 0
    assert _record(env)["log_path"] == str(tmp_path / "mylogs")


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the read-only mode")
def test_an_unwritable_cache_dir_dies_fast(env, tmp_path):
    _git_repo(env, tmp_path)
    cache = pathlib.Path(env["FMTUNE_CACHE_DIR"])
    cache.mkdir()
    cache.chmod(0o555)
    try:
        proc = subprocess.run(
            ["bash", str(_SCRIPT)], env=env, capture_output=True, text=True, timeout=20
        )
    finally:
        cache.chmod(0o755)
    assert proc.returncode != 0
    assert "fm-tune: cannot create lock" in proc.stderr
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


def test_a_new_cache_dir_and_its_missing_parents_are_private(env, tmp_path):
    """Builds run code from the cache (trees, venvs): no other user may write there."""
    env["FMTUNE_CACHE_DIR"] = str(tmp_path / "new" / "cache")
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    for d in (tmp_path / "new", tmp_path / "new" / "cache"):
        assert d.stat().st_mode & 0o777 == 0o700, oct(d.stat().st_mode)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the read-only mode")
def test_a_cache_dir_that_cannot_be_created_dies_naming_it(env, tmp_path):
    """The old per-user SkyPilot cache sat under a shared .fm-tune-cache made by the
    first user (mode 0755): a second user's mkdir -p failed with a bare error."""
    shared = tmp_path / ".fm-tune-cache"
    shared.mkdir()
    shared.chmod(0o555)
    env["FMTUNE_CACHE_DIR"] = str(shared / "1234")
    try:
        proc = _run(env)
    finally:
        shared.chmod(0o755)
    assert proc.returncode == 2
    assert f"fm-tune: cannot create cache dir {shared}/1234" in proc.stderr
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


@pytest.mark.parametrize("mode", [0o775, 0o757, 0o777])
def test_a_cache_dir_others_can_write_is_refused_and_left_alone(env, tmp_path, mode):
    cache = pathlib.Path(env["FMTUNE_CACHE_DIR"])
    cache.mkdir()
    cache.chmod(mode)
    proc = _run(env)
    assert proc.returncode == 2
    assert f"fm-tune: cache dir {cache} is writable by other users" in proc.stderr
    assert f"chmod go-w {cache}" in proc.stderr
    assert cache.stat().st_mode & 0o777 == mode  # refused, not chmod'ed
    assert not list(cache.iterdir())
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root owns the directory")
def test_a_cache_dir_owned_by_another_user_is_refused(env):
    env["FMTUNE_CACHE_DIR"] = (
        "/usr"  # exists and is root's: nothing is ever written there
    )
    proc = _run(env)
    assert proc.returncode == 2
    assert "fm-tune: cache dir /usr is owned by root, not " in proc.stderr
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


def test_a_symlinked_cache_dir_of_ones_own_is_followed(env, tmp_path):
    """Say, to move the cache to a bigger disk: the checks apply to the target."""
    _git_repo(env, tmp_path)
    (tmp_path / "big").mkdir()
    link = pathlib.Path(env["FMTUNE_CACHE_DIR"])
    link.symlink_to(tmp_path / "big")
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert link.is_symlink() and list((tmp_path / "big").glob("tree-*"))
    (tmp_path / "big").chmod(0o777)
    _clear_output(env)
    proc = _run(env)
    assert proc.returncode == 2
    assert f"fm-tune: cache dir {link} is writable by other users" in proc.stderr


@pytest.mark.skipif(os.geteuid() == 0, reason="root owns the link")
def test_a_symlinked_cache_dir_owned_by_another_user_is_refused(env):
    """Its owner could re-point it at their own dir while a build runs from it."""
    link = next(
        (
            p
            for p in ("/tmp", "/etc", "/var", "/bin", "/lib", "/sbin")
            if os.path.islink(p) and os.path.isdir(p) and os.lstat(p).st_uid == 0
        ),
        None,
    )
    if link is None:
        pytest.skip("no root-owned symlink to a directory here")
    env["FMTUNE_CACHE_DIR"] = link
    proc = _run(env)
    assert proc.returncode == 2
    assert f"fm-tune: cache dir {link} is a symlink owned by root, not " in proc.stderr
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root owns the link")
def test_a_trailing_slash_does_not_hide_a_symlinked_cache_dir(env):
    """[ -L /tmp/ ] is false (the slash resolves the link): strip it first."""
    link = next(
        (
            p
            for p in ("/tmp", "/etc", "/var", "/bin", "/lib", "/sbin")
            if os.path.islink(p) and os.path.isdir(p) and os.lstat(p).st_uid == 0
        ),
        None,
    )
    if link is None:
        pytest.skip("no root-owned symlink to a directory here")
    env["FMTUNE_CACHE_DIR"] = link + "//"
    proc = _run(env)
    assert proc.returncode == 2
    assert f"fm-tune: cache dir {link} is a symlink owned by root, not " in proc.stderr


@pytest.mark.skipif(os.geteuid() == 0, reason="root owns /")
def test_a_root_cache_dir_is_not_stripped_to_empty(env):
    env["FMTUNE_CACHE_DIR"] = "/"
    proc = _run(env)
    assert proc.returncode == 2
    assert "fm-tune: cache dir / is owned by root, not " in proc.stderr


def test_an_unrelated_non_empty_output_dir_dies_before_any_clone(env, tmp_path):
    """--model_at_root deletes every other entry in output_dir on success, so a
    mistyped output_path (say /shared) would lose its contents."""
    _git_repo(env, tmp_path)
    git_ran = _git_spy(env, tmp_path)
    out = pathlib.Path(env["FMTUNE_OUTPUT_DIR"])
    out.mkdir()
    (out / "precious.txt").write_text("keep me\n")
    proc = _run(env)
    assert proc.returncode != 0
    assert f"fm-tune: output dir {out} is not empty;" in proc.stderr
    assert "new or empty directory" in proc.stderr
    assert not git_ran.exists()
    assert not pathlib.Path(env["FMTUNE_CACHE_DIR"]).exists()  # no clone, tree or venv
    assert not pathlib.Path(env["STUB_RECORD"]).exists()
    assert (out / "precious.txt").read_text() == "keep me\n"
    assert "GB_ARTIFACT_ID:" not in proc.stdout


def test_a_file_at_the_output_path_dies(env):
    out = pathlib.Path(env["FMTUNE_OUTPUT_DIR"])
    out.write_text("x\n")
    proc = _run(env)
    assert proc.returncode != 0
    assert f"fm-tune: output dir {out} exists and is not a directory" in proc.stderr
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


def test_an_existing_empty_output_dir_is_accepted(env):
    pathlib.Path(env["FMTUNE_OUTPUT_DIR"]).mkdir()
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr


# What a successful run leaves at the output dir's root after --model_at_root.
# Any model dir has one of these (a downloaded HF model has config.json), so none
# proves the dir is this step's to clear.
@pytest.mark.parametrize(
    "marker",
    ["config.json", "adapter_config.json", "model.safetensors", "adapters.safetensors"],
)
def test_a_dir_holding_a_model_is_refused_and_kept(env, marker):
    out = pathlib.Path(env["FMTUNE_OUTPUT_DIR"])
    out.mkdir()
    (out / marker).write_text("{}")
    (out / "tokenizer_config.json").write_text("{}")
    proc = _run(env)
    assert proc.returncode == 2
    assert f"fm-tune: output dir {out} is not empty;" in proc.stderr
    assert sorted(p.name for p in out.iterdir()) == [marker, "tokenizer_config.json"]
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


def test_a_failed_run_leaves_the_output_dir_empty_so_a_retry_succeeds(env):
    """SkyPilot retries a failed run into the same output dir. The failed attempt
    trained in its own staging dir, which is kept (named in the error) for debugging."""
    out = pathlib.Path(env["FMTUNE_OUTPUT_DIR"])
    env["STUB_EXIT"] = "1"
    failed = _run(env)
    assert failed.returncode == 1
    stage = _flag(_record(env)["argv"], "--output_dir")
    assert f"its files are kept in {stage}" in failed.stderr
    assert pathlib.Path(stage).is_dir() and not out.exists()
    del env["STUB_EXIT"]
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert sorted(p.name for p in out.iterdir()) == ["adapter_config.json"]
    assert pathlib.Path(stage).is_dir()  # the failed attempt's files, untouched


def test_a_live_run_sharing_the_output_dir_loses_nothing(env, tmp_path):
    """Two targets of one build share GB_BUILD_ID. While the first trains, a second
    run into the same output dir must not delete the first's checkpoints."""
    out = pathlib.Path(env["FMTUNE_OUTPUT_DIR"])
    env["GB_BUILD_ID"] = "b-1"
    second = dict(env, STUB_RECORD=str(tmp_path / "rec2.json"))
    hook = tmp_path / "hook.py"  # what happens while the first run "trains"
    hook.write_text(
        "import json, os, pathlib, subprocess, sys\n"
        "stage = pathlib.Path(sys.argv[1])\n"
        "(stage / 'checkpoint-1').mkdir()\n"
        f"env = json.loads({json.dumps(json.dumps(second))})\n"
        f"subprocess.run(['bash', {str(_SCRIPT)!r}], env=env, check=True)\n"
    )
    env["STUB_HOOK"] = f'"{sys.executable}" "{hook}" "$STUB_STAGE"'
    env["STUB_EXIT"] = "1"  # then the first run fails
    proc = _run(env)
    assert proc.returncode == 1, proc.stderr
    stage = pathlib.Path(_flag(_record(env)["argv"], "--output_dir"))
    assert (stage / "checkpoint-1").is_dir()  # survived the second run
    assert sorted(p.name for p in out.iterdir()) == ["adapter_config.json"]


def test_a_model_published_while_a_run_trains_is_kept(env):
    """A passes the empty check, then B publishes its model before A finishes:
    A's publish must fail without touching B's model, and keep A's own."""
    out = pathlib.Path(env["FMTUNE_OUTPUT_DIR"])
    env["STUB_HOOK"] = f"mkdir -p '{out}' && echo B > '{out}/config.json'"
    proc = _run(env)
    assert proc.returncode == 2
    stage = _flag(_record(env)["argv"], "--output_dir")
    assert f"cannot move the model into output dir {out}" in proc.stderr
    assert f"the model is kept in {stage}" in proc.stderr
    assert (out / "config.json").read_text() == "B\n"
    assert sorted(p.name for p in out.iterdir()) == ["config.json"]
    assert (pathlib.Path(stage) / "adapter_config.json").is_file()
    assert "GB_ARTIFACT_ID:" not in proc.stdout


@pytest.mark.parametrize("which", ["FMTUNE_BASE_MODEL_DIR", "FMTUNE_DATASET_DIR"])
def test_an_input_inside_the_output_dir_is_never_deleted(env, tmp_path, which):
    """Nothing in the output dir is ever cleared, so an input nested in it is safe,
    even beside a retry mark naming this build (which an earlier version cleared)."""
    out = pathlib.Path(env["FMTUNE_OUTPUT_DIR"])
    env["GB_BUILD_ID"] = "b-1"
    out.mkdir()
    (out / ".fm-tune-incomplete").write_text("b-1\n")
    inp = out / "bound-input"
    if which == "FMTUNE_DATASET_DIR":
        shutil.copytree(env["FMTUNE_DATASET_DIR"], inp)
    else:
        inp.mkdir(parents=True)
        (inp / "config.json").write_text("{}")
    env[which] = str(inp)
    before = sorted(p.name for p in inp.iterdir())
    proc = _run(env)
    assert proc.returncode == 2
    assert f"fm-tune: output dir {out} is not empty;" in proc.stderr
    assert sorted(p.name for p in inp.iterdir()) == before


def test_a_symlinked_output_dir_is_published_through_its_target(env, tmp_path):
    out = pathlib.Path(env["FMTUNE_OUTPUT_DIR"])
    real = tmp_path / "real-out"
    real.mkdir()
    out.symlink_to(real)
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert out.is_symlink()
    assert sorted(p.name for p in real.iterdir()) == ["adapter_config.json"]
    assert f"GB_ARTIFACT_ID:model GB_ARTIFACT_PATH:{out}" in proc.stdout.splitlines()


def test_a_non_empty_output_dir_is_refused_after_a_success(env):
    """A finished model is never overwritten by the next run into the same dir."""
    out = pathlib.Path(env["FMTUNE_OUTPUT_DIR"])
    assert _run(env).returncode == 0
    (out / "precious.txt").write_text("keep me\n")
    proc = _run(env)
    assert proc.returncode == 2
    assert "is not empty;" in proc.stderr
    assert (out / "precious.txt").read_text() == "keep me\n"
    assert (out / "adapter_config.json").is_file()


@pytest.mark.parametrize("which", ["FMTUNE_BASE_MODEL_DIR", "FMTUNE_DATASET_DIR"])
def test_an_output_dir_that_is_an_input_dies_before_any_clone(env, tmp_path, which):
    """Publishing onto an input would put the model in the base weights' (or the
    dataset's) dir; refuse it before any clone or install."""
    _git_repo(env, tmp_path)
    git_ran = _git_spy(env, tmp_path)
    if which == "FMTUNE_BASE_MODEL_DIR":
        (tmp_path / "bm").mkdir()
        env[which] = str(tmp_path / "bm")
    inp = pathlib.Path(env[which])
    (inp / "config.json").write_text("{}")
    (inp / "model.safetensors").write_text("w")
    env["FMTUNE_OUTPUT_DIR"] = f"{inp}/./"  # another spelling of the same dir
    proc = _run(env)
    assert proc.returncode != 0
    assert (
        "fm-tune: output dir " in proc.stderr
        and "is the same directory as" in proc.stderr
    )
    assert not git_ran.exists()
    assert not pathlib.Path(env["FMTUNE_CACHE_DIR"]).exists()
    assert (inp / "model.safetensors").read_text() == "w"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the read-only mode")
def test_an_output_dir_that_cannot_be_listed_dies(env, tmp_path):
    out = pathlib.Path(env["FMTUNE_OUTPUT_DIR"])
    out.mkdir()
    (out / "precious.txt").write_text("keep me\n")
    out.chmod(0o300)  # writable and searchable, not readable
    try:
        proc = _run(env)
    finally:
        out.chmod(0o755)
    assert proc.returncode != 0
    assert f"fm-tune: cannot list output dir {out}" in proc.stderr
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


def test_a_dangling_symlink_at_the_output_path_dies_before_any_clone(env, tmp_path):
    _git_repo(env, tmp_path)
    git_ran = _git_spy(env, tmp_path)
    out = pathlib.Path(env["FMTUNE_OUTPUT_DIR"])
    out.symlink_to(tmp_path / "missing")
    proc = _run(env)
    assert proc.returncode != 0
    assert f"fm-tune: output dir {out} is a dangling symlink" in proc.stderr
    assert not git_ran.exists()
    assert not pathlib.Path(env["FMTUNE_CACHE_DIR"]).exists()


def test_relative_paths_are_made_absolute_against_the_start_dir(env, tmp_path):
    """main.py runs after `cd <fm-tune root>`, and the marker is read off-node."""
    (tmp_path / "bm").mkdir()
    env.update(
        FMTUNE_OUTPUT_DIR="out",
        FMTUNE_DATASET_DIR="data dir",
        FMTUNE_SOURCE_PATH="fm-tune",
        FMTUNE_BASE_MODEL_DIR="bm",
    )
    proc = _run(env, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    rec = _record(env)
    argv = rec["argv"]
    assert _staged_as(argv, f"{tmp_path}/out")
    assert _flag(argv, "--train_file") == f"{tmp_path}/data dir/d_train.jsonl"
    assert _flag(argv, "--validation_file") == f"{tmp_path}/data dir/d_validation.jsonl"
    assert _flag(argv, "--model_name_or_path") == f"{tmp_path}/bm"
    assert (
        _flag(argv, "--config_file")
        == f"{tmp_path}/fm-tune/autotune/configs/autotune.yaml"
    )
    assert rec["cwd"] == f"{tmp_path}/fm-tune"
    assert (
        f"GB_ARTIFACT_ID:model GB_ARTIFACT_PATH:{tmp_path}/out"
        in proc.stdout.splitlines()
    )


def test_a_model_id_is_not_made_absolute(env, tmp_path):
    proc = _run(env, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert _flag(_record(env)["argv"], "--model_name_or_path") == "org/tiny"


def test_a_relative_cache_dir_still_finds_the_venv_python_after_the_cd(env, tmp_path):
    """The venv python is under the cache dir and runs main.py from the fm-tune root.
    A stub interpreter fakes `-m venv` and `-m pip` so no real venv is built."""
    real = sys.executable
    stub = tmp_path / "bin" / "python3"
    stub.parent.mkdir()
    stub.write_text(
        "#!/bin/sh\n"
        'case "$1 $2" in\n'
        '  "-m venv") mkdir -p "$3/bin" && cp "$0" "$3/bin/python"; exit $? ;;\n'
        '  "-m pip") exit 0 ;;\n'
        "esac\n"
        f'exec "{real}" "$@"\n'
    )
    stub.chmod(0o755)
    env.pop("FMTUNE_SKIP_INSTALL")
    env.update(FMTUNE_PYTHON=str(stub), FMTUNE_CACHE_DIR="cache")
    proc = _run(env, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    venv_bin = _record(env)["path"].split(":")[0]
    assert venv_bin.startswith(f"{tmp_path}/cache/venv-") and venv_bin.endswith("/bin")


@pytest.mark.parametrize("slashes", ["/", "//", "///"])
def test_every_trailing_slash_on_the_output_dir_is_stripped(env, slashes):
    out = env["FMTUNE_OUTPUT_DIR"]
    env["FMTUNE_OUTPUT_DIR"] = out + slashes
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert _staged_as(_record(env)["argv"], out)
    assert f"GB_ARTIFACT_ID:model GB_ARTIFACT_PATH:{out}" in proc.stdout.splitlines()


def test_a_root_output_dir_is_not_stripped_to_empty_and_is_refused(env, tmp_path):
    _git_repo(env, tmp_path)
    git_ran = _git_spy(env, tmp_path)
    env["FMTUNE_OUTPUT_DIR"] = "/"
    proc = _run(env)
    assert proc.returncode != 0
    assert "fm-tune: output dir / is not empty;" in proc.stderr
    assert not git_ran.exists()


# The monitors build the binding as '{ "path": "<marker path>" }' and json.loads it:
# a '"' or a control character fails the parse, a '\' is read as a JSON escape, and
# a newline ends the marker line early, registering a truncated path. The bash
# monitor's LocalFileStream rstrip()s each line, so a trailing space is lost too.
@pytest.mark.parametrize("bad", ['a"b', "a\\b", "a\nb", "a\tb", "a "])
def test_an_output_path_that_breaks_the_marker_json_dies_before_any_clone(
    env, tmp_path, bad
):
    _git_repo(env, tmp_path)
    git_ran = _git_spy(env, tmp_path)
    env["FMTUNE_OUTPUT_DIR"] = str(tmp_path / bad)
    proc = _run(env)
    assert proc.returncode != 0
    assert "fm-tune: output dir must not contain" in proc.stderr
    assert not git_ran.exists()
    assert not pathlib.Path(env["STUB_RECORD"]).exists()
    assert "GB_ARTIFACT_ID:" not in proc.stdout


@pytest.mark.slow
def test_venv_is_built_once_and_reused(env):
    """Without FMTUNE_SKIP_INSTALL a venv is built from pyproject and then reused."""
    root = pathlib.Path(env["FMTUNE_SOURCE_PATH"])
    (root / "pyproject.toml").write_text(
        "[project]\nname='autotune'\nversion='0'\n[project.optional-dependencies]\ncore=[]\n"
        "[build-system]\nrequires=['setuptools']\nbuild-backend='setuptools.build_meta'\n"
        "[tool.setuptools]\npy-modules=[]\n"
    )
    env.pop("FMTUNE_SKIP_INSTALL")
    assert _run(env).returncode == 0, "first run"
    venvs = list(pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("venv-*/.installed"))
    assert len(venvs) == 1
    stamp = venvs[0].stat().st_mtime_ns
    _clear_output(env)
    assert _run(env).returncode == 0, "second run"
    assert venvs[0].stat().st_mtime_ns == stamp
    assert not list(pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("*.lock"))


@pytest.mark.slow
def test_a_venv_from_another_interpreter_is_not_reused(env, tmp_path):
    """A venv links to its base python: one built by another interpreter (a patch
    upgrade at a new path, another node type) would run a deleted or wrong python."""
    root = pathlib.Path(env["FMTUNE_SOURCE_PATH"])
    (root / "pyproject.toml").write_text(
        "[project]\nname='autotune'\nversion='0'\n[project.optional-dependencies]\ncore=[]\n"
        "[build-system]\nrequires=['setuptools']\nbuild-backend='setuptools.build_meta'\n"
        "[tool.setuptools]\npy-modules=[]\n"
    )
    env.pop("FMTUNE_SKIP_INSTALL")
    assert _run(env).returncode == 0, "first run"
    # The same python version under another real path, as after an upgrade.
    # A copied binary still finds its stdlib through its compiled-in prefix.
    real = pathlib.Path(os.path.realpath(sys.executable))
    other = tmp_path / "other-python" / real.name
    other.parent.mkdir()
    shutil.copy2(real, other)
    env["FMTUNE_PYTHON"] = str(other)
    _clear_output(env)
    assert _run(env).returncode == 0, "second run"
    venvs = list(pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("venv-*/.installed"))
    assert len(venvs) == 2


@pytest.mark.slow
def test_a_cached_venv_whose_python_no_longer_runs_is_rebuilt(env):
    """Same interpreter path, changed in place: the key matches, the venv is dead."""
    root = pathlib.Path(env["FMTUNE_SOURCE_PATH"])
    (root / "pyproject.toml").write_text(
        "[project]\nname='autotune'\nversion='0'\n[project.optional-dependencies]\ncore=[]\n"
        "[build-system]\nrequires=['setuptools']\nbuild-backend='setuptools.build_meta'\n"
        "[tool.setuptools]\npy-modules=[]\n"
    )
    env.pop("FMTUNE_SKIP_INSTALL")
    assert _run(env).returncode == 0, "first run"
    (venv,) = pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("venv-*")
    py = venv / "bin" / "python"
    py.unlink()
    py.write_text("#!/bin/sh\nexit 127\n")  # a venv python whose base is gone
    py.chmod(0o755)
    _clear_output(env)
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert "building venv" in proc.stdout


@pytest.mark.parametrize("spelling", ["{out}/.", "{out}/./", "{out}/./.", "{d}/./out"])
def test_dot_components_in_the_output_dir_are_dropped(env, tmp_path, spelling):
    """Staging is <output dir>.partial.*: for '<out>/.' that was inside <out>, and
    the publish rename then failed after training."""
    out = env["FMTUNE_OUTPUT_DIR"]
    env["FMTUNE_OUTPUT_DIR"] = spelling.format(out=out, d=tmp_path)
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert _staged_as(_record(env)["argv"], out)
    assert f"GB_ARTIFACT_ID:model GB_ARTIFACT_PATH:{out}" in proc.stdout.splitlines()
    assert sorted(p.name for p in pathlib.Path(out).iterdir()) == [
        "adapter_config.json"
    ]


@pytest.mark.parametrize("spelling", ["{out}/..", "{d}/x/../out"])
def test_a_dot_dot_in_the_output_dir_dies_before_any_clone(env, tmp_path, spelling):
    """Not resolved: through a symlinked parent, a/link/../out is not a/out."""
    _git_repo(env, tmp_path)
    git_ran = _git_spy(env, tmp_path)
    env["FMTUNE_OUTPUT_DIR"] = spelling.format(out=env["FMTUNE_OUTPUT_DIR"], d=tmp_path)
    proc = _run(env)
    assert proc.returncode == 2
    assert "fm-tune: output dir must not contain a '..' component" in proc.stderr
    assert not git_ran.exists()
