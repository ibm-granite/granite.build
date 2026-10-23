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

"""Unit tests for src/fmtune_run.sh, run against a stub fm-tune tree.

The stub ``main.py`` records its argv and cwd as JSON, then fakes a saved model at
``<output_dir>/<output_model_name>/``, so each test asserts the exact command
line fm-tune would get. ``FMTUNE_SKIP_INSTALL=1`` skips the venv, so most tests run
in well under a second. The exception is the ``slow`` venv test, which builds a real
venv and needs network access for pip. Mode 1 only.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest
import yaml

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "src" / "fmtune_run.sh"

_STUB_MAIN = r"""
import json, os, sys
args = sys.argv[1:]
rec = {"argv": args, "cwd": os.getcwd(), "log_path": os.environ.get("LOG_PATH"),
       "path": os.environ.get("PATH")}
if os.environ.get("STUB_CONFIG_COPY"):
    cfg = args[args.index("--config_file") + 1]
    rec["config_text"] = open(cfg).read()
json.dump(rec, open(os.environ["STUB_RECORD"], "w"))
if os.environ.get("STUB_EXIT"):
    sys.exit(int(os.environ["STUB_EXIT"]))
out = args[args.index("--output_dir") + 1]
name = args[args.index("--output_model_name") + 1]
os.makedirs(os.path.join(out, name), exist_ok=True)
open(os.path.join(out, name, "adapter_config.json"), "w").write("{}")
"""


@pytest.fixture
def env(tmp_path):
    """A minimal valid environment: stub fm-tune tree, dataset, output, cache."""
    if shutil.which("bash") is None:  # pragma: no cover - bash is present in CI
        pytest.skip("bash not available")
    src = tmp_path / "fm-tune"
    (src / "autotune" / "configs").mkdir(parents=True)
    (src / "autotune" / "configs" / "autotune.yaml").write_text("tune_config: {}\n")
    (src / "main.py").write_text(_STUB_MAIN)
    (src / "pyproject.toml").write_text("[project]\nname='autotune'\n")
    data = tmp_path / "data dir"  # a space checks the quoting
    data.mkdir()
    (data / "d_train.jsonl").write_text("{}\n")
    (data / "d_validation.jsonl").write_text("{}\n")
    return {
        "PATH": os.environ["PATH"],
        "FMTUNE_SKIP_INSTALL": "1",
        "FMTUNE_PYTHON": sys.executable,
        "FMTUNE_SOURCE_PATH": str(src),
        "FMTUNE_DATASET_DIR": str(data),
        "FMTUNE_TRAIN_FILE": "d_train.jsonl",
        "FMTUNE_VALIDATION_FILE": "d_validation.jsonl",
        "FMTUNE_MODEL": "org/tiny",
        "FMTUNE_RUN_NAME": "run1",
        "FMTUNE_TUNING_ALGO": "lora",
        "FMTUNE_OUTPUT_DIR": str(tmp_path / "out"),
        "FMTUNE_CACHE_DIR": str(tmp_path / "cache"),
        "STUB_RECORD": str(tmp_path / "record.json"),
    }


def _run(env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(_SCRIPT)], env=env, capture_output=True, text=True, check=False
    )


def _record(env: dict) -> dict:
    return json.loads(pathlib.Path(env["STUB_RECORD"]).read_text())


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_success_passes_the_core_flags_and_prints_one_marker(env):
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    argv = _record(env)["argv"]
    data = env["FMTUNE_DATASET_DIR"]
    assert _flag(argv, "--train_file") == f"{data}/d_train.jsonl"
    assert _flag(argv, "--validation_file") == f"{data}/d_validation.jsonl"
    assert _flag(argv, "--model_name_or_path") == "org/tiny"
    assert _flag(argv, "--tuning_algo") == "lora"
    assert _flag(argv, "--run_name") == "run1"
    assert _flag(argv, "--output_model_name") == "run1"
    assert _flag(argv, "--output_dir") == env["FMTUNE_OUTPUT_DIR"]
    for f in ("--model_at_root", "--cleanup", "--save_history"):
        assert f in argv
    for absent in ("--no_autotune", "--rl_algo", "--autotunex_server_url", "--job_id"):
        assert absent not in argv
    markers = [ln for ln in proc.stdout.splitlines() if ln.startswith("GB_ARTIFACT_ID:")]
    assert markers == [f"GB_ARTIFACT_ID:model GB_ARTIFACT_PATH:{env['FMTUNE_OUTPUT_DIR']}"]


def test_main_py_runs_from_the_fm_tune_root(env):
    assert _run(env).returncode == 0
    assert _record(env)["cwd"] == env["FMTUNE_SOURCE_PATH"]


def test_bound_base_model_wins_over_model_id(env, tmp_path):
    (tmp_path / "bm").mkdir()
    env["FMTUNE_BASE_MODEL_DIR"] = str(tmp_path / "bm")
    assert _run(env).returncode == 0
    assert _flag(_record(env)["argv"], "--model_name_or_path") == str(tmp_path / "bm")


@pytest.mark.parametrize(
    "over, flags",
    [
        ({"FMTUNE_AUTOTUNE": "false"}, ["--no_autotune"]),
        ({"FMTUNE_RL_ALGO": "dpo"}, ["--rl_algo", "dpo"]),
        ({"FMTUNE_BACKEND": "mlx"}, ["--backend", "mlx"]),
        ({"FMTUNE_SEED": "7"}, ["--seed", "7"]),
        ({"FMTUNE_SERVER_URL": "http://h:1", "FMTUNE_JOB_ID": "j1"},
         ["--autotunex_server_url", "http://h:1", "--job_id", "j1"]),
        ({"FMTUNE_EXTRA_ARGS": "--data_backend ray_data"}, ["--data_backend", "ray_data"]),
    ],
)
def test_optional_settings_add_their_flags(env, over, flags):
    env.update(over)
    assert _run(env).returncode == 0
    argv = _record(env)["argv"]
    i = argv.index(flags[0])
    assert argv[i : i + len(flags)] == flags


def test_preset_is_used_when_hpo_config_is_empty(env):
    env["FMTUNE_HPO_CONFIG_JSON"] = "{}"
    assert _run(env).returncode == 0
    cfg = _flag(_record(env)["argv"], "--config_file")
    assert cfg == f"{env['FMTUNE_SOURCE_PATH']}/autotune/configs/autotune.yaml"


def test_inline_hpo_config_is_written_and_passed(env):
    cfg = {
        "tune_config": {"num_samples": {"default": 2}},
        "lr": {"default": 5e-06, "values": [1e-06, 5e-06]},
    }
    env["FMTUNE_HPO_CONFIG_JSON"] = json.dumps(cfg)
    env["STUB_CONFIG_COPY"] = "1"
    assert _run(env).returncode == 0
    rec = _record(env)
    # fm-tune reads the file with yaml.safe_load, which reads JSON's 5e-06 as a str.
    loaded = yaml.safe_load(rec["config_text"])
    assert loaded == cfg
    assert isinstance(loaded["lr"]["default"], float)
    assert all(isinstance(v, float) for v in loaded["lr"]["values"])
    assert not _flag(rec["argv"], "--config_file").startswith(env["FMTUNE_SOURCE_PATH"])


def test_invalid_hpo_config_json_fails_before_training(env):
    env["FMTUNE_HPO_CONFIG_JSON"] = "{not json"
    proc = _run(env)
    assert proc.returncode != 0
    assert "fm-tune:" in proc.stderr
    assert not pathlib.Path(env["STUB_RECORD"]).exists()
    assert "GB_ARTIFACT_ID:" not in proc.stdout


def test_failed_training_exits_non_zero_without_a_marker(env):
    env["STUB_EXIT"] = "3"
    proc = _run(env)
    assert proc.returncode == 3
    assert "GB_ARTIFACT_ID:" not in proc.stdout


@pytest.mark.parametrize(
    "over, message",
    [
        ({"FMTUNE_MODEL": ""}, "set fm_tune_config.model or bind the base_model input"),
        ({"FMTUNE_SOURCE_PATH": ""}, "set exactly one of source.path and source.repo"),
        ({"FMTUNE_SOURCE_REPO": "https://x/y.git"}, "set exactly one of source.path and source.repo"),
        ({"FMTUNE_TRAIN_FILE": ""}, "train_file is empty"),
        ({"FMTUNE_TRAIN_FILE": "nope.jsonl"}, "train_file not found"),
        ({"FMTUNE_VALIDATION_FILE": "nope.jsonl"}, "validation_file not found"),
        ({"FMTUNE_CONFIG_PRESET": "missing.yaml"}, "config_preset not found"),
        ({"FMTUNE_DATASET_DIR": ""}, "FMTUNE_DATASET_DIR is empty"),
    ],
)
def test_invalid_input_fails_before_training(env, over, message):
    env.update(over)
    proc = _run(env)
    assert proc.returncode != 0
    assert f"fm-tune: {message}" in proc.stderr
    assert not pathlib.Path(env["STUB_RECORD"]).exists()
    assert "GB_ARTIFACT_ID:" not in proc.stdout


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
    assert _flag(_record(env)["argv"], "--output_dir") == f"{env['LLMB_BASH_OUTPUT_DIR']}/model"
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


def test_no_echoed_line_starts_with_the_marker_prefix_on_failure(env):
    """A run_name carrying a marker is rejected before anything is echoed."""
    env["FMTUNE_RUN_NAME"] = "GB_ARTIFACT_ID:evil"
    env["STUB_EXIT"] = "1"
    proc = _run(env)
    assert proc.returncode != 0
    assert "fm-tune: run_name must match ^[A-Za-z0-9._-]+$" in proc.stderr
    assert not any(ln.startswith("GB_ARTIFACT_") for ln in proc.stdout.splitlines())
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


@pytest.mark.parametrize("name", ["a b", "a/b", "a\nb", "", "\u00e9t\u00e9"])
def test_run_name_outside_the_safe_set_is_rejected(env, name):
    env["FMTUNE_RUN_NAME"] = name
    proc = _run(env)
    if name == "":  # unset-or-empty falls back to the default fm-tune
        assert proc.returncode == 0, proc.stderr
        return
    assert proc.returncode != 0
    assert "fm-tune: run_name must match" in proc.stderr
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


def test_log_path_defaults_to_the_output_dirs_parent(env, tmp_path):
    """fm-tune's bridge logs to $LOG_PATH/logs; unset, that is cwd (the source tree)."""
    assert _run(env).returncode == 0
    assert _record(env)["log_path"] == str(tmp_path)


def test_a_preset_log_path_is_kept(env, tmp_path):
    env["LOG_PATH"] = str(tmp_path / "mylogs")
    assert _run(env).returncode == 0
    assert _record(env)["log_path"] == str(tmp_path / "mylogs")


def test_the_interpreters_dir_is_first_on_path(env):
    assert _run(env).returncode == 0
    assert _record(env)["path"].split(":")[0] == os.path.dirname(env["FMTUNE_PYTHON"])


def test_multi_line_extra_args_keep_every_word(env):
    env["FMTUNE_EXTRA_ARGS"] = "--data_backend ray_data\n--foo bar\n"
    assert _run(env).returncode == 0
    argv = _record(env)["argv"]
    assert argv[-4:] == ["--data_backend", "ray_data", "--foo", "bar"]


def test_whitespace_only_extra_args_add_nothing(env):
    assert _run(env).returncode == 0
    plain = _record(env)["argv"]
    env["FMTUNE_EXTRA_ARGS"] = "  \n "
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert _record(env)["argv"] == plain


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


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
    second = _run(env)  # second run reuses the clone, and the tree: same commit
    assert second.returncode == 0, second.stderr
    assert second.stdout.count("[fm-tune] cloning") == 0
    assert list(cache.glob("src-*")) == clones
    assert list(cache.glob("tree-*")) == trees


def _git_repo(env, tmp_path):
    """A local file:// repo on branch rel holding the stub fm-tune under sub/fm-tune."""
    if shutil.which("git") is None:  # pragma: no cover
        pytest.skip("git not available")
    repo = tmp_path / "repo"
    shutil.copytree(env["FMTUNE_SOURCE_PATH"], repo / "sub" / "fm-tune")
    _git("init", "-q", "-b", "rel", cwd=repo)
    _git("add", ".", cwd=repo)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x", cwd=repo)
    env.update(
        FMTUNE_SOURCE_PATH="",
        FMTUNE_SOURCE_REPO=f"file://{repo}",
        FMTUNE_SOURCE_REF="rel",
        FMTUNE_SOURCE_SUBDIR="sub/fm-tune",
    )
    return repo


def _head(path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_cached_clone_is_updated_to_the_current_ref(env, tmp_path):
    """The cache is shared across builds, so a moving ref must not stay pinned."""
    repo = _git_repo(env, tmp_path)
    assert _run(env).returncode == 0
    (repo / "sub" / "fm-tune" / "NEW").write_text("2\n")
    _git("add", ".", cwd=repo)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "y", cwd=repo)
    second = _run(env)
    assert second.returncode == 0, second.stderr
    assert "[fm-tune] updating" in second.stdout
    assert second.stdout.count("[fm-tune] cloning") == 0
    tree = pathlib.Path(env["FMTUNE_CACHE_DIR"]) / f"tree-{_head(repo)}"
    assert _record(env)["cwd"] == str(tree / "sub" / "fm-tune")
    assert (tree / "sub" / "fm-tune" / "NEW").is_file()


def _commit(repo, msg):
    _git("add", "-A", ".", cwd=repo)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", msg, cwd=repo)


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
    snap = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tree1.rglob("*") if p.is_file()}
    (repo / "sub" / "fm-tune" / "GONE").unlink()
    _commit(repo, "drop GONE")
    second = _run(env)
    assert second.returncode == 0, second.stderr
    tree2 = pathlib.Path(_record(env)["cwd"])
    assert tree2 != tree1 and tree2.parents[1].name == f"tree-{_head(repo)}"
    assert not (tree2 / "GONE").exists()
    assert (tree1 / "GONE").is_file()
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tree1.rglob("*") if p.is_file()} == snap


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


def _clone_lock(env) -> pathlib.Path:
    key = subprocess.run(
        ["bash", "-c", 'printf "%s@%s" "$0" "$1" | cksum | cut -d" " -f1',
         env["FMTUNE_SOURCE_REPO"], env["FMTUNE_SOURCE_REF"]],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return pathlib.Path(env["FMTUNE_CACHE_DIR"]) / f"src-{key}.lock"


def test_stale_clone_lock_from_a_dead_pid_is_taken_over(env, tmp_path):
    _git_repo(env, tmp_path)
    dead = subprocess.Popen(["true"])
    dead.wait()
    lock = _clone_lock(env)
    lock.mkdir(parents=True)
    (lock / "pid").write_text(f"{os.uname().nodename}:{dead.pid}\n")
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert "taking over stale" in proc.stdout
    assert not list(pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("*.lock"))


def test_the_lock_records_host_and_pid(env, tmp_path):
    """While held, the lock's pid file names this host, so other nodes never judge it."""
    _git_repo(env, tmp_path)
    env["STUB_LOCKS"] = env["FMTUNE_CACHE_DIR"]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    real_git = shutil.which("git")
    (bin_dir / "git").write_text(
        "#!/bin/sh\ncat \"$STUB_LOCKS\"/*.lock/pid >> \"$STUB_LOCKS/../owners\" 2>/dev/null\n"
        f'exec "{real_git}" "$@"\n'
    )
    (bin_dir / "git").chmod(0o755)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    assert _run(env).returncode == 0
    owners = set((tmp_path / "owners").read_text().split())
    assert len(owners) == 1
    host, pid = owners.pop().rsplit(":", 1)
    assert host == os.uname().nodename and pid.isdigit()


def test_a_lock_held_by_another_host_is_never_taken_over(env, tmp_path):
    _git_repo(env, tmp_path)
    lock = _clone_lock(env)
    lock.mkdir(parents=True)
    (lock / "pid").write_text("otherhost:1\n")  # pid 1 is alive here, dead there: unknowable
    env["FMTUNE_LOCK_TIMEOUT"] = "2"
    proc = _run(env)
    assert proc.returncode != 0
    assert f"fm-tune: lock {lock}" in proc.stderr
    assert "remove it if no fm-tune build is running" in proc.stderr
    assert (lock / "pid").read_text() == "otherhost:1\n"
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


def test_an_old_pid_less_lock_is_taken_over(env, tmp_path):
    """Killed between mkdir and the pid write: stale once older than the timeout."""
    _git_repo(env, tmp_path)
    lock = _clone_lock(env)
    lock.mkdir(parents=True)
    old = lock.stat().st_mtime - 3600
    os.utime(lock, (old, old))
    env["FMTUNE_LOCK_TIMEOUT"] = "60"
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert "taking over stale" in proc.stdout
    assert not lock.exists()


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


def test_an_annotated_tag_resolves_to_one_tree_and_its_commit(env, tmp_path):
    repo = _git_repo(env, tmp_path)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "tag", "-a", "v1", "-m", "v1", cwd=repo)
    env["FMTUNE_SOURCE_REF"] = "v1"
    commit = subprocess.run(
        ["git", "rev-parse", "v1^{commit}"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    for _ in range(2):
        proc = _run(env)
        assert proc.returncode == 0, proc.stderr
        assert f"@v1 = {commit}" in proc.stdout
    trees = list(pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("tree-*"))
    assert [t.name for t in trees] == [f"tree-{commit}"]


@pytest.mark.parametrize("over", [{"FMTUNE_SOURCE_REF": "-uoops"},
                                  {"FMTUNE_SOURCE_REPO": "--upload-pack=touch /tmp/x"}])
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
    assert "fm-tune: source." in proc.stderr and "must not start with '-'" in proc.stderr
    assert not (tmp_path / "git-ran").exists()


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
    assert _run(env).returncode == 0, "second run"
    assert venvs[0].stat().st_mtime_ns == stamp
    assert not list(pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("*.lock"))
