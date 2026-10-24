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
the root of ``<output_dir>`` (where ``--model_at_root`` leaves it), so each test
asserts the exact command line fm-tune would get. ``FMTUNE_SKIP_INSTALL=1`` skips the venv, so most tests run
in well under a second. The exception is the ``slow`` venv test, which builds a real
venv and needs network access for pip. Mode 1 only.
"""

import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

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
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "adapter_config.json"), "w").write("{}")
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


def _run(env: dict, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(_SCRIPT)], env=env, capture_output=True, text=True, check=False, cwd=cwd
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


def test_job_id_defaults_to_the_build_id_when_the_bridge_is_on(env):
    env.update({"FMTUNE_SERVER_URL": "http://h:1", "GB_BUILD_ID": "b-123"})
    assert _run(env).returncode == 0
    assert _flag(_record(env)["argv"], "--job_id") == "b-123"


def test_an_explicit_job_id_wins_over_the_build_id(env):
    env.update({"FMTUNE_SERVER_URL": "http://h:1", "FMTUNE_JOB_ID": "j1", "GB_BUILD_ID": "b-123"})
    assert _run(env).returncode == 0
    assert _flag(_record(env)["argv"], "--job_id") == "j1"


def test_no_job_id_without_the_bridge(env):
    env["GB_BUILD_ID"] = "b-123"
    assert _run(env).returncode == 0
    assert "--job_id" not in _record(env)["argv"]


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
        ({"FMTUNE_TRAIN_FILE": ""}, "train_file is required"),
        ({"FMTUNE_TRAIN_FILE": "nope.jsonl"}, "train_file not found"),
        ({"FMTUNE_VALIDATION_FILE": "nope.jsonl"}, "validation_file not found"),
        ({"FMTUNE_CONFIG_PRESET": "missing.yaml"}, "config_preset not found"),
        ({"FMTUNE_DATASET_DIR": ""}, "FMTUNE_DATASET_DIR is empty"),
        ({"FMTUNE_TRAIN_FILE": "../x"}, "train_file must be a relative path inside the dataset"),
        ({"FMTUNE_TRAIN_FILE": ".."}, "train_file must be a relative path inside the dataset"),
        ({"FMTUNE_TRAIN_FILE": "a/../../x"}, "train_file must be a relative path inside the dataset"),
        ({"FMTUNE_TRAIN_FILE": "/etc/passwd"}, "train_file must be a relative path inside the dataset"),
        ({"FMTUNE_VALIDATION_FILE": "../x"}, "validation_file must be a relative path inside the dataset"),
        ({"FMTUNE_VALIDATION_FILE": "/abs/v.jsonl"}, "validation_file must be a relative path inside the dataset"),
        ({"FMTUNE_TUNING_ALGO": ""}, "tuning_algo and rl_algo cannot both be empty or none"),
        ({"FMTUNE_TUNING_ALGO": "none", "FMTUNE_RL_ALGO": "None"}, "tuning_algo and rl_algo cannot both be empty or none"),
        ({"FMTUNE_TUNING_ALGO": "", "FMTUNE_RL_ALGO": "dpo"}, "an offline rl_algo (dpo, kto) needs a tuning_algo"),
        ({"FMTUNE_TUNING_ALGO": "None", "FMTUNE_RL_ALGO": "Kto"}, "an offline rl_algo (dpo, kto) needs a tuning_algo"),
    ],
)
def test_invalid_input_fails_before_training(env, over, message):
    env.update(over)
    proc = _run(env)
    assert proc.returncode != 0
    assert f"fm-tune: {message}" in proc.stderr
    assert not pathlib.Path(env["STUB_RECORD"]).exists()
    assert "GB_ARTIFACT_ID:" not in proc.stdout


@pytest.mark.parametrize("name", ["sub/dir/train.jsonl", "a..b.jsonl", "./train.jsonl"])
def test_a_split_file_in_a_subdir_or_with_dots_in_its_name_is_accepted(env, name):
    """The '..' rule matches a whole path component, not a substring."""
    target = pathlib.Path(env["FMTUNE_DATASET_DIR"]) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}\n")
    env["FMTUNE_TRAIN_FILE"] = name
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert _flag(_record(env)["argv"], "--train_file") == f"{env['FMTUNE_DATASET_DIR']}/{name}"


def test_a_split_file_outside_the_dataset_dies_before_any_clone(env, tmp_path):
    _git_repo(env, tmp_path)
    git_ran = _git_spy(env, tmp_path)
    env["FMTUNE_VALIDATION_FILE"] = "../../outside.jsonl"
    proc = _run(env)
    assert proc.returncode != 0
    assert "fm-tune: validation_file must be a relative path inside the dataset" in proc.stderr
    assert not git_ran.exists()


def test_online_rl_needs_no_tuning_algo(env):
    env.update(FMTUNE_TUNING_ALGO="", FMTUNE_RL_ALGO="grpo")
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert "--tuning_algo" not in _record(env)["argv"]


def test_no_algorithm_dies_before_any_clone(env, tmp_path):
    _git_repo(env, tmp_path)
    git_ran = _git_spy(env, tmp_path)
    env.update(FMTUNE_TUNING_ALGO="none", FMTUNE_RL_ALGO="")
    proc = _run(env)
    assert proc.returncode != 0
    assert "tuning_algo and rl_algo cannot both be empty or none" in proc.stderr
    assert not git_ran.exists()


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
    _git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=repo)  # as GitHub serves any SHA
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
    # It died holding the clone lock: the EXIT trap must release it.
    assert not list(pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("*.lock*"))


def _clone_lock(env) -> pathlib.Path:
    key = subprocess.run(
        ["bash", "-c", 'printf "%s@%s" "$0" "$1" | cksum | cut -d" " -f1',
         env["FMTUNE_SOURCE_REPO"], env["FMTUNE_SOURCE_REF"]],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return pathlib.Path(env["FMTUNE_CACHE_DIR"]) / f"src-{key}.lock"


def _stub(bin_dir: pathlib.Path, name: str, body: str) -> None:
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / name).write_text("#!/bin/sh\n" + body)
    (bin_dir / name).chmod(0o755)


def _dead_owner() -> str:
    dead = subprocess.Popen(["true"])
    dead.wait()
    return f"{os.uname().nodename}:{dead.pid}"


def _held_lock(env, owner) -> pathlib.Path:
    """A lock dir as another build left it: owner None = no pid file at all."""
    lock = _clone_lock(env)
    lock.mkdir(parents=True)
    if owner is not None:
        (lock / "pid").write_text(owner)
    return lock


def _sq(path) -> str:
    return "'" + str(path).replace("'", "'\\''") + "'"


def _run_waiting(env, tmp_path, limit=20):
    """Run with the 5 s poll stubbed to 0.05 s; returns (proc, seconds taken)."""
    _stub(tmp_path / "bin", "sleep", f'exec "{shutil.which("sleep")}" 0.05\n')
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    t0 = time.monotonic()
    try:
        proc = subprocess.run(["bash", str(_SCRIPT)], env=env, capture_output=True, check=False,
                              text=True, timeout=limit)
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"still waiting on the lock after {limit}s: {e.stdout!r} {e.stderr!r}")
    return proc, time.monotonic() - t0


def test_a_lock_whose_owner_died_on_this_host_dies_at_once_with_the_rm_hint(env, tmp_path):
    """Never taken over: the run stops at once and says how to clear it."""
    _git_repo(env, tmp_path)
    env["FMTUNE_CACHE_DIR"] = str(tmp_path / "it's cache")  # the hint shell-quotes the path
    owner = _dead_owner()
    lock = _held_lock(env, owner + "\n")
    env["FMTUNE_LOCK_TIMEOUT"] = "60"
    proc, took = _run_waiting(env, tmp_path)
    assert took < 10, took
    assert proc.returncode == 2
    assert (f"fm-tune: lock {lock} is held by {owner}, which is no longer running; "
            f"remove it with: rm -rf {_sq(lock)} if no fm-tune build is running") in proc.stderr
    assert (lock / "pid").read_text() == owner + "\n"
    assert sorted(p.name for p in lock.iterdir()) == ["pid"]
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


def test_an_owner_that_released_the_lock_and_then_exited_is_not_reported(env, tmp_path):
    """The waiter read the owner, then the owner released the lock and exited before
    kill -0 ran: that lock is gone, not stale. A stub cat releases it right after
    the waiter's first read of the pid file."""
    _git_repo(env, tmp_path)
    owner = _dead_owner()
    lock = _held_lock(env, owner + "\n")
    flipped = tmp_path / "flipped"
    _stub(tmp_path / "bin", "cat",
          f'"{shutil.which("cat")}" "$@"; rc=$?\n'
          f'if [ "$1" = "{lock}/pid" ] && [ ! -e "{flipped}" ]; then\n'
          f'  touch "{flipped}"; rm -rf "{lock}"\n'
          'fi\nexit $rc\n')
    env["FMTUNE_LOCK_TIMEOUT"] = "60"
    proc, _ = _run_waiting(env, tmp_path)
    assert flipped.exists()
    assert proc.returncode == 0, proc.stderr
    assert pathlib.Path(env["STUB_RECORD"]).exists()
    assert not lock.exists()


def _assert_timed_out(env, proc, took, lock, owner_text, contents):
    assert 1 <= took < 15, took
    assert proc.returncode == 2
    assert f"fm-tune: lock {lock} is still held after " in proc.stderr
    assert (f"(owner: {owner_text}); remove it with: rm -rf {_sq(lock)} "
            "if no fm-tune build is running") in proc.stderr
    assert sorted(p.name for p in lock.iterdir()) == contents
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


@pytest.mark.parametrize("pid", ["own", "1"])
def test_a_lock_held_by_a_live_owner_on_this_host_is_waited_on_then_reported(env, tmp_path, pid):
    """pid 1 is another user's live process: kill -0 fails (EPERM), but it is not dead."""
    if pid == "1":
        try:
            os.kill(1, 0)
        except PermissionError:
            pass
        else:  # root, or a container whose pid 1 is ours: it would test the alive path
            pytest.skip("this user may signal pid 1")
    _git_repo(env, tmp_path)
    owner = f"{os.uname().nodename}:{os.getpid() if pid == 'own' else 1}"
    lock = _held_lock(env, owner + "\n")
    env["FMTUNE_LOCK_TIMEOUT"] = "2"
    proc, took = _run_waiting(env, tmp_path)
    _assert_timed_out(env, proc, took, lock, owner, ["pid"])
    assert (lock / "pid").read_text() == owner + "\n"


@pytest.mark.parametrize("pid_file", [None, ""], ids=["pid-less", "empty-pid"])
def test_a_lock_with_no_owner_recorded_is_waited_on_then_reported(env, tmp_path, pid_file):
    _git_repo(env, tmp_path)
    lock = _held_lock(env, pid_file)
    env["FMTUNE_LOCK_TIMEOUT"] = "2"
    proc, took = _run_waiting(env, tmp_path)
    _assert_timed_out(env, proc, took, lock, "none recorded", [] if pid_file is None else ["pid"])
    if pid_file is not None:
        assert (lock / "pid").read_text() == ""


def test_a_lock_released_while_waiting_is_then_taken(env, tmp_path):
    _git_repo(env, tmp_path)
    lock = _held_lock(env, f"{os.uname().nodename}:{os.getpid()}\n")
    waited = tmp_path / "waited"
    _stub(tmp_path / "bin", "sleep", f'touch "{waited}"; exec "{shutil.which("sleep")}" 0.05\n')
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["FMTUNE_LOCK_TIMEOUT"] = "60"
    proc = subprocess.Popen(["bash", str(_SCRIPT)], env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 15
    while not waited.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert waited.exists(), proc.communicate(timeout=15)
    shutil.rmtree(lock)  # the holder finishes
    out, err = proc.communicate(timeout=15)
    assert proc.returncode == 0, err
    assert f"[fm-tune] waiting for lock {lock}" in out
    assert pathlib.Path(env["STUB_RECORD"]).exists()
    assert not lock.exists()


def test_a_lock_released_between_the_mkdir_and_the_look_is_then_taken(env, tmp_path):
    """mkdir fails on a held lock, which is released before the waiter looks at it:
    no dir there is not an unwritable cache. A stub mkdir releases it after the
    first failed attempt."""
    _git_repo(env, tmp_path)
    lock = _held_lock(env, "otherhost:1\n")
    flipped = tmp_path / "flipped"
    _stub(tmp_path / "bin", "mkdir",
          f'"{shutil.which("mkdir")}" "$@"; rc=$?\n'
          f'if [ "$1" = "{lock}" ] && [ ! -e "{flipped}" ]; then\n'
          f'  touch "{flipped}"; rm -rf "{lock}"\n'
          'fi\nexit $rc\n')
    env["FMTUNE_LOCK_TIMEOUT"] = "60"
    proc, _ = _run_waiting(env, tmp_path)
    assert flipped.exists()
    assert proc.returncode == 0, proc.stderr
    assert not lock.exists()


def test_concurrent_builds_hold_the_lock_one_at_a_time(env, tmp_path):
    """At most one build may be inside the locked section at a time. A stub git logs
    enter/exit around each call (the clone section runs only git); a stub sleep makes
    the 5 s poll fast."""
    _git_repo(env, tmp_path)
    events = tmp_path / "events"
    real_sleep = shutil.which("sleep")
    _stub(tmp_path / "bin", "git",
          f'echo "$STUB_ID enter" >> "{events}"; "{real_sleep}" 0.1\n'
          f'"{shutil.which("git")}" "$@"; rc=$?\n'
          f'echo "$STUB_ID exit" >> "{events}"; exit $rc\n')
    _stub(tmp_path / "bin", "sleep", f'exec "{real_sleep}" 0.05\n')
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["FMTUNE_LOCK_TIMEOUT"] = "60"
    procs = []
    for i in range(5):
        e = dict(env, STUB_ID=str(i), FMTUNE_OUTPUT_DIR=str(tmp_path / f"out{i}"),
                 STUB_RECORD=str(tmp_path / f"record{i}.json"))
        procs.append(subprocess.Popen(["bash", str(_SCRIPT)], env=e, text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    outs = [p.communicate(timeout=60) for p in procs]
    ids = [ln.split()[0] for ln in events.read_text().splitlines()]
    # One build's git calls form one unbroken run; another build's call inside it overlaps.
    runs = [x for k, x in enumerate(ids) if k == 0 or ids[k - 1] != x]
    assert len(runs) == len(set(runs)) == 5, f"locked sections overlapped: {ids}"
    assert [p.returncode for p in procs] == [0] * 5, [e for _, e in outs]
    assert sorted(p.name for p in pathlib.Path(env["FMTUNE_CACHE_DIR"]).iterdir()
                  if ".lock" in p.name) == []


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
    lock = _held_lock(env, "otherhost:1\n")  # pid 1 is alive here, dead there: unknowable
    env["FMTUNE_LOCK_TIMEOUT"] = "2"
    proc, took = _run_waiting(env, tmp_path)
    _assert_timed_out(env, proc, took, lock, "otherhost:1", ["pid"])
    assert (lock / "pid").read_text() == "otherhost:1\n"


def _gone_rm_stub(tmp_path, lock, body: str) -> None:
    """A stub rm running ``body`` when unlock removes the renamed-away lock dir."""
    _stub(tmp_path / "bin", "rm",
          f'case "$*" in *{lock.name}.gone.*) {body} ;; esac\n'
          f'exec "{shutil.which("rm")}" "$@"\n')


def test_a_failed_unlock_never_removes_a_lock_taken_since(env, tmp_path):
    """rm -rf of the renamed-away dir can fail (NFS .nfsXXXX, EIO): set -e exited
    there, and the EXIT trap's second unlock moved and removed the lock another
    build had taken at the freed path meanwhile."""
    _git_repo(env, tmp_path)
    lock = _clone_lock(env)
    _gone_rm_stub(tmp_path, lock,
                  f'[ -e "{tmp_path}/flipped" ] || {{ touch "{tmp_path}/flipped"; '
                  f'mkdir "{lock}"; echo otherhost:1 > "{lock}/pid"; exit 1; }}')
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    proc = _run(env)
    assert (tmp_path / "flipped").exists()
    assert (lock / "pid").read_text() == "otherhost:1\n", proc.stderr
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_a_build_killed_while_unlocking_never_removes_a_lock_taken_since(env, tmp_path, sig):
    """Killed while rm -rf runs: the EXIT trap ran unlock again on the freed path.
    The signal goes to the process group, as Ctrl-C or a scheduler's kill does."""
    _git_repo(env, tmp_path)
    lock = _clone_lock(env)
    marker = tmp_path / "removing"
    _gone_rm_stub(tmp_path, lock, f'touch "{marker}"; "{shutil.which("sleep")}" 2')
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    proc = subprocess.Popen(["bash", str(_SCRIPT)], env=env, text=True, start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 15
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.exists(), proc.communicate(timeout=15)
    lock.mkdir()  # another build takes the freed lock
    (lock / "pid").write_text("otherhost:1\n")
    os.killpg(proc.pid, sig)
    proc.communicate(timeout=15)
    assert (lock / "pid").read_text() == "otherhost:1\n"


def test_a_failed_owner_write_fails_fast_every_time(env, tmp_path):
    """A claim that cannot record its owner must not leave its lock dir: every later
    run would wait on it for the whole timeout."""
    _git_repo(env, tmp_path)
    _stub(tmp_path / "bin", "mv",
          f'case "$*" in *.lock/pid.new*) exit 1 ;; esac\nexec "{shutil.which("mv")}" "$@"\n')
    env["FMTUNE_LOCK_TIMEOUT"] = "60"
    for _ in range(2):
        proc, took = _run_waiting(env, tmp_path, limit=15)
        assert took < 10, took
        assert proc.returncode == 2
        assert f"fm-tune: cannot write the owner of lock {_clone_lock(env)}" in proc.stderr
    assert not _clone_lock(env).exists()


def test_a_lock_dir_that_cannot_be_renamed_is_still_released(env, tmp_path):
    """Some FUSE mounts cannot rename a dir, and a leftover .gone dir of the same name
    blocks the rename too: the lock leaked, and with no takeover every later run died."""
    _git_repo(env, tmp_path)
    _stub(tmp_path / "bin", "mv",
          f'case "$*" in *.lock.gone.*) exit 1 ;; esac\nexec "{shutil.which("mv")}" "$@"\n')
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    env["FMTUNE_LOCK_TIMEOUT"] = "2"
    cache = pathlib.Path(env["FMTUNE_CACHE_DIR"])
    first = _run(env)
    assert first.returncode == 0, first.stderr
    assert not list(cache.glob("*.lock*"))
    second = _run(env)
    assert second.returncode == 0, second.stderr


def test_unlock_leaves_a_lock_that_is_no_longer_ours(env, tmp_path):
    """Someone removed our lock mid-section and another build took it: our unlock
    must not remove theirs. A stub git swaps the pid file for another owner's."""
    _git_repo(env, tmp_path)
    lock = _clone_lock(env)
    _stub(tmp_path / "bin", "git",
          f'[ -d "{lock}" ] && echo otherhost:1 > "{lock}/pid"\n'
          f'exec "{shutil.which("git")}" "$@"\n')
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert (lock / "pid").read_text() == "otherhost:1\n"


@pytest.mark.parametrize("timeout", ["9223372036854775807", "1234567890"])
def test_a_lock_timeout_too_big_for_shell_arithmetic_is_refused(env, timeout):
    """A value this long would wrap in $(( ))."""
    env["FMTUNE_LOCK_TIMEOUT"] = timeout
    proc = _run(env)
    assert proc.returncode == 2
    assert "fm-tune: FMTUNE_LOCK_TIMEOUT must be a whole number of seconds (at most 9 digits)" in proc.stderr
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


@pytest.mark.parametrize("timeout", ["02", "08"])
def test_a_lock_timeout_with_a_leading_zero_is_read_as_decimal(env, tmp_path, timeout):
    """$(( 08 )) is an octal error in bash, so the wait on a held lock never ended."""
    _git_repo(env, tmp_path)
    lock = _held_lock(env, "otherhost:1\n")
    env["FMTUNE_LOCK_TIMEOUT"] = timeout
    proc, took = _run_waiting(env, tmp_path, limit=30)
    assert int(timeout) - 1 <= took < int(timeout) + 10, took  # $SECONDS is whole seconds
    assert proc.returncode == 2
    assert f"fm-tune: lock {lock} is still held after {int(timeout)}s " in proc.stderr
    assert (lock / "pid").read_text() == "otherhost:1\n"


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
    env["FMTUNE_CACHE_DIR"] = "/usr"  # exists and is root's: nothing is ever written there
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
    proc = _run(env)
    assert proc.returncode == 2
    assert f"fm-tune: cache dir {link} is writable by other users" in proc.stderr


@pytest.mark.skipif(os.geteuid() == 0, reason="root owns the link")
def test_a_symlinked_cache_dir_owned_by_another_user_is_refused(env):
    """Its owner could re-point it at their own dir while a build runs from it."""
    link = next((p for p in ("/tmp", "/etc", "/var", "/bin", "/lib", "/sbin")
                 if os.path.islink(p) and os.path.isdir(p) and os.lstat(p).st_uid == 0), None)
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
    link = next((p for p in ("/tmp", "/etc", "/var", "/bin", "/lib", "/sbin")
                 if os.path.islink(p) and os.path.isdir(p) and os.lstat(p).st_uid == 0), None)
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
    tip = subprocess.run(["git", "-C", str(mirror), "cat-file", "-e", f"{_head(repo)}^{{commit}}"],
                         capture_output=True, check=False)
    assert tip.returncode != 0


def test_a_warm_sha_run_needs_no_git_and_no_remote(env, tmp_path):
    repo = _git_repo(env, tmp_path)
    env["FMTUNE_SOURCE_REF"] = sha = _head(repo)
    assert _run(env).returncode == 0
    pathlib.Path(env["STUB_RECORD"]).unlink()
    shutil.rmtree(repo)  # the remote is gone
    spy = _git_spy(env, tmp_path)
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
        proc = _run(env)
        assert proc.returncode == 0, proc.stderr
        assert f"@{ref} = {sha}" in proc.stdout
    trees = list(pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("tree-*"))
    assert [t.name for t in trees] == [f"tree-{sha}"]


def test_two_refs_extracting_one_commit_at_once_never_leave_an_empty_tree(env, tmp_path):
    """A branch and its SHA hold different locks but write the same tree-<sha>."""
    repo = _git_repo(env, tmp_path)
    sha = _head(repo)
    real_tar, gate = shutil.which("tar"), tmp_path / "gate"
    gate.mkdir()
    # The first extractor waits until the second has started, then both finish.
    _stub(tmp_path / "bin", "tar", f"""\
if mkdir "{gate}/first" 2>/dev/null; then
  i=0; while [ ! -e "{gate}/second" ] && [ $i -lt 100 ]; do sleep 0.1; i=$((i+1)); done
else
  : > "{gate}/second"
fi
exec "{real_tar}" "$@"
""")
    env["PATH"] = f"{tmp_path / 'bin'}:{env['PATH']}"
    procs = []
    for ref in ("rel", sha):
        run_env = dict(env, FMTUNE_SOURCE_REF=ref, STUB_RECORD=str(tmp_path / f"rec-{ref}"))
        procs.append(subprocess.Popen(
            ["bash", str(_SCRIPT)], env=run_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        ))
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
    assert [m.name for m in cache.glob("src-*")] == [_clone_lock(env).name[: -len(".lock")]]
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


@pytest.mark.parametrize("ref", [lambda sha: sha[:7], lambda sha: "no-such-branch"],
                         ids=["short-hex", "missing-branch"])
def test_a_failed_branch_clone_dies_naming_the_full_sha_option(env, tmp_path, ref):
    """A short hex value is not a SHA: it goes to clone --branch like any branch name."""
    repo = _git_repo(env, tmp_path)
    env["FMTUNE_SOURCE_REF"] = ref(_head(repo))
    proc = _run(env)
    assert proc.returncode == 2, proc.stderr
    assert f"fm-tune: could not clone {env['FMTUNE_SOURCE_REPO']} at {env['FMTUNE_SOURCE_REF']}" in proc.stderr
    assert "full 40-character commit SHA" in proc.stderr
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


@pytest.mark.parametrize("obj", ["v1", "HEAD^{tree}"], ids=["tag-object", "tree"])
def test_a_sha_that_is_not_a_commit_dies(env, tmp_path, obj):
    repo = _git_repo(env, tmp_path)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "tag", "-a", "v1", "-m", "v1", cwd=repo)
    env["FMTUNE_SOURCE_REF"] = subprocess.run(
        ["git", "rev-parse", obj], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    proc = _run(env)
    assert proc.returncode == 2, proc.stderr
    assert "is not a commit" in proc.stderr
    assert not pathlib.Path(env["STUB_RECORD"]).exists()
    assert not list(pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("tree-*"))


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


def _git_spy(env, tmp_path):
    """Put a git first on PATH that records each call and fails."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "git").write_text(f"#!/bin/sh\ntouch {tmp_path}/git-ran\nexit 1\n")
    (bin_dir / "git").chmod(0o755)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    return tmp_path / "git-ran"


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
    assert f"fm-tune: output dir {out} is not empty and is not a previous model output" in proc.stderr
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


# What a successful run leaves at the output dir's root after --model_at_root:
# config.json (HF full fine-tune, verl), adapter_config.json (PEFT, MLX LoRA),
# model.safetensors (MLX full fine-tune, which writes no config.json),
# adapters.safetensors (MLX LoRA, whose adapter_config.json write is best-effort).
@pytest.mark.parametrize(
    "marker", ["config.json", "adapter_config.json", "model.safetensors", "adapters.safetensors"]
)
def test_a_previous_model_output_is_accepted(env, marker):
    out = pathlib.Path(env["FMTUNE_OUTPUT_DIR"])
    out.mkdir()
    (out / marker).write_text("{}")
    (out / "tokenizer_config.json").write_text("{}")
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("which", ["FMTUNE_BASE_MODEL_DIR", "FMTUNE_DATASET_DIR"])
def test_an_output_dir_that_is_an_input_dies_before_any_clone(env, tmp_path, which):
    """A base model has config.json at its root, so it passes the model-output check;
    promoting into it would then delete the base weights (or the dataset)."""
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
    assert "fm-tune: output dir " in proc.stderr and "is the same directory as" in proc.stderr
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
    assert _flag(argv, "--output_dir") == f"{tmp_path}/out"
    assert _flag(argv, "--train_file") == f"{tmp_path}/data dir/d_train.jsonl"
    assert _flag(argv, "--validation_file") == f"{tmp_path}/data dir/d_validation.jsonl"
    assert _flag(argv, "--model_name_or_path") == f"{tmp_path}/bm"
    assert _flag(argv, "--config_file") == f"{tmp_path}/fm-tune/autotune/configs/autotune.yaml"
    assert rec["cwd"] == f"{tmp_path}/fm-tune"
    assert f"GB_ARTIFACT_ID:model GB_ARTIFACT_PATH:{tmp_path}/out" in proc.stdout.splitlines()


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


def test_a_trailing_slash_on_the_output_dir_is_stripped(env):
    out = env["FMTUNE_OUTPUT_DIR"]
    env["FMTUNE_OUTPUT_DIR"] = out + "/"
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert _flag(_record(env)["argv"], "--output_dir") == out
    assert f"GB_ARTIFACT_ID:model GB_ARTIFACT_PATH:{out}" in proc.stdout.splitlines()


def test_a_root_output_dir_is_not_stripped_to_empty_and_is_refused(env, tmp_path):
    _git_repo(env, tmp_path)
    git_ran = _git_spy(env, tmp_path)
    env["FMTUNE_OUTPUT_DIR"] = "/"
    proc = _run(env)
    assert proc.returncode != 0
    assert "fm-tune: output dir / is not empty and is not a previous model output" in proc.stderr
    assert not git_ran.exists()


# The monitors build the binding as '{ "path": "<marker path>" }' and json.loads it:
# a '"' or a control character fails the parse, a '\' is read as a JSON escape, and
# a newline ends the marker line early, registering a truncated path. The bash
# monitor's LocalFileStream rstrip()s each line, so a trailing space is lost too.
@pytest.mark.parametrize("bad", ['a"b', "a\\b", "a\nb", "a\tb", "a "])
def test_an_output_path_that_breaks_the_marker_json_dies_before_any_clone(env, tmp_path, bad):
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
    assert _run(env).returncode == 0, "second run"
    assert venvs[0].stat().st_mtime_ns == stamp
    assert not list(pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("*.lock"))
