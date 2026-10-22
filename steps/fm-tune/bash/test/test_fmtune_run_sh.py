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

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "src" / "fmtune_run.sh"

_STUB_MAIN = r"""
import json, os, sys
args = sys.argv[1:]
rec = {"argv": args, "cwd": os.getcwd()}
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
    env["FMTUNE_HPO_CONFIG_JSON"] = json.dumps({"tune_config": {"num_samples": {"default": 2}}})
    env["STUB_CONFIG_COPY"] = "1"
    assert _run(env).returncode == 0
    rec = _record(env)
    assert json.loads(rec["config_text"]) == {"tune_config": {"num_samples": {"default": 2}}}
    assert not _flag(rec["argv"], "--config_file").startswith(env["FMTUNE_SOURCE_PATH"])


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
    env["FMTUNE_RUN_NAME"] = "GB_ARTIFACT_ID:evil"
    env["STUB_EXIT"] = "1"
    proc = _run(env)
    assert not any(ln.startswith("GB_ARTIFACT_") for ln in proc.stdout.splitlines())


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_git_source_is_cloned_and_cached(env, tmp_path):
    """source.repo clones once at ref and runs <clone>/<subdir>/main.py."""
    _git_repo(env, tmp_path)
    first = _run(env)
    assert first.returncode == 0, first.stderr
    assert first.stdout.count("[fm-tune] cloning") == 1
    assert _record(env)["cwd"].endswith("/sub/fm-tune")
    clones = list(pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("src-*"))
    assert len(clones) == 1
    second = _run(env)  # second run reuses the clone
    assert second.returncode == 0, second.stderr
    assert second.stdout.count("[fm-tune] cloning") == 0
    assert list(pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("src-*")) == clones


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
    # src-<cksum> only: the [0-9] tail excludes src-<key>.lock and src-<key>.tmp.
    (clone,) = pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("src-*[0-9]")
    assert _head(clone) == _head(repo)
    assert (clone / "sub" / "fm-tune" / "NEW").is_file()


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


def test_stale_clone_lock_from_a_dead_pid_is_taken_over(env, tmp_path):
    _git_repo(env, tmp_path)
    key = subprocess.run(
        ["bash", "-c", 'printf "%s@%s" "$0" "$1" | cksum | cut -d" " -f1',
         env["FMTUNE_SOURCE_REPO"], env["FMTUNE_SOURCE_REF"]],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dead = subprocess.Popen(["true"])
    dead.wait()
    lock = pathlib.Path(env["FMTUNE_CACHE_DIR"]) / f"src-{key}.lock"
    lock.mkdir(parents=True)
    (lock / "pid").write_text(f"{dead.pid}\n")
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert "taking over stale" in proc.stdout
    assert not list(pathlib.Path(env["FMTUNE_CACHE_DIR"]).glob("*.lock"))


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
