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

"""Unit tests for src/fmtune_run.sh: argv, HPO config, validation and run env.

Run against a stub fm-tune tree (see _fmtune_helpers.py): the stub ``main.py``
records its argv, so each test asserts the exact command line fm-tune would get.
The git source and its cache, the cache lock, and the cache/output paths have
files of their own. Mode 1 only.
"""

import json
import os
import pathlib

import pytest
import yaml
from _fmtune_helpers import (
    _clear_output,
    _flag,
    _git_repo,
    _git_spy,
    _record,
    _run,
    _staged_as,
)


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
    assert _staged_as(argv, env["FMTUNE_OUTPUT_DIR"])
    for f in ("--model_at_root", "--cleanup", "--save_history"):
        assert f in argv
    for absent in ("--no_autotune", "--rl_algo", "--autotunex_server_url", "--job_id"):
        assert absent not in argv
    markers = [
        ln for ln in proc.stdout.splitlines() if ln.startswith("GB_ARTIFACT_ID:")
    ]
    assert markers == [
        f"GB_ARTIFACT_ID:model GB_ARTIFACT_PATH:{env['FMTUNE_OUTPUT_DIR']}"
    ]


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
        (
            {"FMTUNE_SERVER_URL": "http://h:1", "FMTUNE_JOB_ID": "j1"},
            ["--autotunex_server_url", "http://h:1", "--job_id", "j1"],
        ),
        (
            {"FMTUNE_EXTRA_ARGS": "--data_backend ray_data"},
            ["--data_backend", "ray_data"],
        ),
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
    env.update(
        {
            "FMTUNE_SERVER_URL": "http://h:1",
            "FMTUNE_JOB_ID": "j1",
            "GB_BUILD_ID": "b-123",
        }
    )
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


def test_inline_hpo_config_is_private_to_the_run_and_removed(env, tmp_path):
    """Builds whose output dirs share a parent must not share the config file."""
    env["FMTUNE_HPO_CONFIG_JSON"] = json.dumps({"tune_config": {}})
    assert _run(env).returncode == 0
    cfg = pathlib.Path(_flag(_record(env)["argv"], "--config_file"))
    assert cfg.parent != tmp_path
    assert not cfg.parent.exists()


@pytest.mark.parametrize(
    "over",
    [
        {"FMTUNE_HPO_CONFIG_JSON": "{not json"},  # the YAML conversion fails
        {"STUB_EXIT": "3"},  # main.py fails
        {"FMTUNE_SKIP_INSTALL": "", "FMTUNE_PYTHON": "/nonexistent/python3"},  # venv
    ],
    ids=["bad-json", "failed-training", "no-python"],
)
def test_a_failed_run_still_removes_its_hpo_config_dir(env, tmp_path, over):
    """The temp dir goes on every exit, not only after main.py: set -e and die exit
    between its mktemp and the run."""
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    env.update(TMPDIR=str(tmp), FMTUNE_HPO_CONFIG_JSON=json.dumps({"tune_config": {}}))
    env.update(over)
    assert _run(env).returncode != 0
    assert not list(tmp.iterdir())


def test_inline_hpo_config_file_defaults_to_fm_tune_hpo(env):
    env["FMTUNE_HPO_CONFIG_JSON"] = json.dumps({"tune_config": {}})
    assert _run(env).returncode == 0
    assert _flag(_record(env)["argv"], "--config_file").endswith("/fm_tune_hpo.yaml")


def test_config_name_names_the_inline_hpo_config_file(env):
    env["FMTUNE_HPO_CONFIG_JSON"] = json.dumps({"tune_config": {}})
    env["FMTUNE_CONFIG_NAME"] = "lora-sweep_ab12cd34"
    assert _run(env).returncode == 0
    assert _flag(_record(env)["argv"], "--config_file").endswith(
        "/lora-sweep_ab12cd34.yaml"
    )


def test_config_name_does_not_rename_a_preset(env):
    env["FMTUNE_HPO_CONFIG_JSON"] = "{}"
    env["FMTUNE_CONFIG_NAME"] = "lora-sweep"
    assert _run(env).returncode == 0
    cfg = _flag(_record(env)["argv"], "--config_file")
    assert cfg == f"{env['FMTUNE_SOURCE_PATH']}/autotune/configs/autotune.yaml"


@pytest.mark.parametrize("name", ["a/b", "a b", "../x"])
def test_config_name_outside_the_safe_set_is_rejected(env, name):
    env["FMTUNE_CONFIG_NAME"] = name
    proc = _run(env)
    assert proc.returncode != 0
    assert "fm-tune: config_name must match" in proc.stderr
    assert not pathlib.Path(env["STUB_RECORD"]).exists()


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
        (
            {"FMTUNE_SOURCE_REPO": "https://x/y.git"},
            "set exactly one of source.path and source.repo",
        ),
        ({"FMTUNE_TRAIN_FILE": ""}, "train_file is required"),
        ({"FMTUNE_TRAIN_FILE": "nope.jsonl"}, "train_file not found"),
        ({"FMTUNE_VALIDATION_FILE": "nope.jsonl"}, "validation_file not found"),
        ({"FMTUNE_CONFIG_PRESET": "missing.yaml"}, "config_preset not found"),
        ({"FMTUNE_DATASET_DIR": ""}, "FMTUNE_DATASET_DIR is empty"),
        (
            {"FMTUNE_TRAIN_FILE": "../x"},
            "train_file must be a relative path inside the dataset",
        ),
        (
            {"FMTUNE_TRAIN_FILE": ".."},
            "train_file must be a relative path inside the dataset",
        ),
        (
            {"FMTUNE_TRAIN_FILE": "a/../../x"},
            "train_file must be a relative path inside the dataset",
        ),
        (
            {"FMTUNE_TRAIN_FILE": "/etc/passwd"},
            "train_file must be a relative path inside the dataset",
        ),
        (
            {"FMTUNE_VALIDATION_FILE": "../x"},
            "validation_file must be a relative path inside the dataset",
        ),
        (
            {"FMTUNE_VALIDATION_FILE": "/abs/v.jsonl"},
            "validation_file must be a relative path inside the dataset",
        ),
        (
            {"FMTUNE_TUNING_ALGO": ""},
            "tuning_algo and rl_algo cannot both be empty or none",
        ),
        (
            {"FMTUNE_TUNING_ALGO": "none", "FMTUNE_RL_ALGO": "None"},
            "tuning_algo and rl_algo cannot both be empty or none",
        ),
        (
            {"FMTUNE_TUNING_ALGO": "", "FMTUNE_RL_ALGO": "dpo"},
            "an offline rl_algo (dpo, kto) needs a tuning_algo",
        ),
        (
            {"FMTUNE_TUNING_ALGO": "None", "FMTUNE_RL_ALGO": "Kto"},
            "an offline rl_algo (dpo, kto) needs a tuning_algo",
        ),
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
    assert (
        _flag(_record(env)["argv"], "--train_file")
        == f"{env['FMTUNE_DATASET_DIR']}/{name}"
    )


def test_a_split_file_outside_the_dataset_dies_before_any_clone(env, tmp_path):
    _git_repo(env, tmp_path)
    git_ran = _git_spy(env, tmp_path)
    env["FMTUNE_VALIDATION_FILE"] = "../../outside.jsonl"
    proc = _run(env)
    assert proc.returncode != 0
    assert (
        "fm-tune: validation_file must be a relative path inside the dataset"
        in proc.stderr
    )
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


def test_the_interpreters_dir_is_first_on_path(env):
    assert _run(env).returncode == 0
    assert _record(env)["path"].split(":")[0] == os.path.dirname(env["FMTUNE_PYTHON"])


def test_multi_line_extra_args_keep_every_word(env):
    env["FMTUNE_EXTRA_ARGS"] = "--data_backend ray_data\n--foo bar\n"
    assert _run(env).returncode == 0
    argv = _record(env)["argv"]
    assert argv[-4:] == ["--data_backend", "ray_data", "--foo", "bar"]


def test_whitespace_only_extra_args_add_nothing(env):
    def argv_without_staging():  # the staging dir's name is random per run
        argv = _record(env)["argv"]
        del argv[argv.index("--output_dir") + 1]
        return argv

    assert _run(env).returncode == 0
    plain = argv_without_staging()
    env["FMTUNE_EXTRA_ARGS"] = "  \n "
    _clear_output(env)
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert argv_without_staging() == plain
