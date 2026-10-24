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

"""Render test for the skypilot fm-tune step's run block.

Renders ``run:`` with gbserver's strict renderer,
``gbserver.utils.template.fill_template(..., strict=True)`` (the launcher-config
fill in ``targetsteprun.py``: StrictUndefined and the ``raise_error`` global),
then runs it with a stub
src/fmtune_run.sh that dumps its FMTUNE_* environment, so the test checks what the
script actually receives after bash unquotes it. Mode 1 only.
"""

import json
import os
import pathlib
import shutil
import subprocess

import pytest
import yaml

from gbserver.utils.template import fill_objtemplate, fill_template

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_TEMPLATE = yaml.safe_load((_STEP_DIR / "step-template.yaml").read_text())
_SKY = _TEMPLATE["environment_configs"]["Skypilot"]
_LAUNCHER = _SKY["launchers"]["fm-tune"]["config"]
_DATASET = {"dataset": {"binding": {"path": "/staged/data it's"}}}
_STUB = (
    "#!/usr/bin/env bash\n"
    "python3 -c 'import json,os; print(json.dumps({k:v for k,v in os.environ.items() "
    'if k.startswith("FMTUNE_")}))\'\n'
)


# The template defaults leave these empty, which the render-time checks reject.
_VALID = {
    "model": "org/tiny",
    "train_file": "d_train.jsonl",
    "validation_file": "d_validation.jsonl",
    "source": {
        "path": "",
        "repo": "https://g/r.git",
        "ref": "main",
        "subdir": "autotunex/src/fm-tune",
    },
}


def _cfg(
    over: dict | None = None, compute: dict | None = None, launcher: dict | None = None
) -> dict:
    cfg = json.loads(json.dumps(_TEMPLATE["config"]))
    cfg["fm_tune_config"].update(_VALID)
    cfg["fm_tune_config"].update(over or {})
    if compute is not None:
        cfg["compute_config"] = compute
    if launcher is not None:  # the build's config.launcher_config
        cfg["launcher_config"] = launcher
    return cfg


def _render(over: dict | None = None, bindings: dict | None = None) -> str:
    ctx = {"config": _cfg(over), "bindings": _DATASET if bindings is None else bindings}
    return fill_template(_LAUNCHER["run"], ctx, strict=True)


def _fill(over=None, bindings=None, compute=None, launcher=None) -> dict:
    """Fill the whole launcher config, as targetsteprun.py does at step creation."""
    ctx = {
        "config": _cfg(over, compute, launcher),
        "bindings": _DATASET if bindings is None else bindings,
    }
    return fill_objtemplate(_LAUNCHER, ctx, strict=True)


def _render_error(over=None, bindings=None, compute=None, launcher=None) -> str:
    """The raise_error message a failing fill carries (fill_template wraps it)."""
    with pytest.raises(RuntimeError) as exc:
        _fill(over, bindings, compute, launcher)
    err = exc.value
    while err.__cause__ is not None:
        err = err.__cause__
    assert isinstance(err, ValueError), repr(err)
    return str(err)


def _received(tmp_path, over=None, bindings=None) -> dict:
    if shutil.which("bash") is None:  # pragma: no cover
        pytest.skip("bash not available")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "fmtune_run.sh").write_text(_STUB)
    proc = subprocess.run(
        ["bash", "-c", "set -eu\n" + _render(over, bindings)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_run_is_valid_bash():
    assert subprocess.run(["bash", "-n"], input=_render(), text=True).returncode == 0


def test_an_undefined_value_fails_the_render():
    """The strict fill raises on an undefined variable instead of rendering ''."""
    cfg = _cfg()
    del cfg["fm_tune_config"]["model"]
    with pytest.raises(RuntimeError):
        fill_template(
            _LAUNCHER["run"], {"config": cfg, "bindings": _DATASET}, strict=True
        )


def test_defaults_reach_the_script(tmp_path):
    env = _received(tmp_path)
    assert env["FMTUNE_DATASET_DIR"] == "/staged/data it's"
    assert env["FMTUNE_BASE_MODEL_DIR"] == ""
    assert env["FMTUNE_TUNING_ALGO"] == "lora"
    assert env["FMTUNE_HPO_CONFIG_JSON"] == "{}"
    assert env["FMTUNE_CONFIG_NAME"] == ""
    assert env["FMTUNE_OUTPUT_DIR"] == f"{tmp_path}/model"
    # Per user, with no shared level: a shared GB_SHARED_WORKDIR is written by every
    # user's builds, and a shared parent would be made by the first user, at their umask.
    assert env["FMTUNE_CACHE_DIR"] == f"{tmp_path}/.fm-tune-cache-{os.getuid()}"
    assert env["FMTUNE_PYTHON"] == "python3"


def test_quotes_and_json_survive(tmp_path):
    hpo = {"a": 'it\'s "q" $HOME `x`'}
    # model, not run_name: run_name's charset check now rejects a quote at render time.
    env = _received(
        tmp_path, {"hpo_config": hpo, "model": "a b'c", "extra_args": "--x 'y z'"}
    )
    assert json.loads(env["FMTUNE_HPO_CONFIG_JSON"]) == hpo
    assert env["FMTUNE_MODEL"] == "a b'c"
    assert env["FMTUNE_EXTRA_ARGS"] == "--x 'y z'"


@pytest.mark.parametrize(
    "name", ["train.jsonl", "sub/dir/train.jsonl", "a..b.jsonl", "./t.jsonl"]
)
def test_a_split_file_inside_the_dataset_renders(tmp_path, name):
    env = _received(tmp_path, {"train_file": name, "validation_file": name})
    assert env["FMTUNE_TRAIN_FILE"] == env["FMTUNE_VALIDATION_FILE"] == name


def test_a_bound_base_model_satisfies_the_model_check(tmp_path):
    b = dict(_DATASET, base_model={"binding": {"path": "/staged/bm"}})
    assert (
        _received(tmp_path, {"model": ""}, bindings=b)["FMTUNE_BASE_MODEL_DIR"]
        == "/staged/bm"
    )


@pytest.mark.parametrize("nodes", [2, "2", 0])
def test_more_than_one_node_fails_the_fill(nodes):
    msg = _render_error(compute={"num_nodes": nodes})
    assert msg.startswith("fm-tune: compute_config.num_nodes must be 1"), msg


@pytest.mark.parametrize(
    "compute",
    [
        None,
        {},
        {"num_cpus_per_node": 2},
        {"num_nodes": 1},
        {"num_nodes": "1"},
        {"num_nodes": None},
    ],
)
def test_one_node_or_no_num_nodes_renders(compute):
    """The template config has no compute_config; a build may set none at all."""
    assert "bash ./src/fmtune_run.sh" in _fill(compute=compute)["run"]


_VAR = {
    "seed": "FMTUNE_SEED",
    "rl_algo": "FMTUNE_RL_ALGO",
    "tuning_algo": "FMTUNE_TUNING_ALGO",
    "backend": "FMTUNE_BACKEND",
    "extra_args": "FMTUNE_EXTRA_ARGS",
}


@pytest.mark.parametrize(
    "key", ["seed", "rl_algo", "tuning_algo", "backend", "extra_args"]
)
def test_a_null_value_renders_empty(key, tmp_path):
    """AutoTuneX's LaunchContext can emit null; it must not reach the script as 'None'."""
    # tuning_algo alone null would trip the "both empty or none" check.
    over = {key: None, "rl_algo": "grpo"} if key == "tuning_algo" else {key: None}
    assert _received(tmp_path, over)[_VAR[key]] == ""


def test_null_hpo_config_and_autotune_render_empty(tmp_path):
    env = _received(tmp_path, {"hpo_config": None, "autotune": None})
    # Empty: the script then uses the preset and its default autotune=true.
    assert env["FMTUNE_HPO_CONFIG_JSON"] == env["FMTUNE_AUTOTUNE"] == ""


def test_zero_and_false_are_not_treated_as_null(tmp_path):
    env = _received(tmp_path, {"seed": 0, "autotune": False})
    assert (env["FMTUNE_SEED"], env["FMTUNE_AUTOTUNE"]) == ("0", "false")


def test_null_python_and_output_path_fall_back_to_their_defaults(tmp_path):
    env = _received(tmp_path, {"python": None, "output_path": None})
    assert env["FMTUNE_PYTHON"] == "python3"
    assert env["FMTUNE_OUTPUT_DIR"] == f"{tmp_path}/model"


@pytest.mark.parametrize("gpus", [1, 8, "2"])
@pytest.mark.parametrize(
    "launcher",
    [
        None,
        {},
        {"envs": {"A": "1"}},
        {"resources": {}},
        {"resources": {"accelerators": ""}},
    ],
)
def test_gpus_without_an_accelerators_request_fail_the_fill(gpus, launcher):
    """gbserver does not request GPUs from compute_config (#536); a GPU count
    alone would provision a CPU-only node, so the build fails instead."""
    msg = _render_error(
        compute={"num_gpus_per_node": gpus, "gpu_model": "H100"}, launcher=launcher
    )
    assert msg.startswith(f"fm-tune: compute_config.num_gpus_per_node is {gpus}"), msg
    assert 'config.launcher_config.resources.accelerators, e.g. "H100:' in msg, msg


def test_the_gpu_error_without_a_gpu_model_names_a_placeholder():
    msg = _render_error(compute={"num_gpus_per_node": 2})
    assert 'e.g. "<gpu_model>:2"' in msg, msg


@pytest.mark.parametrize("acc", ["H100:1", ":2"])
def test_an_explicit_accelerators_request_satisfies_the_gpu_check(acc):
    filled = _fill(
        compute={"num_gpus_per_node": 1}, launcher={"resources": {"accelerators": acc}}
    )
    # The template never writes accelerators itself: gbserver merges the build's
    # config.launcher_config.resources over this {} (skypilot.py, override_res).
    assert filled["resources"] == {}
    assert "bash ./src/fmtune_run.sh" in filled["run"]


@pytest.mark.parametrize(
    "compute",
    [
        None,
        {},
        {"num_gpus_per_node": 0},
        {"num_gpus_per_node": None},
        {"num_cpus_per_node": 2, "num_gpus_per_node": 0},
    ],
)
def test_cpu_only_renders_without_accelerators(compute):
    """The slurm fixture's shape (cpus=2, gpus=0) and no compute_config at all."""
    filled = _fill(compute=compute)
    assert filled["resources"] == {}


@pytest.mark.parametrize(
    "compute, launcher",
    [
        (
            {"num_nodes": 1},
            {"num_nodes": 2},
        ),  # launcher_config wins in _num_nodes_from_configs
        (None, {"num_nodes": 2}),
        ({}, {"num_nodes": "3"}),
    ],
)
def test_a_launcher_config_num_nodes_other_than_1_fails_the_fill(compute, launcher):
    """skypilot.py _num_nodes_from_configs reads compute_config, then the build's
    config.launcher_config.num_nodes (last wins), so both are checked."""
    msg = _render_error(compute=compute, launcher=launcher)
    assert msg.startswith("fm-tune: launcher_config.num_nodes must be 1"), msg


@pytest.mark.parametrize(
    "compute, launcher",
    [
        ({"num_nodes": 1}, {"num_nodes": 1}),
        ({"num_nodes": 1.0}, None),
        ({}, {"num_nodes": "1"}),
        ({"num_nodes": None}, {"num_nodes": None}),
    ],
)
def test_one_node_from_every_source_renders(compute, launcher):
    assert (
        "bash ./src/fmtune_run.sh" in _fill(compute=compute, launcher=launcher)["run"]
    )


@pytest.mark.parametrize("bad", ["abc", "1.5", 1.5, -1, " ", True, [1]])
def test_a_non_numeric_gpu_count_fails_the_fill(bad):
    msg = _render_error(compute={"num_gpus_per_node": bad})
    assert msg.startswith(
        "fm-tune: compute_config.num_gpus_per_node must be a whole number >= 0"
    ), msg


@pytest.mark.parametrize(
    "bad", ["abc", "1.5", 1.5, True, [1], ""]
)  # "": gbserver int("") raises
@pytest.mark.parametrize("where", ["compute_config", "launcher_config"])
def test_a_non_numeric_node_count_fails_the_fill(bad, where):
    kw = (
        {"compute": {"num_nodes": bad}}
        if where == "compute_config"
        else {"launcher": {"num_nodes": bad}}
    )
    msg = _render_error(**kw)
    assert msg.startswith(f"fm-tune: {where}.num_nodes must be a whole number"), msg


@pytest.mark.parametrize("gpus", ["", None, 0, "0", 0.0])
def test_an_empty_or_zero_gpu_count_is_cpu_only(gpus):
    assert _fill(compute={"num_gpus_per_node": gpus})["resources"] == {}


@pytest.mark.parametrize("where", ["compute_config", "launcher_config"])
@pytest.mark.parametrize("bad", ["x", ["a"], 3])
def test_a_compute_or_launcher_config_that_is_not_a_mapping_fails_with_a_named_error(
    where, bad
):
    kw = {"compute": bad} if where == "compute_config" else {"launcher": bad}
    msg = _render_error(**kw)
    assert msg.startswith(f"fm-tune: config.{where} must be a mapping"), msg


@pytest.mark.parametrize("where", ["compute_config", "launcher_config"])
def test_a_null_compute_or_launcher_config_is_unset(where):
    cfg = _cfg()
    cfg[where] = None
    filled = fill_objtemplate(
        _LAUNCHER, {"config": cfg, "bindings": _DATASET}, strict=True
    )
    assert "bash ./src/fmtune_run.sh" in filled["run"]


def test_cache_dir_is_per_user_under_the_shared_workdir():
    assert (
        'FMTUNE_CACHE_DIR="${GB_SHARED_WORKDIR:-$PWD}/.fm-tune-cache-$(id -u)"'
        in _render()
    )


def test_output_path_overrides_the_workdir_default(tmp_path):
    assert (
        _received(tmp_path, {"output_path": "/shared/out"})["FMTUNE_OUTPUT_DIR"]
        == "/shared/out"
    )


def test_bound_base_model_reaches_the_script(tmp_path):
    b = dict(_DATASET, base_model={"binding": {"path": "/staged/bm"}})
    assert _received(tmp_path, bindings=b)["FMTUNE_BASE_MODEL_DIR"] == "/staged/bm"


def test_slurm_only_bare_node_and_ships_src():
    assert _SKY["subtypes"] == ["slurm"]
    assert _LAUNCHER["image_id"] == ""
    assert _LAUNCHER["file_mounts"] == {"src": "src"}
