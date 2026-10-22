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

Renders ``run:`` with gbserver's strict renderer, then runs it with a stub
src/fmtune_run.sh that dumps its FMTUNE_* environment, so the test checks what the
script actually receives after bash unquotes it. Mode 1 only.
"""

import json
import pathlib
import shutil
import subprocess

import pytest
import yaml

from gbcommon.utils.template import fill_template

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_TEMPLATE = yaml.safe_load((_STEP_DIR / "step-template.yaml").read_text())
_SKY = _TEMPLATE["environment_configs"]["Skypilot"]
_LAUNCHER = _SKY["launchers"]["fm-tune"]["config"]
_DATASET = {"dataset": {"binding": {"path": "/staged/data it's"}}}
_STUB = (
    "#!/usr/bin/env bash\n"
    "python3 -c 'import json,os; print(json.dumps({k:v for k,v in os.environ.items() "
    "if k.startswith(\"FMTUNE_\")}))'\n"
)


def _cfg(over: dict | None = None) -> dict:
    cfg = json.loads(json.dumps(_TEMPLATE["config"]))
    cfg["fm_tune_config"].update(over or {})
    return cfg


def _render(over: dict | None = None, bindings: dict | None = None) -> str:
    ctx = {"config": _cfg(over), "bindings": _DATASET if bindings is None else bindings}
    return fill_template(_LAUNCHER["run"], ctx, strict=True)


def _received(tmp_path, over=None, bindings=None) -> dict:
    if shutil.which("bash") is None:  # pragma: no cover
        pytest.skip("bash not available")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "fmtune_run.sh").write_text(_STUB)
    proc = subprocess.run(
        ["bash", "-c", "set -eu\n" + _render(over, bindings)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_run_is_valid_bash():
    assert subprocess.run(["bash", "-n"], input=_render(), text=True).returncode == 0


def test_defaults_reach_the_script(tmp_path):
    env = _received(tmp_path)
    assert env["FMTUNE_DATASET_DIR"] == "/staged/data it's"
    assert env["FMTUNE_BASE_MODEL_DIR"] == ""
    assert env["FMTUNE_TUNING_ALGO"] == "lora"
    assert env["FMTUNE_HPO_CONFIG_JSON"] == "{}"
    assert env["FMTUNE_OUTPUT_DIR"] == f"{tmp_path}/model"
    assert env["FMTUNE_CACHE_DIR"] == f"{tmp_path}/.fm-tune-cache"
    assert env["FMTUNE_PYTHON"] == "python3"


def test_quotes_and_json_survive(tmp_path):
    hpo = {"a": "it's \"q\" $HOME `x`"}
    env = _received(tmp_path, {"hpo_config": hpo, "run_name": "a b'c", "extra_args": "--x 'y z'"})
    assert json.loads(env["FMTUNE_HPO_CONFIG_JSON"]) == hpo
    assert env["FMTUNE_RUN_NAME"] == "a b'c"
    assert env["FMTUNE_EXTRA_ARGS"] == "--x 'y z'"


def test_output_path_overrides_the_workdir_default(tmp_path):
    assert _received(tmp_path, {"output_path": "/shared/out"})["FMTUNE_OUTPUT_DIR"] == "/shared/out"


def test_bound_base_model_reaches_the_script(tmp_path):
    b = dict(_DATASET, base_model={"binding": {"path": "/staged/bm"}})
    assert _received(tmp_path, bindings=b)["FMTUNE_BASE_MODEL_DIR"] == "/staged/bm"


def test_slurm_only_bare_node_and_ships_src():
    assert _SKY["subtypes"] == ["slurm"]
    assert _LAUNCHER["image_id"] == ""
    assert _LAUNCHER["file_mounts"] == {"src": "src"}
