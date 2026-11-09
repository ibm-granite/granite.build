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

"""Unit tests for the multi-challenge local-scoring recipes.

build-local.yaml and score-local.yaml run their workloads as `command` steps, so the
command is a `>-` folded scalar holding `<% if %>` blocks for the optional flags. A
line made only of those blocks renders empty when every flag is unset, and YAML then
folds that empty line into a literal newline: `bash -c` runs the command in two parts,
the first without the judge options. Only the full-benchmark default hits that case,
so every flag combination is rendered here.
"""

import itertools
import pathlib

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_RECIPE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "recipes"
    / "evals"
    / "judged"
    / "multi-challenge"
    / "skypilot"
)
_REQUIRED = [
    "JUDGE_BASE_URL=https://judge.example/v1",
    "GENERATION_PATH=/tmp/run/multi-challenge/generation.json",
]
_OPTIONAL = [
    ("LIMIT=5", "--limit 5"),
    ("REPEATS=2", "--repeats 2"),
    ("SERVED_MODEL_NAME=served", '--served-model-name "served"'),
    ("JUDGE_MODEL=judge", "judge_model=judge"),
    ("JUDGE_TAG=tag", "judge_tag=tag"),
]
_COMBOS = list(itertools.product([False, True], repeat=len(_OPTIONAL)))


def _commands(build_file, params, tmp_path):
    contents = (_RECIPE / build_file).read_text(encoding="utf-8")
    defaults = get_params_from_file(str(_RECIPE / "parameters.yaml"))
    rendered = yaml.safe_load(
        apply_parameters(contents, params, defaults, str(tmp_path))
    )
    return {
        name: target["steps"][0]["config"]["command_config"]["command"]
        for name, target in rendered["granite.build"]["targets"].items()
        if "command_config" in target["steps"][0]["config"]
    }


@pytest.mark.parametrize("build_file", ["build-local.yaml", "score-local.yaml"])
@pytest.mark.parametrize(
    "combo", _COMBOS, ids=lambda c: "".join("1" if x else "0" for x in c)
)
def test_commands_render_to_one_line(build_file, combo, tmp_path):
    params = _REQUIRED + [p for (p, _), on in zip(_OPTIONAL, combo) if on]
    for name, command in _commands(build_file, params, tmp_path).items():
        assert (
            "\n" not in command
        ), f"{build_file}:{name} renders a newline: {command!r}"


def test_full_run_scores_with_the_judge_options(tmp_path):
    command = _commands(
        "build-local.yaml", _REQUIRED + ["JUDGE_MODEL=judge"], tmp_path
    )["multi-challenge"]
    run = command[command.index("granite-evals run") :]
    assert "--limit" not in run and "--repeats" not in run
    assert '--workers 16 "${OPTS[@]}"' in run
    assert "judge_base_url=https://judge.example/v1 judge_model=judge" in command


@pytest.mark.parametrize("flag,expected", _OPTIONAL[:3])
def test_optional_flags_reach_the_score_command(flag, expected, tmp_path):
    command = _commands("build-local.yaml", _REQUIRED + [flag], tmp_path)[
        "multi-challenge"
    ]
    assert expected in command


def test_build_yaml_has_no_local_targets(tmp_path):
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    defaults = get_params_from_file(str(_RECIPE / "parameters.yaml"))
    rendered = yaml.safe_load(apply_parameters(contents, [], defaults, str(tmp_path)))
    assert set(rendered["granite.build"]["targets"]) == {
        "multi-challenge-generate",
        "multi-challenge",
    }
