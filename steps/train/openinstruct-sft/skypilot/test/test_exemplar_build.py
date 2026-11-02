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

"""The builds this step ships stay consistent with the step contract.

Checks every build.yaml (the exemplar a user copies and each cluster fixture)
against what the template's render-time validation would accept, so a fixture
cannot drift from the step without a fast test failing.
"""

import re
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parent / "test-data"
_TEMPLATE = yaml.safe_load((_HERE.parent / "step-template.yaml").read_text())
_PARAM_RE = re.compile(r"\$\$\{([A-Z0-9_]+)\}")
_FIXTURES = ["aws/build.yaml", "slurm-bluevela/build.yaml"]
BUILDS = ["build.yaml", *_FIXTURES]


def _parse_step_owned():
    """Return the finetune.py flags the template owns, from its Jinja `owned` list.

    The list appears twice (setup and run copies, pinned identical by another
    test); the first match is used.
    """
    text = (_HERE.parent / "step-template.yaml").read_text()
    matches = re.findall(r"\{%-?\s*set owned = \[(.*?)\]\s*-?%\}", text)
    assert matches, "no `{%- set owned = [...] -%}` line in step-template.yaml"
    return frozenset(re.findall(r'"([a-z_]+)"', matches[0]))


_STEP_OWNED = _parse_step_owned()


def _load(name):
    return yaml.safe_load((_DATA / name).read_text())


def _target(build):
    targets = build["granite.build"]["targets"]
    assert len(targets) == 1
    return next(iter(targets.values()))


def _step(target):
    assert len(target["steps"]) == 1
    return target["steps"][0]


@pytest.mark.parametrize("build", BUILDS)
def test_step_uri(build):
    assert (
        _step(_target(_load(build)))["step_uri"]
        == "space://steps/train/openinstruct-sft"
    )


@pytest.mark.parametrize("build", BUILDS)
def test_output_names_a_declared_model_output(build):
    target = _target(_load(build))
    cfg = _step(target)["config"]["openinstruct_sft_config"]
    output = cfg.get("output", _TEMPLATE["config"]["openinstruct_sft_config"]["output"])
    assert target["outputs"][output]["type"] == "model"


@pytest.mark.parametrize("build", BUILDS)
def test_exactly_one_data_source(build):
    target = _target(_load(build))
    mixer = (
        _step(target)["config"]["openinstruct_sft_config"].get("dataset_mixer_list")
        or {}
    )
    assert bool(mixer) != ("dataset" in target["inputs"])


@pytest.mark.parametrize("build", BUILDS)
def test_base_model_bound(build):
    assert "base_model" in _target(_load(build))["inputs"]


@pytest.mark.parametrize("build", BUILDS)
def test_gpus_requested_by_the_build(build):
    res = _step(_target(_load(build)))["config"]["launcher_config"]["resources"]
    assert str(res["accelerators"]).strip()


def test_step_owned_list_is_parsed():
    assert len(_STEP_OWNED) == 15, sorted(_STEP_OWNED)
    assert {"model_name_or_path", "resume_from_checkpoint"} <= _STEP_OWNED


@pytest.mark.parametrize("build", BUILDS)
def test_no_step_owned_args(build):
    args = _step(_target(_load(build)))["config"]["openinstruct_sft_config"].get(
        "args", {}
    )
    assert not _STEP_OWNED & set(args)


@pytest.mark.parametrize("build", BUILDS)
def test_single_node(build):
    compute = _step(_target(_load(build)))["config"].get("compute_config", {})
    assert compute.get("num_nodes", 1) == 1


def test_bluevela_fixture_matches_the_environment():
    repo = _HERE.parents[4]
    env = yaml.safe_load(
        (
            repo
            / "configurations/assets/environments/skypilot/slurm/ibm-bluevela/environment.yaml"
        ).read_text()
    )
    target = _target(_load("slurm-bluevela/build.yaml"))
    assert (
        target["environment_uri"] == "space://environments/skypilot/slurm/ibm-bluevela"
    )
    acc = _step(target)["config"]["launcher_config"]["resources"]["accelerators"]
    assert acc.split(":")[0].lower() == env["config"]["sbatch_options"]["constraint"]


def test_every_referenced_param_is_defined():
    referenced = set(_PARAM_RE.findall((_DATA / "build.yaml").read_text()))
    missing = referenced - set(_load("parameters.yaml"))
    assert not missing, sorted(missing)
