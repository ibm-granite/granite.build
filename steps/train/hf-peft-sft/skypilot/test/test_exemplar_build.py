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

"""Every build.yaml the step ships stays inside the step contract.

The exemplar a user copies and each cluster fixture are checked against what
the template's render-time validation accepts, so a fixture cannot drift from
the step without a fast failure. Multi-target fixtures are checked per target.
"""

import re
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parent / "test-data"
_TEMPLATE_TEXT = (_HERE.parent / "step-template.yaml").read_text()
_DEFAULTS = yaml.safe_load(_TEMPLATE_TEXT)["config"]["hf_peft_sft_config"]
_PARAM_RE = re.compile(r"\$\$\{([A-Z0-9_]+)\}")
BUILDS = ["build.yaml", "aws/build.yaml", "slurm-bluevela/build.yaml"]
_OWNED = frozenset(
    re.findall(
        r'"([a-z_0-9]+)"', re.findall(r"set owned = \[(.*?)\]", _TEMPLATE_TEXT)[0]
    )
)


def _targets(name):
    build = yaml.safe_load((_DATA / name).read_text())
    return list(build["granite.build"]["targets"].values())


def _cases():
    return [(b, i) for b in BUILDS for i in range(len(_targets(b)))]


def _target(build, i):
    return _targets(build)[i]


def _step(target):
    assert len(target["steps"]) == 1
    return target["steps"][0]


def _cfg(target):
    return {**_DEFAULTS, **_step(target)["config"]["hf_peft_sft_config"]}


def _count(acc):
    parts = str(acc).split(":")
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1


@pytest.mark.parametrize("build,i", _cases())
def test_step_uri(build, i):
    assert _step(_target(build, i))["step_uri"] == "space://steps/train/hf-peft-sft"


@pytest.mark.parametrize("build,i", _cases())
def test_output_names_a_declared_model_output(build, i):
    target = _target(build, i)
    assert target["outputs"][_cfg(target)["output"]]["type"] == "model"


@pytest.mark.parametrize("build,i", _cases())
def test_exactly_one_data_source(build, i):
    target = _target(build, i)
    assert bool(_cfg(target)["dataset_name"]) != ("dataset" in target["inputs"])


@pytest.mark.parametrize("build,i", _cases())
def test_base_model_bound(build, i):
    assert "base_model" in _target(build, i)["inputs"]


@pytest.mark.parametrize("build,i", _cases())
def test_gpus_requested_and_enough_for_sharding(build, i):
    target = _target(build, i)
    acc = _step(target)["config"]["launcher_config"]["resources"]["accelerators"]
    assert str(acc).strip()
    if _cfg(target)["distributed"] != "ddp" and "$${" not in str(acc):
        assert _count(acc) >= 2, acc


@pytest.mark.parametrize("build,i", _cases())
def test_alora_iff_invocation(build, i):
    cfg = _cfg(_target(build, i))
    assert (cfg["method"] == "alora") == bool(cfg["alora_invocation_string"])


@pytest.mark.parametrize("build,i", _cases())
def test_no_step_owned_args(build, i):
    args = {k.replace("-", "_") for k in _cfg(_target(build, i)).get("args", {})}
    assert not _OWNED & args


@pytest.mark.parametrize("build,i", _cases())
def test_single_node(build, i):
    assert (
        _step(_target(build, i))["config"].get("compute_config", {}).get("num_nodes", 1)
        == 1
    )


@pytest.mark.parametrize("build,i", _cases())
def test_fixture_records_a_portable_base(build, i):
    assert _cfg(_target(build, i))["base_model_id"]


def test_fixtures_cover_every_method_and_sharding_mode():
    seen = {
        (_cfg(t)["method"], _cfg(t)["distributed"])
        for b in BUILDS[1:]
        for t in _targets(b)
    }
    assert seen == {
        ("qlora", "ddp"),
        ("alora", "ddp"),
        ("lora", "fsdp"),
        ("qlora", "deepspeed"),
    }


def test_bluevela_fixture_matches_the_environment():
    repo = _HERE.parents[4]
    env = yaml.safe_load(
        (
            repo
            / "configurations/assets/environments/skypilot/slurm/ibm-bluevela/environment.yaml"
        ).read_text()
    )
    for target in _targets("slurm-bluevela/build.yaml"):
        assert (
            target["environment_uri"]
            == "space://environments/skypilot/slurm/ibm-bluevela"
        )
        acc = _step(target)["config"]["launcher_config"]["resources"]["accelerators"]
        assert (
            acc.split(":")[0].lower() == env["config"]["sbatch_options"]["constraint"]
        )


def test_every_referenced_param_is_defined():
    referenced = set(_PARAM_RE.findall((_DATA / "build.yaml").read_text()))
    missing = referenced - set(yaml.safe_load((_DATA / "parameters.yaml").read_text()))
    assert not missing, sorted(missing)
