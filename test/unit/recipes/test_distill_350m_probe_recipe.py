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

"""Unit tests for the distill-probe recipe.

Every target runs the ``distill-probe`` step; the probes and the tests of their logic
live with the step (steps/distill/probe/skypilot). What the recipe owns is the
wiring -- which probe each target asks for, against which checkpoint -- so each target's
config is rendered through the PUBLISHED step template and run against a stand-in
interpreter that records its argv.

The recipe carries no parameters.yaml: it has no ``$${...}`` markers, because nothing
about it varies per run.
"""

import json
import pathlib
import subprocess

import pytest
import yaml
from unit.recipes.published_step import render_run

_BUILD = (
    pathlib.Path(__file__).resolve().parents[3]
    / "recipes"
    / "granite4-350m"
    / "lsf"
    / "distill-probe"
    / "build.yaml"
)

_TARGETS = ["load-student", "tokenizer", "tokenizer-fit"]
_CHECKPOINT = (
    "/proj/granite-build/g4os/skypilot-test/sft/checkpoints/"
    "v0-20260529-20260529_212150-hf/epoch_hf_2"
)


@pytest.fixture(scope="module")
def build():
    return yaml.safe_load(_BUILD.read_text(encoding="utf-8"))


def _step(build, name):
    (step,) = build["granite.build"]["targets"][name]["steps"]
    return step


def test_the_recipe_declares_all_three_probes(build):
    assert list(build["granite.build"]["targets"]) == _TARGETS


def test_it_takes_no_parameters():
    """Nothing about a probe varies per run, so there is no parameters.yaml and
    there must be no markers left behind expecting one."""
    assert "$${" not in _BUILD.read_text(encoding="utf-8")


def test_no_retries(build):
    """A probe that needed a retry has already told you something."""
    assert build["granite.build"]["retries"]["max_retries"] == 0


def test_no_probe_declares_an_output(build):
    """The answers are decisions for a human, not artifacts for a step."""
    for target in build["granite.build"]["targets"].values():
        assert "outputs" not in target


@pytest.mark.parametrize("name", _TARGETS)
def test_every_target_runs_the_probe_step(build, name):
    """No inline scripts: the recipe passes configuration, the step holds the code."""
    step = _step(build, name)
    assert step["step_uri"] == "space://steps/distill/probe"
    assert "command_config" not in step["config"]


@pytest.mark.parametrize("name", ["load-student", "tokenizer-fit"])
def test_the_gpu_probes_get_a_gpu(build, name):
    config = _step(build, name)["config"]
    assert config["compute_config"]["num_gpus_per_node"] == 1
    assert config["launcher_config"]["resources"]["accelerators"] == "H100:1"


def test_the_tokenizer_probe_is_cpu_only(build):
    config = _step(build, "tokenizer")["config"]
    assert "num_gpus_per_node" not in config["compute_config"]
    assert "accelerators" not in config["launcher_config"]["resources"]


@pytest.mark.parametrize("name", _TARGETS)
def test_each_target_asks_its_own_probe_of_the_same_checkpoint(build, name, tmp_path):
    """Renders the target through the published step and runs it. The three probes
    only answer one question together if they all read the same checkpoint."""
    argv_log = tmp_path / "argv.json"
    stub = tmp_path / "python-stub"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"json.dump(sys.argv[1:], open({str(argv_log)!r}, 'w'))\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    script, step_dir = render_run(
        "distill/probe",
        _step(build, name)["config"],
        probe_config={"python": str(stub)},
    )

    result = subprocess.run(
        ["bash", "-c", script], cwd=step_dir, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    argv = json.loads(argv_log.read_text(encoding="utf-8"))
    assert argv[:4] == ["./src/distill_probe.py", name, "--checkpoint", _CHECKPOINT]
    if name == "tokenizer-fit":
        assert argv[4:] == [
            "--corpus",
            "/proj/granite-build/g4os/gbtest/gold-distill-smoke/smoke_2000_nothink.jsonl",
            "--rows",
            "200",
        ]
