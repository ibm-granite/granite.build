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

"""Unit tests for src/nemo_rl_guard.sh — the step's pre-flight contract check.

Runs the REAL script with a clean env, both directions: it must refuse an
invalid contract with a one-line `nemo-rl:` reason, and stay silent on a valid
one. The guard is what stops a bad build before the expensive setup phase.
"""

import pathlib
import shutil
import subprocess

import pytest

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _STEP_DIR / "src" / "nemo_rl_guard.sh"

_OK = dict(
    algorithm="sft",
    config_name="sft.yaml",
    output="model",
    num_nodes="1",
    base_model="/staged/base",
    dataset="",
    dataset_key="",
)


def _run(**over) -> subprocess.CompletedProcess:
    cfg = dict(_OK, **over)
    argv = [
        "bash",
        str(_SCRIPT),
        "--algorithm",
        cfg["algorithm"],
        "--config-name",
        cfg["config_name"],
        "--output",
        cfg["output"],
        "--num-nodes",
        cfg["num_nodes"],
        "--base-model",
        cfg["base_model"],
        "--dataset",
        cfg["dataset"],
        "--dataset-key",
        cfg["dataset_key"],
    ]
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )


def _reasons(proc: subprocess.CompletedProcess) -> list[str]:
    return [ln for ln in proc.stderr.splitlines() if ln.startswith("nemo-rl:")]


class TestAcceptsValidContract:
    def test_minimal_valid_is_silent(self):
        proc = _run()
        assert proc.returncode == 0, proc.stderr
        assert _reasons(proc) == []

    def test_bound_dataset_with_key_is_valid(self):
        proc = _run(dataset="/staged/ds", dataset_key="data.train.input_file")
        assert proc.returncode == 0, proc.stderr


class TestRefusesBadContract:
    @pytest.mark.parametrize("field", ["algorithm", "config_name", "output"])
    def test_required_strings_may_not_be_empty(self, field):
        proc = _run(**{field: ""})
        assert proc.returncode == 1
        assert _reasons(proc)

    def test_output_with_space_is_refused(self):
        proc = _run(output="my model")
        assert proc.returncode == 1
        assert any("whitespace" in r or "double quote" in r for r in _reasons(proc))

    def test_multi_node_is_refused(self):
        proc = _run(num_nodes="2")
        assert proc.returncode == 1
        assert any("num_nodes" in r for r in _reasons(proc))

    def test_empty_base_model_is_refused(self):
        proc = _run(base_model="")
        assert proc.returncode == 1
        assert any("base_model" in r for r in _reasons(proc))

    def test_unrendered_jinja_base_model_is_refused(self):
        proc = _run(base_model="{{ bindings.bse_model.binding.path }}")
        assert proc.returncode == 1
        assert any("Jinja" in r for r in _reasons(proc))

    def test_bound_dataset_without_key_is_refused(self):
        proc = _run(dataset="/staged/ds", dataset_key="")
        assert proc.returncode == 1
        assert any("dataset_override_key" in r for r in _reasons(proc))


class TestScriptIsValidShell:
    def test_bash_parses_the_script(self):
        assert subprocess.run(["bash", "-n", str(_SCRIPT)]).returncode == 0

    def test_shellcheck_is_clean(self):
        if shutil.which("shellcheck") is None:
            pytest.skip("shellcheck not installed")
        assert subprocess.run(["shellcheck", str(_SCRIPT)]).returncode == 0
