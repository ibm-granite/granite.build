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

"""Behaviour tests for the bundled ``src/dd_guard.sh``.

Runs the real script. Each refusal case asserts a non-zero exit and that the message
names the offending field, since a guard whose message is unhelpful only moves the
confusion earlier. Cluster-agnostic, Mode 1 only.
"""

import pathlib
import shutil
import subprocess

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "src" / "dd_guard.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)

_VALID = {
    "--output": "questions",
    "--num-records": "10",
    "--resume": "if_possible",
    "--export-format": "",
    "--min-records": "",
    "--seed-path": "",
    "--dd-version": "0.9.5",
    "--dd-image": "",
    "--has-config": "x",
    "--has-providers": "x",
    "--local-runtime": "",
    "--local-model": "",
    "--local-port": "11434",
}


def _guard(**over) -> subprocess.CompletedProcess:
    opts = dict(_VALID)
    opts.update({f"--{k.replace('_', '-')}": v for k, v in over.items()})
    argv = [w for kv in opts.items() for w in kv]
    return subprocess.run(["bash", str(_SCRIPT), *argv], capture_output=True, text=True)


def test_valid_config_passes():
    proc = _guard()
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""


@pytest.mark.parametrize(
    "over,needle",
    [
        ({"has_config": ""}, "data_designer_config.config is required"),
        ({"has_providers": ""}, "model_providers is required"),
        ({"output": ""}, "output is required"),
        ({"output": "my output"}, "whitespace or a double quote"),
        ({"output": 'out"put'}, "whitespace or a double quote"),
        ({"num_records": "0"}, "num_records must be a positive integer"),
        ({"num_records": "10.5"}, "num_records must be a positive integer"),
        ({"num_records": ""}, "num_records must be a positive integer"),
        ({"min_records": "-1"}, "min_records must be"),
        ({"resume": "sometimes"}, "resume must be one of"),
        ({"export_format": "xlsx"}, "export_format must be empty or one of"),
        ({"dd_version": ""}, "dd_version is required"),
        ({"seed_path": "{{ bindings.sede.binding.path }}"}, "unrendered Jinja"),
        ({"local_runtime": "vllm"}, "local_model.runtime must be empty or 'ollama'"),
        (
            {"local_runtime": "ollama", "local_model": ""},
            "local_model.model is required",
        ),
        (
            {"local_runtime": "ollama", "local_model": "m", "local_port": "0"},
            "local_model.port",
        ),
        (
            {"local_runtime": "ollama", "local_model": "m", "local_port": "x1"},
            "local_model.port",
        ),
        (
            {"local_runtime": "ollama", "local_model": "m", "local_port": "65536"},
            "at most 65535",
        ),
        (
            {
                "local_runtime": "ollama",
                "local_model": "m",
                "dd_image": "quay.io/o/dd:1",
            },
            "cannot be used together with dd_image",
        ),
        ({"num_records": "4", "min_records": "5"}, "min_records (5) is greater than"),
    ],
)
def test_refusals(over, needle):
    proc = _guard(**over)
    assert proc.returncode != 0
    assert needle in proc.stderr


def test_image_mode_needs_no_version():
    assert _guard(dd_version="", dd_image="quay.io/org/dd:1").returncode == 0


@pytest.mark.parametrize("fmt", ["jsonl", "csv", "parquet"])
def test_export_formats(fmt):
    assert _guard(export_format=fmt).returncode == 0


def test_accented_output_is_not_whitespace():
    """LC_ALL=C keeps [[:space:]] ASCII; an accented id is not refused as whitespace."""
    assert _guard(output="questionnaire_é").returncode == 0


def test_local_model_satisfies_providers():
    """A model served on the node is a provider; no model_providers needed."""
    assert (
        _guard(
            has_providers="", local_runtime="ollama", local_model="granite4:350m"
        ).returncode
        == 0
    )


def test_no_providers_and_no_local_model_names_both_options():
    proc = _guard(has_providers="")
    assert proc.returncode != 0 and "local_model" in proc.stderr


def test_min_records_equal_to_num_records_passes():
    assert _guard(num_records="4", min_records="4").returncode == 0


def test_dd_image_with_api_providers_still_allowed():
    """Only the dd_image + local_model combination is refused."""
    assert _guard(dd_version="", dd_image="quay.io/o/dd:1").returncode == 0


@pytest.mark.parametrize("mode", ["if_possible", "always", "restart", "never"])
def test_every_resume_mode_is_accepted(mode):
    assert _guard(resume=mode).returncode == 0
