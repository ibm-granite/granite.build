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

"""The bash and skypilot fm-tune steps ship the same run script, and the same config.

Each env dir publishes its own src/, so the script is duplicated on purpose. This
test is what keeps the copies from drifting; edit both, or copy one over the other.
Mode 1 only.
"""

import os
import pathlib

import yaml

_STEPS = pathlib.Path(__file__).resolve().parents[2]


def test_run_script_copies_are_identical():
    bash = (_STEPS / "bash" / "src" / "fmtune_run.sh").read_bytes()
    sky = (_STEPS / "skypilot" / "src" / "fmtune_run.sh").read_bytes()
    assert bash == sky, "copy steps/tune/fm-tune/bash/src/fmtune_run.sh over the skypilot copy"


def test_both_copies_are_executable():
    for env in ("bash", "skypilot"):
        assert os.access(_STEPS / env / "src" / "fmtune_run.sh", os.X_OK), env


def test_both_templates_share_config_and_io():
    """The user-facing contract (fm_tune_config, I/O, name, version) must match.

    Only config.fm_tune_config is compared, not the whole config: the bash template
    also sets config.bash.skip_finding_output_artifacts, an env-specific knob for
    the bash wrapper that has no meaning on skypilot.
    """
    t = {e: yaml.safe_load((_STEPS / e / "step-template.yaml").read_text()) for e in ("bash", "skypilot")}
    assert t["bash"]["config"]["fm_tune_config"] == t["skypilot"]["config"]["fm_tune_config"]
    for key in ("inputs", "outputs", "name", "version"):
        assert t["bash"][key] == t["skypilot"][key], key
