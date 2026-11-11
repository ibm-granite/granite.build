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

"""USAGE.md stays in step with the template: every config field and every
step-owned flag is documented, the exemplar is pasted verbatim, and no link
would rot when publish-step copies USAGE.md into configurations/."""

import pathlib
import re

import yaml

_STEP = pathlib.Path(__file__).resolve().parents[1]
_USAGE = (_STEP / "USAGE.md").read_text()
_README = (_STEP / "README.md").read_text()
_TEXT = (_STEP / "step-template.yaml").read_text()
_DEFAULTS = yaml.safe_load(_TEXT)["config"]["hf_peft_sft_config"]
_OWNED = re.findall(r'"([a-z_0-9]+)"', re.findall(r"set owned = \[(.*?)\]", _TEXT)[0])


def test_every_config_field_documented():
    for field in _DEFAULTS:
        assert f"| `{field}` |" in _USAGE, field


def test_every_owned_flag_listed():
    for flag in _OWNED:
        assert flag in _USAGE, flag


def test_exemplar_targets_pasted_verbatim():
    build = (_STEP / "test-data" / "build.yaml").read_text()
    body = build[build.index("    train:") :].rstrip()
    assert body in _USAGE


def test_only_links_that_survive_publishing():
    for link in re.findall(r"\]\(([^)]+)\)", _USAGE):
        assert link.startswith(("#", "src/", "https://")), link


def test_readme_points_to_usage_and_records_the_torch_decision():
    assert "[USAGE.md](USAGE.md)" in _README
    assert re.search(r"torch \d+\.\d+", _README)
    assert "driver" in _README
