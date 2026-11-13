# Copyright Granite.Build Authors
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

"""Tests that fm-tune's base install is slim and the heavy stack lives in extras."""

import re
from pathlib import Path

import tomllib  # Python 3.11+

_PYPROJECT = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())


def test_base_dependencies_are_slim():
    assert _PYPROJECT["project"]["dependencies"] == ["pyyaml>=6.0.2"]


def _names(reqs):
    """Requirement names without versions, extras or markers ("verl[vllm]==0.9.1; ..." -> "verl")."""
    return {re.split(r"[\[=<>~!; ]", r, maxsplit=1)[0].lower() for r in reqs}


def test_core_carries_the_training_stack():
    """`core` is the lean CPU/demo install (single-device SFT + DPO/KTO); pyproject explains why it exists."""
    core = _names(_PYPROJECT["project"]["optional-dependencies"]["core"])
    for dep in (
        "torch",
        "transformers",
        "peft",
        "trl",
        "bitsandbytes",
        "accelerate",
        "hyperopt",
        "tokenizers",
        "sentencepiece",
        "tabulate",
        "tensorboardx",
        "packaging",
        "datasets",
        "ray",
    ):
        assert dep in core, dep
    # The GPU-only stack must stay out of core (it does not build on CPU / macOS).
    assert not core & {"verl", "flash-attn", "deepspeed", "vllm"}


def test_full_is_core_plus_the_gpu_stack():
    full = _PYPROJECT["project"]["optional-dependencies"]["full"]
    assert "autotune[core]" in full  # every shared pin lives in core only
    assert {"verl", "flash-attn", "deepspeed"} <= _names(full)


def test_configs_are_shipped_as_package_data():
    assert _PYPROJECT["tool"]["setuptools"]["package-data"]["autotune"] == ["configs/*.yaml"]
