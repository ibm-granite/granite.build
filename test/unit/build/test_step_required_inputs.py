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

"""A target that does not bind a step's required inputs fails build validation (#453).

This is the point of declaring inputs: the failure happens when the build is created,
before anything is queued, instead of on the cluster after queue time, where an
unresolved `{{ bindings.<name>... }}` used to arrive as literal text.

Each step with required inputs is resolved through the standalone `local` space, the
same `space://steps/...` lookup a real build does, with nothing launched.
"""

import pathlib

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "configurations" / "assets"
LOCAL_SPACE = REPO_ROOT / "configurations" / "spaces" / "local"

_LSF_ENV = "space://environments/skypilot/lsf/ibm-bluevela"
_BASH_ENV = "space://environments/bash"

# step path (under space://steps/) -> the environment it is resolved in
STEPS = {
    "distill/tokenizer-align": _LSF_ENV,
    "distill/corpus-prep": _LSF_ENV,
    "distill/corpus-pin-check": _LSF_ENV,
    "distill/sft": _LSF_ENV,
    "distill/gold": _LSF_ENV,
    "distill/eval": _LSF_ENV,
    "distill/hf-export": _LSF_ENV,
    "distill/logit-precompute": _LSF_ENV,
    "distill/vllm-server": _LSF_ENV,
    "inference": _BASH_ENV,
    "inference-lora": _BASH_ENV,
    "lora-finetune": _BASH_ENV,
}


def _step_yaml(step: str, env: str) -> dict:
    env_dir = env.removeprefix("space://")
    path = ASSETS_DIR / env_dir / "steps" / step / "step.yaml"
    if not path.is_file():  # environment-agnostic steps live one level up
        path = ASSETS_DIR / "environments" / "skypilot" / "steps" / step / "step.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _build(tmp_path, monkeypatch, step: str, env: str, inputs: dict):
    from gbserver.build.build import Build
    from gbserver.build.space import Space

    monkeypatch.setenv("GB_HOME_DIR", str(tmp_path / "gb_home"))
    declared = _step_yaml(step, env)
    outputs = {
        name: {"type": spec["type"]}
        for name, spec in (
            (declared.get("outputs") or {}).get("required") or {}
        ).items()
    }
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    target = {"environment_uri": env, "steps": [{"step_uri": f"space://steps/{step}"}]}
    if inputs:
        target["inputs"] = inputs
    if outputs:
        target["outputs"] = outputs
    (build_dir / "build.yaml").write_text(
        yaml.safe_dump({"granite.build": {"name": "t", "targets": {"t": target}}})
    )
    return Build(
        build_dir=build_dir,
        space=Space(uri=f"file://{LOCAL_SPACE}", username="t"),
        username="t",
    )


def _bound(step: str, env: str) -> dict:
    required = _step_yaml(step, env)["inputs"]["required"]
    return {
        name: {"uri": f"env:///inputs/{name}", "type": spec["type"]}
        for name, spec in required.items()
    }


@pytest.mark.parametrize("step", sorted(STEPS))
def test_the_step_has_required_inputs(step):
    assert _step_yaml(step, STEPS[step])["inputs"]["required"]


@pytest.mark.parametrize("step", sorted(STEPS))
def test_binding_every_required_input_validates(tmp_path, monkeypatch, step):
    _build(tmp_path, monkeypatch, step, STEPS[step], _bound(step, STEPS[step]))


@pytest.mark.parametrize("step", sorted(STEPS))
def test_a_missing_required_input_fails_validation(tmp_path, monkeypatch, step):
    env = STEPS[step]
    bound = _bound(step, env)
    for missing in sorted(bound):
        inputs = {k: v for k, v in bound.items() if k != missing}
        sub = tmp_path / missing
        sub.mkdir()
        with pytest.raises(ValueError) as raised:
            _build(sub, monkeypatch, step, env, inputs)
        cause = str(raised.value.__cause__)
        assert f"Required input `{missing}`" in cause and "is missing" in cause, cause
