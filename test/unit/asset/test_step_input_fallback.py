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

"""The deprecated config-key fallback of the steps shipped in v0.3.x (#453).

Those steps now declare their inputs as OPTIONAL and read them through `bindings`.
For one deprecation window an old recipe that still sets the config key keeps working,
with a DEPRECATED warning; a bound input wins over the key. The launcher's `run` is
rendered STRICT by gbserver (targetsteprun.py), so an unbound optional input must be
guarded rather than referenced — which is what rendering it here, strict, checks.

dpk's fallback is covered by its own step tests (steps/dpk/skypilot/test).
"""

import pathlib

import pytest
import yaml

from gbserver.utils.template import fill_template

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
LSF = REPO_ROOT / "configurations/assets/environments/skypilot/lsf/ibm-bluevela/steps"

# (step, config section, config key, input, binding field, shell variable)
CASES = [
    ("sage-eval", "sage_eval_config", "model_path", "model", "path", "MODEL_PATH"),
    ("bfcl-eval", "bfcl_config", "model_path", "model", "path", "MODEL_PATH"),
    ("openinstruct-sft", "sft_config", "model_path", "model", "path", "MODEL_PATH"),
    (
        "openinstruct-rl",
        "rl_config",
        "rm_server_url",
        "rm_url",
        "state",
        "RM_SERVER_URL",
    ),
    (
        "openinstruct-rl",
        "rl_config",
        "code_server_url",
        "code_url",
        "state",
        "CODE_SERVER_URL",
    ),
]
IDS = [f"{c[0]}:{c[3]}" for c in CASES]


def _render(step, section, key, value, bindings):
    raw = yaml.safe_load((LSF / step / "step.yaml").read_text(encoding="utf-8"))
    (launcher,) = raw["environment_configs"]["Skypilot"]["launchers"].values()
    config = dict(raw["config"])
    config[section] = dict(config[section], **{key: value})
    return fill_template(
        templ=launcher["config"]["run"],
        data={"config": config, "bindings": bindings},
        strict=True,
    )


def _assigned(rendered, var):
    """The value the rendered script assigns to `var` (with or without `export`)."""
    for line in rendered.splitlines():
        line = line.strip().removeprefix("export ")
        if line.startswith(f"{var}="):
            return line[len(var) + 1 :].strip('"')
    return None


@pytest.mark.parametrize("step,section,key,inp,field,var", CASES, ids=IDS)
def test_the_input_is_declared_optional(step, section, key, inp, field, var):
    raw = yaml.safe_load((LSF / step / "step.yaml").read_text(encoding="utf-8"))
    assert inp in raw["inputs"]["optional"]
    assert raw["inputs"]["allow_unknown"] is True


@pytest.mark.parametrize("step,section,key,inp,field,var", CASES, ids=IDS)
def test_a_bound_input_is_used_without_warnings(step, section, key, inp, field, var):
    bindings = {inp: {"binding": {field: "/from/binding"}}}
    rendered = _render(step, section, key, "", bindings)
    assert _assigned(rendered, var) == "/from/binding"
    assert ": DEPRECATED:" not in rendered
    assert "is ignored" not in rendered


@pytest.mark.parametrize("step,section,key,inp,field,var", CASES, ids=IDS)
def test_the_config_key_is_a_deprecated_fallback(step, section, key, inp, field, var):
    rendered = _render(step, section, key, "/from/config", {})
    assert _assigned(rendered, var) == "/from/config"
    assert f"DEPRECATED: {section}.{key}" in rendered
    assert f"input '{inp}'" in rendered


@pytest.mark.parametrize("step,section,key,inp,field,var", CASES, ids=IDS)
def test_the_bound_input_wins_over_the_config_key(step, section, key, inp, field, var):
    bindings = {inp: {"binding": {field: "/from/binding"}}}
    rendered = _render(step, section, key, "/from/config", bindings)
    assert _assigned(rendered, var) == "/from/binding"
    assert f"{section}.{key} is ignored" in rendered


@pytest.mark.parametrize("step,section,key,inp,field,var", CASES, ids=IDS)
def test_the_same_value_in_both_is_flagged_as_redundant(
    step, section, key, inp, field, var
):
    """A recipe that still copies the binding into the key is not 'ignored', but the
    key is still deprecated, so it is told to drop it."""
    bindings = {inp: {"binding": {field: "/same"}}}
    rendered = _render(step, section, key, "/same", bindings)
    assert _assigned(rendered, var) == "/same"
    assert "is ignored" not in rendered
    assert f"DEPRECATED: {section}.{key} duplicates the bound '{inp}' input" in rendered


@pytest.mark.parametrize("step,section,key,inp,field,var", CASES, ids=IDS)
def test_neither_renders_without_error(step, section, key, inp, field, var):
    """Strict rendering with the input unbound and the key empty must not raise."""
    rendered = _render(step, section, key, "", {})
    assert ": DEPRECATED:" not in rendered
