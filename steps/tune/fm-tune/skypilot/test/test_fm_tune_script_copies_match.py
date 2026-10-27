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
It also keeps the render-time ``validate`` macro the same in both templates, and in
step with the script's own checks: one table of bad values must be rejected by both
layers. Mode 1 only.
"""

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest
import yaml

from gbserver.utils.template import fill_objtemplate, fill_template

_STEPS = pathlib.Path(__file__).resolve().parents[2]


def test_run_script_copies_are_identical():
    bash = (_STEPS / "bash" / "src" / "fmtune_run.sh").read_bytes()
    sky = (_STEPS / "skypilot" / "src" / "fmtune_run.sh").read_bytes()
    assert (
        bash == sky
    ), "copy steps/tune/fm-tune/bash/src/fmtune_run.sh over the skypilot copy"


def test_both_copies_are_executable():
    for env in ("bash", "skypilot"):
        assert os.access(_STEPS / env / "src" / "fmtune_run.sh", os.X_OK), env


def test_both_templates_share_config_and_io():
    """The user-facing contract (fm_tune_config, I/O, name, version) must match.

    Only config.fm_tune_config is compared, not the whole config: the bash template
    also sets config.bash.skip_finding_output_artifacts, an env-specific knob for
    the bash wrapper that has no meaning on skypilot.
    """
    t = {
        e: yaml.safe_load((_STEPS / e / "step-template.yaml").read_text())
        for e in ("bash", "skypilot")
    }
    assert (
        t["bash"]["config"]["fm_tune_config"]
        == t["skypilot"]["config"]["fm_tune_config"]
    )
    for key in ("inputs", "outputs", "name", "version"):
        assert t["bash"][key] == t["skypilot"][key], key


_MACRO = re.compile(r"^( *)\{%- macro validate\(.*?\{%- endmacro -%\}$", re.M | re.S)


def _validate_macro(env: str) -> str:
    text = (_STEPS / env / "step-template.yaml").read_text()
    (m,) = list(_MACRO.finditer(text))
    return "\n".join(line[len(m.group(1)) :] for line in m.group(0).splitlines())


def test_both_templates_carry_the_same_validate_macro():
    assert _validate_macro("bash") == _validate_macro("skypilot")


# One row per rule: an fm_tune_config override and that rule's own message fragment,
# which both layers' messages must contain. A bare key is not enough: with a rule
# removed from the script, a later die naming the same key ("train_file not found")
# would still pass the row.
# Not here: an empty run_name, which the script defaults to fm-tune (an unset env
# var) but the template rejects, since a build that sets it empty made a mistake.
_SRC = {"path": "", "repo": "", "ref": "main", "subdir": "sub"}
_BAD = [
    ({"train_file": ""}, "train_file is required"),
    ({"validation_file": ""}, "validation_file is required"),
    ({"train_file": "../x"}, "train_file must be a relative path inside the dataset"),
    ({"train_file": ".."}, "train_file must be a relative path inside the dataset"),
    (
        {"train_file": "a/../../x"},
        "train_file must be a relative path inside the dataset",
    ),
    (
        {"train_file": "/abs/t.jsonl"},
        "train_file must be a relative path inside the dataset",
    ),
    (
        {"validation_file": "../v"},
        "validation_file must be a relative path inside the dataset",
    ),
    (
        {"validation_file": "/abs/v.jsonl"},
        "validation_file must be a relative path inside the dataset",
    ),
    (
        {"validation_file": "/v"},
        "validation_file must be a relative path inside the dataset",
    ),
    ({"source": dict(_SRC, path="/a", repo="https://g/r.git")}, "set exactly one of"),
    ({"source": _SRC}, "set exactly one of"),
    ({"model": ""}, "set fm_tune_config.model or bind the base_model input"),
    ({"run_name": "a b"}, "run_name must match ^[A-Za-z0-9._-]+$"),
    ({"run_name": "\u00e9t\u00e9"}, "run_name must match ^[A-Za-z0-9._-]+$"),
    ({"run_name": "a/b"}, "run_name must match ^[A-Za-z0-9._-]+$"),
    ({"config_name": "a/b"}, "config_name must match ^[A-Za-z0-9._-]+$"),
    ({"config_name": "a b"}, "config_name must match ^[A-Za-z0-9._-]+$"),
    # --no_autotune is passed for false alone, so anything else would run a sweep.
    ({"autotune": "no"}, "autotune must be true or false"),
    ({"autotune": 0}, "autotune must be true or false"),
    ({"source": dict(_SRC, repo="-x")}, "source.repo must not start with '-'"),
    (
        {"source": dict(_SRC, repo="--upload-pack=x")},
        "source.repo must not start with '-'",
    ),
    ({"source": dict(_SRC, path="/a", ref="-x")}, "source.ref must not start with '-'"),
    ({"source": dict(_SRC, path="/a", ref="-u")}, "source.ref must not start with '-'"),
    # fm-tune's pipeline.py raises once training starts, after clone and venv.
    (
        {"tuning_algo": "", "rl_algo": ""},
        "tuning_algo and rl_algo cannot both be empty or none",
    ),
    (
        {"tuning_algo": "none", "rl_algo": ""},
        "tuning_algo and rl_algo cannot both be empty or none",
    ),
    (
        {"tuning_algo": "None", "rl_algo": "NONE"},
        "tuning_algo and rl_algo cannot both be empty or none",
    ),
    (
        {"tuning_algo": "", "rl_algo": "none"},
        "tuning_algo and rl_algo cannot both be empty or none",
    ),
    # pipeline.py: offline RL (AUTOTUNE_OFFLINE_RL = dpo, kto) with tuning_algo none.
    (
        {"tuning_algo": "", "rl_algo": "dpo"},
        "an offline rl_algo (dpo, kto) needs a tuning_algo",
    ),
    (
        {"tuning_algo": "none", "rl_algo": "dpo"},
        "an offline rl_algo (dpo, kto) needs a tuning_algo",
    ),
    (
        {"tuning_algo": "NONE", "rl_algo": "KTO"},
        "an offline rl_algo (dpo, kto) needs a tuning_algo",
    ),
]
# Rules only the render can check (the script gets flat strings): both templates.
_RENDER_ONLY = [
    ({"run_name": ""}, "run_name must match ^[A-Za-z0-9._-]+$"),
    ({"seed": [1]}, "fm_tune_config.seed must be a scalar"),
    (
        {"source": dict(_SRC, path="/a", subdir={"a": 1})},
        "fm_tune_config.source.subdir must be a scalar",
    ),
    (
        {"autotunex": {"server_url": ["u"], "job_id": ""}},
        "fm_tune_config.autotunex.server_url must be a scalar",
    ),
    ({"source": None}, "fm_tune_config.source must be a mapping"),
    ({"autotunex": "x"}, "fm_tune_config.autotunex must be a mapping"),
    ({"hpo_config": ["a"]}, "fm_tune_config.hpo_config must be a mapping or null"),
    (
        {"tuning_algo": None, "rl_algo": None},
        "tuning_algo and rl_algo cannot both be empty or none",
    ),
    (
        {"tuning_algo": None, "rl_algo": "dpo"},
        "an offline rl_algo (dpo, kto) needs a tuning_algo",
    ),
]
_GOOD = [
    {"train_file": n, "validation_file": n}
    for n in ("t.jsonl", "sub/dir/t.jsonl", "a..b.jsonl")
] + [
    {"tuning_algo": "", "rl_algo": "grpo"},  # online RL needs no tuning algorithm
    {"tuning_algo": "none", "rl_algo": "ppo"},
    {"tuning_algo": "lora", "rl_algo": ""},
    {"tuning_algo": "lora", "rl_algo": "dpo"},
    {"tuning_algo": "sft", "rl_algo": "kto"},
    {"autotune": False},
    {"autotune": "True"},
    {"autotune": ""},  # unset: the script's default (true)
]
_LAUNCHERS = {"bash": ("Bash", "fm-tune"), "skypilot": ("Skypilot", "fm-tune")}


def _template(env):
    tpl = yaml.safe_load((_STEPS / env / "step-template.yaml").read_text())
    key, name = _LAUNCHERS[env]
    return tpl, tpl["environment_configs"][key]["launchers"][name]["config"]


@pytest.fixture
def layers(tmp_path):
    """Render both templates' launcher configs, and run the script on the bash env map."""
    if shutil.which("bash") is None:  # pragma: no cover
        pytest.skip("bash not available")
    src = tmp_path / "fm-tune"
    (src / "autotune" / "configs").mkdir(parents=True)
    (src / "autotune" / "configs" / "autotune.yaml").write_text("{}\n")
    (src / "main.py").write_text("")
    data = tmp_path / "data"
    for n in ("t.jsonl", "sub/dir/t.jsonl", "a..b.jsonl"):
        (data / n).parent.mkdir(parents=True, exist_ok=True)
        (data / n).write_text("{}\n")

    def cfg(tpl, over):
        c = json.loads(json.dumps(tpl["config"]))
        c["fm_tune_config"].update(
            model="org/tiny",
            train_file="t.jsonl",
            validation_file="t.jsonl",
            source=dict(_SRC, path=str(src)),
        )
        c["fm_tune_config"].update(over)
        return {"config": c, "bindings": {"dataset": {"binding": {"path": str(data)}}}}

    def render_error(env, over):
        """The raise_error message the strict fill of one template fails with."""
        tpl, launcher = _template(env)
        with pytest.raises(RuntimeError) as exc:
            fill_objtemplate(launcher, cfg(tpl, over), strict=True)
        assert isinstance(exc.value.__cause__, ValueError), repr(exc.value.__cause__)
        return str(exc.value.__cause__)

    def render(env, over):
        tpl, launcher = _template(env)
        fill_objtemplate(launcher, cfg(tpl, over), strict=True)

    def script(over):
        tpl, launcher = _template("bash")
        ctx = cfg(tpl, over)
        env = {
            k: fill_template(v, ctx, strict=True) for k, v in launcher["env"].items()
        }
        env.update(
            PATH=os.environ["PATH"],
            FMTUNE_SKIP_INSTALL="1",
            FMTUNE_PYTHON=sys.executable,
            FMTUNE_OUTPUT_DIR=str(tmp_path / "out"),
            FMTUNE_CACHE_DIR=str(tmp_path / "cache"),
        )
        return subprocess.run(
            ["bash", str(_STEPS / "bash" / "src" / "fmtune_run.sh")],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    return render_error, render, script


@pytest.mark.parametrize("env", ["bash", "skypilot"])
@pytest.mark.parametrize("over, rule", _BAD)
def test_both_renders_reject_a_bad_value_with_the_rules_message(
    layers, env, over, rule
):
    render_error, _, _ = layers
    msg = render_error(env, over)
    assert msg.startswith("fm-tune: ") and rule in msg, msg


@pytest.mark.parametrize("over, rule", _BAD)
def test_the_script_rejects_a_bad_value_with_the_rules_message(layers, over, rule):
    *_, script = layers
    proc = script(over)
    assert proc.returncode == 2, proc.stderr
    assert f"fm-tune: " in proc.stderr and rule in proc.stderr, proc.stderr


@pytest.mark.parametrize("env", ["bash", "skypilot"])
@pytest.mark.parametrize("over, rule", _RENDER_ONLY)
def test_both_renders_reject_a_render_only_bad_value(layers, env, over, rule):
    render_error, _, _ = layers
    msg = render_error(env, over)
    assert msg.startswith("fm-tune: ") and rule in msg, msg


@pytest.mark.parametrize("over", _GOOD)
def test_both_layers_accept_the_same_good_values(layers, over):
    _, render, script = layers
    render("bash", over)
    render("skypilot", over)
    proc = script(over)
    assert proc.returncode == 0, proc.stderr


# The validate macro's type checks, which only the render can make (the script gets
# flat strings). One copy here, over both templates: the macro text is shared.
_SCALAR_KEYS = [
    "extra",
    "model",
    "train_file",
    "validation_file",
    "tuning_algo",
    "rl_algo",
    "run_name",
    "autotune",
    "backend",
    "seed",
    "config_preset",
    "config_name",
    "output_path",
    "extra_args",
    "python",
]


@pytest.mark.parametrize("env", ["bash", "skypilot"])
@pytest.mark.parametrize("key", _SCALAR_KEYS)
@pytest.mark.parametrize("bad", [[1, 2], {"a": 1}])
def test_a_list_or_mapping_for_a_scalar_key_fails_the_fill(layers, env, key, bad):
    render_error, _, _ = layers
    assert render_error(env, {key: bad}) == (
        f"fm-tune: fm_tune_config.{key} must be a scalar (string/number/bool), "
        "got a list/mapping; write it as a string"
    )


@pytest.mark.parametrize("env", ["bash", "skypilot"])
@pytest.mark.parametrize("key", ["path", "repo", "ref", "subdir"])
def test_a_list_for_a_source_key_fails_the_fill(layers, env, key):
    render_error, _, _ = layers
    src = dict(_SRC, path="/a")
    src[key] = ["x"]
    msg = render_error(env, {"source": src})
    assert f"fm_tune_config.source.{key} must be a scalar" in msg, msg


@pytest.mark.parametrize("env", ["bash", "skypilot"])
@pytest.mark.parametrize("key", ["server_url", "job_id"])
def test_a_mapping_for_an_autotunex_key_fails_the_fill(layers, env, key):
    render_error, _, _ = layers
    ax = {"server_url": "", "job_id": "", key: {"a": 1}}
    msg = render_error(env, {"autotunex": ax})
    assert f"fm_tune_config.autotunex.{key} must be a scalar" in msg, msg


@pytest.mark.parametrize("env", ["bash", "skypilot"])
@pytest.mark.parametrize("key", ["source", "autotunex"])
@pytest.mark.parametrize("bad", [None, "x", ["a"], 3])
def test_a_source_or_autotunex_that_is_not_a_mapping_fails_with_a_named_error(
    layers, env, key, bad
):
    """Checked before anything reads source.* / autotunex.*: no bare UndefinedError."""
    render_error, _, _ = layers
    msg = render_error(env, {key: bad})
    assert msg.startswith(f"fm-tune: fm_tune_config.{key} must be a mapping"), msg


@pytest.mark.parametrize("env", ["bash", "skypilot"])
@pytest.mark.parametrize("bad", [["a"], "x", 3, True])
def test_an_hpo_config_that_is_not_a_mapping_or_null_fails_the_fill(layers, env, bad):
    render_error, _, _ = layers
    msg = render_error(env, {"hpo_config": bad})
    assert msg.startswith(
        "fm-tune: fm_tune_config.hpo_config must be a mapping or null"
    ), msg
