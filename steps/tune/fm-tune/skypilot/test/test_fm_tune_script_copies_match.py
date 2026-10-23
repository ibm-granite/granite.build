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


_MACRO = re.compile(r"^( *)\{%- macro validate\(.*?\{%- endmacro -%\}$", re.M | re.S)


def _validate_macro(env: str) -> str:
    text = (_STEPS / env / "step-template.yaml").read_text()
    (m,) = list(_MACRO.finditer(text))
    return "\n".join(line[len(m.group(1)):] for line in m.group(0).splitlines())


def test_both_templates_carry_the_same_validate_macro():
    assert _validate_macro("bash") == _validate_macro("skypilot")


# One row per rule, as an fm_tune_config override and the key both layers must name.
# Not here: an empty run_name, which the script defaults to fm-tune (an unset env
# var) but the template rejects, since a build that sets it empty made a mistake.
_SRC = {"path": "", "repo": "", "ref": "main", "subdir": "sub"}
_BAD = [
    ({"train_file": ""}, "train_file"),
    ({"validation_file": ""}, "validation_file"),
    ({"train_file": "../x"}, "train_file"),
    ({"train_file": ".."}, "train_file"),
    ({"train_file": "a/../../x"}, "train_file"),
    ({"validation_file": "/abs/v.jsonl"}, "validation_file"),
    ({"source": dict(_SRC, path="/a", repo="https://g/r.git")}, "source.path and"),
    ({"source": _SRC}, "source.path and"),
    ({"model": ""}, "bind the base_model input"),
    ({"run_name": "a b"}, "run_name must match"),
    ({"run_name": "\u00e9t\u00e9"}, "run_name must match"),
    ({"source": dict(_SRC, repo="-x")}, "source.repo must not start with '-'"),
    ({"source": dict(_SRC, path="/a", ref="-x")}, "source.ref must not start with '-'"),
]
_GOOD = [{"train_file": n, "validation_file": n} for n in ("t.jsonl", "sub/dir/t.jsonl", "a..b.jsonl")]


@pytest.fixture
def layers(tmp_path):
    """Render the bash launcher config, and run the script on the env it maps."""
    if shutil.which("bash") is None:  # pragma: no cover
        pytest.skip("bash not available")
    tpl = yaml.safe_load((_STEPS / "bash" / "step-template.yaml").read_text())
    launcher = tpl["environment_configs"]["Bash"]["launchers"]["fm-tune"]["config"]
    src = tmp_path / "fm-tune"
    (src / "autotune" / "configs").mkdir(parents=True)
    (src / "autotune" / "configs" / "autotune.yaml").write_text("{}\n")
    (src / "main.py").write_text("")
    data = tmp_path / "data"
    for n in ("t.jsonl", "sub/dir/t.jsonl", "a..b.jsonl"):
        (data / n).parent.mkdir(parents=True, exist_ok=True)
        (data / n).write_text("{}\n")

    def cfg(over):
        c = json.loads(json.dumps(tpl["config"]))
        c["fm_tune_config"].update(
            model="org/tiny", train_file="t.jsonl", validation_file="t.jsonl",
            source=dict(_SRC, path=str(src)),
        )
        c["fm_tune_config"].update(over)
        return {"config": c, "bindings": {"dataset": {"binding": {"path": str(data)}}}}

    def render(over):
        fill_objtemplate(launcher, cfg(over), strict=True)

    def script(over):
        ctx = cfg(over)
        env = {k: fill_template(v, ctx, strict=True) for k, v in launcher["env"].items()}
        env.update(
            PATH=os.environ["PATH"], FMTUNE_SKIP_INSTALL="1", FMTUNE_PYTHON=sys.executable,
            FMTUNE_OUTPUT_DIR=str(tmp_path / "out"), FMTUNE_CACHE_DIR=str(tmp_path / "cache"),
        )
        return subprocess.run(
            ["bash", str(_STEPS / "bash" / "src" / "fmtune_run.sh")],
            env=env, capture_output=True, text=True, check=False,
        )

    return render, script


@pytest.mark.parametrize("over, key", _BAD)
def test_both_layers_reject_the_same_bad_values(layers, over, key):
    render, script = layers
    with pytest.raises(RuntimeError) as exc:
        render(over)
    assert key in str(exc.value.__cause__)
    proc = script(over)
    assert proc.returncode == 2, proc.stderr
    assert "fm-tune: " in proc.stderr and key in proc.stderr, proc.stderr


@pytest.mark.parametrize("over", _GOOD)
def test_both_layers_accept_the_same_good_values(layers, over):
    render, script = layers
    render(over)
    proc = script(over)
    assert proc.returncode == 0, proc.stderr
