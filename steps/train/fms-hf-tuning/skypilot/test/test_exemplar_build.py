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

"""Every build.yaml the step ships stays inside the step contract.

The exemplar a user copies and each cluster fixture are checked against what
the template's step-creation validation accepts, so a fixture cannot drift from
the step without a fast failure. Multi-target fixtures are checked per target.
"""

import re
from pathlib import Path

import pytest
import yaml

from gbserver.utils.template import fill_objtemplate, fill_template

_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parent / "test-data"
_TEMPLATE_TEXT = (_HERE.parent / "step-template.yaml").read_text()
_DEFAULTS = yaml.safe_load(_TEMPLATE_TEXT)["config"]["fms_hf_tuning_config"]
_PARAM_RE = re.compile(r"\$\$\{([A-Z0-9_]+)\}")
BUILDS = ["build.yaml", "aws/build.yaml", "slurm-bluevela/build.yaml"]
_OWNED = frozenset(
    re.findall(
        r'"([a-z_0-9]+)"', re.findall(r"set owned = \[(.*?)\]", _TEMPLATE_TEXT)[0]
    )
)
_CHAT = ("dataset_conversation_field", "instruction_template", "response_template")


def _targets(name):
    build = yaml.safe_load((_DATA / name).read_text())
    return list(build["granite.build"]["targets"].values())


def _cases():
    return [(b, i) for b in BUILDS for i in range(len(_targets(b)))]


def _target(build, i):
    return _targets(build)[i]


def _step(target):
    assert len(target["steps"]) == 1
    return target["steps"][0]


def _cfg(target):
    return {**_DEFAULTS, **_step(target)["config"]["fms_hf_tuning_config"]}


def _count(acc):
    parts = str(acc).split(":")
    return int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else 1


@pytest.mark.parametrize("build,i", _cases())
def test_step_uri(build, i):
    assert _step(_target(build, i))["step_uri"] == "space://steps/train/fms-hf-tuning"


@pytest.mark.parametrize("build,i", _cases())
def test_output_names_a_declared_model_output(build, i):
    target = _target(build, i)
    assert target["outputs"][_cfg(target)["output"]]["type"] == "model"


@pytest.mark.parametrize("build,i", _cases())
def test_exactly_one_data_source(build, i):
    target = _target(build, i)
    cfg = _cfg(target)
    if cfg["data_config"]:
        assert not cfg["dataset_name"] and "dataset" not in target["inputs"]
        for ds in cfg["data_config"]["datasets"]:
            for path in ds["data_paths"]:
                m = re.fullmatch(r"\{\{ bindings\.(\w+)\.binding\.path \}\}", path)
                assert m and m.group(1) in target["inputs"], path
    else:
        assert bool(cfg["dataset_name"]) != ("dataset" in target["inputs"])


@pytest.mark.parametrize("build,i", _cases())
def test_chat_data_has_upstreams_markers(build, i):
    cfg = _cfg(_target(build, i))
    if not cfg["data_config"]:
        assert all(k in cfg["args"] for k in _CHAT), sorted(cfg["args"])


@pytest.mark.parametrize("build,i", _cases())
def test_base_model_bound(build, i):
    assert "base_model" in _target(build, i)["inputs"]


@pytest.mark.parametrize("build,i", _cases())
def test_gpus_requested_and_enough_for_sharding(build, i):
    target = _target(build, i)
    acc = _step(target)["config"]["launcher_config"]["resources"]["accelerators"]
    assert str(acc).strip()
    if _cfg(target)["distributed"] == "fsdp" and "$${" not in str(acc):
        assert _count(acc) >= 2, acc


@pytest.mark.parametrize("build,i", _cases())
def test_alora_iff_invocation(build, i):
    cfg = _cfg(_target(build, i))
    assert (cfg["peft_method"] == "alora") == bool(
        cfg["args"].get("alora_invocation_string")
    )


@pytest.mark.parametrize("build,i", _cases())
def test_no_step_owned_args(build, i):
    args = {k.replace("-", "_") for k in _cfg(_target(build, i))["args"]}
    assert not _OWNED & args


@pytest.mark.parametrize("build,i", _cases())
def test_single_node(build, i):
    assert (
        _step(_target(build, i))["config"].get("compute_config", {}).get("num_nodes", 1)
        == 1
    )


@pytest.mark.parametrize("build,i", _cases())
def test_adapters_record_a_portable_base(build, i):
    cfg = _cfg(_target(build, i))
    if cfg["peft_method"] == "none":
        assert not cfg["base_model_id"]
    else:
        assert cfg["base_model_id"]


def test_fixtures_cover_every_method_and_mode():
    seen = {
        (_cfg(t)["peft_method"], _cfg(t)["distributed"])
        for b in BUILDS[1:]
        for t in _targets(b)
    }
    assert seen == {
        ("none", "ddp"),
        ("alora", "ddp"),
        ("lora", "fsdp"),
        ("none", "fsdp"),
    }


def test_fixtures_cover_both_data_paths():
    cfgs = [_cfg(t) for b in BUILDS[1:] for t in _targets(b)]
    assert any(c["data_config"] for c in cfgs)
    assert any(c["dataset_name"] for c in cfgs)
    assert any("dataset" in t["inputs"] for t in _targets("slurm-bluevela/build.yaml"))


def test_bluevela_fixture_stages_a_small_base_model():
    # BlueVela's hfpull runs in the cpu partition's 1 GiB cgroup and pulls the
    # whole repo; SmolLM2-135M-Instruct's 1.7 GB of onnx exports get it
    # OOM-killed (#597). The base repo has no chat template, so the Instruct
    # ChatML template is passed inline, inside {% raw %} so gbserver keeps it.
    for target in _targets("slurm-bluevela/build.yaml"):
        assert (
            target["inputs"]["base_model"]["uri"] == "hf:///HuggingFaceTB/SmolLM2-135M"
        )
        tmpl = _cfg(target)["args"]["chat_template"]
        assert tmpl.startswith("{% raw %}") and tmpl.endswith("{% endraw %}")
        assert "<|im_start|>" in tmpl


def test_bluevela_fixture_matches_the_environment():
    repo = _HERE.parents[4]
    env = yaml.safe_load(
        (
            repo
            / "configurations/assets/environments/skypilot/slurm/bluevela/environment.yaml"
        ).read_text()
    )
    for target in _targets("slurm-bluevela/build.yaml"):
        assert (
            target["environment_uri"] == "space://environments/skypilot/slurm/bluevela"
        )
        acc = _step(target)["config"]["launcher_config"]["resources"]["accelerators"]
        assert (
            acc.split(":")[0].lower() == env["config"]["sbatch_options"]["constraint"]
        )


def test_every_referenced_param_is_defined():
    referenced = set(_PARAM_RE.findall((_DATA / "build.yaml").read_text()))
    missing = referenced - set(yaml.safe_load((_DATA / "parameters.yaml").read_text()))
    assert not missing, sorted(missing)


_LAUNCHER = yaml.safe_load(_TEMPLATE_TEXT)["environment_configs"]["Skypilot"][
    "launchers"
]["fms_hf_tuning"]["config"]
_COMPUTE_DEFAULT = yaml.safe_load(_TEMPLATE_TEXT)["config"]["compute_config"]


def _substituted_targets(name):
    # The targets as a user gets them: $${NAME} replaced by parameters.yaml.
    text = (_DATA / name).read_text()
    params = yaml.safe_load((_DATA / "parameters.yaml").read_text())
    text = _PARAM_RE.sub(lambda m: str(params[m.group(1)]), text)
    return list(yaml.safe_load(text)["granite.build"]["targets"].values())


def _bindings(target):
    return {n: {"binding": {"path": f"/staged/{n}"}} for n in target["inputs"]}


def _inner_filled(target):
    # gbserver's non-strict inner config fill, run before the launcher fill.
    bindings = _bindings(target)
    config = fill_objtemplate(
        _step(target)["config"], {"bindings": bindings}, strict=False
    )
    return config, bindings


@pytest.mark.parametrize("build,i", _cases())
@pytest.mark.parametrize("block", ["setup", "run"])
def test_every_target_passes_the_templates_validation(build, i, block):
    config, bindings = _inner_filled(_substituted_targets(build)[i])
    cfg = {
        **_DEFAULTS,
        **config["fms_hf_tuning_config"],
        "image_id": "docker:quay.io/x/img:t",
    }
    ctx = {
        "config": {
            "fms_hf_tuning_config": cfg,
            "compute_config": config.get("compute_config", _COMPUTE_DEFAULT),
            "launcher_config": config["launcher_config"],
        },
        "bindings": bindings,
    }
    try:
        fill_template(_LAUNCHER[block], ctx, strict=True)
    except RuntimeError as exc:
        pytest.fail(f"{build} target {i} {block}: {exc.__cause__}")


def test_inner_fill_renders_raw_templates_and_bindings():
    for target in _targets("slurm-bluevela/build.yaml"):
        config, _ = _inner_filled(target)
        tmpl = config["fms_hf_tuning_config"]["args"]["chat_template"]
        assert "{% raw %}" not in tmpl
        assert tmpl.startswith("{% for message in messages %}")
    build = yaml.safe_load((_DATA / "slurm-bluevela/build.yaml").read_text())
    full, _ = _inner_filled(build["granite.build"]["targets"]["full-fsdp"])
    paths = full["fms_hf_tuning_config"]["data_config"]["datasets"][0]["data_paths"]
    assert paths == ["/staged/capybara"]
