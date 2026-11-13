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

"""Render tests for step-template.yaml with gbserver's strict renderer.

A missing key or a raise_error fails here as it would at step creation. The
rendered blocks then run under bash with the /opt/fms-hf-tuning scripts swapped
for argv printers. Mode 1 only.
"""

import base64
import json
import pathlib
import re
import subprocess

import pytest
import yaml

from gbserver.utils.template import fill_template

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_TEXT = (_STEP_DIR / "step-template.yaml").read_text()
_TEMPLATE = yaml.safe_load(_TEXT)
_SKY = _TEMPLATE["environment_configs"]["Skypilot"]
_LAUNCHER = _SKY["launchers"]["fms_hf_tuning"]["config"]
_DEFAULTS = _TEMPLATE["config"]["fms_hf_tuning_config"]
_BASE = {"base_model": {"binding": {"path": "/staged/base it's"}}}
_DS = dict(_BASE, dataset={"binding": {"path": "/staged/ds"}})
_SCRIPTS = ("fms_hf_tuning_guard.sh", "fms_hf_tuning_setup.sh", "fms_hf_tuning_run.sh")
_GPU = object()
_DC = {"datasets": [{"name": "d", "data_paths": ["/staged/ds"]}]}


def _ctx(bindings=None, compute=None, launcher_config=_GPU, **over):
    cfg = dict(
        _DEFAULTS, image_id="docker:quay.io/x/img:t", dataset_name="trl-lib/Capybara"
    )
    cfg.update(over)
    if launcher_config is _GPU:
        launcher_config = {"resources": {"accelerators": "A10G:2"}}
    config = {
        "fms_hf_tuning_config": cfg,
        "compute_config": compute or {"num_nodes": 1},
    }
    if launcher_config is not None:
        config["launcher_config"] = launcher_config
    return {"config": config, "bindings": _BASE if bindings is None else bindings}


def _render(block, ctx):
    return fill_template(_LAUNCHER[block], ctx, strict=True)


def _render_error(block, ctx) -> str:
    with pytest.raises(RuntimeError) as exc:
        _render(block, ctx)
    return str(exc.value.__cause__)


def _argv(block, ctx, measure):
    text = _render(block, ctx)
    for s in _SCRIPTS:
        stub = (
            'f() { for a in "$@"; do echo "ARG:$a"; done; }; f'
            if s == measure
            else "true"
        )
        text = text.replace(f"bash /opt/fms-hf-tuning/{s}", stub)
    proc = subprocess.run(
        ["bash", "-c", text], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return [ln[4:] for ln in proc.stdout.splitlines() if ln.startswith("ARG:")]


def _opt(argv, name):
    return argv[argv.index(name) + 1]


def _b64(argv, name):
    return json.loads(base64.b64decode(_opt(argv, name)))


def _owned():
    return set(
        re.findall(r'"([a-z_0-9]+)"', re.findall(r"set owned = \[(.*?)\]", _TEXT)[0])
    )


class TestContract:
    def test_identity(self):
        assert _TEMPLATE["name"] == "fms-hf-tuning"
        assert _SKY["default_launcher"] == "fms_hf_tuning"

    def test_io_schema(self):
        assert set(_TEMPLATE["inputs"]["required"]) == {"base_model"}
        assert set(_TEMPLATE["inputs"]["optional"]) == {"dataset"}
        assert _TEMPLATE["outputs"]["optional"]["model"]["type"] == "model"
        assert _TEMPLATE["inputs"]["allow_unknown"]
        assert _TEMPLATE["outputs"]["allow_unknown"]

    def test_defaults(self):
        assert _DEFAULTS["peft_method"] == "none"
        assert _DEFAULTS["distributed"] == "ddp"
        assert _DEFAULTS["output"] == "model"
        assert _DEFAULTS["args"] == {} and _DEFAULTS["data_config"] == {}
        assert _DEFAULTS["image_id"] == "docker:${IMAGE_REF}"
        assert _TEMPLATE["config"]["compute_config"] == {"num_nodes": 1}

    def test_shared_monitor_with_step_progress(self):
        mon = _SKY["monitors"]["skypilot_monitor"]
        assert mon["ref"] == "space://monitors/skypilot"
        assert "event_configs" not in mon["config"]
        rule = mon["config"]["extra_event_configs"][0]
        assert rule["event_type"] == "WORKLOAD_STATUS_EVENT"
        line = "Step: 3/10 loss=0.7012"
        # gbserver narrows the line to the line_regex match before field regexes run.
        assert re.search(rule["line_regex"], line)[0] == line
        msg = [f for f in rule["event_fields"] if f["field_name"] == "message"][0]
        assert re.search(msg["field_regex"], line)[0] == line

    def test_both_blocks_carry_the_same_validate_macro(self):
        def macro(block):
            text = _LAUNCHER[block]
            return text[
                text.index("{%- macro validate") : text.index(
                    "{%- endmacro %}\n{%- set c"
                )
            ]

        assert macro("setup") == macro("run")

    def test_no_gpu_request_in_the_template(self):
        assert _LAUNCHER["resources"] == {}

    def test_owned_matches_job_script(self):
        import fms_hf_tuning_job as job  # src/ is on PYTHONPATH

        assert _owned() == set(job.OWNED)


class TestImageId:
    @pytest.mark.parametrize(
        "given,want",
        [
            ("docker:quay.io/x/i:t", "docker:quay.io/x/i:t"),
            ("quay.io/x/i:t", "docker:quay.io/x/i:t"),
            ("/proj/cache/i.sqsh", "docker:/proj/cache/i.sqsh"),
        ],
    )
    def test_docker_prefix(self, given, want):
        assert _render("image_id", _ctx(image_id=given)) == want


class TestValidation:
    @pytest.mark.parametrize("block", ["setup", "run"])
    @pytest.mark.parametrize(
        "over,needle",
        [
            ({"peft_method": "qlora"}, "peft_method must be none, lora or alora"),
            ({"distributed": "deepspeed"}, "distributed must be ddp or fsdp"),
            (
                {"clearml_enabled": "maybe"},
                "clearml_enabled must be auto, true or false",
            ),
            ({"args": [1]}, "args must be a mapping"),
            ({"data_config": [1], "dataset_name": ""}, "data_config must be a mapping"),
            (
                {"data_config": _DC},
                "data_config or fms_hf_tuning_config.dataset_name, not both",
            ),
            ({"dataset_name": ""}, "no training data"),
            ({"peft_method": "alora"}, "needs args.alora_invocation_string"),
            ({"args": {"alora_invocation_string": "x"}}, "only for peft_method: alora"),
            ({"base_model_id": "org/m"}, "does not apply to peft_method: none"),
            ({"output": ""}, "output is required"),
            ({"output": "a b"}, "whitespace or a double quote"),
            ({"output": 'a"b'}, "whitespace or a double quote"),
            ({"args": {"--lr": 1}}, "must be a bare fms-hf-tuning argument name"),
            ({"args": {"lr=1": 1}}, "must be a bare fms-hf-tuning argument name"),
            ({"args": {"a b": 1}}, "must be a bare fms-hf-tuning argument name"),
            ({"args": {"output_dir": "/x"}}, "args.output_dir is set by the step"),
            (
                {"args": {"save-model-dir": "/x"}},
                "args.save-model-dir is set by the step",
            ),
        ],
    )
    def test_config_errors(self, block, over, needle):
        msg = _render_error(block, _ctx(**over))
        assert msg.startswith("fms-hf-tuning: ")
        assert needle in msg

    @pytest.mark.parametrize("key", sorted(_owned()))
    def test_every_owned_key_refused(self, key):
        assert "is set by the step" in _render_error("setup", _ctx(args={key: "x"}))

    @pytest.mark.parametrize("block", ["setup", "run"])
    @pytest.mark.parametrize(
        "key",
        [
            "push_to_hub_token",
            "push_to_hub_model_id",
            "push_to_hub_organization",
            "use_cpu",
            "no_cuda",
        ],
    )
    def test_aliases_of_owned_keys_refused(self, block, key):
        msg = _render_error(block, _ctx(args={key: "x"}))
        assert f"args.{key} is set by the step" in msg

    def test_bound_dataset_and_dataset_name(self):
        assert "use one" in _render_error("setup", _ctx(bindings=_DS))

    def test_bound_dataset_with_data_config_is_fine(self):
        _render("setup", _ctx(bindings=_DS, dataset_name="", data_config=_DC))

    @pytest.mark.parametrize("block", ["setup", "run"])
    @pytest.mark.parametrize(
        "flag, value",
        [
            ("packing", True),
            # Every spelling transformers' string_to_bool reads as true.
            *[("packing", v) for v in ["true", "TRUE", "1", "yes", "Y", "t", "T"]],
            ("packing", 1),
            # padding_free takes a list of methods (nargs="+"), e.g. [huggingface].
            ("padding_free", ["huggingface"]),
            ("padding_free", "huggingface"),
            ("padding_free", True),
        ],
    )
    def test_alora_refuses_packing(self, block, flag, value):
        msg = _render_error(
            block,
            _ctx(
                peft_method="alora", args={"alora_invocation_string": "x", flag: value}
            ),
        )
        assert "cannot train with packing or padding_free" in msg

    @pytest.mark.parametrize("block", ["setup", "run"])
    @pytest.mark.parametrize(
        "flag, value",
        [
            ("packing", False),
            *[("packing", v) for v in ["false", "0", "no", "n", "f", "F"]],
            ("packing", 0),
            ("packing", None),
            ("padding_free", None),
        ],
    )
    def test_alora_accepts_packing_off(self, block, flag, value):
        _render(
            block,
            _ctx(
                peft_method="alora",
                args={"alora_invocation_string": "x", flag: value},
            ),
        )

    @pytest.mark.parametrize("value", ["t", "y", ["huggingface"]])
    def test_other_methods_may_pack(self, value):
        key = "padding_free" if isinstance(value, list) else "packing"
        _render("setup", _ctx(peft_method="lora", args={key: value}))

    @pytest.mark.parametrize("block", ["setup", "run"])
    @pytest.mark.parametrize("layer", ["compute", "launcher"])
    @pytest.mark.parametrize("n", [1, 1.0, "1", "01", " 1", "+1", "1 "])
    def test_whole_one_num_nodes_is_fine(self, block, layer, n):
        # gbserver's _num_nodes_from_configs takes int(value): a whole float
        # (YAML 1.0) and any string int() reads as 1 (a substituted parameter).
        if layer == "compute":
            ctx = _ctx(compute={"num_nodes": n})
        else:
            ctx = _ctx(
                launcher_config={
                    "resources": {"accelerators": "A10G:1"},
                    "num_nodes": n,
                }
            )
        _render(block, ctx)

    @pytest.mark.parametrize("block", ["setup", "run"])
    @pytest.mark.parametrize("n", ["2", "02", "1.0", "0x1", "", "${NODES}"])
    def test_string_num_nodes_other_than_one_refused(self, block, n):
        # "1.0" and "0x1" fail gbserver's int() too; "${NODES}" is an
        # unsubstituted parameter.
        lc = {"resources": {"accelerators": "A10G:1"}, "num_nodes": n}
        assert _render_error(block, _ctx(launcher_config=lc)) == (
            f"fms-hf-tuning: launcher_config.num_nodes must be 1 (got '{n}'); "
            "this step is single-node."
        )

    def test_num_nodes(self):
        assert "compute_config.num_nodes must be 1" in _render_error(
            "setup", _ctx(compute={"num_nodes": 2})
        )
        assert "launcher_config.num_nodes must be 1" in _render_error(
            "setup",
            _ctx(
                launcher_config={
                    "num_nodes": 2,
                    "resources": {"accelerators": "A10G:1"},
                }
            ),
        )
        _render("setup", _ctx(compute={"num_nodes": 1.0}))

    @pytest.mark.parametrize("lc", [None, {}, {"resources": {}}])
    def test_no_gpu_request(self, lc):
        assert "no GPU request" in _render_error("setup", _ctx(launcher_config=lc))

    def test_instance_type_is_a_gpu_request(self):
        _render(
            "setup", _ctx(launcher_config={"resources": {"instance_type": "g5.xlarge"}})
        )

    @pytest.mark.parametrize("acc", ["H200:1", "H200", "A10G: 1"])
    def test_fsdp_needs_two_gpus(self, acc):
        msg = _render_error(
            "setup",
            _ctx(
                distributed="fsdp", launcher_config={"resources": {"accelerators": acc}}
            ),
        )
        assert "fsdp needs at least 2 GPUs" in msg

    def test_fsdp_on_two_gpus(self):
        _render(
            "setup",
            _ctx(
                distributed="fsdp",
                launcher_config={"resources": {"accelerators": "H200:2"}},
            ),
        )

    @pytest.mark.parametrize("method", ["lora", "alora"])
    def test_base_model_id_for_adapters(self, method):
        args = {"alora_invocation_string": "x"} if method == "alora" else {}
        _render("setup", _ctx(peft_method=method, base_model_id="org/m", args=args))


class TestArgv:
    def test_guard_argv_in_both_blocks(self):
        for block in ("setup", "run"):
            argv = _argv(
                block, _ctx(bindings=_DS, dataset_name=""), "fms_hf_tuning_guard.sh"
            )
            assert argv == [
                "--base-model",
                "/staged/base it's",
                "--dataset",
                "/staged/ds",
            ]

    def test_setup_runs_the_image_check(self):
        assert _argv("setup", _ctx(), "fms_hf_tuning_setup.sh") == []
        assert "bash /opt/fms-hf-tuning/fms_hf_tuning_setup.sh" in _render(
            "setup", _ctx()
        )

    def test_run_defaults(self):
        a = _argv("run", _ctx(), "fms_hf_tuning_run.sh")
        assert _opt(a, "--base-model") == "/staged/base it's"
        assert _opt(a, "--output-id") == "model"
        assert _opt(a, "--output-path") == "./model_output"
        assert _opt(a, "--peft-method") == "none"
        assert _opt(a, "--distributed") == "ddp"
        assert _opt(a, "--data-path") == "trl-lib/Capybara"
        assert _b64(a, "--data-config-b64") == {}
        assert _b64(a, "--args-b64") == {}
        assert _opt(a, "--base-model-id") == ""
        assert _opt(a, "--clearml-mode") == "auto"
        assert _opt(a, "--clearml-project") == ""
        assert _opt(a, "--clearml-task") == ""

    def test_bound_dataset_is_the_data_path(self):
        a = _argv("run", _ctx(bindings=_DS, dataset_name=""), "fms_hf_tuning_run.sh")
        assert _opt(a, "--data-path") == "/staged/ds"

    def test_data_config_empties_the_data_path(self):
        a = _argv(
            "run",
            _ctx(bindings=_DS, dataset_name="", data_config=_DC),
            "fms_hf_tuning_run.sh",
        )
        assert _opt(a, "--data-path") == ""
        assert _b64(a, "--data-config-b64") == _DC

    def test_args_round_trip_byte_for_byte(self):
        args = {
            "response_template": "\n### Answer:",
            "chat_template": "{% for m in messages %}{{ m['content'] }}{% endfor %}",
            "quote": 'it\'s "q" ü',
            "learning_rate": 1e-05,
            "target_modules": ["q_proj", "v_proj"],
            "packing": False,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "seed": None,
        }
        a = _argv("run", _ctx(args=args), "fms_hf_tuning_run.sh")
        assert _b64(a, "--args-b64") == args

    @pytest.mark.parametrize(
        "given,mode",
        [
            ("auto", "auto"),
            (True, "on"),
            ("on", "on"),
            (1, "on"),
            (False, "off"),
            ("off", "off"),
            ("0", "off"),
        ],
    )
    def test_clearml_mode(self, given, mode):
        a = _argv("run", _ctx(clearml_enabled=given), "fms_hf_tuning_run.sh")
        assert _opt(a, "--clearml-mode") == mode

    def test_names_and_paths(self):
        a = _argv(
            "run",
            _ctx(
                peft_method="lora",
                distributed="fsdp",
                output="adapter",
                output_path="/shared/o it's",
                base_model_id="org/m",
                clearml_project="p",
                clearml_task_name="t",
            ),
            "fms_hf_tuning_run.sh",
        )
        assert _opt(a, "--peft-method") == "lora"
        assert _opt(a, "--distributed") == "fsdp"
        assert _opt(a, "--output-id") == "adapter"
        assert _opt(a, "--output-path") == "/shared/o it's"
        assert _opt(a, "--base-model-id") == "org/m"
        assert _opt(a, "--clearml-project") == "p"
        assert _opt(a, "--clearml-task") == "t"

    def test_null_values_render_empty(self):
        ctx = _ctx(peft_method="lora", base_model_id=None, clearml_project=None)
        text = _render("run", ctx)
        assert "--base-model-id ''" in text and "--clearml-project ''" in text
        assert "None" not in text
        a = _argv("run", ctx, "fms_hf_tuning_run.sh")
        assert _opt(a, "--base-model-id") == ""
        assert _opt(a, "--clearml-project") == ""

    def test_rendered_blocks_are_valid_bash(self):
        for block in ("setup", "run"):
            proc = subprocess.run(
                ["bash", "-n", "-c", _render(block, _ctx())],
                capture_output=True,
                text=True,
                check=False,
            )
            assert proc.returncode == 0, proc.stderr
