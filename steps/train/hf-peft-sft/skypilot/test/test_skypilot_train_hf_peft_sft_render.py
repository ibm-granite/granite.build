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
rendered blocks then run with the /opt/hf-peft-sft scripts swapped for argv
printers. Mode 1 only.
"""

import pathlib
import re
import subprocess

import hf_peft_sft_train as train
import pytest
import yaml

from gbserver.utils.template import fill_template

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_TEMPLATE = yaml.safe_load((_STEP_DIR / "step-template.yaml").read_text())
_LAUNCHER = _TEMPLATE["environment_configs"]["Skypilot"]["launchers"]["hf_peft_sft"][
    "config"
]
_DEFAULTS = _TEMPLATE["config"]["hf_peft_sft_config"]
_BASE = {"base_model": {"binding": {"path": "/staged/base it's"}}}
_DS = dict(_BASE, dataset={"binding": {"path": "/staged/ds"}})
_SCRIPTS = ("hf_peft_sft_guard.sh", "hf_peft_sft_setup.sh", "hf_peft_sft_run.sh")
_GPU_REQUEST = object()


def _ctx(bindings=None, compute=None, launcher_config=_GPU_REQUEST, **over):
    cfg = dict(
        _DEFAULTS, image_id="docker:quay.io/x/img:t", dataset_name="trl-lib/Capybara"
    )
    cfg.update(over)
    if launcher_config is _GPU_REQUEST:
        launcher_config = {"resources": {"accelerators": "A10G:2"}}
    config = {"hf_peft_sft_config": cfg, "compute_config": compute or {"num_nodes": 1}}
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
        text = text.replace(f"bash /opt/hf-peft-sft/{s}", stub)
    proc = subprocess.run(
        ["bash", "-c", text], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return [ln[4:] for ln in proc.stdout.splitlines() if ln.startswith("ARG:")]


def _opt(argv, name):
    return argv[argv.index(name) + 1]


def _tail(argv):
    return argv[argv.index("--") + 1 :]


def _owned():
    text = (_STEP_DIR / "step-template.yaml").read_text()
    return set(
        re.findall(r'"([a-z_0-9]+)"', re.findall(r"set owned = \[(.*?)\]", text)[0])
    )


class TestContract:
    def test_identity(self):
        assert _TEMPLATE["name"] == "hf-peft-sft"
        assert (
            _TEMPLATE["environment_configs"]["Skypilot"]["default_launcher"]
            == "hf_peft_sft"
        )

    def test_io_schema(self):
        assert set(_TEMPLATE["inputs"]["required"]) == {"base_model"}
        assert set(_TEMPLATE["inputs"]["optional"]) == {"dataset"}
        assert _TEMPLATE["outputs"]["optional"]["adapter"]["type"] == "model"
        assert (
            _TEMPLATE["inputs"]["allow_unknown"]
            and _TEMPLATE["outputs"]["allow_unknown"]
        )

    def test_uses_shared_monitor_with_step_progress(self):
        mon = _TEMPLATE["environment_configs"]["Skypilot"]["monitors"][
            "skypilot_monitor"
        ]
        assert mon["ref"] == "space://monitors/skypilot"
        assert "event_configs" not in mon["config"]
        assert re.match(
            mon["config"]["extra_event_configs"][0]["line_regex"],
            train.step_line(3, 9, {"loss": 1.0}),
        )

    def test_requests_no_gpus_itself(self):
        assert _LAUNCHER["resources"] == {}

    @pytest.mark.parametrize("name", ["validate", "q"])
    def test_macro_copies_are_identical(self, name):
        macro = re.compile(r"\{%- macro " + name + r"\(.*?\{%- endmacro %\}", re.S)
        setup = macro.findall(_LAUNCHER["setup"])
        assert len(setup) == 1
        assert setup == macro.findall(_LAUNCHER["run"])

    def test_template_carries_the_image_ref_token(self):
        assert _DEFAULTS["image_id"] == "docker:${IMAGE_REF}"

    def test_owned_covers_trainer_flags_and_the_run_scripts_flags(self):
        owned = _owned()
        assert set(train.STEP_ARG_FIELDS) <= owned
        assert {
            "output_dir",
            "report_to",
            "save_strategy",
            "bf16",
            "fp16",
            "push_to_hub",
            "deepspeed",
        } <= owned


class TestImage:
    @pytest.mark.parametrize(
        "given,expected",
        [
            ("docker:quay.io/x/img:t", "docker:quay.io/x/img:t"),
            ("quay.io/x/img:t", "docker:quay.io/x/img:t"),
            ("/proj/images/p.sqsh", "docker:/proj/images/p.sqsh"),
        ],
    )
    def test_docker_prefix_added_once(self, given, expected):
        assert _render("image_id", _ctx(image_id=given)) == expected


@pytest.mark.parametrize("block", ["setup", "run"])
def test_blocks_are_valid_bash(block):
    assert (
        subprocess.run(
            ["bash", "-n"], input=_render(block, _ctx()), text=True
        ).returncode
        == 0
    )


class TestValidation:
    @pytest.mark.parametrize("block", ["setup", "run"])
    def test_no_data_source(self, block):
        assert "no training data" in _render_error(block, _ctx(dataset_name=""))

    @pytest.mark.parametrize("block", ["setup", "run"])
    def test_two_data_sources(self, block):
        assert "use one" in _render_error(block, _ctx(bindings=_DS))

    def test_empty_split(self):
        assert "dataset_split" in _render_error("run", _ctx(dataset_split=""))

    @pytest.mark.parametrize(
        "field,bad",
        [
            ("method", "dora"),
            ("distributed", "tp"),
            ("mixed_precision", "fp8"),
            ("clearml_enabled", "maybe"),
        ],
    )
    def test_bad_enum(self, field, bad):
        assert f"hf_peft_sft_config.{field}" in _render_error(
            "run", _ctx(**{field: bad})
        )

    def test_alora_needs_invocation(self):
        assert "alora_invocation_string is required" in _render_error(
            "run", _ctx(method="alora")
        )

    def test_invocation_without_alora_refused(self):
        assert "only for method: alora" in _render_error(
            "run", _ctx(alora_invocation_string="x")
        )

    @pytest.mark.parametrize("flag", ["packing", "padding_free", "padding-free"])
    def test_alora_refuses_packing(self, flag):
        err = _render_error(
            "run", _ctx(method="alora", alora_invocation_string="x", args={flag: True})
        )
        assert "packing" in err

    def test_alora_packing_false_is_fine(self):
        _render(
            "run",
            _ctx(method="alora", alora_invocation_string="x", args={"packing": False}),
        )

    def test_qlora_refuses_fp16(self):
        assert "fp16" in _render_error(
            "run", _ctx(method="qlora", mixed_precision="fp16")
        )

    @pytest.mark.parametrize("bad", ["", "my adapter", 'a"b'])
    def test_bad_output(self, bad):
        assert "hf_peft_sft_config.output" in _render_error("run", _ctx(output=bad))

    def test_multi_node_refused(self):
        assert "num_nodes must be 1" in _render_error(
            "run", _ctx(compute={"num_nodes": 2})
        )

    @pytest.mark.parametrize("block", ["setup", "run"])
    @pytest.mark.parametrize(
        "launcher_config",
        [
            None,
            {},
            {"resources": None},
            {"resources": {}},
            {"resources": {"accelerators": ""}},
        ],
    )
    def test_no_gpu_request_refused(self, block, launcher_config):
        err = _render_error(block, _ctx(launcher_config=launcher_config))
        assert "no GPU request" in err

    @pytest.mark.parametrize("acc", ["A10G", "A10G:1", "H200:1"])
    @pytest.mark.parametrize("dist", ["fsdp", "deepspeed"])
    def test_sharding_on_one_gpu_refused(self, acc, dist):
        err = _render_error(
            "run",
            _ctx(
                distributed=dist, launcher_config={"resources": {"accelerators": acc}}
            ),
        )
        assert "needs at least 2 GPUs" in err

    @pytest.mark.parametrize(
        "resources",
        [
            {"accelerators": "H200:2"},
            {"instance_type": "p4d.24xlarge"},
            {"accelerators": {"A100": 2}},
        ],
    )
    def test_sharding_check_skipped_when_count_unknown_or_large(self, resources):
        _render(
            "run", _ctx(distributed="fsdp", launcher_config={"resources": resources})
        )

    @pytest.mark.parametrize(
        "lora,msg",
        [
            ("x", "lora must be a mapping"),
            ({"r": 0}, "lora.r"),
            ({"alpha": -1}, "lora.alpha"),
            ({"r": "8"}, "lora.r"),
            ({"dropout": 1.0}, "lora.dropout"),
            ({"target_modules": []}, "lora.target_modules"),
            ({"target_modules": "q_proj"}, "lora.target_modules"),
            ({"rank": 8}, "unknown key lora.rank"),
            ({"modules_to_save": "lm_head"}, "lora.modules_to_save"),
        ],
    )
    def test_bad_lora(self, lora, msg):
        merged = dict(_DEFAULTS["lora"], **lora) if isinstance(lora, dict) else lora
        assert msg in _render_error("run", _ctx(lora=merged))

    @pytest.mark.parametrize(
        "key",
        [
            "output_dir",
            "report_to",
            "save_strategy",
            "bf16",
            "deepspeed",
            "method",
            "lora_r",
            "base_model_id",
        ],
    )
    def test_step_owned_args_refused(self, key):
        assert f"args.{key} is set by the step" in _render_error(
            "run", _ctx(args={key: "x"})
        )

    @pytest.mark.parametrize(
        "key", ["output-dir", "push-to-hub", "lora-target-modules"]
    )
    def test_step_owned_dash_aliases_refused(self, key):
        assert "is set by the step" in _render_error("run", _ctx(args={key: "x"}))

    def test_mapping_arg_refused(self):
        assert "scalar or a list" in _render_error("run", _ctx(args={"x": {"a": 1}}))

    def test_empty_list_arg_refused(self):
        assert "args.foo is an empty list" in _render_error(
            "run", _ctx(args={"foo": []})
        )


class TestGuardArgv:
    @pytest.mark.parametrize("block", ["setup", "run"])
    def test_paths_pass_verbatim(self, block):
        assert _argv(block, _ctx(), "hf_peft_sft_guard.sh") == [
            "--base-model",
            "/staged/base it's",
            "--dataset",
            "",
        ]

    def test_bound_dataset_reaches_guard(self):
        argv = _argv(
            "setup", _ctx(bindings=_DS, dataset_name=""), "hf_peft_sft_guard.sh"
        )
        assert _opt(argv, "--dataset") == "/staged/ds"


class TestRunArgv:
    def test_defaults(self):
        argv = _argv("run", _ctx(), "hf_peft_sft_run.sh")
        assert argv[: argv.index("--")] == [
            "--base-model",
            "/staged/base it's",
            "--output-id",
            "adapter",
            "--output-path",
            "./adapter_output",
            "--method",
            "lora",
            "--distributed",
            "ddp",
            "--mixed-precision",
            "bf16",
            "--clearml-mode",
            "auto",
            "--clearml-project",
            "",
            "--clearml-task",
            "",
        ]
        assert _tail(argv) == [
            "--dataset_path",
            "trl-lib/Capybara",
            "--dataset_split",
            "train",
            "--lora_r",
            "16",
            "--lora_alpha",
            "32",
            "--lora_dropout",
            "0.05",
            "--lora_target_modules",
            "all-linear",
        ]

    def test_bound_dataset_is_the_path(self):
        tail = _tail(
            _argv("run", _ctx(bindings=_DS, dataset_name=""), "hf_peft_sft_run.sh")
        )
        assert tail[:2] == ["--dataset_path", "/staged/ds"]

    def test_lora_lists_and_alora(self):
        ctx = _ctx(
            method="alora",
            alora_invocation_string="<|im_start|>assistant",
            base_model_id="org/m",
            lora=dict(
                _DEFAULTS["lora"],
                target_modules=["q_proj", "v_proj"],
                modules_to_save=["lm_head"],
            ),
        )
        tail = _tail(_argv("run", ctx, "hf_peft_sft_run.sh"))
        i = tail.index("--lora_target_modules")
        assert tail[i : i + 3] == ["--lora_target_modules", "q_proj", "v_proj"]
        assert tail[tail.index("--lora_modules_to_save") + 1] == "lm_head"
        assert _opt(tail, "--alora_invocation_string") == "<|im_start|>assistant"
        assert _opt(tail, "--base_model_id") == "org/m"

    def test_args_rendering_after_step_flags(self):
        args = {
            "max_steps": 3,
            "gradient_checkpointing": True,
            "packing": False,
            "seed": None,
            "x": ["a b", "c'd"],
        }
        tail = _tail(_argv("run", _ctx(args=args), "hf_peft_sft_run.sh"))
        rest = tail[tail.index("all-linear") + 1 :]
        assert rest == [
            "--max_steps",
            "3",
            "--gradient_checkpointing",
            "true",
            "--packing",
            "false",
            "--x",
            "a b",
            "c'd",
        ]

    @pytest.mark.parametrize(
        "given,mode",
        [
            ("true", "on"),
            ("false", "off"),
            ("auto", "auto"),
            (True, "on"),
            (False, "off"),
        ],
    )
    def test_clearml_mode(self, given, mode):
        argv = _argv("run", _ctx(clearml_enabled=given), "hf_peft_sft_run.sh")
        assert _opt(argv, "--clearml-mode") == mode

    def test_output_path_and_clearml_names(self):
        argv = _argv(
            "run",
            _ctx(output_path="/shared/a", clearml_project="p q", clearml_task_name="t"),
            "hf_peft_sft_run.sh",
        )
        assert _opt(argv, "--output-path") == "/shared/a"
        assert _opt(argv, "--clearml-project") == "p q"
        assert _opt(argv, "--clearml-task") == "t"
