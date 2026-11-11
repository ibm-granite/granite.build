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

import importlib.util
import pathlib
import re
import subprocess

import hf_peft_sft_train as train
import pytest
import yaml

from gbserver.environment.skypilot import validate_launch_num_nodes
from gbserver.types.stepconfig import StepEnvironmentTypeConfig
from gbserver.utils.template import fill_template

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_SOURCE = (_STEP_DIR / "step-template.yaml").read_text()
_VALIDATE = (_STEP_DIR / "macros" / "validate.j2").read_text()


def _expanded_template() -> dict:
    """The template as `make space` renders it: gb-include lines expanded."""
    script = _STEP_DIR.parents[2] / "scripts" / "render_step_template.py"
    spec = importlib.util.spec_from_file_location("render_step_template", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return yaml.safe_load(module.expand(_STEP_DIR / "step-template.yaml"))


_TEMPLATE = _expanded_template()
_LAUNCHER = _TEMPLATE["environment_configs"]["Skypilot"]["launchers"]["hf_peft_sft"][
    "config"
]
_DEFAULTS = _TEMPLATE["config"]["hf_peft_sft_config"]
_ENV = StepEnvironmentTypeConfig.model_validate(
    _TEMPLATE["environment_configs"]["Skypilot"]
)
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
    text = _VALIDATE
    return set(
        re.findall(r'"([a-z_0-9]+)"', re.findall(r"set owned = \[(.*?)\]", text)[0])
    )


def _exempt():
    text = _VALIDATE
    found = re.findall(r"set full_flags = \[(.*?)\]", text)
    return set(re.findall(r'"([a-z_0-9]+)"', found[0])) if found else set()


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

    def test_step_progress_message_is_the_whole_line(self):
        # gbserver narrows the log line to the line_regex match before the
        # field regexes run, so line_regex must span the line for the message
        # to carry the total and the loss.
        rule = _TEMPLATE["environment_configs"]["Skypilot"]["monitors"][
            "skypilot_monitor"
        ]["config"]["extra_event_configs"][0]
        line = train.step_line(3, 10, {"loss": 0.70123})
        matched = re.match(rule["line_regex"], line)[0]
        field = next(f for f in rule["event_fields"] if f["field_name"] == "message")
        assert re.search(field["field_regex"], matched)[0] == line

    def test_requests_no_gpus_itself(self):
        assert _LAUNCHER["resources"] == {}

    @pytest.mark.parametrize("block", ["setup", "run"])
    def test_blocks_include_the_shared_macros(self, block):
        source = yaml.safe_load(_SOURCE)["environment_configs"]["Skypilot"][
            "launchers"
        ]["hf_peft_sft"]["config"][block]
        for marker in (
            "{#- gb-include ../../shared/trainer_args.j2 -#}",
            "{#- gb-include macros/validate.j2 -#}",
        ):
            assert source.count(marker) == 1, marker
        assert "{%- macro" not in _SOURCE

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

    @pytest.mark.parametrize("block", ["setup", "run"])
    def test_alora_check_runs_after_the_shared_args_checks(self, block):
        err = _render_error(
            block,
            _ctx(method="alora", alora_invocation_string="x", args={"packing": []}),
        )
        assert "is an empty list" in err
        assert "cannot train with packing" not in err

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

    @pytest.mark.parametrize("block", ["setup", "run"])
    @pytest.mark.parametrize("dist", ["fsdp", "deepspeed"])
    def test_qlora_sharded_refuses_fp32(self, block, dist):
        err = _render_error(
            block, _ctx(method="qlora", distributed=dist, mixed_precision="no")
        )
        assert err == (
            f"hf-peft-sft: method: qlora with distributed: {dist} stores the sharded "
            "4-bit weights in bf16, so mixed_precision must be bf16 (got 'no'). Use "
            "mixed_precision: bf16, or distributed: ddp for fp32 compute."
        )

    def test_qlora_ddp_fp32_is_fine(self):
        _render("run", _ctx(method="qlora", distributed="ddp", mixed_precision="no"))

    @pytest.mark.parametrize("dist", ["fsdp", "deepspeed"])
    def test_qlora_sharded_bf16_is_fine(self, dist):
        _render("run", _ctx(method="qlora", distributed=dist, mixed_precision="bf16"))

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

    @pytest.mark.parametrize("block", ["setup", "run"])
    @pytest.mark.parametrize(
        "tm", [["all-linear", "lm_head"], ["all-linear"], ["q_proj", "all-linear"]]
    )
    def test_all_linear_in_a_list_refused(self, block, tm):
        err = _render_error(
            block, _ctx(lora=dict(_DEFAULTS["lora"], target_modules=tm))
        )
        assert err == (
            "hf-peft-sft: hf_peft_sft_config.lora.target_modules: all-linear is a "
            "peft keyword, not a module name; use target_modules: all-linear on its "
            "own, or list module names (with lora.modules_to_save for extra "
            "trainable modules like lm_head)."
        )

    def test_module_name_list_is_fine(self):
        _render(
            "run",
            _ctx(lora=dict(_DEFAULTS["lora"], target_modules=["q_proj", "v_proj"])),
        )

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
            "dataset_config",
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

    @pytest.mark.parametrize("block", ["setup", "run"])
    @pytest.mark.parametrize(
        "key,owned",
        [
            ("output_d", "output_dir"),
            ("push_to", "push_to_hub"),
            ("report_", "report_to"),
            ("meth", "method"),
            ("hub_tok", "hub_token"),
            ("output-d", "output_dir"),
        ],
    )
    def test_abbreviated_owned_args_refused(self, block, key, owned):
        assert _render_error(block, _ctx(args={key: "x"})) == (
            f"hf-peft-sft: hf_peft_sft_config.args.{key} abbreviates the step-owned "
            f"flag {owned}; abbreviated flags are not accepted."
        )

    @pytest.mark.parametrize(
        "key", ["completion_only_loss", "learning_rate", "dataset_text_field"]
    )
    def test_non_owned_non_prefix_args_are_fine(self, key):
        # Real SFTConfig fields (trl 1.17.1) that are not a prefix of an owned
        # flag; dataset_text_field shares the dataset_ stem with owned flags.
        _render("run", _ctx(args={key: "x"}))

    @pytest.mark.parametrize(
        "key", ["gradient_checkpointing", "gradient-checkpointing"]
    )
    def test_full_flag_that_prefixes_an_owned_one_is_fine(self, key):
        # argparse takes an exact match before any prefix, so this is the
        # SFTConfig flag itself, not an abbreviation of gradient_checkpointing_kwargs.
        _render("run", _ctx(args={key: False}))

    def test_full_flags_are_not_owned(self):
        assert _exempt() and not _exempt() & _owned()

    @pytest.mark.parametrize("block", ["setup", "run"])
    @pytest.mark.parametrize(
        "key", ["report_to=wandb", "--report_to", "-x", "report to", "", "a\tb"]
    )
    def test_non_bare_args_key_refused(self, block, key):
        # '--report_to=wandb' would set report_to even with allow_abbrev=False.
        assert _render_error(block, _ctx(args={key: "--use_cpu"})) == (
            f"hf-peft-sft: hf_peft_sft_config.args key '{key}' must be a bare "
            "SFTConfig field name (no '=', '-' prefix or whitespace)."
        )

    @pytest.mark.parametrize("key", ["max_steps", "max-steps", "learning_rate"])
    def test_bare_args_key_renders(self, key):
        tail = _tail(_argv("run", _ctx(args={key: 3}), "hf_peft_sft_run.sh"))
        assert tail[-2:] == [f"--{key}", "3"]

    def test_mapping_arg_refused(self):
        assert "scalar or a list" in _render_error("run", _ctx(args={"x": {"a": 1}}))

    def test_empty_list_arg_refused(self):
        assert "args.foo is an empty list" in _render_error(
            "run", _ctx(args={"foo": []})
        )


class TestSingleNode:
    """gbserver, not the template, refuses multi-node: the step declares
    single_node and the num_nodes resolver enforces it at step creation, on
    every layer it reads."""

    def test_declared(self):
        assert _ENV.single_node is True

    @pytest.mark.parametrize(
        "config",
        [
            {"compute_config": {"num_nodes": 2}},
            {"compute_config": {"num_nodes": 1}, "launcher_config": {"num_nodes": 2}},
        ],
    )
    def test_gbserver_refuses_more_than_one_node(self, config):
        with pytest.raises(ValueError, match="this step is single-node"):
            validate_launch_num_nodes(_ENV, _LAUNCHER, config)

    @pytest.mark.parametrize("n", [1, 1.0, "1", "01"])
    def test_gbserver_accepts_one_node(self, n):
        assert (
            validate_launch_num_nodes(
                _ENV, _LAUNCHER, {"compute_config": {"num_nodes": n}}
            )
            == 1
        )

    @pytest.mark.parametrize("block", ["setup", "run"])
    def test_template_leaves_num_nodes_to_gbserver(self, block):
        lc = {"resources": {"accelerators": "A10G:2"}, "num_nodes": 2}
        assert "hf_peft_sft_guard.sh" in _render(
            block, _ctx(compute={"num_nodes": 2}, launcher_config=lc)
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
            "--dataset_config",
            "",
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

    @pytest.mark.parametrize("bindings", [None, _DS])
    def test_dataset_config_reaches_the_trainer(self, bindings):
        ctx = _ctx(
            bindings=bindings,
            dataset_name="" if bindings else "HuggingFaceTB/smoltalk",
            dataset_config="everyday-conversations",
        )
        tail = _tail(_argv("run", ctx, "hf_peft_sft_run.sh"))
        assert _opt(tail, "--dataset_config") == "everyday-conversations"

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
