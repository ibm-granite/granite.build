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

"""Render tests for the openinstruct-sft step's step-template.yaml.

Renders the launcher blocks with gbserver's production strict renderer (so a
missing key or a raise_error fails here, as at step creation), then runs them with
the /opt/openinstruct-sft scripts swapped for argv printers. Mode 1 only.
"""

import pathlib
import re
import subprocess

import pytest
import yaml

from gbserver.environment.environment import Environment, EventLogLineParserConfig
from gbserver.environment.skypilot import validate_launch_num_nodes
from gbserver.types.buildevent import BuildEventType, EntityRunMetadata
from gbserver.types.stepconfig import StepEnvironmentTypeConfig
from gbserver.utils.template import fill_template

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_TEMPLATE = yaml.safe_load((_STEP_DIR / "step-template.yaml").read_text())
_LAUNCHER = _TEMPLATE["environment_configs"]["Skypilot"]["launchers"][
    "openinstruct_sft"
]["config"]
_DEFAULTS = _TEMPLATE["config"]["openinstruct_sft_config"]
_ENV = StepEnvironmentTypeConfig.model_validate(
    _TEMPLATE["environment_configs"]["Skypilot"]
)
_MIX = {"allenai/tulu-3-sft-personas-algebra": 64}
_BASE = {"base_model": {"binding": {"path": "/staged/base it's"}}}
_SCRIPTS = (
    "openinstruct_sft_guard.sh",
    "openinstruct_sft_setup.sh",
    "openinstruct_sft_run.sh",
)


# _ctx's default: a real build's GPU request, merged into config.launcher_config
# by gbserver. Pass launcher_config=None for a build that sets none.
_GPU_REQUEST = object()


def _ctx(bindings=None, compute=None, launcher_config=_GPU_REQUEST, **over):
    cfg = dict(_DEFAULTS, image_id="docker:quay.io/x/img:t", dataset_mixer_list=_MIX)
    cfg.update(over)
    if launcher_config is _GPU_REQUEST:
        launcher_config = {"resources": {"accelerators": "A10G:1"}}
    config = {
        "openinstruct_sft_config": cfg,
        "compute_config": compute or {"num_nodes": 1},
    }
    if launcher_config is not None:
        config["launcher_config"] = launcher_config
    return {
        "config": config,
        "bindings": _BASE if bindings is None else bindings,
    }


def _render(block, ctx):
    return fill_template(_LAUNCHER[block], ctx, strict=True)


def _render_error(block, ctx) -> str:
    with pytest.raises(RuntimeError) as exc:
        _render(block, ctx)
    return str(exc.value.__cause__)


def _argv(block, ctx, measure):
    """argv the rendered block passes to one /opt script (others are no-ops)."""
    text = _render(block, ctx)
    for s in _SCRIPTS:
        stub = (
            'f() { for a in "$@"; do echo "ARG:$a"; done; }; f'
            if s == measure
            else "true"
        )
        text = text.replace(f"bash /opt/openinstruct-sft/{s}", stub)
    proc = subprocess.run(
        ["bash", "-c", text], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return [ln[4:] for ln in proc.stdout.splitlines() if ln.startswith("ARG:")]


def _opt(argv, name):
    return argv[argv.index(name) + 1]


def _tail(argv):
    return argv[argv.index("--") + 1 :]


class TestContract:
    def test_identity(self):
        assert _TEMPLATE["name"] == "openinstruct-sft"
        assert (
            _TEMPLATE["environment_configs"]["Skypilot"]["default_launcher"]
            == "openinstruct_sft"
        )

    def test_io_schema(self):
        assert set(_TEMPLATE["inputs"]["required"]) == {"base_model"}
        assert set(_TEMPLATE["inputs"]["optional"]) == {"dataset"}
        assert _TEMPLATE["outputs"]["optional"]["model"]["type"] == "model"

    def test_uses_shared_monitor(self):
        mon = _TEMPLATE["environment_configs"]["Skypilot"]["monitors"][
            "skypilot_monitor"
        ]
        assert mon["ref"] == "space://monitors/skypilot"
        assert "event_configs" not in mon["config"]

    @pytest.mark.asyncio
    async def test_step_progress_message_carries_loss(self):
        # gbserver narrows the log line to the line_regex match before the field
        # regexes run, so a narrow line_regex leaves the message at "Step: 3".
        # The line: SkyPilot's worker prefix + open-instruct's logger_utils format
        # + finetune.py's progress f-string (open-instruct 26de9960).
        progress = "Step: 3, LR: 1.9e-05, Loss: 0.7012, TPS: 1234.5"
        line = (
            "(head, rank=0, pid=4242) 2026-11-03 12:00:00 - INFO - "
            f"finetune.py:949 -   {progress}"
        )
        rule = _TEMPLATE["environment_configs"]["Skypilot"]["monitors"][
            "skypilot_monitor"
        ]["config"]["extra_event_configs"][0]
        events = await Environment.get_events_from_log_line(
            line,
            [EventLogLineParserConfig(**rule)],
            entityrun_metadata=EntityRunMetadata(),
        )
        assert len(events) == 1
        assert events[0].type is BuildEventType.WORKLOAD_STATUS_EVENT
        assert events[0].payload.message == progress

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


class TestImage:
    @pytest.mark.parametrize(
        "given,expected",
        [
            ("docker:quay.io/x/img:t", "docker:quay.io/x/img:t"),
            ("quay.io/x/img:t", "docker:quay.io/x/img:t"),
            ("/proj/images/oi.sqsh", "docker:/proj/images/oi.sqsh"),
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
        assert "no training data" in _render_error(block, _ctx(dataset_mixer_list={}))

    @pytest.mark.parametrize("block", ["setup", "run"])
    def test_two_data_sources(self, block):
        b = dict(_BASE, dataset={"binding": {"path": "/staged/ds"}})
        assert "use one" in _render_error(block, _ctx(bindings=b))

    def test_mixer_must_be_mapping(self):
        assert "must map" in _render_error("run", _ctx(dataset_mixer_list=["a", "1.0"]))

    @pytest.mark.parametrize("bad", ["train", []])
    def test_splits_must_be_a_non_empty_list(self, bad):
        assert "dataset_mixer_list_splits must be a non-empty list" in _render_error(
            "run", _ctx(dataset_mixer_list_splits=bad)
        )

    @pytest.mark.parametrize("bad", ["", "my model", 'a"b'])
    def test_bad_output(self, bad):
        assert "openinstruct_sft_config.output" in _render_error(
            "run", _ctx(output=bad)
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
        # Without a GPU request SkyPilot gives a CPU node: finetune.py fails only
        # after provisioning and the image pull. Fail at step creation instead.
        err = _render_error(block, _ctx(launcher_config=launcher_config))
        assert "no GPU request" in err
        assert "accelerators" in err

    @pytest.mark.parametrize(
        "resources",
        [{"accelerators": "H200:1"}, {"instance_type": "g5.xlarge"}],
    )
    def test_gpu_request_accepted(self, resources):
        _render("run", _ctx(launcher_config={"resources": resources}))

    def test_deepspeed_config_is_a_file_name(self):
        assert "deepspeed_config" in _render_error(
            "run", _ctx(deepspeed_config="../x.conf")
        )

    def test_bad_clearml_mode(self):
        assert "clearml_enabled" in _render_error("run", _ctx(clearml_enabled="maybe"))

    @pytest.mark.parametrize(
        "key",
        [
            "output_dir",
            "model_name_or_path",
            "report_to",
            "push_to_hub",
            "dataset_local_cache_dir",
            # hf would push a dataset-mix-cached dataset to the Hub.
            "dataset_cache_mode",
        ],
    )
    def test_step_owned_args_refused(self, key):
        assert f"args.{key} is set by the step" in _render_error(
            "run", _ctx(args={key: "x"})
        )

    @pytest.mark.parametrize("key", ["output-dir", "push-to-hub", "model-name-or-path"])
    def test_step_owned_dash_aliases_refused(self, key):
        assert "is set by the step" in _render_error("run", _ctx(args={key: "x"}))

    @pytest.mark.parametrize(
        "key",
        [
            "no_push_to_hub",
            "no-push-to-hub",
            "no_try_launch_beaker_eval_jobs",
            "no_try_auto_save_to_beaker",
        ],
    )
    def test_no_complement_of_step_owned_refused(self, key):
        # HfArgumentParser adds --no_<f> for a bool field defaulting to True; it
        # writes the same field as --<f>.
        assert f"args.{key} is set by the step" in _render_error(
            "run", _ctx(args={key: True})
        )

    @pytest.mark.parametrize("block", ["setup", "run"])
    @pytest.mark.parametrize(
        "key,flag",
        [
            ("no_fused_optimizer", "fused_optimizer"),
            ("no-fused-optimizer", "fused-optimizer"),
            ("no_use_fast", "use_fast"),
            ("no_clean_checkpoints_at_end", "clean_checkpoints_at_end"),
        ],
    )
    def test_no_complement_keys_refused(self, block, key, flag):
        # HfArgumentParser's --no_<f> is store_false and takes no value, but args
        # always renders one: `--no_fused_optimizer true` fails the run on 'true'.
        assert _render_error(block, _ctx(args={key: True})) == (
            f"openinstruct-sft: openinstruct_sft_config.args.{key} is a no_* flag, "
            f"which takes no value; write {flag}: false instead."
        )

    @pytest.mark.parametrize("block", ["setup", "run"])
    @pytest.mark.parametrize(
        "key,owned",
        [
            ("output_d", "output_dir"),
            ("push_to", "push_to_hub"),
            ("report_", "report_to"),
            ("model_name", "model_name_or_path"),
            ("resume", "resume_from_checkpoint"),
            ("output-d", "output_dir"),
            ("no_push", "no_push_to_hub"),
            ("dataset_cache", "dataset_cache_mode"),
        ],
    )
    def test_abbreviated_owned_args_refused(self, block, key, owned):
        # argparse's allow_abbrev expands a unique prefix, so `--output_d` would
        # override the step's --output_dir.
        assert _render_error(block, _ctx(args={key: "x"})) == (
            f"openinstruct-sft: openinstruct_sft_config.args.{key} abbreviates the "
            f"step-owned flag {owned}; abbreviated flags are not accepted."
        )

    @pytest.mark.parametrize(
        "key,value,rendered",
        [
            ("dataset_transform_fn", ["sft_tulu_filter_v1"], ["sft_tulu_filter_v1"]),
            ("dataset_target_columns", ["input_ids"], ["input_ids"]),
            ("model_revision", "main", ["main"]),
            ("fused_optimizer", False, ["false"]),
        ],
    )
    def test_flags_sharing_an_owned_stem_are_fine(self, key, value, rendered):
        # Real finetune.py flags (open-instruct 26de9960) that start like an owned
        # flag but are not a prefix of one, with values finetune.py would parse.
        tail = _tail(_argv("run", _ctx(args={key: value}), "openinstruct_sft_run.sh"))
        assert tail[-1 - len(rendered) :] == [f"--{key}", *rendered]

    @pytest.mark.parametrize("block", ["setup", "run"])
    @pytest.mark.parametrize(
        "key", ["report_to=wandb", "--report_to", "-x", "report to", "", "a\tb"]
    )
    def test_non_bare_args_key_refused(self, block, key):
        # '--report_to=wandb' would set report_to even with allow_abbrev off.
        assert _render_error(block, _ctx(args={key: "x"})) == (
            f"openinstruct-sft: openinstruct_sft_config.args key '{key}' must be a "
            "bare finetune.py flag name (no '=', '-' prefix or whitespace)."
        )

    def test_mapping_arg_refused(self):
        assert "scalar or a list" in _render_error("run", _ctx(args={"x": {"a": 1}}))

    @pytest.mark.parametrize("block", ["setup", "run"])
    def test_empty_list_arg_refused(self, block):
        # `--foo` with no value would make finetune.py read the next flag as
        # foo's value.
        err = _render_error(block, _ctx(args={"foo": [], "max_train_steps": 3}))
        assert "args.foo is an empty list" in err


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

    def test_gbserver_accepts_one_node(self):
        assert validate_launch_num_nodes(_ENV, _LAUNCHER, _ctx()["config"]) == 1

    @pytest.mark.parametrize("block", ["setup", "run"])
    def test_template_leaves_num_nodes_to_gbserver(self, block):
        lc = {"resources": {"accelerators": "A10G:1"}, "num_nodes": 2}
        assert "openinstruct_sft_guard.sh" in _render(
            block, _ctx(compute={"num_nodes": 2}, launcher_config=lc)
        )


class TestGuardArgv:
    @pytest.mark.parametrize("block", ["setup", "run"])
    def test_paths_pass_verbatim(self, block):
        argv = _argv(block, _ctx(), "openinstruct_sft_guard.sh")
        assert argv == ["--base-model", "/staged/base it's", "--dataset", ""]

    def test_bound_dataset_reaches_guard(self):
        b = dict(_BASE, dataset={"binding": {"path": "/staged/ds"}})
        argv = _argv(
            "setup",
            _ctx(bindings=b, dataset_mixer_list={}),
            "openinstruct_sft_guard.sh",
        )
        assert _opt(argv, "--dataset") == "/staged/ds"


class TestRunArgv:
    def test_defaults(self):
        argv = _argv("run", _ctx(), "openinstruct_sft_run.sh")
        assert _opt(argv, "--base-model") == "/staged/base it's"
        assert _opt(argv, "--output-id") == "model"
        assert _opt(argv, "--output-path") == "./model_output"
        assert _opt(argv, "--mixed-precision") == "bf16"
        assert (
            _opt(argv, "--deepspeed-config") == "stage3_no_offloading_accelerate.conf"
        )
        assert _opt(argv, "--clearml-mode") == "auto"
        assert _tail(argv) == [
            "--dataset_mixer_list",
            "allenai/tulu-3-sft-personas-algebra",
            "64",
            "--dataset_mixer_list_splits",
            "train",
        ]

    @pytest.mark.parametrize(
        "given,mode", [("true", "on"), ("false", "off"), ("auto", "auto"), (True, "on")]
    )
    def test_clearml_mode(self, given, mode):
        argv = _argv("run", _ctx(clearml_enabled=given), "openinstruct_sft_run.sh")
        assert _opt(argv, "--clearml-mode") == mode

    def test_bound_dataset_becomes_the_only_mixer_entry(self):
        b = dict(_BASE, dataset={"binding": {"path": "/staged/ds"}})
        argv = _argv(
            "run", _ctx(bindings=b, dataset_mixer_list={}), "openinstruct_sft_run.sh"
        )
        assert _tail(argv)[:3] == ["--dataset_mixer_list", "/staged/ds", "1.0"]

    def test_multiple_splits(self):
        argv = _argv(
            "run",
            _ctx(dataset_mixer_list_splits=["train", "test"]),
            "openinstruct_sft_run.sh",
        )
        tail = _tail(argv)
        i = tail.index("--dataset_mixer_list_splits")
        assert tail[i : i + 3] == ["--dataset_mixer_list_splits", "train", "test"]

    def test_args_rendering(self):
        args = {
            "max_train_steps": 3,
            "gradient_checkpointing": True,
            "use_lora": False,
            "seed": None,
            "dataset_transform_fn": ["a b", "c'd"],
        }
        tail = _tail(_argv("run", _ctx(args=args), "openinstruct_sft_run.sh"))
        rest = tail[tail.index("--dataset_mixer_list_splits") + 2 :]
        assert rest == [
            "--max_train_steps",
            "3",
            "--gradient_checkpointing",
            "true",
            "--use_lora",
            "false",
            "--dataset_transform_fn",
            "a b",
            "c'd",
        ]

    def test_output_path_and_clearml_names(self):
        argv = _argv(
            "run",
            _ctx(output_path="/shared/m", clearml_project="p q", clearml_task_name="t"),
            "openinstruct_sft_run.sh",
        )
        assert _opt(argv, "--output-path") == "/shared/m"
        assert _opt(argv, "--clearml-project") == "p q"
        assert _opt(argv, "--clearml-task") == "t"
