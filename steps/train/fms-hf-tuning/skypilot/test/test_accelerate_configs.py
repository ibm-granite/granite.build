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

"""The step-owned accelerate configs: what each launch mode gets.

Static checks need only PyYAML; the accelerate parse check is skipped unless
accelerate imports (run it in the CPU venv, README.md "Tests").
"""

import pathlib
import re

import pytest
import yaml

_STEP = pathlib.Path(__file__).resolve().parents[1]
_DIR = _STEP / "src" / "accelerate"
_RUN = (_STEP / "src" / "fms_hf_tuning_run.sh").read_text()
_NAMES = ("single_gpu.yaml", "ddp.yaml", "fsdp.yaml")


def _cfg(name):
    return yaml.safe_load((_DIR / name).read_text())


def test_exactly_the_configs_the_run_script_uses():
    assert sorted(p.name for p in _DIR.glob("*.yaml")) == sorted(_NAMES)
    used = set(re.findall(r'cfg="([a-z_]+\.yaml)"', _RUN))
    assert used == set(_NAMES)


@pytest.mark.parametrize("name", _NAMES)
def test_common_shape(name):
    c = _cfg(name)
    assert c["compute_environment"] == "LOCAL_MACHINE"
    assert c["num_machines"] == 1 and c["machine_rank"] == 0
    # Precision comes from fms-hf-tuning's training args, not accelerate.
    assert c.get("mixed_precision", "no") == "no"


def test_modes():
    assert _cfg("single_gpu.yaml")["distributed_type"] == "NO"
    assert _cfg("ddp.yaml")["distributed_type"] == "MULTI_GPU"
    f = _cfg("fsdp.yaml")
    assert f["distributed_type"] == "FSDP"


def test_fsdp_is_upstreams_fsdp1_defaults():
    f = _cfg("fsdp.yaml")["fsdp_config"]
    assert "fsdp_version" not in f  # FSDP1, as upstream ships it
    assert f["fsdp_sharding_strategy"] == 1  # FULL_SHARD
    assert f["fsdp_state_dict_type"] == "FULL_STATE_DICT"
    assert f["fsdp_auto_wrap_policy"] == "TRANSFORMER_BASED_WRAP"
    assert f["fsdp_cpu_ram_efficient_loading"] is True
    assert f["fsdp_sync_module_states"] is True
    assert "fixtures/accelerate_fsdp_defaults.yaml" in (_DIR / "fsdp.yaml").read_text()


@pytest.mark.parametrize(
    "name,kind",
    [("single_gpu.yaml", "single"), ("ddp.yaml", "multi_gpu"), ("fsdp.yaml", "fsdp")],
)
def test_accelerate_reads_each_config(name, kind):
    pytest.importorskip("accelerate.commands.launch")
    from accelerate.commands.launch import (  # pylint: disable=import-outside-toplevel
        _validate_launch_command,
        launch_command_parser,
    )

    args = launch_command_parser().parse_args(
        [
            "--config_file",
            str(_DIR / name),
            "--num_machines",
            "1",
            "--num_processes",
            "2",
            "x.py",
        ]
    )
    args, _, _ = _validate_launch_command(args)
    got = "fsdp" if args.use_fsdp else ("multi_gpu" if args.multi_gpu else "single")
    assert got == kind
    if kind == "fsdp":
        assert int(args.fsdp_version) == 1
