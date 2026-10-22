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

"""The V2-only `torch_save` override, decided in the driver's own process.

NeMo-RL's V2 DTensor worker (policy.dtensor_cfg._v2: true) saves sharded HF
safetensors with no DCP `.metadata`, which convert_dcp_to_hf.py requires, so a V2
run needs `++checkpointing.model_save_format=torch_save`. A V1 worker REJECTS a
non-null model_save_format. Whether a run is V2 depends on the config the driver
sees — `defaults:` inheritance plus the Hydra overrides — so clearml_launch.py
resolves it with NeMo-RL's own loader (nemo_rl.utils.config, as every
examples/run_<algo>.py does) just before running the driver, in-process.

nemo_rl.utils.config exists only in the image, so the loader is a recording
double here; the real loader was checked against v0.7.0's configs.
"""

import sys
import types

import clearml_launch
import pytest

_V2 = "++checkpointing.model_save_format=torch_save"


class TestUsesV2:
    # Mirrors lm_policy: policy.dtensor_cfg._v2 picks the V2 worker.
    @pytest.mark.parametrize(
        "cfg,want",
        [
            ({"policy": {"dtensor_cfg": {"_v2": True}}}, True),
            ({"policy": {"dtensor_cfg": {"_v2": False}}}, False),
            ({"policy": {"dtensor_cfg": {"enabled": True}}}, False),
            ({"policy": {}}, False),
            ({}, False),
        ],
    )
    def test_decision(self, cfg, want):
        assert clearml_launch.uses_v2(cfg) is want


class _FakeConfigModule(types.ModuleType):
    """Stands in for nemo_rl.utils.config and records what it is handed."""

    def __init__(self, loaded):
        super().__init__("nemo_rl.utils.config")
        self.loaded = loaded
        self.calls = []

    def register_omegaconf_resolvers(self):
        self.calls.append(("register",))

    def load_config(self, path):
        self.calls.append(("load", path))
        return {"policy": {"dtensor_cfg": dict(self.loaded)}}

    def parse_hydra_overrides(self, cfg, overrides):
        self.calls.append(("parse", list(overrides)))
        for o in overrides:
            if o == "bad.key=1":
                raise ValueError("Failed to parse Hydra overrides: bad.key")
            if o.startswith("policy.dtensor_cfg._v2="):
                cfg["policy"]["dtensor_cfg"]["_v2"] = o.endswith("=true")
        return cfg


@pytest.fixture
def launch(monkeypatch):
    """Run clearml_launch.main with a fake loader and a recording driver."""

    def go(loaded, driver_argv, mode="off"):
        mod = _FakeConfigModule(loaded)
        for name in ("nemo_rl", "nemo_rl.utils"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "nemo_rl.utils.config", mod)
        seen = {}

        def fake_run_path(path, run_name=None):
            seen["argv"] = list(sys.argv[1:])

        monkeypatch.setattr(clearml_launch.runpy, "run_path", fake_run_path)
        monkeypatch.setattr(sys, "path", list(sys.path))
        monkeypatch.setattr(sys, "argv", list(sys.argv))
        clearml_launch.main(
            ["--clearml-mode", mode, "--v2-save-format", "--driver", "/d/run_sft.py"]
            + ["--"]
            + driver_argv
        )
        return mod, seen.get("argv")

    return go


class TestInjection:
    def test_v2_config_gets_torch_save(self, launch):
        _, argv = launch({"_v2": True}, ["--config", "/c/sft.yaml", "a=1"])
        assert argv == ["--config", "/c/sft.yaml", "a=1", _V2]

    def test_v1_config_is_left_alone(self, launch):
        _, argv = launch({"_v2": False}, ["--config", "/c/dpo.yaml", "a=1"])
        assert argv == ["--config", "/c/dpo.yaml", "a=1"]

    def test_user_override_of_v2_is_respected(self, launch):
        _, argv = launch(
            {"_v2": True}, ["--config", "/c/sft.yaml", "policy.dtensor_cfg._v2=false"]
        )
        assert _V2 not in argv

    def test_resolves_the_drivers_config_and_overrides(self, launch):
        mod, _ = launch({"_v2": True}, ["--config", "/c/child.yaml", "a=1", "b=2"])
        assert ("load", "/c/child.yaml") in mod.calls
        assert ("parse", ["a=1", "b=2"]) in mod.calls

    def test_bad_override_raises_before_the_driver_runs(self, launch):
        with pytest.raises(ValueError, match="bad.key"):
            launch({"_v2": True}, ["--config", "/c/sft.yaml", "bad.key=1"])

    def test_off_without_the_flag(self, monkeypatch):
        # Without --v2-save-format the wrapper never imports nemo_rl.
        monkeypatch.setitem(sys.modules, "nemo_rl.utils.config", None)
        seen = {}
        monkeypatch.setattr(
            clearml_launch.runpy,
            "run_path",
            lambda p, run_name=None: seen.update(argv=list(sys.argv[1:])),
        )
        monkeypatch.setattr(sys, "path", list(sys.path))
        monkeypatch.setattr(sys, "argv", list(sys.argv))
        clearml_launch.main(
            [
                "--clearml-mode",
                "off",
                "--driver",
                "/d/run_sft.py",
                "--",
                "--config",
                "c",
            ]
        )
        assert seen["argv"] == ["--config", "c"]
