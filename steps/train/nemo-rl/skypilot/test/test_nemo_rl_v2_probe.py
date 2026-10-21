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

"""Unit tests for src/nemo_rl_v2_probe.py.

The probe decides whether the step forces checkpointing.model_save_format=
torch_save. It must answer for the config the DRIVER will see — `defaults:`
inheritance plus the Hydra overrides — so it delegates loading to NeMo-RL's own
nemo_rl.utils.config. That module only exists in the image, so the wiring test
stands in a recording double for it; the decision itself is tested on plain
mappings.
"""

import sys
import types

import nemo_rl_v2_probe
import pytest


class TestUsesV2:
    # Mirrors lm_policy.save_checkpoint: it rejects a non-null model_save_format
    # unless policy.dtensor_cfg._v2 is truthy.
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
        assert nemo_rl_v2_probe.uses_v2(cfg) is want


class _FakeConfigModule(types.ModuleType):
    """Stands in for nemo_rl.utils.config: records what main() hands it."""

    def __init__(self, loaded):
        super().__init__("nemo_rl.utils.config")
        self.loaded = loaded
        self.calls = []

    def register_omegaconf_resolvers(self):
        self.calls.append(("register",))

    def load_config(self, path):
        self.calls.append(("load", path))
        return self.loaded

    def parse_hydra_overrides(self, cfg, overrides):
        self.calls.append(("parse", list(overrides)))
        out = {"policy": {"dtensor_cfg": dict(cfg["policy"]["dtensor_cfg"])}}
        for o in overrides:
            if o == "policy.dtensor_cfg._v2=false":
                out["policy"]["dtensor_cfg"]["_v2"] = False
        return out


@pytest.fixture
def fake_loader(monkeypatch):
    def install(loaded):
        mod = _FakeConfigModule(loaded)
        for name in ("nemo_rl", "nemo_rl.utils"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "nemo_rl.utils.config", mod)
        return mod

    return install


class TestMain:
    _V2 = {"policy": {"dtensor_cfg": {"_v2": True}}}

    def test_resolves_with_nemo_rl_loader_and_overrides(self, fake_loader, capsys):
        mod = fake_loader(self._V2)
        nemo_rl_v2_probe.main(
            ["/cfg/child.yaml", "a=1", "policy.dtensor_cfg._v2=false"]
        )
        assert ("load", "/cfg/child.yaml") in mod.calls
        assert ("parse", ["a=1", "policy.dtensor_cfg._v2=false"]) in mod.calls
        assert capsys.readouterr().out.splitlines()[-1] == "NEMO_RL_V2=false"

    def test_reports_true_for_v2(self, fake_loader, capsys):
        fake_loader(self._V2)
        nemo_rl_v2_probe.main(["/cfg/sft.yaml"])
        assert capsys.readouterr().out.splitlines()[-1] == "NEMO_RL_V2=true"

    def test_no_overrides_skips_parse(self, fake_loader, capsys):
        mod = fake_loader(self._V2)
        nemo_rl_v2_probe.main(["/cfg/sft.yaml"])
        assert not any(c[0] == "parse" for c in mod.calls)

    def test_requires_a_config_path(self):
        with pytest.raises(SystemExit):
            nemo_rl_v2_probe.main([])
