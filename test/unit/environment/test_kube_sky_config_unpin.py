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

"""Unit tests for scripts/kube/sky_config_unpin.py (run by `make kube-teardown`).

It must remove exactly what skypilot/kubernetes/local leaves in ~/.sky/config.yaml
(the allowed_contexts pin and that context's context_configs entry) and nothing a
user put there. Runs against temporary config files, never the real one.
"""

import importlib.util
import pathlib
import stat

import yaml

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[3] / "scripts/kube/sky_config_unpin.py"
)
_spec = importlib.util.spec_from_file_location("sky_config_unpin", _SCRIPT)
unpin_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(unpin_mod)

CTX = "kind-skypilot"


def _write(tmp_path, data, mode=0o600):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    path.chmod(mode)
    return path


def _env_keys():
    """What the local env merges into the file."""
    return {
        "allowed_contexts": [CTX],
        "context_configs": {CTX: {"pod_config": {"spec": {"volumes": []}}}},
    }


class TestUnpin:
    def test_removes_the_pin_and_keeps_everything_else(self, tmp_path):
        path = _write(
            tmp_path,
            {
                "slurm": {"cluster_configs": {"slurm-docker": {"workdir": "/shared"}}},
                "kubernetes": {
                    **_env_keys(),
                    "provision_timeout": 30,
                },
            },
        )
        assert unpin_mod.main(["x", CTX, str(path)]) == 0
        assert yaml.safe_load(path.read_text()) == {
            "slurm": {"cluster_configs": {"slurm-docker": {"workdir": "/shared"}}},
            "kubernetes": {"provision_timeout": 30},
        }

    def test_drops_a_kubernetes_section_left_empty(self, tmp_path):
        path = _write(tmp_path, {"slurm": {"a": 1}, "kubernetes": _env_keys()})
        unpin_mod.main(["x", CTX, str(path)])
        assert yaml.safe_load(path.read_text()) == {"slurm": {"a": 1}}

    def test_keeps_other_contexts_configs(self, tmp_path):
        keys = _env_keys()
        keys["context_configs"]["prod"] = {"pod_config": {"x": 1}}
        path = _write(tmp_path, {"kubernetes": keys})
        unpin_mod.main(["x", CTX, str(path)])
        assert yaml.safe_load(path.read_text()) == {
            "kubernetes": {"context_configs": {"prod": {"pod_config": {"x": 1}}}}
        }

    def test_leaves_an_allowed_contexts_list_the_user_edited(self, tmp_path, capsys):
        path = _write(
            tmp_path, {"kubernetes": {"allowed_contexts": [CTX, "prod-cluster"]}}
        )
        before = path.read_bytes()
        assert unpin_mod.main(["x", CTX, str(path)]) == 0
        assert path.read_bytes() == before
        assert "not exactly" in capsys.readouterr().out

    def test_no_pin_leaves_the_file_byte_for_byte(self, tmp_path):
        path = _write(tmp_path, {"slurm": {"a": 1}})
        before = path.read_bytes()
        assert unpin_mod.main(["x", CTX, str(path)]) == 0
        assert path.read_bytes() == before

    def test_missing_file_is_not_an_error(self, tmp_path, capsys):
        assert unpin_mod.main(["x", CTX, str(tmp_path / "absent.yaml")]) == 0
        assert "nothing to undo" in capsys.readouterr().out

    def test_file_permissions_are_preserved(self, tmp_path):
        path = _write(tmp_path, {"kubernetes": _env_keys()}, mode=0o640)
        unpin_mod.main(["x", CTX, str(path)])
        assert stat.S_IMODE(path.stat().st_mode) == 0o640

    def test_wrong_arguments_exit_two(self):
        assert unpin_mod.main(["x"]) == 2
