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

"""``kube_configs``: an environment supplies its Kubernetes credentials (#505)."""

import asyncio
import os
import stat
from unittest.mock import patch

import pytest
import yaml

from gbserver.environment import skypilot_config as sc
from gbserver.types.environmentconfig import KubeConfigEntry
from gbserver.types.errors import SkypilotConfigCollisionError


def _kubeconfig(context, server="https://a:6443", token="t1"):
    return yaml.safe_dump(
        {
            "apiVersion": "v1",
            "kind": "Config",
            "clusters": [{"name": f"{context}-c", "cluster": {"server": server}}],
            "users": [{"name": f"{context}-u", "user": {"token": token}}],
            "contexts": [
                {
                    "name": context,
                    "context": {"cluster": f"{context}-c", "user": f"{context}-u"},
                }
            ],
            "current-context": context,
        }
    )


@pytest.fixture(autouse=True)
def _isolated_kubeconfig(tmp_path, monkeypatch):
    """No test reads the developer's real kubeconfig."""
    monkeypatch.setenv("KUBECONFIG", str(tmp_path / "user-kubeconfig"))


def _managed(tmp_path):
    return yaml.safe_load(sc.kubeconfig_path(tmp_path).read_text())


def _entry(context="remote", secret="remote-kc"):
    return KubeConfigEntry(context=context, kubeconfig=secret)


class TestMerge:
    def test_writes_context_cluster_user_0600(self, tmp_path):
        assert sc.merge_kube_configs(
            [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
        )
        doc = _managed(tmp_path)
        assert [c["name"] for c in doc["contexts"]] == ["remote"]
        assert [c["name"] for c in doc["clusters"]] == ["remote-c"]
        assert [u["name"] for u in doc["users"]] == ["remote-u"]
        # Set, so a merged KUBECONFIG is valid for the kubernetes client.
        assert doc["current-context"] == "remote"
        mode = stat.S_IMODE(os.stat(sc.kubeconfig_path(tmp_path)).st_mode)
        assert mode == 0o600

    def test_same_entry_again_is_a_noop(self, tmp_path):
        secrets = {"remote-kc": _kubeconfig("remote")}
        sc.merge_kube_configs([_entry()], secrets, "envA", home=tmp_path)
        assert not sc.merge_kube_configs([_entry()], secrets, "envA", home=tmp_path)

    def test_same_env_replaces_its_rotated_token(self, tmp_path):
        sc.merge_kube_configs(
            [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
        )
        sc.merge_kube_configs(
            [_entry()],
            {"remote-kc": _kubeconfig("remote", token="t2")},
            "envA",
            home=tmp_path,
        )
        users = _managed(tmp_path)["users"]
        assert [u["user"]["token"] for u in users] == ["t2"]

    def test_other_env_conflicting_context_is_refused(self, tmp_path):
        sc.merge_kube_configs(
            [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
        )
        with pytest.raises(SkypilotConfigCollisionError, match="environment 'envA'"):
            sc.merge_kube_configs(
                [_entry()],
                {"remote-kc": _kubeconfig("remote", token="other")},
                "envB",
                home=tmp_path,
            )
        # envA's credentials are untouched.
        assert _managed(tmp_path)["users"][0]["user"]["token"] == "t1"

    def test_other_env_different_contexts_coexist(self, tmp_path):
        sc.merge_kube_configs(
            [_entry("a", "a-kc")], {"a-kc": _kubeconfig("a")}, "envA", home=tmp_path
        )
        sc.merge_kube_configs(
            [_entry("b", "b-kc")], {"b-kc": _kubeconfig("b")}, "envB", home=tmp_path
        )
        assert sorted(c["name"] for c in _managed(tmp_path)["contexts"]) == ["a", "b"]

    def test_users_own_differing_context_is_refused(self, tmp_path, monkeypatch):
        user = tmp_path / "user-kubeconfig"
        user.write_text(_kubeconfig("remote", server="https://mine:6443"))
        with pytest.raises(SkypilotConfigCollisionError, match="your own kubeconfig"):
            sc.merge_kube_configs(
                [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
            )
        assert not sc.kubeconfig_path(tmp_path).exists()

    def test_missing_secret_names_the_secret_not_its_value(self, tmp_path):
        with pytest.raises(ValueError, match="'remote-kc'"):
            sc.merge_kube_configs([_entry()], {}, "envA", home=tmp_path)

    @pytest.mark.parametrize(
        "text, match",
        [
            (": not yaml: [", "not YAML"),
            ("- a list", "not a mapping"),
            (_kubeconfig("other"), "does not define context 'remote'"),
        ],
    )
    def test_bad_secret_content(self, tmp_path, text, match):
        with pytest.raises(ValueError, match=match):
            sc.merge_kube_configs(
                [_entry()], {"remote-kc": text}, "envA", home=tmp_path
            )

    def test_secret_values_never_logged(self, tmp_path, caplog):
        caplog.set_level("DEBUG")
        sc.merge_kube_configs(
            [_entry()],
            {"remote-kc": _kubeconfig("remote", token="S3CR3T")},
            "envA",
            home=tmp_path,
        )
        assert "S3CR3T" not in caplog.text

    def test_release_removes_only_that_envs_entries(self, tmp_path):
        sc.merge_kube_configs(
            [_entry("a", "a-kc")], {"a-kc": _kubeconfig("a")}, "envA", home=tmp_path
        )
        sc.merge_kube_configs(
            [_entry("b", "b-kc")], {"b-kc": _kubeconfig("b")}, "envB", home=tmp_path
        )
        assert sc.release_kube_configs("envA", home=tmp_path)
        doc = _managed(tmp_path)
        assert [c["name"] for c in doc["contexts"]] == ["b"]
        assert doc["current-context"] == "b"
        assert not sc.release_kube_configs("envA", home=tmp_path)


class TestActivate:
    def test_prepends_managed_and_keeps_existing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KUBECONFIG", "/x/one" + os.pathsep + "/x/two")
        sc.activate_kubeconfig(home=tmp_path)
        managed = str(sc.kubeconfig_path(tmp_path))
        assert os.environ["KUBECONFIG"].split(os.pathsep) == [
            managed,
            "/x/one",
            "/x/two",
        ]
        sc.activate_kubeconfig(home=tmp_path)  # idempotent
        assert os.environ["KUBECONFIG"].split(os.pathsep).count(managed) == 1

    def test_defaults_to_the_users_kube_config(self, tmp_path, monkeypatch):
        monkeypatch.delenv("KUBECONFIG")
        sc.activate_kubeconfig(home=tmp_path)
        assert os.environ["KUBECONFIG"].split(os.pathsep)[1] == "~/.kube/config"

    def test_merged_kubeconfig_lists_both_and_keeps_users_current(
        self, tmp_path, monkeypatch
    ):
        """The kubernetes client SkyPilot uses accepts the merged list."""
        kube_config = pytest.importorskip("kubernetes.config")
        user = tmp_path / "user-kubeconfig"
        user.write_text(_kubeconfig("mine", server="https://mine:6443"))
        sc.merge_kube_configs(
            [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
        )
        sc.activate_kubeconfig(home=tmp_path)
        contexts, current = kube_config.list_kube_config_contexts(
            os.environ["KUBECONFIG"]
        )
        assert sorted(c["name"] for c in contexts) == ["mine", "remote"]
        assert current["name"] == "mine"

    def test_merged_kubeconfig_valid_when_user_has_none(self, tmp_path, monkeypatch):
        kube_config = pytest.importorskip("kubernetes.config")
        sc.merge_kube_configs(
            [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
        )
        sc.activate_kubeconfig(home=tmp_path)
        contexts, current = kube_config.list_kube_config_contexts(
            os.environ["KUBECONFIG"]
        )
        assert [c["name"] for c in contexts] == ["remote"]
        assert current["name"] == "remote"


class TestEnvironmentWiring:
    def _env(self, config, secrets=None):
        from gbserver.environment.skypilot import Skypilot
        from gbserver.types.environmentconfig import EnvironmentConfig

        cfg = EnvironmentConfig(name="env-k", type="Skypilot", config=config)
        return Skypilot(
            event_q=asyncio.Queue(), environment_config=cfg, secrets=secrets or {}
        )

    def _config(self, allowed):
        return {
            "default_cloud": "kubernetes",
            "kube_configs": [{"context": "remote", "kubeconfig": "remote-kc"}],
            "cloud_config": {"kubernetes": {"allowed_contexts": allowed}},
        }

    def test_context_must_be_in_allowed_contexts(self):
        env = self._env(self._config(["remot"]))
        with pytest.raises(ValueError, match="allowed_contexts"):
            env._kube_config_entries()

    def test_no_allowed_contexts_at_all_is_refused(self):
        cfg = self._config([])
        del cfg["cloud_config"]
        with pytest.raises(ValueError, match="allowed_contexts"):
            self._env(cfg)._kube_config_entries()

    def test_materialize_merges_then_activates(self):
        env = self._env(self._config(["remote"]), {"remote-kc": _kubeconfig("remote")})
        calls = []
        with (
            patch(
                "gbserver.environment.skypilot_config.merge_kube_configs",
                side_effect=lambda e, s, n: calls.append(("merge", n, e[0].context)),
            ),
            patch(
                "gbserver.environment.skypilot_config.activate_kubeconfig",
                side_effect=lambda: calls.append(("activate",)),
            ),
        ):
            env._ensure_inline_configs_materialized()
        assert calls == [("merge", "env-k", "remote"), ("activate",)]

    def test_no_kube_configs_touches_nothing(self):
        env = self._env({"default_cloud": "kubernetes"})
        with patch("gbserver.environment.skypilot_config.merge_kube_configs") as merge:
            env._ensure_inline_configs_materialized()
        merge.assert_not_called()
