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
    """The managed kubeconfig, minus the placeholder default entries."""
    doc = yaml.safe_load(sc.kubeconfig_path(tmp_path).read_text())
    for section in ("contexts", "clusters", "users"):
        doc[section] = [
            e for e in doc.get(section) or [] if e["name"] != sc._KUBE_PLACEHOLDER
        ]
    return doc


def _raw_managed(tmp_path):
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
        # The secret's own cluster/user names are replaced (see the collision
        # tests below for why).
        assert [c["name"] for c in doc["clusters"]] == ["gbserver/remote"]
        assert [u["name"] for u in doc["users"]] == ["gbserver/remote"]
        assert doc["contexts"][0]["context"] == {
            "cluster": "gbserver/remote",
            "user": "gbserver/remote",
        }
        # The placeholder, not a real context, so a merged KUBECONFIG is valid
        # without making an env's cluster the process-wide default.
        assert doc["current-context"] == sc._KUBE_PLACEHOLDER
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
            (": not yaml: [", "not valid YAML"),
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

    def test_generic_cluster_user_names_cannot_hijack_users_context(self, tmp_path):
        """A secret reusing the user's cluster/user names leaves theirs intact.

        kubernetes-client keeps the first definition of each cluster/user name
        across KUBECONFIG, so a secret's `default` cluster must not shadow the
        user's `default`.
        """
        kube_config = pytest.importorskip("kubernetes.config.kube_config")

        def config(context, server, token):
            return yaml.safe_dump(
                {
                    "apiVersion": "v1",
                    "kind": "Config",
                    "clusters": [{"name": "default", "cluster": {"server": server}}],
                    "users": [{"name": "default", "user": {"token": token}}],
                    "contexts": [
                        {
                            "name": context,
                            "context": {"cluster": "default", "user": "default"},
                        }
                    ],
                    "current-context": context,
                }
            )

        (tmp_path / "user-kubeconfig").write_text(
            config("local", "https://mine", "MINE")
        )
        sc.merge_kube_configs(
            [_entry()],
            {"remote-kc": config("remote", "https://remote", "REMOTE")},
            "envA",
            home=tmp_path,
        )
        sc.activate_kubeconfig(home=tmp_path)
        for ctx, server, token in (
            ("local", "https://mine", "MINE"),
            ("remote", "https://remote", "REMOTE"),
        ):
            loader = kube_config._get_kube_config_loader(
                filename=os.environ["KUBECONFIG"], active_context=ctx
            )
            assert loader._cluster["server"] == server
            assert loader._user["token"] == token

    def test_two_contexts_from_one_env_sharing_a_cluster_name(self, tmp_path):
        """Different secrets naming their cluster alike no longer clobber."""
        sc.merge_kube_configs(
            [_entry("a", "a-kc"), _entry("b", "b-kc")],
            {
                "a-kc": _kubeconfig("a", server="https://a"),
                "b-kc": yaml.safe_dump(
                    {
                        "clusters": [
                            {"name": "a-c", "cluster": {"server": "https://b"}}
                        ],
                        "users": [{"name": "a-u", "user": {"token": "tb"}}],
                        "contexts": [
                            {"name": "b", "context": {"cluster": "a-c", "user": "a-u"}}
                        ],
                    }
                ),
            },
            "envA",
            home=tmp_path,
        )
        servers = {
            c["name"]: c["cluster"]["server"] for c in _managed(tmp_path)["clusters"]
        }
        assert servers == {"gbserver/a": "https://a", "gbserver/b": "https://b"}

    def test_same_context_listed_twice_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="more than once"):
            sc.merge_kube_configs(
                [_entry(), _entry()],
                {"remote-kc": _kubeconfig("remote")},
                "envA",
                home=tmp_path,
            )

    def test_identical_shared_context_keeps_first_owner(self, tmp_path):
        """envB using envA's exact context does not take it over."""
        secrets = {"remote-kc": _kubeconfig("remote")}
        sc.merge_kube_configs([_entry()], secrets, "envA", home=tmp_path)
        sc.merge_kube_configs([_entry()], secrets, "envB", home=tmp_path)
        assert _managed(tmp_path)["x-gbserver-owners"] == {"remote": "envA"}
        # envA can still rotate its own token...
        sc.merge_kube_configs(
            [_entry()],
            {"remote-kc": _kubeconfig("remote", token="t2")},
            "envA",
            home=tmp_path,
        )
        assert _managed(tmp_path)["users"][0]["user"]["token"] == "t2"
        # ...and releasing envB removes nothing of envA's.
        assert not sc.release_kube_configs("envB", home=tmp_path)
        assert [c["name"] for c in _managed(tmp_path)["contexts"]] == ["remote"]

    def test_yaml_error_does_not_chain_the_secret_text(self, tmp_path):
        """PyYAML's error echoes the offending line; it must not ride along."""
        bad = "users:\n- user: {token: S3CR3T: x}\n"
        with pytest.raises(ValueError) as info:
            sc.merge_kube_configs([_entry()], {"remote-kc": bad}, "envA", home=tmp_path)
        assert info.value.__cause__ is None and info.value.__suppress_context__
        assert "S3CR3T" not in str(info.value)

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
        assert doc["current-context"] == sc._KUBE_PLACEHOLDER
        assert not sc.release_kube_configs("envA", home=tmp_path)
        # Releasing the last owner removes the placeholder too.
        assert sc.release_kube_configs("envB", home=tmp_path)
        raw = _raw_managed(tmp_path)
        assert raw["contexts"] == [] and "current-context" not in raw


class TestRobustness:
    def test_tmp_file_is_created_0600(self, tmp_path, monkeypatch):
        """The temp file never exists with looser permissions, even briefly."""
        seen = []
        real_open = os.open

        def spy(path, flags, mode=0o777, *a, **k):
            if str(path).endswith(".gbtmp"):
                seen.append(mode)
            return real_open(path, flags, mode, *a, **k)

        monkeypatch.setattr(os, "open", spy)
        sc.merge_kube_configs(
            [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
        )
        assert seen == [0o600]

    def test_corrupt_managed_file_is_a_valueerror_without_its_text(self, tmp_path):
        dest = sc.kubeconfig_path(tmp_path)
        dest.parent.mkdir(parents=True)
        dest.write_text("users:\n- user: {token: S3CR3T: x}\n")
        with pytest.raises(ValueError, match="not valid YAML") as info:
            sc.merge_kube_configs(
                [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
            )
        assert "S3CR3T" not in str(info.value) and info.value.__cause__ is None
        with pytest.raises(ValueError):
            sc.release_kube_configs("envA", home=tmp_path)

    def test_malformed_context_entry_is_a_valueerror(self, tmp_path):
        bad = yaml.safe_dump({"contexts": [{"name": "remote", "context": "oops"}]})
        with pytest.raises(ValueError, match="malformed"):
            sc.merge_kube_configs([_entry()], {"remote-kc": bad}, "envA", home=tmp_path)

    def test_placeholder_name_is_reserved(self, tmp_path):
        with pytest.raises(ValueError, match="reserved"):
            sc.merge_kube_configs(
                [_entry(sc._KUBE_PLACEHOLDER, "x")],
                {"x": _kubeconfig(sc._KUBE_PLACEHOLDER)},
                "envA",
                home=tmp_path,
            )


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
        names = {c["name"] for c in contexts} - {sc._KUBE_PLACEHOLDER}
        assert names == {"mine", "remote"}
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
        assert {c["name"] for c in contexts} == {"remote", sc._KUBE_PLACEHOLDER}
        # The default is the unreachable placeholder, never an env's cluster.
        assert current["name"] == sc._KUBE_PLACEHOLDER
        from kubernetes.config import kube_config as kc

        loader = kc._get_kube_config_loader(filename=os.environ["KUBECONFIG"])
        assert loader._cluster["server"] == "https://192.0.2.1:1"
        assert not loader._user


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

    @pytest.mark.parametrize(
        "cloud_config, expected",
        [
            ({"kubernetes": {"allowed_contexts": ["a"]}}, ["a"]),
            (
                {
                    "kubernetes": {"allowed_contexts": ["a"]},
                    "workspaces": {
                        "default": {"kubernetes": {"allowed_contexts": ["w"]}}
                    },
                },
                ["w"],
            ),
            (
                {
                    "active_workspace": "team",
                    "kubernetes": {"allowed_contexts": ["a"]},
                    "workspaces": {"team": {"kubernetes": {"allowed_contexts": ["t"]}}},
                },
                ["t"],
            ),
            (
                {
                    "active_workspace": "team",
                    "kubernetes": {"allowed_contexts": ["a"]},
                    "workspaces": {
                        "other": {"kubernetes": {"allowed_contexts": ["o"]}}
                    },
                },
                ["a"],
            ),
            ({"kubernetes": {"allowed_contexts": "all"}}, []),
            ({}, []),
        ],
    )
    def test_allowed_contexts_resolved_like_skypilot(self, cloud_config, expected):
        from gbserver.environment.skypilot import _kube_allowed_contexts

        assert _kube_allowed_contexts(cloud_config) == expected

    def test_workspace_pinned_context_is_accepted(self):
        cfg = {
            "default_cloud": "kubernetes",
            "kube_configs": [{"context": "remote", "kubeconfig": "remote-kc"}],
            "cloud_config": {
                "workspaces": {
                    "default": {"kubernetes": {"allowed_contexts": ["remote"]}}
                }
            },
        }
        assert [e.context for e in self._env(cfg)._kube_config_entries()] == ["remote"]

    def test_teardown_paths_log_instead_of_raising(self, caplog):
        """A bad kube_configs fails setup; it must not also block cleanup."""
        env = self._env(self._config(["typo"]))
        with pytest.raises(ValueError):
            env._ensure_kube_configs()
        env._ensure_kube_configs(strict=False)
        assert "kube_configs not applied" in caplog.text

    def test_every_cluster_path_activates(self):
        """Every Skypilot method that sends a SkyPilot request ensures credentials.

        Scans the class instead of naming methods, so a new request path that
        forgets _ensure_kube_configs fails here.
        """
        import inspect
        import re

        from gbserver.environment.skypilot import Skypilot

        request = re.compile(
            r"\bsky\.(launch|down|job_status|status|queue|tail_logs|cancel|stop|exec)\b"
            r"|_launch_with_cloud_config|\bself\._teardown\("
        )
        missing = []
        for name, fn in inspect.getmembers(Skypilot, inspect.isfunction):
            if fn.__qualname__.split(".")[0] != "Skypilot":
                continue
            src = inspect.getsource(fn)
            body = src.split('"""', 2)[-1] if src.count('"""') >= 2 else src
            ensures = (
                "_ensure_kube_configs" in body
                or "_ensure_inline_configs_materialized()" in body
            )
            if request.search(body) and not ensures:
                missing.append(name)
        # Methods that only reach SkyPilot through a method that ensures it.
        delegating = {
            "_abort_provision",  # -> self._teardown
            "_await_cleanup_job_status",  # called by teardown_skypilot
            "_provision_with_retry",  # called by launch_skypilot
            "_launch_cleanup_vm",  # called by teardown_skypilot
        }
        assert sorted(set(missing) - delegating) == []
