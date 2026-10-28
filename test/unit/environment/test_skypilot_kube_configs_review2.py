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

"""``kube_configs``: hardening from the PR 528 review."""

import asyncio
import os
from unittest.mock import patch

import pytest
import yaml

from gbserver.environment import skypilot_config as sc
from gbserver.types.environmentconfig import KubeConfigEntry
from gbserver.types.errors import SkypilotConfigCollisionError


def _kubeconfig(context, user=None, cluster=None):
    return yaml.safe_dump(
        {
            "apiVersion": "v1",
            "kind": "Config",
            "clusters": [
                {"name": "c", "cluster": cluster or {"server": "https://a:6443"}}
            ],
            "users": [{"name": "u", "user": user or {"token": "t1"}}],
            "contexts": [{"name": context, "context": {"cluster": "c", "user": "u"}}],
        }
    )


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("KUBECONFIG", str(tmp_path / "user-kubeconfig"))


def _merge(tmp_path, text, owner="s/envA", context="remote"):
    return sc.merge_kube_configs(
        [KubeConfigEntry(context=context, kubeconfig="kc")],
        {"kc": text},
        owner,
        home=tmp_path,
    )


class TestCredentialAllowlist:
    """Review note 1: no exec/auth-provider plugins, no file paths."""

    @pytest.mark.parametrize(
        "user",
        [
            {"exec": {"command": "touch", "args": ["/tmp/PWNED"]}},
            {"auth-provider": {"name": "oidc", "config": {}}},
            {"client-key": "relative/key.pem", "client-certificate": "c.pem"},
            {"tokenFile": "/var/run/token"},
            {"token": "t", "exec": {"command": "x"}},
        ],
    )
    def test_dangerous_user_fields_refused(self, tmp_path, user):
        with pytest.raises(ValueError, match="not allowed in kube_configs") as e:
            _merge(tmp_path, _kubeconfig("remote", user=user))
        # Only field names are reported, never values.
        assert "PWNED" not in str(e.value) and "/var/run" not in str(e.value)
        assert not sc.kubeconfig_path(tmp_path).exists()

    def test_ca_file_path_refused(self, tmp_path):
        cluster = {"server": "https://a", "certificate-authority": "ca.crt"}
        with pytest.raises(ValueError, match="cluster fields"):
            _merge(tmp_path, _kubeconfig("remote", cluster=cluster))

    @pytest.mark.parametrize(
        "user",
        [
            {"token": "t"},
            {"client-certificate-data": "Y2VydA==", "client-key-data": "a2V5"},
            {"username": "u", "password": "p"},
        ],
    )
    def test_static_credentials_accepted(self, tmp_path, user):
        cluster = {"server": "https://a", "certificate-authority-data": "Y2E="}
        assert _merge(tmp_path, _kubeconfig("remote", user=user, cluster=cluster))

    def test_malformed_user_body_refused(self, tmp_path):
        bad = yaml.safe_load(_kubeconfig("remote"))
        bad["users"][0]["user"] = "token-as-a-string"
        with pytest.raises(ValueError, match="malformed"):
            _merge(tmp_path, yaml.safe_dump(bad))


class TestRobustness:
    def test_non_dict_entries_in_managed_file_are_repaired(self, tmp_path):
        """Review note 8: _without tolerates junk instead of AttributeError."""
        _merge(tmp_path, _kubeconfig("remote"))
        dest = sc.kubeconfig_path(tmp_path)
        doc = yaml.safe_load(dest.read_text())
        doc["contexts"].append("junk")
        doc["users"].append(None)
        dest.write_text(yaml.safe_dump(doc))
        _merge(tmp_path, _kubeconfig("remote", user={"token": "t2"}))
        doc = yaml.safe_load(dest.read_text())
        assert all(isinstance(e, dict) for e in doc["contexts"] + doc["users"])

    def test_lock_wait_is_bounded(self, tmp_path, monkeypatch):
        """Review note 4: a stuck lock holder surfaces as an error."""
        from filelock import FileLock, Timeout

        monkeypatch.setattr(sc, "_KUBE_LOCK_TIMEOUT_S", 0.2)
        held = FileLock(str(tmp_path / ".sky" / "locks" / "gbserver-kubeconfig.lock"))
        (tmp_path / ".sky" / "locks").mkdir(parents=True)
        import threading

        held.acquire()
        try:
            err = []
            t = threading.Thread(
                target=lambda: err.append(
                    pytest.raises(Timeout, _merge, tmp_path, _kubeconfig("remote"))
                )
            )
            t.start()
            t.join(5)
            assert not t.is_alive() and err
            assert issubclass(Timeout, OSError)  # caught by the non-strict paths
        finally:
            held.release()


class _Env:
    @staticmethod
    def make(config, secrets=None, space_name=None):
        from gbserver.environment.skypilot import Skypilot
        from gbserver.types.environmentconfig import EnvironmentConfig

        return Skypilot(
            event_q=asyncio.Queue(),
            environment_config=EnvironmentConfig(
                name="env-k", type="Skypilot", config=config
            ),
            secrets=secrets or {},
            space_name=space_name,
        )

    @staticmethod
    def config(contexts=("remote",), kube=True):
        cfg = {
            "default_cloud": "kubernetes",
            "cloud_config": {"kubernetes": {"allowed_contexts": list(contexts)}},
        }
        if kube:
            cfg["kube_configs"] = [{"context": "remote", "kubeconfig": "kc"}]
        return cfg


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point the managed kubeconfig at tmp_path for env-level tests."""
    real = sc.kubeconfig_path
    monkeypatch.setattr(sc, "kubeconfig_path", lambda home=None: real(tmp_path))
    real_home = sc._home
    monkeypatch.setattr(sc, "_home", lambda home=None: real_home(tmp_path))
    return tmp_path


class TestOwnerAndRemerge:
    def test_owner_comes_from_the_instance_space_not_the_thread(self, home):
        """Review note 3."""
        from gbcommon.uri.uri import URI

        env = _Env.make(_Env.config(), {"kc": _kubeconfig("remote")}, "spaceA")
        with patch.object(
            URI, "get_space_config", return_value={"space": {"name": "other"}}
        ):
            assert env._kube_owner() == "spaceA/env-k"
            env._ensure_kube_configs()
        assert sc.managed_kube_contexts() == {"remote": {"spaceA/env-k"}}

    def test_two_spaces_same_env_name_are_distinct_owners(self, home):
        a = _Env.make(_Env.config(), {"kc": _kubeconfig("remote")}, "spaceA")
        b = _Env.make(
            _Env.config(), {"kc": _kubeconfig("remote", user={"token": "B"})}, "spaceB"
        )
        a._ensure_kube_configs()
        with pytest.raises(SkypilotConfigCollisionError, match="spaceA/env-k"):
            b._ensure_kube_configs()

    def test_rotated_secret_is_remerged(self, home):
        """Review note 6: the cache no longer pins the first credentials."""
        secrets = {"kc": _kubeconfig("remote")}
        env = _Env.make(_Env.config(), secrets, "s")
        env._ensure_kube_configs()
        secrets["kc"] = _kubeconfig("remote", user={"token": "rotated"})
        env._ensure_kube_configs()
        doc = yaml.safe_load(sc.kubeconfig_path().read_text())
        tokens = [u["user"].get("token") for u in doc["users"]]
        assert "rotated" in tokens

    def test_unchanged_inputs_do_not_rewrite(self, home):
        env = _Env.make(_Env.config(), {"kc": _kubeconfig("remote")}, "s")
        env._ensure_kube_configs()
        with patch("gbserver.environment.skypilot_config.merge_kube_configs") as m:
            env._ensure_kube_configs()
        m.assert_not_called()

    def test_async_wrapper_runs_off_the_loop(self, home):
        import threading

        env = _Env.make(_Env.config(), {"kc": _kubeconfig("remote")}, "s")
        seen = []
        real = env._ensure_kube_configs

        def record(strict=True):
            seen.append(threading.current_thread() is threading.main_thread())
            return real(strict)

        env._ensure_kube_configs = record
        asyncio.run(env._ensure_kube_configs_async(strict=False))
        assert seen == [False]


class TestForeignPin:
    """Review note 2: pinning another environment's managed context is refused."""

    def test_other_env_pinning_a_managed_context_is_refused(self, home):
        owner = _Env.make(_Env.config(), {"kc": _kubeconfig("remote")}, "spaceA")
        owner._ensure_kube_configs()
        thief = _Env.make(_Env.config(kube=False), {}, "spaceB")
        with pytest.raises(SkypilotConfigCollisionError, match="another environment"):
            thief._ensure_kube_configs()

    def test_non_strict_paths_log_instead(self, home, caplog):
        owner = _Env.make(_Env.config(), {"kc": _kubeconfig("remote")}, "spaceA")
        owner._ensure_kube_configs()
        thief = _Env.make(_Env.config(kube=False), {}, "spaceB")
        thief._ensure_kube_configs(strict=False)
        assert "kube_configs not applied" in caplog.text

    def test_pinning_unmanaged_contexts_is_unaffected(self, home):
        env = _Env.make(_Env.config(contexts=("kind-skypilot",), kube=False), {}, "s")
        env._ensure_kube_configs()  # no managed contexts: nothing to refuse

    def test_co_owner_with_its_own_identical_entry_may_pin(self, home):
        text = _kubeconfig("remote")
        _Env.make(_Env.config(), {"kc": text}, "spaceA")._ensure_kube_configs()
        _Env.make(_Env.config(), {"kc": text}, "spaceB")._ensure_kube_configs()
        assert sc.managed_kube_contexts() == {
            "remote": {"spaceA/env-k", "spaceB/env-k"}
        }


class TestNativeK8sIgnoresManagedFile:
    def test_filter_drops_only_the_managed_path(self, home, monkeypatch):
        from gbserver.environment.k8s import _kubeconfig_without_skypilot_managed

        managed = str(sc.kubeconfig_path())
        monkeypatch.setenv(
            "KUBECONFIG", os.pathsep.join(["/x/mine", managed, "/x/other"])
        )
        assert _kubeconfig_without_skypilot_managed() == os.pathsep.join(
            ["/x/mine", "/x/other"]
        )
        monkeypatch.setenv("KUBECONFIG", managed)
        assert _kubeconfig_without_skypilot_managed() is None
        monkeypatch.delenv("KUBECONFIG")
        assert _kubeconfig_without_skypilot_managed() is None


class TestReleaseAtTeardown:
    """Review note 5: contexts are released when the env's last setup ends."""

    def _env(self):
        return _Env.make(_Env.config(), {"kc": _kubeconfig("remote")}, "s")

    def test_released_after_last_setup_only(self, home):
        env = self._env()
        env._ensure_kube_configs()
        env._kube_active_setups.update({"a", "b"})
        asyncio.run(env._release_kube_configs_if_idle("a"))
        assert sc.managed_kube_contexts() == {"remote": {"s/env-k"}}  # b still up
        asyncio.run(env._release_kube_configs_if_idle("b"))
        assert sc.managed_kube_contexts() == {}
        # The next request merges again.
        env._ensure_kube_configs()
        assert sc.managed_kube_contexts() == {"remote": {"s/env-k"}}

    def test_shared_context_survives_one_owners_release(self, home):
        text = _kubeconfig("remote")
        a = _Env.make(_Env.config(), {"kc": text}, "spaceA")
        b = _Env.make(_Env.config(), {"kc": text}, "spaceB")
        a._ensure_kube_configs()
        b._ensure_kube_configs()
        a._kube_active_setups.add("x")
        asyncio.run(a._release_kube_configs_if_idle("x"))
        assert sc.managed_kube_contexts() == {"remote": {"spaceB/env-k"}}

    def test_teardown_skypilot_releases_even_with_nothing_to_reap(self, home):
        env = self._env()
        env._ensure_kube_configs()
        env._kube_active_setups.add("only")
        asyncio.run(env.teardown_skypilot("only"))
        assert sc.managed_kube_contexts() == {}

    def test_release_failure_is_logged_not_raised(self, home, caplog):
        env = self._env()
        env._ensure_kube_configs()
        env._kube_active_setups.add("x")
        with patch(
            "gbserver.environment.skypilot_config.release_kube_configs",
            side_effect=OSError("disk"),
        ):
            asyncio.run(env._release_kube_configs_if_idle("x"))
        assert "kube_configs not released" in caplog.text
