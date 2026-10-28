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
    monkeypatch.delenv("SKYPILOT_ALLOW_ALL_KUBERNETES_CONTEXTS", raising=False)
    # The host's ~/.sky/config.yaml must not leak into allowed_contexts checks.
    real_host = sc.host_sky_config
    monkeypatch.setattr(sc, "host_sky_config", lambda home=None: real_host(tmp_path))


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
            assert env._kube_owner().startswith("spaceA/env-k#")
            env._ensure_kube_configs()
        assert sc.managed_kube_contexts() == {"remote": {env._kube_owner()}}

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
        a = _Env.make(_Env.config(), {"kc": text}, "spaceA")
        b = _Env.make(_Env.config(), {"kc": text}, "spaceB")
        a._ensure_kube_configs()
        b._ensure_kube_configs()
        assert sc.managed_kube_contexts() == {
            "remote": {a._kube_owner(), b._kube_owner()}
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
        assert sc.managed_kube_contexts() == {"remote": {env._kube_owner()}}  # b up
        asyncio.run(env._release_kube_configs_if_idle("b"))
        assert sc.managed_kube_contexts() == {}
        # The next request merges again.
        env._ensure_kube_configs()
        assert sc.managed_kube_contexts() == {"remote": {env._kube_owner()}}

    def test_shared_context_survives_one_owners_release(self, home):
        text = _kubeconfig("remote")
        a = _Env.make(_Env.config(), {"kc": text}, "spaceA")
        b = _Env.make(_Env.config(), {"kc": text}, "spaceB")
        a._ensure_kube_configs()
        b._ensure_kube_configs()
        a._kube_active_setups.add("x")
        asyncio.run(a._release_kube_configs_if_idle("x"))
        assert sc.managed_kube_contexts() == {"remote": {b._kube_owner()}}

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


class TestValueTypesAndContextFields:
    """A wrong-typed value would make kubectl refuse the whole merged file."""

    @pytest.mark.parametrize(
        "user",
        [{"token": {"a": 1}}, {"token": 123}, {"username": ["u"], "password": "p"}],
    )
    def test_non_string_user_values_refused(self, tmp_path, user):
        with pytest.raises(ValueError, match="wrong type"):
            _merge(tmp_path, _kubeconfig("remote", user=user))
        assert not sc.kubeconfig_path(tmp_path).exists()

    def test_bool_flag_must_be_bool(self, tmp_path):
        cluster = {"server": "https://a", "insecure-skip-tls-verify": "yes"}
        with pytest.raises(ValueError, match="wrong type"):
            _merge(tmp_path, _kubeconfig("remote", cluster=cluster))
        cluster["insecure-skip-tls-verify"] = True
        assert _merge(tmp_path, _kubeconfig("remote", cluster=cluster))

    def test_context_extension_fields_refused(self, tmp_path):
        doc = yaml.safe_load(_kubeconfig("remote"))
        doc["contexts"][0]["context"]["extensions"] = [{"name": "x"}]
        with pytest.raises(ValueError, match="context fields"):
            _merge(tmp_path, yaml.safe_dump(doc))

    def test_namespace_is_kept(self, tmp_path):
        doc = yaml.safe_load(_kubeconfig("remote"))
        doc["contexts"][0]["context"]["namespace"] = "team-a"
        _merge(tmp_path, yaml.safe_dump(doc))
        merged = yaml.safe_load(sc.kubeconfig_path(tmp_path).read_text())
        (ctx,) = [c for c in merged["contexts"] if c["name"] == "remote"]
        assert ctx["context"] == {
            "namespace": "team-a",
            "cluster": "gbserver/remote",
            "user": "gbserver/remote",
        }


class TestThreadLockTimeout:
    def test_in_process_lock_wait_is_bounded(self, tmp_path, monkeypatch):
        import threading

        monkeypatch.setattr(sc, "_KUBE_LOCK_TIMEOUT_S", 0.2)
        holding, done = threading.Event(), threading.Event()

        def hold():
            with sc._THREAD_LOCK:
                holding.set()
                done.wait(5)

        t = threading.Thread(target=hold)
        t.start()
        try:
            holding.wait(5)
            with pytest.raises(TimeoutError, match="kubeconfig lock"):
                _merge(tmp_path, _kubeconfig("remote"))
        finally:
            done.set()
            t.join(5)


class TestPerInstanceOwnership:
    """Two builds of the same environment share a context safely."""

    def test_one_instances_release_keeps_the_others_context(self, home):
        text = _kubeconfig("remote")
        a = _Env.make(_Env.config(), {"kc": text}, "s")
        b = _Env.make(_Env.config(), {"kc": text}, "s")
        assert a._kube_owner() != b._kube_owner()
        a._ensure_kube_configs()
        b._ensure_kube_configs()
        a._kube_active_setups.add("x")
        asyncio.run(a._release_kube_configs_if_idle("x"))
        assert sc.managed_kube_contexts() == {"remote": {b._kube_owner()}}

    def test_rotation_refused_while_another_build_co_owns(self, home):
        a = _Env.make(_Env.config(), {"kc": _kubeconfig("remote")}, "s")
        b = _Env.make(_Env.config(), {"kc": _kubeconfig("remote")}, "s")
        a._ensure_kube_configs()
        b._ensure_kube_configs()
        a.secrets["kc"] = _kubeconfig("remote", user={"token": "rotated"})
        with pytest.raises(SkypilotConfigCollisionError):
            a._ensure_kube_configs()
        # Once b is done, the rotation goes through.
        b._kube_active_setups.add("y")
        asyncio.run(b._release_kube_configs_if_idle("y"))
        a._ensure_kube_configs()
        doc = yaml.safe_load(sc.kubeconfig_path().read_text())
        (user,) = [u for u in doc["users"] if u["name"] == "gbserver/remote"]
        assert user["user"]["token"] == "rotated"

    def test_setup_after_release_merges_again(self, home):
        from unittest.mock import MagicMock

        env = _Env.make(_Env.config(), {"kc": _kubeconfig("remote")}, "s")
        meta = MagicMock(build_id="b", targetrun_id="t1", target_name="x")
        asyncio.run(env.setup_skypilot("one", meta))
        asyncio.run(env.teardown_skypilot("one"))
        assert sc.managed_kube_contexts() == {}
        asyncio.run(env.setup_skypilot("two", meta))
        assert sc.managed_kube_contexts() == {"remote": {env._kube_owner()}}


class TestForeignPinEveryCall:
    def test_pin_made_before_the_owner_merged_is_refused_next_time(self, home):
        thief = _Env.make(_Env.config(kube=False), {}, "spaceB")
        thief._ensure_kube_configs()  # nothing managed yet
        _Env.make(
            _Env.config(), {"kc": _kubeconfig("remote")}, "spaceA"
        )._ensure_kube_configs()
        with pytest.raises(SkypilotConfigCollisionError, match="another environment"):
            thief._ensure_kube_configs()

    def test_allowed_contexts_all_refused_while_others_are_managed(self, home):
        _Env.make(
            _Env.config(), {"kc": _kubeconfig("remote")}, "spaceA"
        )._ensure_kube_configs()
        cfg = _Env.config(kube=False)
        cfg["cloud_config"]["kubernetes"]["allowed_contexts"] = "all"
        env = _Env.make(cfg, {}, "spaceB")
        with pytest.raises(SkypilotConfigCollisionError, match="allowed_contexts: all"):
            env._ensure_kube_configs()

    def test_allowed_contexts_all_fine_with_nothing_managed(self, home):
        cfg = _Env.config(kube=False)
        cfg["cloud_config"]["kubernetes"]["allowed_contexts"] = "all"
        _Env.make(cfg, {}, "s")._ensure_kube_configs()

    def test_users_own_context_is_never_foreign(self, home, monkeypatch):
        _Env.make(
            _Env.config(), {"kc": _kubeconfig("remote")}, "spaceA"
        )._ensure_kube_configs()
        monkeypatch.setattr(sc, "user_kube_context_names", lambda home=None: {"remote"})
        _Env.make(_Env.config(kube=False), {}, "spaceB")._ensure_kube_configs()


class TestUserKubeContextNames:
    def test_lists_the_users_contexts_only(self, tmp_path, monkeypatch):
        user = tmp_path / "user-kubeconfig"
        user.write_text(_kubeconfig("mine"))
        _merge(tmp_path, _kubeconfig("remote"))
        monkeypatch.setenv(
            "KUBECONFIG",
            os.pathsep.join([str(user), str(sc.kubeconfig_path(tmp_path))]),
        )
        assert sc.user_kube_context_names(tmp_path) == {"mine"}


class TestKubeToolSubprocessEnv:
    def test_managed_file_is_filtered_from_tool_env(self, home, monkeypatch):
        from gbserver.environment.k8s import _kube_tool_env

        managed = str(sc.kubeconfig_path())
        monkeypatch.setenv("KUBECONFIG", os.pathsep.join(["/x/mine", managed]))
        assert _kube_tool_env()["KUBECONFIG"] == "/x/mine"
        monkeypatch.setenv("KUBECONFIG", managed)
        assert "KUBECONFIG" not in _kube_tool_env()
        assert os.environ["KUBECONFIG"] == managed  # process env untouched

    def test_kubectl_fallback_runner_passes_filtered_env(self, home, monkeypatch):
        """Every kubectl/oc/helm launch in k8s.py passes env=_kube_tool_env()."""
        import inspect

        from gbserver.environment import k8s

        src = inspect.getsource(k8s)
        launches = src.count("launch_command_and_raise_errors(\n")
        subprocs = src.count("asyncio.create_subprocess_exec(")
        assert launches + subprocs == src.count("env=_kube_tool_env()")


class TestEnvironmentCacheKey:
    """space_name joins the cache key without changing what is cached."""

    def _resolve(self, raw, resolved):
        from gbserver.environment.environment import Environment

        cfg = type("Cfg", (), {"type": "Skypilot"})()
        asset = type("Asset", (), {"uristr": resolved})()
        made = []

        class Fake:
            def __init__(self, **kw):
                made.append(kw["event_q"])

        with (
            patch.object(
                Environment, "load_environment_config", return_value=(cfg, asset)
            ),
            patch.dict(Environment.environment_types, {"Skypilot": Fake}),
        ):
            if hasattr(Environment._thread_local, "environments"):
                Environment._thread_local.environments.clear()
            q1, q2 = asyncio.Queue(), asyncio.Queue()
            a = Environment.get_environment(raw, q1, space_name="s")
            b = Environment.get_environment(raw, q2, space_name="s")
            Environment._thread_local.environments.clear()
        return a, b, made

    def test_templated_uri_gets_a_fresh_instance_and_queue(self):
        """Regression: keying the store by the raw (template) URI made a later
        build reuse an earlier build's instance, whose event_q nobody reads, so
        its artifact events were lost."""
        a, b, made = self._resolve("space://environments/x", "file:///resolved/x")
        assert a is not b and len(made) == 2

    def test_resolved_uri_is_cached_per_space(self):
        a, b, made = self._resolve("file:///resolved/x", "file:///resolved/x")
        assert a is b and len(made) == 1


class TestAllowAllPrecedence:
    """'all' is detected the way SkyPilot resolves allowed_contexts."""

    def _env_with(self, k8s=None, workspace=None):
        cfg = {"default_cloud": "kubernetes", "cloud_config": {}}
        if k8s is not None:
            cfg["cloud_config"]["kubernetes"] = k8s
        if workspace is not None:
            cfg["cloud_config"]["workspaces"] = {"default": {"kubernetes": workspace}}
        return _Env.make(cfg, {}, "spaceB")

    @pytest.fixture
    def owned(self, home):
        _Env.make(
            _Env.config(), {"kc": _kubeconfig("remote")}, "spaceA"
        )._ensure_kube_configs()

    def test_workspace_all_refused_even_with_a_top_level_list(self, owned):
        env = self._env_with(
            {"allowed_contexts": ["mine"]}, {"allowed_contexts": "all"}
        )
        with pytest.raises(SkypilotConfigCollisionError, match="allows all"):
            env._ensure_kube_configs()

    def test_workspace_pin_of_managed_context_refused(self, owned):
        env = self._env_with(
            {"allowed_contexts": ["mine"]}, {"allowed_contexts": ["remote"]}
        )
        with pytest.raises(SkypilotConfigCollisionError, match="another environment"):
            env._ensure_kube_configs()

    def test_workspace_list_overrides_top_level_all(self, owned):
        env = self._env_with(
            {"allowed_contexts": "all"}, {"allowed_contexts": ["mine"]}
        )
        env._ensure_kube_configs()

    def test_unset_with_allow_all_env_var_refused(self, owned, monkeypatch):
        monkeypatch.setenv("SKYPILOT_ALLOW_ALL_KUBERNETES_CONTEXTS", "true")
        with pytest.raises(SkypilotConfigCollisionError, match="allows all"):
            self._env_with()._ensure_kube_configs()

    def test_unset_without_env_var_is_fine(self, owned, monkeypatch):
        monkeypatch.delenv("SKYPILOT_ALLOW_ALL_KUBERNETES_CONTEXTS", raising=False)
        self._env_with()._ensure_kube_configs()

    def test_pin_check_runs_under_the_launch_lock(self, monkeypatch):
        from gbserver.environment import skypilot as sp

        order = []
        monkeypatch.setattr(sp, "_apply_cloud_config", lambda *a: order.append("apply"))
        monkeypatch.setattr(
            sp.sky, "launch", lambda *a, **k: order.append("launch"), raising=False
        )

        def check():
            assert sp._CLOUD_CONFIG_LOCK.locked()
            order.append("check")

        sp._launch_with_cloud_config("e", {}, "task", pre_submit=check)
        assert order == ["check", "apply", "launch"]

        def refuse():
            raise SkypilotConfigCollisionError("no")

        order.clear()
        with pytest.raises(SkypilotConfigCollisionError):
            sp._launch_with_cloud_config("e", {}, "task", pre_submit=refuse)
        assert order == []


class TestStaleOwners:
    """Owners from a gbserver process that no longer runs don't block forever."""

    def _dead_owner(self, owner):
        import socket

        # pid 1 runs, but never with this start time: a pid reused by a later
        # process counts as gone.
        return f"{owner}@{socket.gethostname()}:1:12345.00"

    def test_tag_names_this_live_process(self):
        tag = sc.kube_process_tag()
        assert tag.startswith("@") and str(os.getpid()) in tag
        assert not sc._owner_is_stale("s/env#abc" + tag)

    def test_dead_or_reused_pid_is_stale(self):
        assert sc._owner_is_stale(self._dead_owner("s/env#abc"))
        import socket

        assert sc._owner_is_stale(f"s/env#a@{socket.gethostname()}:999999999:1.00")

    def test_untagged_or_other_host_owner_is_kept(self):
        assert not sc._owner_is_stale("s/env-legacy")
        assert not sc._owner_is_stale("s/env#a@some-other-host:1:1.00")

    def test_rotation_after_restart_succeeds(self, tmp_path):
        dead = self._dead_owner("s/envA#old")
        _merge(tmp_path, _kubeconfig("remote"), owner=dead)
        assert sc.managed_kube_contexts(tmp_path) == {}  # stale owners hidden
        live = "s/envA#new" + sc.kube_process_tag()
        _merge(tmp_path, _kubeconfig("remote", user={"token": "rotated"}), owner=live)
        assert sc.managed_kube_contexts(tmp_path) == {"remote": {live}}
        doc = yaml.safe_load(sc.kubeconfig_path(tmp_path).read_text())
        assert doc["x-gbserver-owners"] == {"remote": [live]}

    def test_release_drops_contexts_only_dead_owners_held(self, tmp_path):
        _merge(tmp_path, _kubeconfig("a"), owner=self._dead_owner("s/x#1"), context="a")
        live = "s/y#2" + sc.kube_process_tag()
        _merge(tmp_path, _kubeconfig("b"), owner=live, context="b")
        sc.release_kube_configs("s/nobody", home=tmp_path)
        doc = yaml.safe_load(sc.kubeconfig_path(tmp_path).read_text())
        assert {c["name"] for c in doc["contexts"]} - {sc._KUBE_PLACEHOLDER} == {"b"}

    def test_collision_message_hides_instance_and_process(self, tmp_path):
        a = "s/envA#aaa" + sc.kube_process_tag()
        _merge(tmp_path, _kubeconfig("remote"), owner=a)
        b = "s/envA#bbb" + sc.kube_process_tag()
        _merge(tmp_path, _kubeconfig("remote"), owner=b)
        with pytest.raises(SkypilotConfigCollisionError) as e:
            _merge(tmp_path, _kubeconfig("remote", user={"token": "x"}), owner=a)
        assert "#" not in str(e.value) and "@" not in str(e.value)
        assert "'s/envA'" in str(e.value)

    def test_env_owner_key_carries_the_process_tag(self, home):
        env = _Env.make(_Env.config(), {"kc": _kubeconfig("remote")}, "s")
        assert env._kube_owner().endswith(sc.kube_process_tag())


class TestMalformedSections:
    @pytest.mark.parametrize("section", ["contexts", "clusters", "users"])
    @pytest.mark.parametrize("value", [5, True, "x", {"a": 1}])
    def test_non_list_section_in_secret_is_a_value_error(
        self, tmp_path, section, value
    ):
        doc = yaml.safe_load(_kubeconfig("remote"))
        doc[section] = value
        with pytest.raises(ValueError, match="is not a list"):
            _merge(tmp_path, yaml.safe_dump(doc))

    def test_non_list_section_in_managed_file_is_repaired(self, tmp_path):
        _merge(tmp_path, _kubeconfig("remote"))
        dest = sc.kubeconfig_path(tmp_path)
        doc = yaml.safe_load(dest.read_text())
        doc["users"] = 5
        dest.write_text(yaml.safe_dump(doc))
        _merge(tmp_path, _kubeconfig("remote", user={"token": "t2"}))
        assert isinstance(yaml.safe_load(dest.read_text())["users"], list)
        sc.release_kube_configs("s/envA", home=tmp_path)

    def test_non_list_section_in_user_kubeconfig_is_ignored(self, tmp_path):
        (tmp_path / "user-kubeconfig").write_text(yaml.safe_dump({"contexts": 5}))
        assert _merge(tmp_path, _kubeconfig("remote"))

    def test_teardown_survives_any_kube_error(self, home):
        """A non-ValueError from the merge must not skip teardown work."""
        env = _Env.make(_Env.config(), {"kc": _kubeconfig("remote")}, "s")
        env._kube_active_setups.add("x")
        reaped = []

        async def fake_reap(*a, **k):
            reaped.append(True)

        env._setup_workdirs["x"] = "/shared/builds/b/runs/r"
        env._reap_per_run_workdir = fake_reap
        env._shared_fs_providers = lambda: []
        env._workdir_mount = lambda: None
        with patch(
            "gbserver.environment.skypilot_config.merge_kube_configs",
            side_effect=TypeError("boom"),
        ):
            asyncio.run(env.teardown_skypilot("x"))
        assert reaped == [True]

    def test_launch_teardown_survives_any_kube_error(self, home):
        env = _Env.make(_Env.config(), {"kc": _kubeconfig("remote")}, "s")
        with patch(
            "gbserver.environment.skypilot_config.merge_kube_configs",
            side_effect=KeyError("boom"),
        ):
            asyncio.run(env.launch_skypilot_teardown(config={}))


class TestSiblingKeysDropped:
    def test_only_cluster_and_user_bodies_are_copied(self, tmp_path):
        doc = yaml.safe_load(_kubeconfig("remote"))
        doc["users"][0]["exec"] = {"command": "evil"}
        doc["clusters"][0]["extensions"] = [{"name": "x"}]
        _merge(tmp_path, yaml.safe_dump(doc))
        merged = yaml.safe_load(sc.kubeconfig_path(tmp_path).read_text())
        (user,) = [u for u in merged["users"] if u["name"] == "gbserver/remote"]
        (cluster,) = [c for c in merged["clusters"] if c["name"] == "gbserver/remote"]
        assert set(user) == {"name", "user"} and set(cluster) == {"name", "cluster"}
        assert "evil" not in sc.kubeconfig_path(tmp_path).read_text()


class TestReleaseVsConcurrentSetup:
    def test_release_waits_for_a_merge_in_progress_and_then_skips(self, home):
        """A setup registered while a release waits for the lock keeps its contexts."""
        import threading

        env = _Env.make(_Env.config(), {"kc": _kubeconfig("remote")}, "s")
        env._ensure_kube_configs()
        env._kube_active_setups.add("old")
        released = threading.Event()

        with env._kube_lock:  # a merge "in progress"
            t = threading.Thread(
                target=lambda: (
                    asyncio.run(env._release_kube_configs_if_idle("old")),
                    released.set(),
                )
            )
            t.start()
            assert not released.wait(0.3)  # blocked behind the lock
            env._kube_active_setups.add("new")  # the concurrent setup registers
        t.join(5)
        assert released.is_set()
        assert sc.managed_kube_contexts() == {"remote": {env._kube_owner()}}

    def test_release_then_setup_remerges(self, home):
        env = _Env.make(_Env.config(), {"kc": _kubeconfig("remote")}, "s")
        env._ensure_kube_configs()
        env._kube_active_setups.add("old")
        asyncio.run(env._release_kube_configs_if_idle("old"))
        assert env._kube_configs_applied is None and sc.managed_kube_contexts() == {}
        env._kube_active_setups.add("new")
        env._ensure_kube_configs()
        assert sc.managed_kube_contexts() == {"remote": {env._kube_owner()}}


class TestReview3:
    """Second-review findings."""

    def _aws(self, extra=None):
        cfg = {"default_cloud": "aws", "cloud_config": {"aws": {"vpc_name": "v"}}}
        cfg.update(extra or {})
        return _Env.make(cfg, {}, "spaceB")

    def test_non_kube_env_never_reads_a_malformed_managed_file(self, home):
        sc.kubeconfig_path().parent.mkdir(parents=True, exist_ok=True)
        sc.kubeconfig_path().write_text("[1, 2")
        self._aws()._ensure_kube_configs()  # no ValueError

    def test_unpinned_kube_env_skips_file_io(self, home):
        sc.kubeconfig_path().parent.mkdir(parents=True, exist_ok=True)
        sc.kubeconfig_path().write_text("[1, 2")
        env = _Env.make({"default_cloud": "kubernetes"}, {}, "spaceB")
        with patch.object(sc, "managed_kube_contexts") as m:
            env._check_no_foreign_pins()
        m.assert_not_called()

    def test_allow_all_env_var_ignored_for_non_kube_env(self, home, monkeypatch):
        _Env.make(
            _Env.config(), {"kc": _kubeconfig("remote")}, "spaceA"
        )._ensure_kube_configs()
        monkeypatch.setenv("SKYPILOT_ALLOW_ALL_KUBERNETES_CONTEXTS", "true")
        self._aws()._ensure_kube_configs()

    def test_host_config_all_is_refused(self, home):
        _Env.make(
            _Env.config(), {"kc": _kubeconfig("remote")}, "spaceA"
        )._ensure_kube_configs()
        (home / ".sky" / "config.yaml").write_text(
            yaml.safe_dump({"kubernetes": {"allowed_contexts": "all"}})
        )
        env = _Env.make({"default_cloud": "kubernetes"}, {}, "spaceB")
        with pytest.raises(SkypilotConfigCollisionError, match="allows all"):
            env._ensure_kube_configs()

    def test_host_config_pin_of_managed_context_is_refused(self, home):
        _Env.make(
            _Env.config(), {"kc": _kubeconfig("remote")}, "spaceA"
        )._ensure_kube_configs()
        (home / ".sky" / "config.yaml").write_text(
            yaml.safe_dump({"kubernetes": {"allowed_contexts": ["remote"]}})
        )
        env = _Env.make({"default_cloud": "kubernetes"}, {}, "spaceB")
        with pytest.raises(SkypilotConfigCollisionError, match="another environment"):
            env._ensure_kube_configs()

    def test_env_list_overrides_host_all(self, home):
        _Env.make(
            _Env.config(), {"kc": _kubeconfig("remote")}, "spaceA"
        )._ensure_kube_configs()
        (home / ".sky" / "config.yaml").write_text(
            yaml.safe_dump({"kubernetes": {"allowed_contexts": "all"}})
        )
        _Env.make(
            _Env.config(contexts=("mine",), kube=False), {}, "spaceB"
        )._ensure_kube_configs()

    def test_gbservers_own_recorded_keys_are_not_the_hosts(self, home):
        (home / ".sky").mkdir(exist_ok=True)
        (home / ".sky" / "config.yaml").write_text(
            yaml.safe_dump({"kubernetes": {"allowed_contexts": "all"}, "x": 1})
        )
        (home / ".sky" / sc._CLOUD_CONFIG_RECORD).write_text(
            yaml.safe_dump(
                {"cloud_config": {"kubernetes": {"allowed_contexts": "all"}}}
            )
        )
        assert sc.host_sky_config(home) == {"x": 1}

    def test_late_non_strict_call_after_release_does_not_remerge(self, home):
        env = _Env.make(_Env.config(), {"kc": _kubeconfig("remote")}, "s")
        env._kube_active_setups.add("x")
        env._ensure_kube_configs()
        asyncio.run(env._release_kube_configs_if_idle("x"))
        env._ensure_kube_configs(strict=False)  # a late poll / cleanup
        assert sc.managed_kube_contexts() == {}
        env._kube_active_setups.add("y")
        env._ensure_kube_configs()  # the next setup merges again
        assert sc.managed_kube_contexts() == {"remote": {env._kube_owner()}}

    def test_launch_teardown_after_release_merges_and_releases(self, home):
        env = _Env.make(_Env.config(), {"kc": _kubeconfig("remote")}, "s")
        env._kube_active_setups.add("x")
        env._ensure_kube_configs()
        asyncio.run(env._release_kube_configs_if_idle("x"))
        seen = []
        real = sc.merge_kube_configs
        with patch.object(
            sc,
            "merge_kube_configs",
            side_effect=lambda *a, **k: seen.append(1) or real(*a, **k),
        ):
            asyncio.run(env.launch_skypilot_teardown(config={}))
        assert seen == [1] and sc.managed_kube_contexts() == {}

    def test_process_tag_is_computed_once(self, monkeypatch):
        first = sc.kube_process_tag()
        monkeypatch.setattr(sc, "_start_time", lambda proc: "changed")
        assert sc.kube_process_tag() == first

    def test_start_time_ignores_clock_steps(self):
        """Boot-relative where psutil has it (Linux): a clock step moves only the
        wall-clock value, which must not make a live owner look stale."""

        class Proc:
            def __init__(self, wall):
                self._proc = self
                self.wall = wall

            def create_time(self, monotonic=False):
                return 100.0 if monotonic else self.wall

        assert sc._start_time(Proc(1000.0)) == sc._start_time(Proc(4600.0))

    def test_messages_show_only_space_and_env(self, tmp_path):
        owner = "s/envA#abc" + sc.kube_process_tag()
        entry = KubeConfigEntry(context="remote", kubeconfig="kc")
        for entries, secrets in (
            ([entry, entry], {"kc": _kubeconfig("remote")}),
            ([entry], {}),
        ):
            with pytest.raises(ValueError) as e:
                sc.merge_kube_configs(entries, secrets, owner, home=tmp_path)
            assert "'s/envA'" in str(e.value) and "#" not in str(e.value)
        (tmp_path / "user-kubeconfig").write_text(_kubeconfig("remote"))
        with pytest.raises(SkypilotConfigCollisionError) as e:
            sc.merge_kube_configs(
                [entry], {"kc": _kubeconfig("remote")}, owner, home=tmp_path
            )
        assert "#" not in str(e.value) and "@" not in str(e.value)

    @pytest.mark.parametrize(
        "text",
        [
            None,
            123,
            "[" * 20000,
            yaml.safe_dump(
                {
                    "clusters": [{"name": "c", "cluster": {"server": "s"}}],
                    "users": [{"name": "u", "user": {1: "a", "exec": "b"}}],
                    "contexts": [
                        {"name": "remote", "context": {"cluster": "c", "user": "u"}}
                    ],
                }
            ),
            yaml.safe_dump(
                {
                    "clusters": [{"cluster": {"server": "s"}}],
                    "users": [{"user": {"token": "t"}}],
                    "contexts": [{"name": "remote", "context": {}}],
                }
            ),
        ],
        ids=["none", "int", "deep", "non-str-key", "missing-refs"],
    )
    def test_any_malformed_secret_is_a_value_error(self, tmp_path, text):
        with pytest.raises(ValueError):
            sc._extract_kube_context(text, "remote")
