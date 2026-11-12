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

"""``kube_configs``: one kubeconfig per environment instance, per request (#571)."""

import ast
import asyncio
import inspect
import os
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from gbserver.environment import skypilot as sp
from gbserver.environment import skypilot_kube as sk
from gbserver.types.environmentconfig import EnvironmentConfig, KubeConfigEntry
from gbserver.types.errors import SkypilotConfigCollisionError

pytestmark = pytest.mark.skipif(not sp.HAS_SKYPILOT, reason="needs skypilot")


def _kubeconfig(context="remote", user=None, cluster=None, ctx_extra=None):
    return yaml.safe_dump(
        {
            "apiVersion": "v1",
            "kind": "Config",
            "clusters": [
                {"name": "c", "cluster": cluster or {"server": "https://a:6443"}}
            ],
            "users": [{"name": "u", "user": user or {"token": "t1"}}],
            "contexts": [
                {
                    "name": context,
                    "context": {"cluster": "c", "user": "u", **(ctx_extra or {})},
                }
            ],
        }
    )


def _user_kubeconfig(path, *names, current=None):
    path.write_text(
        yaml.safe_dump(
            {
                "clusters": [
                    {"name": n, "cluster": {"server": "https://u"}} for n in names
                ],
                "users": [{"name": n, "user": {"token": "u"}} for n in names],
                "contexts": [
                    {"name": n, "context": {"cluster": n, "user": n}} for n in names
                ],
                **({"current-context": current} if current else {}),
            }
        )
    )


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Never touch the real ~/.sky or kubeconfig."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("KUBECONFIG", str(tmp_path / "user-kubeconfig"))
    monkeypatch.delenv("SKYPILOT_IN_CLUSTER_CONTEXT_NAME", raising=False)


@pytest.fixture
def local_server(monkeypatch):
    """A local server that keeps the forwarded KUBECONFIG; no real server."""
    monkeypatch.setattr(sp, "_ensure_skypilot_api_running", lambda: None)
    monkeypatch.setattr(sk, "check_server_forwards_kubeconfig", lambda env: None)


def _build(text, context="remote", secret="kc"):
    return sk.build_instance_kubeconfig(
        [KubeConfigEntry(context=context, kubeconfig=secret)], {secret: text}, "env"
    )


def _env(config=None, secrets=None, name="env-k"):
    return sp.Skypilot(
        event_q=asyncio.Queue(),
        environment_config=EnvironmentConfig(
            name=name, type="Skypilot", config=config or _config()
        ),
        secrets=secrets if secrets is not None else {"kc": _kubeconfig()},
    )


def _config(contexts=("remote",), kube=True):
    cfg = {
        "default_cloud": "kubernetes",
        "cloud_config": {"kubernetes": {"allowed_contexts": list(contexts)}},
    }
    if kube:
        cfg["kube_configs"] = [{"context": "remote", "kubeconfig": "kc"}]
    return cfg


# --------------------------------------------------------------------------- #
# Validation of the secret
# --------------------------------------------------------------------------- #


class TestSecretValidation:
    @pytest.mark.parametrize(
        "user",
        [
            {"exec": {"command": "touch", "args": ["/tmp/PWNED"]}},
            {"auth-provider": {"name": "oidc", "config": {}}},
            {"client-key": "k.pem", "client-certificate": "c.pem"},
            {"tokenFile": "/var/run/token"},
        ],
    )
    def test_commands_and_file_paths_refused(self, user):
        with pytest.raises(ValueError, match="not allowed in kube_configs") as e:
            _build(_kubeconfig(user=user))
        assert "PWNED" not in str(e.value) and "/var/run" not in str(e.value)

    def test_ca_file_path_refused(self):
        with pytest.raises(ValueError, match="cluster fields"):
            _build(_kubeconfig(cluster={"server": "s", "certificate-authority": "ca"}))

    @pytest.mark.parametrize(
        "user",
        [
            {"token": "t"},
            {"client-certificate-data": "Y2VydA==", "client-key-data": "a2V5"},
            {"username": "u", "password": "p"},
        ],
    )
    def test_static_credentials_accepted(self, user):
        doc = _build(_kubeconfig(user=user))
        assert doc["users"][0] == {"name": "gbserver/remote", "user": user}

    @pytest.mark.parametrize(
        "user", [{"token": {"a": 1}}, {"token": 1}, {"client-key-data": "not b64!"}]
    )
    def test_bad_values_refused(self, user):
        with pytest.raises(ValueError, match="invalid values"):
            _build(_kubeconfig(user=user))

    def test_line_wrapped_base64_accepted(self):
        """kubectl accepts certs pasted as YAML block scalars."""
        wrapped = "Y2Vy\ndEhl\ncmU=\n"  # "certhere" wrapped
        doc = _build(
            _kubeconfig(cluster={"server": "s", "certificate-authority-data": wrapped})
        )
        assert doc["clusters"][0]["cluster"]["certificate-authority-data"] == wrapped

    def test_bool_flags_must_be_bool(self):
        with pytest.raises(ValueError, match="invalid values"):
            _build(
                _kubeconfig(cluster={"server": "s", "insecure-skip-tls-verify": "y"})
            )
        _build(_kubeconfig(cluster={"server": "s", "insecure-skip-tls-verify": True}))

    def test_context_only_keeps_namespace(self):
        doc = _build(_kubeconfig(ctx_extra={"namespace": "team-a"}))
        assert doc["contexts"][0]["context"] == {
            "namespace": "team-a",
            "cluster": "gbserver/remote",
            "user": "gbserver/remote",
        }
        with pytest.raises(ValueError, match="context fields"):
            _build(_kubeconfig(ctx_extra={"extensions": []}))

    @pytest.mark.parametrize(
        "ns", ["a;touch /tmp/pwn;", "Team", "-a", "a" * 64, "a b", "a\n"]
    )
    def test_bad_namespace_refused(self, ns):
        """SkyPilot puts it unquoted into an SSH ProxyCommand on the host."""
        with pytest.raises(ValueError, match="namespace"):
            _build(_kubeconfig(ctx_extra={"namespace": ns}))

    @pytest.mark.parametrize("ns", ["default", "team-a", "a" * 63, "x1"])
    def test_good_namespace_accepted(self, ns):
        _build(_kubeconfig(ctx_extra={"namespace": ns}))

    def test_sibling_keys_dropped(self):
        doc = yaml.safe_load(_kubeconfig())
        doc["users"][0]["exec"] = {"command": "evil"}
        out = _build(yaml.safe_dump(doc))
        assert "evil" not in yaml.safe_dump(out)

    @pytest.mark.parametrize(
        "text",
        [None, 5, "[" * 20000, "a: [", "- 1", yaml.safe_dump({"contexts": 5})],
    )
    def test_malformed_secret_is_a_value_error(self, text):
        with pytest.raises(ValueError) as e:
            _build(text)
        assert e.value.__cause__ is None

    @pytest.mark.parametrize(
        "name", ["mine ", "a\tb", "-x", "in-cluster\u200b", "", "remote\n"]
    )
    def test_odd_context_names_refused(self, name):
        with pytest.raises(ValueError, match="may use only"):
            _build(_kubeconfig(context=name), context=name)

    @pytest.mark.parametrize(
        "name", ["in-cluster", "ssh-pool", "gbserver-no-default-context"]
    )
    def test_reserved_names_refused(self, name):
        with pytest.raises(ValueError, match="reserved"):
            _build(_kubeconfig(context=name), context=name)

    def test_missing_secret_does_not_echo_the_value(self):
        with pytest.raises(ValueError, match="not in the space's secrets") as e:
            sk.build_instance_kubeconfig(
                [KubeConfigEntry(context="remote", kubeconfig="apiVersion: v1 SECRET")],
                {},
                "env",
            )
        assert "SECRET" not in str(e.value)

    def test_repeated_context_refused(self):
        e = KubeConfigEntry(context="remote", kubeconfig="kc")
        with pytest.raises(ValueError, match="more than once"):
            sk.build_instance_kubeconfig([e, e], {"kc": _kubeconfig()}, "env")


# --------------------------------------------------------------------------- #
# The user's own kubeconfig
# --------------------------------------------------------------------------- #


class TestUserKubeconfig:
    @pytest.mark.parametrize("clash", ["remote", "gbserver/remote"])
    def test_name_clashes_refused(self, tmp_path, clash):
        _user_kubeconfig(tmp_path / "user-kubeconfig", clash)
        with pytest.raises(SkypilotConfigCollisionError, match="your own kubeconfig"):
            _build(_kubeconfig())

    def test_users_current_context_naming_it_refused(self, tmp_path):
        (tmp_path / "user-kubeconfig").write_text(
            yaml.safe_dump({"current-context": "remote"})
        )
        with pytest.raises(SkypilotConfigCollisionError, match="current-context"):
            _build(_kubeconfig())

    def test_placeholder_default_without_a_user_default(self):
        doc = _build(_kubeconfig())
        assert doc["current-context"] == "gbserver-no-default-context"
        names = {c["name"] for c in doc["contexts"]}
        assert names == {"remote", "gbserver-no-default-context"}

    def test_no_placeholder_when_the_user_has_a_default(self, tmp_path):
        _user_kubeconfig(tmp_path / "user-kubeconfig", "mine", current="mine")
        doc = _build(_kubeconfig())
        assert "current-context" not in doc
        assert [c["name"] for c in doc["contexts"]] == ["remote"]

    def test_user_files_come_first_on_the_request(self, tmp_path):
        f = tmp_path / "instance.yaml"
        assert sk.request_kubeconfig(f) == os.pathsep.join(
            [str(tmp_path / "user-kubeconfig"), str(f)]
        )


# --------------------------------------------------------------------------- #
# Instance files
# --------------------------------------------------------------------------- #


class TestInstanceFile:
    def test_written_0600_with_only_its_contexts(self, tmp_path):
        path = sk.instance_kubeconfig_path("abc")
        sk.write_instance_kubeconfig(_build(_kubeconfig()), path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert path.parent == tmp_path / ".sky" / sk.KUBE_DIR
        assert path.name.startswith(f"{os.getpid()}-")

    def test_sweep_keeps_live_processes_and_removes_the_rest(self, tmp_path):
        d = tmp_path / ".sky" / sk.KUBE_DIR
        d.mkdir(parents=True)
        me = sk._process_tag()
        pid, start = me.split("-")
        files = {
            "mine": d / f"{me}-a.yaml",
            "mine_tmp": d / f"{me}-a.x.tmp",
            "dead_pid": d / "999999999-1-b.yaml",
            "reused_pid": d / f"{pid}-{int(start) - 1000}-c.yaml",
            "other": d / "notes.txt",
        }
        for f in files.values():
            f.write_text("x")
        sk.sweep_stale_instance_files(d)
        assert {k for k, f in files.items() if f.exists()} == {
            "mine",
            "mine_tmp",
            "other",
        }

    def test_existing_directory_is_tightened_to_0700(self, tmp_path):
        path = sk.instance_kubeconfig_path("abc")
        path.parent.mkdir(parents=True, mode=0o755)
        os.chmod(path.parent, 0o755)
        sk.write_instance_kubeconfig({"a": 1}, path)
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert not list(path.parent.glob("*.tmp"))

    def test_concurrent_writes_never_collide(self, tmp_path):
        import concurrent.futures

        path = sk.instance_kubeconfig_path("same")
        doc = _build(_kubeconfig())
        with concurrent.futures.ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda _: sk.write_instance_kubeconfig(doc, path), range(64)))
        assert yaml.safe_load(path.read_text()) == doc

    def test_file_removed_when_the_instance_is_collected(self, local_server):
        import gc

        env = _env()
        env._kube_request_kubeconfig(True)
        path = env._kube_file
        assert path.exists()
        del env
        gc.collect()
        assert not path.exists()


# --------------------------------------------------------------------------- #
# The SkyPilot client patch
# --------------------------------------------------------------------------- #


class TestRequestPatch:
    def test_forwards_the_contextvar_only_when_set(self, monkeypatch):
        from sky.server import common
        from sky.server.requests import payloads

        monkeypatch.setattr(common, "is_api_server_local", lambda *a: True)
        sk.install()
        assert "KUBECONFIG" not in payloads.request_body_env_vars() or (
            payloads.request_body_env_vars()["KUBECONFIG"] == os.environ["KUBECONFIG"]
        )
        token = sk.set_request_kubeconfig("/a:/b")
        try:
            assert payloads.request_body_env_vars()["KUBECONFIG"] == "/a:/b"
            body = payloads.RequestBody()
            assert body.env_vars["KUBECONFIG"] == "/a:/b"
        finally:
            sk.reset_request_kubeconfig(token)

    def test_not_forwarded_to_a_remote_server(self, monkeypatch):
        from sky.server import common
        from sky.server.requests import payloads

        monkeypatch.setattr(common, "is_api_server_local", lambda *a: False)
        sk.install()
        token = sk.set_request_kubeconfig("/a:/b")
        try:
            assert payloads.request_body_env_vars().get("KUBECONFIG") != "/a:/b"
        finally:
            sk.reset_request_kubeconfig(token)

    def test_upgrade_guard(self):
        """Fails on a SkyPilot upgrade that breaks the single-entry-point assumption."""
        import pydantic
        from sky.server.requests import payloads

        original = getattr(payloads.request_body_env_vars, "__wrapped__", None)
        assert original is not None, "skypilot_kube.install() didn't patch"
        assert inspect.signature(original).parameters == {}
        assert "KUBECONFIG" in payloads.EXTERNAL_LOCAL_ENV_VARS
        assert "request_body_env_vars()" in inspect.getsource(
            payloads.RequestBody.__init__
        )
        others = {
            n
            for n, c in inspect.getmembers(payloads, inspect.isclass)
            if issubclass(c, pydantic.BaseModel)
            and c.__module__ == payloads.__name__
            and not issubclass(c, payloads.RequestBody)
        }
        # These only read an existing request (get, stream, status).
        assert others <= {
            "BasePayload",
            "RequestPayload",
            "RequestStatusBody",
            "StreamBody",
            "UploadZipFileResponse",
        }, others

    def test_concurrent_tasks_build_bodies_with_their_own_value(self, monkeypatch):
        from sky.server import common
        from sky.server.requests import payloads

        monkeypatch.setattr(common, "is_api_server_local", lambda *a: True)
        sk.install()

        async def body_for(value):
            token = sk.set_request_kubeconfig(value)
            try:
                await asyncio.sleep(0)
                return await asyncio.to_thread(
                    lambda: payloads.RequestBody().env_vars["KUBECONFIG"]
                )
            finally:
                sk.reset_request_kubeconfig(token)

        async def run():
            return await asyncio.gather(*(body_for(f"/v{i}") for i in range(10)))

        assert asyncio.run(run()) == [f"/v{i}" for i in range(10)]

    def test_only_logs_requests_share_server_threads(self):
        """Guard: SkyPilot runs only /logs and /hook_logs on shared server threads.

        Those share the server's Kubernetes client cache, so gbserver never sends
        them for a kube_configs instance (tail_logs is only used by the stream
        log mode, which is downgraded; hook_logs is never used). A SkyPilot
        upgrade that moves more requests there fails this test.
        """
        import re

        from sky.server import server as sky_server

        src = Path(sky_server.__file__).read_text()
        routes = set()
        for m in re.finditer(r"execute_request_in_coroutine\(", src):
            routes.add(re.findall(r"@app\.\w+\('([^']+)'", src[: m.start()])[-1])
        assert routes == {"/logs", "/hook_logs"}, routes
        gb = Path(sp.__file__).parent.parent
        users = {
            str(f.relative_to(gb))
            for f in gb.rglob("*.py")
            if re.search(r"sky\.(tail_logs|hook_logs)\(|\.hook_logs\(", f.read_text())
        }
        assert users <= {"monitoring/streams/skypilot_log_stream.py"}, users

    def test_sky_executor_carries_the_contextvar(self):
        async def run():
            token = sk.set_request_kubeconfig("/scoped")
            try:
                return await sp._sky_submit_to_thread(sk.current_request_kubeconfig)
            finally:
                sk.reset_request_kubeconfig(token)

        assert asyncio.run(run()) == "/scoped"


# --------------------------------------------------------------------------- #
# The environment instance
# --------------------------------------------------------------------------- #


class TestEnvironmentInstance:
    def test_no_kube_configs_means_no_change(self):
        env = _env(_config(kube=False), secrets={})

        async def call():
            return sk.current_request_kubeconfig()

        with patch.object(sp, "_ensure_skypilot_api_running") as started:
            assert asyncio.run(sp._kube_scoped(True)(lambda self: call())(env)) is None
        started.assert_not_called()

    def test_scope_sets_and_resets_the_request_kubeconfig(self, local_server):
        env = _env()
        seen = []

        async def body(self):
            seen.append(sk.current_request_kubeconfig())
            seen.append(await asyncio.to_thread(sk.current_request_kubeconfig))

        asyncio.run(sp._kube_scoped(True)(body)(env))
        assert seen[0] == seen[1] == sk.request_kubeconfig(env._kube_file)
        assert sk.current_request_kubeconfig() is None
        assert os.environ["KUBECONFIG"].endswith("user-kubeconfig")  # untouched

    def test_two_instances_same_context_name_stay_apart(self, local_server):
        a = _env(secrets={"kc": _kubeconfig(user={"token": "A"})}, name="env-k")
        b = _env(secrets={"kc": _kubeconfig(user={"token": "B"})}, name="env-k")
        va, vb = a._kube_request_kubeconfig(True), b._kube_request_kubeconfig(True)
        assert va != vb and str(b._kube_file) not in va and str(a._kube_file) not in vb
        tokens = {
            p: yaml.safe_load(p.read_text())["users"][0]["user"]["token"]
            for p in (a._kube_file, b._kube_file)
        }
        assert tokens == {a._kube_file: "A", b._kube_file: "B"}

    def test_secrets_are_fixed_per_instance(self, local_server):
        """A rotated secret reaches the next build's instance, not a running one."""
        a = _env(secrets={"kc": _kubeconfig(user={"token": "old"})})
        a._kube_request_kubeconfig(True)
        b = _env(secrets={"kc": _kubeconfig(user={"token": "new"})})
        b._kube_request_kubeconfig(True)

        def tok(e):
            return yaml.safe_load(e._kube_file.read_text())["users"][0]["user"]["token"]

        assert (tok(a), tok(b)) == ("old", "new")

    def test_concurrent_first_calls_on_one_instance(self, local_server, monkeypatch):
        """Parallel steps of one environment share an instance: one file, one value."""
        import concurrent.futures
        import threading
        import time

        env = _env()
        real = sk.instance_kubeconfig_path

        def slow_path(instance_id, home=None):
            time.sleep(0.05)  # widen the first-use window
            return real(instance_id, home)

        monkeypatch.setattr(sk, "instance_kubeconfig_path", slow_path)
        barrier = threading.Barrier(8)

        def call(_):
            barrier.wait()
            return env._kube_request_kubeconfig(True)

        with concurrent.futures.ThreadPoolExecutor(8) as ex:
            results = list(ex.map(call, range(8)))
        assert len(set(results)) == 1 and results[0]
        assert len(list(env._kube_file.parent.glob("*.yaml"))) == 1

    def test_missing_file_is_rewritten(self, local_server):
        env = _env()
        env._kube_request_kubeconfig(True)
        env._kube_file.unlink()
        env._kube_request_kubeconfig(False)
        assert env._kube_file.exists()

    def test_bad_secret_fails_only_its_instance(self, local_server):
        bad = _env(secrets={"kc": "not: [yaml"})
        good = _env()
        with pytest.raises(ValueError):
            bad._kube_request_kubeconfig(True)
        assert good._kube_request_kubeconfig(True)

    def test_non_strict_failure_logs_once_and_sends_no_file(self, caplog):
        env = _env(secrets={})
        assert env._kube_request_kubeconfig(False) is None
        assert env._kube_request_kubeconfig(False) is None
        assert caplog.text.count("kube_configs not applied") == 1

    @pytest.mark.parametrize(
        "cloud_config,ok",
        [
            ({"kubernetes": {"allowed_contexts": ["remote"]}}, True),
            ({"kubernetes": {"allowed_contexts": "all"}}, True),
            ({"kubernetes": {"allowed_contexts": ["other"]}}, False),
            ({}, False),
            (
                {
                    "kubernetes": {"allowed_contexts": ["other"]},
                    "workspaces": {
                        "default": {"kubernetes": {"allowed_contexts": ["remote"]}}
                    },
                },
                True,
            ),
        ],
    )
    def test_contexts_must_be_pinned(self, cloud_config, ok):
        cfg = {**_config(), "cloud_config": cloud_config}
        env = _env(cfg)
        if ok:
            assert env._kube_config_entries()
        else:
            with pytest.raises(ValueError, match="must also be listed"):
                env._kube_config_entries()

    def test_malformed_entry_does_not_echo_the_input(self):
        cfg = {
            **_config(),
            "kube_configs": [{"context": "remote", "kubeconfig": {"token": "S3CRET"}}],
        }
        with pytest.raises(ValueError, match="malformed") as e:
            _env(cfg)._kube_config_entries()
        assert "S3CRET" not in str(e.value)

    def test_stream_log_retrieval_downgraded_with_kube_configs(self, caplog):
        assert _env()._kube_safe_log_mode("stream") == sp.LOG_RETRIEVAL_PERIODIC
        assert "not supported with kube_configs" in caplog.text
        for mode in ("on_completion", "periodic", "startup_window"):
            assert _env()._kube_safe_log_mode(mode) == mode
        plain = _env(_config(kube=False), secrets={})
        assert plain._kube_safe_log_mode("stream") == "stream"

    def test_no_kube_configs_leaves_the_instance_untouched(self):
        env = _env(_config(kube=False), secrets={})
        assert env._kube_file is None and env._kube_request_kubeconfig(True) is None

    def test_strict_checks_the_server_first(self, monkeypatch):
        monkeypatch.setattr(sp, "_ensure_skypilot_api_running", lambda: None)

        def refuse(env_name):
            raise ValueError("server drops it")

        monkeypatch.setattr(sk, "check_server_forwards_kubeconfig", refuse)
        env = _env()
        with pytest.raises(ValueError, match="server drops it"):
            env._kube_request_kubeconfig(True)
        assert env._kube_file is None


# --------------------------------------------------------------------------- #
# The API server precondition
# --------------------------------------------------------------------------- #


class TestServerPrecondition:
    def _response(self, ok=True, forwarded=True):
        r = MagicMock(ok=ok)
        r.json.return_value = {"forwarded": forwarded}
        return r

    def test_remote_server_refused(self, monkeypatch):
        from sky.server import common

        monkeypatch.setattr(common, "is_api_server_local", lambda *a: False)
        with pytest.raises(ValueError, match="local SkyPilot API server"):
            sk.check_server_forwards_kubeconfig("e")

    @pytest.mark.parametrize(
        "response,ok",
        [((True, True), True), ((True, False), False), ((False, True), False)],
    )
    def test_endpoint_decides(self, monkeypatch, response, ok):
        from sky.server import common

        monkeypatch.setattr(common, "is_api_server_local", lambda *a: True)
        monkeypatch.setattr(
            common,
            "make_authenticated_request",
            lambda *a, **k: self._response(*response),
        )
        if ok:
            sk.check_server_forwards_kubeconfig("e")
        else:
            with pytest.raises(ValueError, match="does not apply"):
                sk.check_server_forwards_kubeconfig("e")

    @pytest.mark.parametrize(
        "flag,forwarded", [("0", True), ("1", False), (None, False)]
    )
    def test_endpoint_reports_the_servers_flag(self, monkeypatch, flag, forwarded):
        from sky.skylet import constants

        if flag is None:
            monkeypatch.delenv(
                constants.ENV_VAR_ENFORCE_MOUNT_CONTAINMENT, raising=False
            )
        else:
            monkeypatch.setenv(constants.ENV_VAR_ENFORCE_MOUNT_CONTAINMENT, flag)
        from gbserver.utils import sky_patches

        assert sky_patches.kubeconfig_forwarding_endpoint() == {"forwarded": forwarded}

    def test_plugin_serves_the_endpoint(self):
        from gbserver.utils import sky_patches

        app = MagicMock()
        with patch.object(sky_patches, "apply"):
            sky_patches.SkyPatchesPlugin().install(MagicMock(app=app))
        app.add_api_route.assert_called_once_with(
            sky_patches.KUBECONFIG_FORWARDING_ROUTE,
            sky_patches.kubeconfig_forwarding_endpoint,
            methods=["GET"],
        )

    def test_plugin_does_not_import_the_environment_package(self):
        """The API server must not load gbserver's environments (or its client patch)."""
        import subprocess
        import sys

        code = (
            "import sys; from gbserver.utils import sky_patches; "
            "print('gbserver.environment' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False"

    def test_non_json_response_is_refused(self, monkeypatch):
        from sky.server import common

        r = MagicMock(ok=True)
        r.json.side_effect = ValueError("not json")
        monkeypatch.setattr(common, "is_api_server_local", lambda *a: True)
        monkeypatch.setattr(common, "make_authenticated_request", lambda *a, **k: r)
        with pytest.raises(ValueError, match="does not apply"):
            sk.check_server_forwards_kubeconfig("e")


# --------------------------------------------------------------------------- #
# Managed jobs and wiring
# --------------------------------------------------------------------------- #


def test_managed_jobs_refuse_kube_configs():
    from gbserver.environment.skypilot_managed import Skypilot_managed

    with pytest.raises(ValueError, match="not supported for Skypilot_managed"):
        Skypilot_managed(
            event_q=asyncio.Queue(),
            environment_config=EnvironmentConfig(
                name="m", type="Skypilot_managed", config=_config()
            ),
            secrets={"kc": _kubeconfig()},
        )


def test_every_request_path_is_scoped():
    """Each Skypilot method that can reach a SkyPilot request must be scoped.

    Builds the module's call graph: functions that send a request directly
    (sky.<verb>), the log stream (sky.tail_logs in another module), and
    everything that reaches them. Every such method with no scoped caller
    must itself carry @_kube_scoped, so its requests carry the instance's
    KUBECONFIG. A new request path that forgets it fails here.
    """
    verbs = {
        "launch", "down", "status", "job_status", "tail_logs", "download_logs",
        "cancel", "exec", "stop", "queue", "api_cancel", "start", "autostop",
    }  # fmt: skip
    tree = ast.parse(Path(sp.__file__).read_text())
    funcs, scoped = {}, set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node
        if isinstance(node, ast.ClassDef) and node.name == "Skypilot":
            for n in node.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    funcs["self." + n.name] = n
                    if any("_kube_scoped" in ast.unparse(d) for d in n.decorator_list):
                        scoped.add("self." + n.name)

    def sends(node):
        for n in ast.walk(node):
            if isinstance(n, ast.Attribute):
                parts = ast.unparse(n).split(".")
                if parts[0] == "sky" and parts[-1] in verbs and len(parts) <= 3:
                    return True
        return False

    def callees(node):
        out = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Attribute) and ast.unparse(n.value) == "self":
                out.add("self." + n.attr)
            if isinstance(n, ast.Name):
                out.add(n.id)
        return out & set(funcs)

    graph = {k: callees(v) for k, v in funcs.items()}
    senders = {k for k, v in funcs.items() if sends(v)} | {
        "self._start_log_stream_task"
    }
    assert senders >= {"_launch_with_cloud_config", "self._teardown"}  # sanity
    reach, changed = set(senders), True
    while changed:
        changed = False
        for k, cs in graph.items():
            if k not in reach and cs & reach:
                reach.add(k)
                changed = True
    # A function is covered when it is scoped, or it has callers and every
    # caller that reaches a request is covered. Iterate to a fixed point.
    callers = {
        k: {c for c, cs in graph.items() if k in cs and c in reach} for k in reach
    }
    covered = set(scoped & reach)
    changed = True
    while changed:
        changed = False
        for k in reach - covered:
            if callers[k] and callers[k] <= covered:
                covered.add(k)
                changed = True
    # retry_workload is called by the RetryHandler, whose task the scoped
    # monitor_skypilot_monitor starts (Environment._with_retry_handler), so it
    # runs in the monitor's scope.
    monitor_src = ast.unparse(funcs["self.monitor_skypilot_monitor"])
    assert "_with_retry_handler" in monitor_src
    covered.add("self.retry_workload")
    assert reach <= covered, sorted(reach - covered)
    # The entry points gbserver's dispatch calls by name are scoped, strict for
    # setup and launch.
    for name, strict in {
        "setup_skypilot": True,
        "launch_skypilot": True,
        "monitor_skypilot_monitor": False,
        "cleanup_skypilot": False,
        "teardown_skypilot": False,
        "launch_skypilot_teardown": False,
    }.items():
        assert getattr(sp.Skypilot, name).__kube_scoped__ is strict, name
    # retry_workload runs inside the monitor's scope and calls the scoped
    # cleanup/launch; scoping it again would hide its own errors.
    assert not hasattr(sp.Skypilot.retry_workload, "__kube_scoped__")
