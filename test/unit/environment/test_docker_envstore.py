#!/usr/bin/env python3

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

"""Docker env:// assets: inputs bind-mounted read-only; outputs registered at host paths."""

import asyncio
import logging
import os

import pytest

from gbcommon.uri.uri import URI
from gbserver.environment.docker import Docker
from gbserver.environment.environment import BINDING_KEY, Environment
from gbserver.types.buildevent import EntityRunMetadata


@pytest.fixture
def docker_env():
    return Docker(event_q=asyncio.Queue())


def _env_uri(path) -> str:
    # ``env://{{ binding.path }}`` with an absolute path renders as env:///abs/...
    return f"env://{path}"


@pytest.mark.asyncio
async def test_file_input_is_mounted_read_only(docker_env, tmp_path):
    f = tmp_path / "gen" / "generation.json"
    f.parent.mkdir()
    f.write_text("{}")
    uri = _env_uri(f)
    assert uri.startswith("env:///")
    binding, step = await docker_env.pullasset_envstore(uri=uri)
    assert step is None
    path = binding[BINDING_KEY]["path"]
    assert path.startswith("/gb-env-inputs/") and path.endswith("/generation.json")
    vol = docker_env._env_input_volumes[str(f.parent)]
    assert vol["mode"] == "ro"
    assert path == f"{vol['bind']}/generation.json"
    # Not a shared mount: only launches whose bindings reference it get it.
    assert docker_env._extra_volumes == {}
    assert docker_env._resolve_host_path(path) == str(f)


@pytest.mark.asyncio
async def test_dir_input_is_mounted_whole(docker_env, tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    binding, _ = await docker_env.pullasset_envstore(uri=_env_uri(d))
    vol = docker_env._env_input_volumes[str(d)]
    assert vol["mode"] == "ro"
    assert binding[BINDING_KEY]["path"] == vol["bind"]


@pytest.mark.asyncio
async def test_missing_path_raises(docker_env, tmp_path):
    with pytest.raises(FileNotFoundError):
        await docker_env.pullasset_envstore(uri=_env_uri(tmp_path / "nope.json"))


@pytest.mark.asyncio
async def test_distinct_inputs_get_distinct_mounts_shared_parent_reused(
    docker_env, tmp_path
):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    a1 = a_dir / "one.json"
    a2 = a_dir / "two.json"
    b1 = b_dir / "one.json"
    for f in (a1, a2, b1):
        f.write_text("{}")

    pa1 = (await docker_env.pullasset_envstore(uri=_env_uri(a1)))[0][BINDING_KEY]
    pa2 = (await docker_env.pullasset_envstore(uri=_env_uri(a2)))[0][BINDING_KEY]
    pb1 = (await docker_env.pullasset_envstore(uri=_env_uri(b1)))[0][BINDING_KEY]

    mount_a = docker_env._env_input_volumes[str(a_dir)]["bind"]
    mount_b = docker_env._env_input_volumes[str(b_dir)]["bind"]
    assert mount_a != mount_b
    assert pa1["path"] == f"{mount_a}/one.json"
    assert pa2["path"] == f"{mount_a}/two.json"
    assert pb1["path"] == f"{mount_b}/one.json"
    assert pa1["path"] != pb1["path"]
    assert len(docker_env._env_input_volumes) == 2


@pytest.mark.asyncio
async def test_reuses_existing_read_only_mount(docker_env, tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    docker_env._extra_volumes[str(snap)] = {
        "bind": "/gb-hf-models/org/repo",
        "mode": "ro",
    }
    binding, _ = await docker_env.pullasset_envstore(uri=_env_uri(snap / "config.json"))
    assert binding[BINDING_KEY]["path"] == "/gb-hf-models/org/repo/config.json"
    assert len(docker_env._extra_volumes) == 1
    assert docker_env._env_input_volumes == {}


@pytest.mark.asyncio
async def test_reuses_hf_mount_through_symlinked_path(docker_env, tmp_path):
    real = tmp_path / "real-snap"
    real.mkdir()
    (real / "config.json").write_text("{}")
    link = tmp_path / "snap-link"
    link.symlink_to(real)
    # The HF pull keys the mount by its unresolved path.
    docker_env._extra_volumes[str(link)] = {"bind": "/gb-hf-models/o/r", "mode": "ro"}
    binding, _ = await docker_env.pullasset_envstore(uri=_env_uri(real / "config.json"))
    assert binding[BINDING_KEY]["path"] == "/gb-hf-models/o/r/config.json"
    assert docker_env._env_input_volumes == {}


# ---------------------------------------------------------------------------
# env:// outputs: a /gb-workspace binding is registered at its host path.
# ---------------------------------------------------------------------------

_RESULTS = "/gb-workspace/outputs/x/results.json"


def _push_env(docker_env):
    """Route every URI to a minimal store of the URI's own scheme."""
    from types import SimpleNamespace

    def _storeconfig(uri, **kw):
        scheme = URI.get_uristr(uri).split("://", 1)[0]
        return (
            SimpleNamespace(type=f"{scheme}store", get_secrets=lambda: {}),
            SimpleNamespace(push=None, pull=None),
        )

    docker_env._get_storeconfig = _storeconfig
    Environment._thread_local.asset_events = {}
    return docker_env


def _register_launch(docker_env, tsr_id, ws):
    docker_env._workspaces_by_targetsteprun[tsr_id] = str(ws)


async def _push(docker_env, binding, uristr="env://{{ binding.path }}", tsr="tsrB"):
    task = docker_env.pushasset(
        task_group=None,  # type: ignore[arg-type]
        binding=binding,
        uristr=uristr,
        binding_id="granite_results",
        run_metadata=EntityRunMetadata(targetsteprun_id=tsr),
    )
    uri = await task
    events = []
    while not docker_env.event_q.empty():
        events.append(docker_env.event_q.get_nowait())
    return URI.get_uristr(uri), events


@pytest.mark.asyncio
async def test_env_output_in_workspace_registers_host_path(docker_env, tmp_path):
    env = _push_env(docker_env)
    _register_launch(env, "tsrB", tmp_path / "wsB")
    binding = {"path": _RESULTS}
    uristr, events = await _push(env, binding)
    expected = f"env://{tmp_path}/wsB/outputs/x/results.json"
    assert uristr == expected
    assert {e.payload.uri for e in events} == {expected}
    # env:// pushes are never cached, so a pull always reaches the store.
    assert expected not in env.asset_bindings
    # The caller's binding object is not mutated.
    assert binding == {"path": _RESULTS}


@pytest.mark.asyncio
async def test_env_output_resolves_to_producing_launch(docker_env, tmp_path):
    env = _push_env(docker_env)
    _register_launch(env, "tsrA", tmp_path / "wsA")
    _register_launch(env, "tsrB", tmp_path / "wsB")
    # Launch order must not matter: A registered last still loses for B's push.
    _register_launch(env, "tsrA", tmp_path / "wsA")
    uristr, _ = await _push(env, {"path": _RESULTS}, tsr="tsrB")
    assert uristr == f"env://{tmp_path}/wsB/outputs/x/results.json"


@pytest.mark.asyncio
async def test_env_output_host_path_unchanged(docker_env, tmp_path):
    env = _push_env(docker_env)
    _register_launch(env, "tsrB", tmp_path / "wsB")
    host = str(tmp_path / "elsewhere" / "results.json")
    uristr, _ = await _push(env, {"path": host})
    assert uristr == f"env://{host}"


@pytest.mark.asyncio
async def test_file_output_binding_untouched(docker_env, tmp_path):
    env = _push_env(docker_env)
    _register_launch(env, "tsrB", tmp_path / "wsB")
    seen = {}

    async def _file_handler(self, binding=None, uri=None, **kw):
        seen["binding"] = binding
        return uri

    env.pushasset_types = {"filestore": _file_handler}
    uristr, _ = await _push(
        env, {"path": _RESULTS}, uristr=f"file://{tmp_path}/out/results.json"
    )
    assert seen["binding"] == {"path": _RESULTS}
    assert uristr == f"file://{tmp_path}/out/results.json"


@pytest.mark.asyncio
async def test_env_output_unknown_targetsteprun_raises(docker_env):
    env = _push_env(docker_env)
    with pytest.raises(LookupError, match="nope"):
        await _push(env, {"path": _RESULTS}, tsr="nope")
    # Nothing registered: the container path exists on no host.
    assert env.event_q.empty()


@pytest.mark.asyncio
async def test_env_output_unknown_targetsteprun_raises_in_task_group(docker_env):
    env = _push_env(docker_env)
    with pytest.raises(ExceptionGroup) as excinfo:
        async with asyncio.TaskGroup() as tg:
            env.pushasset(
                task_group=tg,
                binding={"path": _RESULTS},
                uristr="env://{{ binding.path }}",
                binding_id="granite_results",
                run_metadata=EntityRunMetadata(targetsteprun_id="nope"),
            )
    assert excinfo.group_contains(LookupError)


@pytest.mark.asyncio
async def test_launch_records_workspace_by_targetsteprun(docker_env, tmp_path):
    from unittest.mock import MagicMock

    client = MagicMock()
    docker_env._get_docker = MagicMock(return_value=(MagicMock(), client))
    docker_env._resolve_image = MagicMock(return_value="img:latest")
    docker_env._pull_image = MagicMock()
    docker_env._release_monitors = MagicMock()
    docker_env.get_launch_env_vars = MagicMock(return_value={})
    await docker_env.launch_docker(
        launch_id="launchid123",
        targetsteprun_asset_dir=tmp_path / "wsB",
        config={},
        launcher_config={"command": "true"},
        step={"name": "s"},
        run_metadata={"targetsteprun_id": "tsrB"},
    )
    assert docker_env._workspaces_by_targetsteprun == {"tsrB": str(tmp_path / "wsB")}
    await docker_env.cleanup_docker(launch_id="launchid123")
    # Survives cleanup so a push that races teardown still resolves.
    assert docker_env._workspaces_by_targetsteprun == {"tsrB": str(tmp_path / "wsB")}


# ---------------------------------------------------------------------------
# Per-launch workspace mounts: one instance serves every Docker target.
# ---------------------------------------------------------------------------


def _mock_launch_env(docker_env):
    from unittest.mock import MagicMock

    client = MagicMock()
    docker_env._get_docker = MagicMock(return_value=(MagicMock(), client))
    docker_env._resolve_image = MagicMock(return_value="img:latest")
    docker_env._pull_image = MagicMock()
    docker_env._release_monitors = MagicMock()
    docker_env.get_launch_env_vars = MagicMock(return_value={})
    return client


async def _launch(docker_env, launch_id, ws, tsr_id):
    await docker_env.launch_docker(
        launch_id=launch_id,
        targetsteprun_asset_dir=ws,
        config={},
        launcher_config={"command": "true"},
        step={"name": "s"},
        run_metadata={"targetsteprun_id": tsr_id},
    )


def _workspace_binds(volumes):
    return {h: v for h, v in volumes.items() if v["bind"] == "/gb-workspace"}


@pytest.mark.asyncio
async def test_second_launch_mounts_only_its_own_workspace(docker_env, tmp_path):
    client = _mock_launch_env(docker_env)
    ro = {str(tmp_path / "hf"): {"bind": "/gb-hf-models/x", "mode": "ro"}}
    docker_env._extra_volumes.update(ro)
    await _launch(docker_env, "launchAAAA", tmp_path / "wsA", "tsrA")
    await _launch(docker_env, "launchBBBB", tmp_path / "wsB", "tsrB")
    vols_a = client.containers.run.call_args_list[0].kwargs["volumes"]
    vols_b = client.containers.run.call_args_list[1].kwargs["volumes"]
    assert _workspace_binds(vols_a) == {
        str(tmp_path / "wsA"): {"bind": "/gb-workspace", "mode": "rw"}
    }
    assert _workspace_binds(vols_b) == {
        str(tmp_path / "wsB"): {"bind": "/gb-workspace", "mode": "rw"}
    }
    # Shared read-only mounts reach every container; launches never add to them.
    assert vols_b[str(tmp_path / "hf")] == ro[str(tmp_path / "hf")]
    assert docker_env._extra_volumes == ro


# ---------------------------------------------------------------------------
# file:// outputs copy from the producing launch's workspace.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_push_copies_from_producing_launch(docker_env, tmp_path):
    _mock_launch_env(docker_env)
    for name in ("wsA", "wsB"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "r.txt").write_text(name)
    await _launch(docker_env, "launchBBBB", tmp_path / "wsB", "tsrB")
    await _launch(docker_env, "launchAAAA", tmp_path / "wsA", "tsrA")
    dest = tmp_path / "out" / "r.txt"
    dest.parent.mkdir()
    await docker_env.pushasset_filestore(
        binding={"path": "/gb-workspace/r.txt"},
        uri=f"file://{dest}",
        run_metadata=EntityRunMetadata(targetsteprun_id="tsrB"),
    )
    assert dest.read_text() == "wsB"


@pytest.mark.asyncio
async def test_file_push_unknown_launch_raises(docker_env, tmp_path):
    with pytest.raises(LookupError, match="nope"):
        await docker_env.pushasset_filestore(
            binding={"path": "/gb-workspace/r.txt"},
            uri=f"file://{tmp_path}/r.txt",
            run_metadata=EntityRunMetadata(targetsteprun_id="nope"),
        )


@pytest.mark.asyncio
async def test_file_push_host_path_unchanged(docker_env, tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("host")
    dest = tmp_path / "dest.txt"
    await docker_env.pushasset_filestore(
        binding={"path": str(src)}, uri=f"file://{dest}"
    )
    assert dest.read_text() == "host"


# ---------------------------------------------------------------------------
# Docker -> Docker env:// chaining on one instance.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_env_output_of_earlier_docker_target_is_mounted_on_pull(
    docker_env, tmp_path
):
    env = _push_env(docker_env)
    ws = tmp_path / "wsA"
    (ws / "outputs" / "x").mkdir(parents=True)
    (ws / "outputs" / "x" / "results.json").write_text("{}")
    _register_launch(env, "tsrA", ws)
    uristr, _ = await _push(env, {"path": _RESULTS}, tsr="tsrA")
    host = f"{ws}/outputs/x/results.json"
    assert uristr == f"env://{host}"
    assert uristr not in env.asset_bindings

    binding, step = await env.pullasset(task_group=None, uri=URI.get_uri(uristr))
    assert step is None
    path = binding[BINDING_KEY]["path"]
    vol = env._env_input_volumes[str(ws / "outputs" / "x")]
    assert vol["mode"] == "ro"
    assert path == f"{vol['bind']}/results.json"
    assert path.startswith("/gb-env-inputs/")


# ---------------------------------------------------------------------------
# Mount scope: each refusal names the rule that fired.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_root_dir_input_refused(docker_env):
    with pytest.raises(ValueError, match="contains the home directory"):
        await docker_env.pullasset_envstore(uri="env:///")
    assert docker_env._env_input_volumes == {}


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    from pathlib import Path

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


@pytest.mark.asyncio
async def test_file_in_home_dir_refused(docker_env, fake_home):
    (fake_home / "generation.json").write_text("{}")
    with pytest.raises(ValueError, match="contains the home directory"):
        await docker_env.pullasset_envstore(uri=_env_uri(fake_home / "generation.json"))
    assert docker_env._env_input_volumes == {}


@pytest.mark.asyncio
async def test_home_dir_input_refused(docker_env, fake_home):
    with pytest.raises(ValueError, match="contains the home directory"):
        await docker_env.pullasset_envstore(uri=_env_uri(fake_home))


@pytest.mark.asyncio
async def test_home_dir_via_symlink_refused(docker_env, fake_home, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(fake_home)
    (fake_home / "generation.json").write_text("{}")
    with pytest.raises(ValueError, match="contains the home directory"):
        await docker_env.pullasset_envstore(uri=_env_uri(link / "generation.json"))


@pytest.mark.asyncio
async def test_subdir_of_home_is_mounted(docker_env, fake_home):
    sub = fake_home / "run1"
    sub.mkdir()
    (sub / "generation.json").write_text("{}")
    binding, _ = await docker_env.pullasset_envstore(
        uri=_env_uri(sub / "generation.json")
    )
    assert docker_env._env_input_volumes[str(sub)]["mode"] == "ro"
    assert binding[BINDING_KEY]["path"].endswith("/generation.json")


@pytest.mark.asyncio
async def test_parent_of_home_dir_refused(docker_env, fake_home):
    # env:///home/report.json mounts /home, which holds the user's home.
    (fake_home.parent / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="contains the home directory"):
        await docker_env.pullasset_envstore(
            uri=_env_uri(fake_home.parent / "report.json")
        )


@pytest.mark.asyncio
async def test_hidden_home_subdir_refused(docker_env, fake_home):
    ssh = fake_home / ".ssh" / "keys"
    ssh.mkdir(parents=True)
    with pytest.raises(ValueError, match="hidden home directory"):
        await docker_env.pullasset_envstore(uri=_env_uri(ssh))


@pytest.mark.asyncio
async def test_gb_home_dir_refused(docker_env, tmp_path, monkeypatch):
    gb_home = tmp_path / "gbhome"
    (gb_home / "secrets").mkdir(parents=True)
    monkeypatch.setenv("GB_HOME_DIR", str(gb_home))
    with pytest.raises(ValueError, match="GB home directory"):
        await docker_env.pullasset_envstore(uri=_env_uri(gb_home / "secrets"))
    # A dir that contains the GB home is refused too.
    with pytest.raises(ValueError, match="GB home directory"):
        await docker_env.pullasset_envstore(uri=_env_uri(tmp_path))


@pytest.mark.asyncio
async def test_system_dir_input_refused(docker_env):
    # env:///etc/hosts would mount all of /etc.
    with pytest.raises(ValueError, match="system directory"):
        await docker_env.pullasset_envstore(uri="env:///etc/hosts")


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads any dir"
)
@pytest.mark.asyncio
async def test_unreadable_dir_refused(docker_env, tmp_path):
    # The daemon mounts as root: never hand the container a dir the server
    # user cannot read itself (e.g. /root, /var/lib/docker).
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        with pytest.raises(ValueError, match="not readable"):
            await docker_env.pullasset_envstore(uri=_env_uri(locked))
    finally:
        locked.chmod(0o700)


@pytest.mark.parametrize("src", ["/root", "/var/lib/docker", "/var", "/var/log"])
def test_root_only_and_container_state_dirs_refused(docker_env, src, monkeypatch):
    from pathlib import Path

    # Isolate the system rule: pretend the server user can read everything.
    monkeypatch.setattr("gbserver.environment.docker.os.access", lambda *a: True)
    with pytest.raises(ValueError, match="system directory"):
        docker_env._check_env_input_mount_src(Path(src).resolve(), f"env://{src}")


# ---------------------------------------------------------------------------
# env:// input mounts belong to the launch whose bindings reference them.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_env_input_mounted_only_into_referencing_launch(docker_env, tmp_path):
    client = _mock_launch_env(docker_env)
    d = tmp_path / "gen"
    d.mkdir()
    (d / "generation.json").write_text("{}")
    binding, _ = await docker_env.pullasset_envstore(
        uri=_env_uri(d / "generation.json")
    )

    await docker_env.launch_docker(
        launch_id="launchAAAA",
        targetsteprun_asset_dir=tmp_path / "wsA",
        config={},
        launcher_config={"command": "true"},
        step={"name": "s"},
        run_metadata={"targetsteprun_id": "tsrA"},
        bindings={"gen": binding},
    )
    await _launch(docker_env, "launchBBBB", tmp_path / "wsB", "tsrB")

    vols_a = client.containers.run.call_args_list[0].kwargs["volumes"]
    vols_b = client.containers.run.call_args_list[1].kwargs["volumes"]
    assert vols_a[str(d)] == {"bind": "/gb-env-inputs/0", "mode": "ro"}
    assert str(d) not in vols_b
    assert docker_env._extra_volumes == {}


@pytest.mark.asyncio
async def test_launch_records_absolute_workspace(docker_env, tmp_path, monkeypatch):
    _mock_launch_env(docker_env)
    monkeypatch.chdir(tmp_path)
    from pathlib import Path

    await _launch(docker_env, "launchAAAA", Path("rel") / "ws", "tsrA")
    assert docker_env._workspaces_by_targetsteprun["tsrA"] == str(
        tmp_path / "rel" / "ws"
    )


@pytest.mark.asyncio
async def test_launch_without_targetsteprun_id_warns(docker_env, tmp_path, caplog):
    _mock_launch_env(docker_env)
    with caplog.at_level(logging.WARNING):
        await _launch(docker_env, "launchAAAA", tmp_path / "wsA", None)
    assert docker_env._workspaces_by_targetsteprun == {}
    assert "no targetsteprun_id" in caplog.text


# ---------------------------------------------------------------------------
# Base Environment: env:// is never served from the asset_bindings cache.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_env_pull_skips_cache_even_after_push(docker_env, tmp_path):
    # A pull racing the producer's push must still reach pullasset_envstore,
    # whatever asset_bindings holds by then.
    env = _push_env(docker_env)
    d = tmp_path / "gen"
    d.mkdir()
    uristr = _env_uri(d)
    env.asset_bindings[uristr] = {BINDING_KEY: {"path": str(d)}}
    Environment._thread_local.asset_events[uristr] = asyncio.Event()
    Environment._thread_local.asset_events[uristr].set()
    env.pullasset_types = {"envstore": Docker.pullasset_envstore}
    binding, _ = await env.pullasset(task_group=None, uri=URI.get_uri(uristr))
    assert binding[BINDING_KEY]["path"].startswith("/gb-env-inputs/")


def test_only_env_uris_skip_the_cache():
    assert not Environment._is_cached_uri("env:///a/b")
    assert Environment._is_cached_uri("file:///a/b")
    assert Environment._is_cached_uri("hf://org/repo")


# ---------------------------------------------------------------------------
# Output roots gbserver writes to are allowed inside the GB home directory.
# ---------------------------------------------------------------------------


@pytest.fixture
def gb_home_in_fake_home(fake_home, monkeypatch):
    # The default layout: GB home is the hidden ~/.granite.build.
    gb_home = fake_home / ".granite.build"
    gb_home.mkdir()
    monkeypatch.setenv("GB_HOME_DIR", str(gb_home))
    return gb_home


@pytest.mark.asyncio
async def test_bash_output_in_gb_home_workdir_is_mounted(
    docker_env, gb_home_in_fake_home
):
    # Bash writes outputs under <GB home>/workdir by default (bash.py).
    out = gb_home_in_fake_home / "workdir" / "llm-build-x" / "t" / "outputs"
    out.mkdir(parents=True)
    (out / "result.json").write_text("{}")
    binding, _ = await docker_env.pullasset_envstore(uri=_env_uri(out / "result.json"))
    assert docker_env._env_input_volumes[str(out)]["mode"] == "ro"
    assert binding[BINDING_KEY]["path"].endswith("/result.json")


@pytest.mark.asyncio
async def test_rest_of_gb_home_still_refused(docker_env, gb_home_in_fake_home):
    for sub in ("user_secrets", "space_secrets"):
        (gb_home_in_fake_home / sub).mkdir()
        with pytest.raises(ValueError, match="GB home directory"):
            await docker_env.pullasset_envstore(
                uri=_env_uri(gb_home_in_fake_home / sub)
            )
    (gb_home_in_fake_home / "llmb-server.db").write_text("")
    with pytest.raises(ValueError, match="GB home directory"):
        await docker_env.pullasset_envstore(
            uri=_env_uri(gb_home_in_fake_home / "llmb-server.db")
        )
    # The output root itself would expose every Bash build's outputs.
    (gb_home_in_fake_home / "workdir").mkdir()
    with pytest.raises(ValueError, match="GB home directory"):
        await docker_env.pullasset_envstore(
            uri=_env_uri(gb_home_in_fake_home / "workdir")
        )


@pytest.mark.asyncio
async def test_gb_home_rule_alone_refuses_outside_home(
    docker_env, tmp_path, fake_home, monkeypatch
):
    # GB home outside the home dir, so only the GB-home rule can fire.
    gb_home = tmp_path / "gbhome"
    (gb_home / "user_secrets").mkdir(parents=True)
    monkeypatch.setenv("GB_HOME_DIR", str(gb_home))
    with pytest.raises(ValueError, match="GB home directory"):
        await docker_env.pullasset_envstore(uri=_env_uri(gb_home / "user_secrets"))


@pytest.mark.asyncio
async def test_other_hidden_home_dir_still_refused(docker_env, gb_home_in_fake_home):
    ssh = gb_home_in_fake_home.parent / ".ssh"
    ssh.mkdir()
    with pytest.raises(ValueError, match="hidden home directory"):
        await docker_env.pullasset_envstore(uri=_env_uri(ssh))


@pytest.mark.asyncio
async def test_earlier_docker_workspace_in_hidden_dir_is_mounted(
    docker_env, fake_home, tmp_path
):
    # The real workspace root (e.g. workspace_dir: ~/.gbws) is whatever the
    # launch used, recorded at launch time, not a constant.
    _mock_launch_env(docker_env)
    ws = fake_home / ".gbws" / "builds" / "b1" / "targetruns" / "tsrA"
    (ws / "outputs").mkdir(parents=True)
    await _launch(docker_env, "launchAAAA", ws, "tsrA")
    await docker_env.pullasset_envstore(uri=_env_uri(ws / "outputs"))
    assert str(ws / "outputs") in docker_env._env_input_volumes


@pytest.mark.asyncio
async def test_workspace_root_clones_still_refused(docker_env, fake_home):
    # Only recorded launch workspaces are allowed, not the root that also
    # holds the build watcher's PR clones (git config with a GitHub token).
    _mock_launch_env(docker_env)
    root = fake_home / ".gbws"
    ws = root / "builds" / "b1" / "targetruns" / "tsrA"
    clone = root / "gbserver-buildwatcher-workspace" / "repos" / "r"
    clone.mkdir(parents=True)
    ws.mkdir(parents=True)
    await _launch(docker_env, "launchAAAA", ws, "tsrA")
    with pytest.raises(ValueError, match="hidden home directory"):
        await docker_env.pullasset_envstore(uri=_env_uri(clone))


def test_output_root_containing_home_or_gb_home_is_dropped(
    docker_env, fake_home, tmp_path, monkeypatch
):
    monkeypatch.setenv("GB_HOME_DIR", str(tmp_path / "gbhome"))
    # A launch workspace of $HOME (or above GB home) must not reopen ~/.ssh.
    docker_env._workspaces_by_targetsteprun = {"a": str(fake_home), "b": str(tmp_path)}
    assert docker_env._env_output_roots() == [
        (tmp_path / "gbhome" / "workdir").resolve()
    ]


@pytest.mark.asyncio
async def test_workspace_of_home_does_not_reopen_hidden_dirs(
    docker_env, fake_home, tmp_path, monkeypatch
):
    monkeypatch.setenv("GB_HOME_DIR", str(tmp_path / "gbhome"))
    docker_env._workspaces_by_targetsteprun = {"a": str(fake_home)}
    (fake_home / ".ssh").mkdir()
    with pytest.raises(ValueError, match="hidden home directory"):
        await docker_env.pullasset_envstore(uri=_env_uri(fake_home / ".ssh"))


# ---------------------------------------------------------------------------
# The checked path is the mounted path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mount_uses_resolved_dir_and_keeps_file_name(docker_env, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "data.json").write_text("{}")
    link = tmp_path / "link"
    link.symlink_to(real)
    (real / "alias.json").symlink_to(real / "data.json")

    b1, _ = await docker_env.pullasset_envstore(uri=_env_uri(link / "alias.json"))
    b2, _ = await docker_env.pullasset_envstore(uri=_env_uri(real / "data.json"))
    # One mount, keyed by (and sourced from) the resolved directory.
    assert list(docker_env._env_input_volumes) == [str(real.resolve())]
    bind = docker_env._env_input_volumes[str(real.resolve())]["bind"]
    # The in-container file name is the one the user gave.
    assert b1[BINDING_KEY]["path"] == f"{bind}/alias.json"
    assert b2[BINDING_KEY]["path"] == f"{bind}/data.json"


# ---------------------------------------------------------------------------
# The env:// cache rule holds for every environment, not only Docker.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_docker_env_output_pull_returns_uri_path(tmp_path):
    from gbserver.environment.bash import Bash

    env = _push_env(Bash(event_q=asyncio.Queue()))
    host = tmp_path / "outputs" / "result.json"
    # The producer's own view of the path differs from the registered URI.
    task = env.pushasset(
        task_group=None,  # type: ignore[arg-type]
        binding={"path": str(host), "extra": "producer-only"},
        uristr=f"env://{host}",
        binding_id="result",
        run_metadata=EntityRunMetadata(targetsteprun_id="tsrA"),
    )
    uristr = URI.get_uristr(await task)
    assert uristr not in env.asset_bindings
    binding, step = await env.pullasset(task_group=None, uri=URI.get_uri(uristr))
    assert step is None
    assert binding == {BINDING_KEY: {"path": str(host)}}


# ---------------------------------------------------------------------------
# A container-path producer binding is never cached for another container.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_output_with_workspace_path_is_not_cached(docker_env, tmp_path):
    # A cached /gb-workspace path would point into the *consumer's* workspace.
    env = _push_env(docker_env)
    _register_launch(env, "tsrA", tmp_path / "wsA")

    async def _file_handler(self, uri=None, **kw):
        return uri

    env.pushasset_types = {"filestore": _file_handler}
    uristr, _ = await _push(
        env, {"path": "/gb-workspace/m.json"}, uristr=f"file://{tmp_path}/m.json"
    )
    assert uristr not in env.asset_bindings
    # Docker has no file:// pull, so the consumer fails loudly, not silently.
    Environment._thread_local.asset_events[uristr].set()
    with pytest.raises(RuntimeError, match="failed to load"):
        await env.pullasset(task_group=None, uri=URI.get_uri(uristr))


@pytest.mark.asyncio
async def test_file_output_with_host_path_is_still_cached(docker_env, tmp_path):
    env = _push_env(docker_env)

    async def _file_handler(self, uri=None, **kw):
        return uri

    env.pushasset_types = {"filestore": _file_handler}
    host = str(tmp_path / "m.json")
    uristr, _ = await _push(env, {"path": host}, uristr=f"file://{tmp_path}/out.json")
    assert env.asset_bindings[uristr] == {BINDING_KEY: {"path": host}}


# ---------------------------------------------------------------------------
# env:// outputs at a read-only input mount map to the input's host path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_env_output_of_reemitted_input_registers_host_path(docker_env, tmp_path):
    env = _push_env(docker_env)
    d = tmp_path / "gen"
    d.mkdir()
    (d / "m.json").write_text("{}")
    binding, _ = await env.pullasset_envstore(uri=_env_uri(d / "m.json"))
    container = binding[BINDING_KEY]["path"]
    assert container.startswith("/gb-env-inputs/")
    uristr, _ = await _push(env, {"path": container})
    assert uristr == f"env://{d.resolve()}/m.json"


@pytest.mark.asyncio
async def test_env_output_at_unmounted_container_path_raises(docker_env):
    env = _push_env(docker_env)
    with pytest.raises(LookupError, match="no Docker mount"):
        await _push(env, {"path": "/gb-env-inputs/7/m.json"})


@pytest.mark.asyncio
async def test_literal_env_uri_at_container_path_refused(docker_env):
    env = _push_env(docker_env)
    env.pushasset_types = {"envstore": Docker.pushasset_envstore}
    with pytest.raises(ValueError, match="container path"):
        await _push(
            env,
            {"path": "/somewhere/else"},
            uristr="env:///gb-workspace/outputs/x",
        )
