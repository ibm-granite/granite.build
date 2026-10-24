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
    vol = docker_env._extra_volumes[str(f.parent)]
    assert vol["mode"] == "ro"
    assert path == f"{vol['bind']}/generation.json"
    assert docker_env._resolve_host_path(path) == str(f)


@pytest.mark.asyncio
async def test_dir_input_is_mounted_whole(docker_env, tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    binding, _ = await docker_env.pullasset_envstore(uri=_env_uri(d))
    vol = docker_env._extra_volumes[str(d)]
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

    mount_a = docker_env._extra_volumes[str(a_dir)]["bind"]
    mount_b = docker_env._extra_volumes[str(b_dir)]["bind"]
    assert mount_a != mount_b
    assert pa1["path"] == f"{mount_a}/one.json"
    assert pa2["path"] == f"{mount_a}/two.json"
    assert pb1["path"] == f"{mount_b}/one.json"
    assert pa1["path"] != pb1["path"]
    env_mounts = [
        v
        for v in docker_env._extra_volumes.values()
        if v["bind"].startswith("/gb-env-inputs/")
    ]
    assert len(env_mounts) == 2


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
            SimpleNamespace(push=None),
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
    assert env.asset_bindings[expected][BINDING_KEY]["path"] == (
        f"{tmp_path}/wsB/outputs/x/results.json"
    )
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
async def test_env_output_unknown_targetsteprun_unchanged(docker_env, caplog):
    env = _push_env(docker_env)
    with caplog.at_level(logging.WARNING):
        uristr, _ = await _push(env, {"path": _RESULTS}, tsr="nope")
    assert uristr == f"env://{_RESULTS}"
    assert "nope" in caplog.text


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
    # The base push cached the host binding; a pull must not hand it back as is.
    assert env.asset_bindings[uristr][BINDING_KEY]["path"] == host

    binding, step = await env.pullasset(task_group=None, uri=URI.get_uri(uristr))
    assert step is None
    path = binding[BINDING_KEY]["path"]
    vol = env._extra_volumes[str(ws / "outputs" / "x")]
    assert vol["mode"] == "ro"
    assert path == f"{vol['bind']}/results.json"
    assert path.startswith("/gb-env-inputs/")


# ---------------------------------------------------------------------------
# Mount scope: never / or the gbserver user's home directory.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_root_dir_input_refused(docker_env):
    with pytest.raises(ValueError, match="its own directory"):
        await docker_env.pullasset_envstore(uri="env:///")
    assert docker_env._extra_volumes == {}


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
    with pytest.raises(ValueError, match="its own directory"):
        await docker_env.pullasset_envstore(uri=_env_uri(fake_home / "generation.json"))
    assert docker_env._extra_volumes == {}


@pytest.mark.asyncio
async def test_home_dir_input_refused(docker_env, fake_home):
    with pytest.raises(ValueError, match="its own directory"):
        await docker_env.pullasset_envstore(uri=_env_uri(fake_home))


@pytest.mark.asyncio
async def test_home_dir_via_symlink_refused(docker_env, fake_home, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(fake_home)
    (fake_home / "generation.json").write_text("{}")
    with pytest.raises(ValueError, match="its own directory"):
        await docker_env.pullasset_envstore(uri=_env_uri(link / "generation.json"))


@pytest.mark.asyncio
async def test_subdir_of_home_is_mounted(docker_env, fake_home):
    sub = fake_home / "run1"
    sub.mkdir()
    (sub / "generation.json").write_text("{}")
    binding, _ = await docker_env.pullasset_envstore(
        uri=_env_uri(sub / "generation.json")
    )
    assert docker_env._extra_volumes[str(sub)]["mode"] == "ro"
    assert binding[BINDING_KEY]["path"].endswith("/generation.json")
