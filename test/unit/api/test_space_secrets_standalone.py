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

"""API tests for the /space_secrets endpoints in standalone mode.

Exercises the pluggable SpaceSecretManager path (no IBM dependency): a local
backend supports full CRUD, an env backend is read-only (writes -> 405). This is
the path `gb secret list`/`get`/... (space scope) takes in standalone.
"""

import base64
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_client(monkeypatch, space_dir):
    monkeypatch.setenv("GB_ENVIRONMENT", "STANDALONE")

    import gbserver.api.secrets as secrets_module

    # The manager cache is a module-global keyed by (name, uri); clear it so each
    # test's space.yaml is re-read rather than served from a prior test's instance.
    secrets_module._space_secret_manager_cache.clear()

    # The space-secret handlers resolve the space via _get_space_for_admin; return a
    # standalone space dict whose git_repo_uri points at our space.yaml dir.
    space = {
        "name": "standalone",
        "git_repo_uri": f"file://{space_dir}",
        "is_admin": True,
    }
    monkeypatch.setattr(
        secrets_module, "_get_space_for_admin", lambda username, space_name: space
    )

    app = FastAPI()

    @app.middleware("http")
    async def _inject_user(request, call_next):
        request.state.data = {"user": SimpleNamespace(login="alice", email="alice@x")}
        return await call_next(request)

    app.mount("", secrets_module.secrets_api)
    return TestClient(app)


def _write_space_yaml(space_dir, secret_manager_yaml):
    space_dir.mkdir(parents=True, exist_ok=True)
    (space_dir / "space.yaml").write_text("name: standalone\n" + secret_manager_yaml)


@pytest.fixture
def local_client(tmp_path, monkeypatch):
    space_dir = tmp_path / "space"
    secrets_dir = tmp_path / "space_secrets"
    _write_space_yaml(
        space_dir,
        f"secret_manager:\n  type: local\n  config:\n    secrets_dir: {secrets_dir}\n",
    )
    return _build_client(monkeypatch, space_dir)


@pytest.fixture
def env_client(tmp_path, monkeypatch):
    space_dir = tmp_path / "space"
    _write_space_yaml(space_dir, "secret_manager:\n  type: env\n  config: {}\n")
    return _build_client(monkeypatch, space_dir)


@pytest.fixture
def local_default_client(tmp_path, monkeypatch):
    """A `type: local` space.yaml with `config: {}` and no secrets_dir.

    Mirrors configurations/spaces/local/space.yaml: LocalSpaceSecretManager should
    default secrets_dir to <gb_home>/space_secrets. GB_HOME_DIR is pointed at a temp
    dir so the default location is isolated and assertable.
    """
    gb_home = tmp_path / "gb_home"
    monkeypatch.setenv("GB_HOME_DIR", str(gb_home))
    space_dir = tmp_path / "space"
    _write_space_yaml(space_dir, "secret_manager:\n  type: local\n  config: {}\n")
    client = _build_client(monkeypatch, space_dir)
    return client, gb_home


def _decode(resp_json):
    return base64.b64decode(resp_json["secret_value"]).decode("utf-8")


def test_local_space_secret_crud(local_client):
    base = "/space_secrets/standalone"

    # create
    r = local_client.post(
        base,
        json={"secret_name": "API_KEY", "secret_value": "hunter2", "encoding": "plain"},
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"result": "success"}

    # list
    r = local_client.get(base)
    assert r.status_code == 200
    assert r.json()["secrets"] == ["API_KEY"]

    # get (base64-encoded value)
    r = local_client.get(f"{base}/API_KEY")
    assert r.status_code == 200
    body = r.json()
    assert body["space_name"] == "standalone"
    assert body["encoding"] == "base64"
    assert _decode(body) == "hunter2"

    # update
    r = local_client.put(
        f"{base}/API_KEY", json={"secret_value": "newval", "encoding": "plain"}
    )
    assert r.status_code == 200
    assert _decode(local_client.get(f"{base}/API_KEY").json()) == "newval"

    # delete
    assert local_client.delete(f"{base}/API_KEY").status_code == 200
    assert local_client.get(base).json()["secrets"] == []


def test_env_space_secret_read_only(env_client, monkeypatch):
    base = "/space_secrets/standalone"
    monkeypatch.setenv("GBSERVER_SECRET_API_KEY", "envval")

    # reads work
    assert env_client.get(base).json()["secrets"] == ["API_KEY"]
    assert _decode(env_client.get(f"{base}/API_KEY").json()) == "envval"

    # writes are rejected as 405 Method Not Allowed (read-only backend)
    assert (
        env_client.post(
            base,
            json={"secret_name": "X", "secret_value": "y", "encoding": "plain"},
        ).status_code
        == 405
    )
    assert (
        env_client.put(
            f"{base}/X", json={"secret_value": "y", "encoding": "plain"}
        ).status_code
        == 405
    )
    assert env_client.delete(f"{base}/X").status_code == 405


def test_get_missing_space_secret_returns_404(local_client):
    assert local_client.get("/space_secrets/standalone/NOPE").status_code == 404


def test_local_default_secrets_dir_crud(local_default_client):
    """`type: local` with no secrets_dir defaults to <gb_home>/space_secrets and is writable."""
    client, gb_home = local_default_client
    base = "/space_secrets/standalone"

    r = client.post(
        base,
        json={"secret_name": "API_KEY", "secret_value": "hunter2", "encoding": "plain"},
    )
    assert r.status_code == 200, r.text

    # The secret file landed under the defaulted <gb_home>/space_secrets dir.
    default_dir = gb_home / "space_secrets"
    assert default_dir.is_dir()
    assert any(default_dir.iterdir()), "expected a secret file under the default dir"

    # Round-trips via the API.
    assert client.get(base).json()["secrets"] == ["API_KEY"]
    assert _decode(client.get(f"{base}/API_KEY").json()) == "hunter2"
    assert client.delete(f"{base}/API_KEY").status_code == 200
