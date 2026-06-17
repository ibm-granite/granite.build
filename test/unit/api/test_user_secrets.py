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

"""API tests for the /user_secrets endpoints against the local backend.

These exercise the full handler path (factory selection + UserSecretManager) in a
standalone-like configuration with no IBM Secret Manager available, and assert the
response shapes are unchanged from the previous IBM-backed behavior.
"""

import base64
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_client():
    """A TestClient for secrets_api with a fake authenticated user 'alice'.

    The factory reads the backend env vars at call time, so the per-test monkeypatch
    of GBSERVER_USER_SECRET_MANAGER / dir is picked up without reloading modules.
    """
    import gbserver.api.secrets as secrets_module

    app = FastAPI()

    @app.middleware("http")
    async def _inject_user(request, call_next):
        request.state.data = {"user": SimpleNamespace(login="alice", email="alice@x")}
        return await call_next(request)

    app.mount("", secrets_module.secrets_api)
    return TestClient(app)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient backed by the editable local user-secret backend."""
    monkeypatch.setenv("GBSERVER_USER_SECRET_MANAGER", "local")
    monkeypatch.setenv("GBSERVER_USER_SECRET_DIR", str(tmp_path / "user_secrets"))
    return _build_client()


@pytest.fixture
def env_client(monkeypatch):
    """TestClient backed by the read-only env user-secret backend."""
    monkeypatch.setenv("GBSERVER_USER_SECRET_MANAGER", "env")
    return _build_client()


def _decode(resp_json):
    return base64.b64decode(resp_json["secret_value"]).decode("utf-8")


def test_create_list_get_update_delete_roundtrip(client):
    # create
    r = client.post(
        "/user_secrets",
        json={"secret_name": "API_KEY", "secret_value": "hunter2", "encoding": "plain"},
    )
    assert r.status_code == 200
    assert r.json() == {"result": "success"}

    # list
    r = client.get("/user_secrets")
    assert r.status_code == 200
    body = r.json()
    assert body["user"] == "alice"
    assert body["secrets"] == ["API_KEY"]

    # get (value returned base64-encoded with encoding marker)
    r = client.get("/user_secrets/API_KEY")
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == "alice"
    assert body["secret_name"] == "API_KEY"
    assert body["encoding"] == "base64"
    assert _decode(body) == "hunter2"

    # update
    r = client.put(
        "/user_secrets/API_KEY",
        json={"secret_value": "newval", "encoding": "plain"},
    )
    assert r.status_code == 200
    assert _decode(client.get("/user_secrets/API_KEY").json()) == "newval"

    # delete
    r = client.delete("/user_secrets/API_KEY")
    assert r.status_code == 200
    assert client.get("/user_secrets").json()["secrets"] == []


def test_create_accepts_base64_encoding(client):
    encoded = base64.b64encode(b"sekret").decode("ascii")
    r = client.post(
        "/user_secrets",
        json={"secret_name": "K", "secret_value": encoded, "encoding": "base64"},
    )
    assert r.status_code == 200
    assert _decode(client.get("/user_secrets/K").json()) == "sekret"


def test_get_missing_returns_404(client):
    assert client.get("/user_secrets/NOPE").status_code == 404


def test_readonly_backend_create_returns_405(env_client, monkeypatch):
    """Writing to the read-only env backend must return 405, not 404."""
    monkeypatch.setenv("GBSERVER_USER_SECRET_ALICE_API_KEY", "v")
    # reads still work on the env backend
    assert env_client.get("/user_secrets").json()["secrets"] == ["API_KEY"]
    # writes are rejected as Method Not Allowed (not 404 Not Found)
    r = env_client.post(
        "/user_secrets",
        json={"secret_name": "X", "secret_value": "y", "encoding": "plain"},
    )
    assert r.status_code == 405


def test_readonly_backend_update_delete_return_405(env_client):
    assert (
        env_client.put(
            "/user_secrets/X", json={"secret_value": "y", "encoding": "plain"}
        ).status_code
        == 405
    )
    assert env_client.delete("/user_secrets/X").status_code == 405
