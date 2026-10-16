# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Tests for the api-bridge write-route bearer-token guard.

``server.require_write_token`` gates the 5 write routes (``record_logs``,
``record_metrics``, ``record_trial``/``insert_trials``, ``update_status``,
``insert_trial_result``)
so a network caller can no longer write to production MySQL with no credential
at all. These tests exercise the guard through the real FastAPI app on an
in-memory SQLite database, covering both the token-configured and
token-unset (backward-compatible) configurations.

``api_bridge.server`` reads ``AUTOTUNEX_DATABASE_URL`` and ``API_BRIDGE_TOKEN``
at *import* time, so each configuration under test needs a fresh module
execution rather than a dependency override — ``_load_app`` sets the env vars
and then ``importlib.reload``s the module.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from api_bridge.tables import metadata

UPDATE_STATUS_BODY = {"id": "11111111-1111-1111-1111-111111111111", "status": "RUNNING"}


def _load_app(monkeypatch: pytest.MonkeyPatch, token: str | None):
    """(Re)import ``api_bridge.server`` with a fresh in-memory DB and given token.

    Returns the reloaded module so a test can build a ``TestClient`` from its
    ``app``.
    """
    monkeypatch.setenv("AUTOTUNEX_DATABASE_URL", "sqlite://")
    if token is None:
        monkeypatch.delenv("API_BRIDGE_TOKEN", raising=False)
    else:
        monkeypatch.setenv("API_BRIDGE_TOKEN", token)

    import api_bridge.server as server

    server = importlib.reload(server)
    # A fresh in-memory SQLite database starts empty; create the schema on the
    # same engine the module-level `database.Database()` above just opened, so
    # the app's startup lifespan (test_db_connection_and_structure) finds every
    # required table instead of failing closed with a 500.
    metadata.create_all(server.db._engine)
    return server


def test_update_status_returns_401_with_no_authorization_header_when_token_set(monkeypatch):
    server = _load_app(monkeypatch, token="s3cret")

    with TestClient(server.app) as client:
        response = client.post("/fmtune/api/update_status", json=UPDATE_STATUS_BODY)

    assert response.status_code == 401


def test_update_status_returns_401_with_the_wrong_bearer_token(monkeypatch):
    server = _load_app(monkeypatch, token="s3cret")

    with TestClient(server.app) as client:
        response = client.post(
            "/fmtune/api/update_status",
            json=UPDATE_STATUS_BODY,
            headers={"Authorization": "Bearer wrong"},
        )

    assert response.status_code == 401


def test_update_status_succeeds_with_the_correct_bearer_token(monkeypatch):
    server = _load_app(monkeypatch, token="s3cret")

    with TestClient(server.app) as client:
        response = client.post(
            "/fmtune/api/update_status",
            json=UPDATE_STATUS_BODY,
            headers={"Authorization": "Bearer s3cret"},
        )

    assert response.status_code == 200


def test_update_status_is_reachable_with_no_header_when_token_is_unset(monkeypatch):
    # Backward-compatible: the live tuning pipeline sends no Authorization header
    # today, so an unset API_BRIDGE_TOKEN must not start rejecting it.
    server = _load_app(monkeypatch, token=None)

    with TestClient(server.app) as client:
        response = client.post("/fmtune/api/update_status", json=UPDATE_STATUS_BODY)

    assert response.status_code == 200


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/fmtune/api/record_logs", [{"job_id": None, "trial_id": None}]),
        (
            "/fmtune/api/record_metrics",
            {
                "job_id": "11111111-1111-1111-1111-111111111111",
                "trial_id": "t1",
                "global_step": 10,
                "epoch": 0.1,
                "loss": 1.0,
                "grad_norm": 1.0,
                "learning_rate": 1e-6,
                "split": "train",
                "extra": {},
            },
        ),
        (
            "/fmtune/api/record_trial",
            {
                "id": "t1",
                "job_id": "11111111-1111-1111-1111-111111111111",
                "status": "PENDING",
                "config": None,
            },
        ),
        ("/fmtune/api/insert_trial_result", {"id": "job-1"}),
    ],
)
def test_other_write_routes_also_require_the_bearer_token(monkeypatch, path, body):
    server = _load_app(monkeypatch, token="s3cret")

    with TestClient(server.app) as client:
        response = client.post(path, json=body)

    assert response.status_code == 401
