# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""``/api/job/bootstrap`` must stay idempotent when two attempts race.

fm-tune retries a timed-out bootstrap while the first attempt may still be
running, so both can miss the job_id short-circuit. The unique keys stop a
second row; the losing attempt used to surface the constraint error ("Dataset
name must be unique", or insert_job's 404) instead of the job the winner made.
The loser can also hit the constraint before the winner's job exists, when its
config or dataset lookup ran before the winner's insert committed.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from api_bridge.tables import metadata

JOB_ID = "11111111-1111-1111-1111-111111111111"
BODY = {
    "job_id": JOB_ID,
    "build_id": None,
    "config": {"name": "cfg", "tuner_type": "lora", "config_data": {"lr": 1}},
    "dataset": {"name": "ds", "artifact_uri": "hf:///datasets/org/ds"},
    "job": {"model": "m", "experiment_name": "e", "tuning_type": "lora", "seed": 42},
}


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("AUTOTUNEX_DATABASE_URL", "sqlite://")
    monkeypatch.delenv("API_BRIDGE_TOKEN", raising=False)
    import api_bridge.server as server

    server = importlib.reload(server)
    metadata.create_all(server.db._engine)
    from api_bridge import dependencies

    server.app.dependency_overrides[dependencies.get_database] = lambda: server.db
    yield server
    server.app.dependency_overrides.clear()


def test_repeat_bootstrap_returns_same_ids(server):
    with TestClient(server.app) as client:
        first = client.post("/fmtune/api/job/bootstrap", json=BODY)
        second = client.post("/fmtune/api/job/bootstrap", json=BODY)
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["created"] is True
    assert second.json()["created"] is False
    for key in ("config_id", "dataset_id", "job_id"):
        assert second.json()[key] == first.json()[key]


def test_losing_racer_returns_winner_job(server, monkeypatch):
    """The second attempt passes the short-circuit, then the winner commits first."""
    from api_bridge.services import job_service

    with TestClient(server.app) as client:
        winner = client.post("/fmtune/api/job/bootstrap", json=BODY).json()

        real_get = job_service.Job.get_by_id
        calls = {"n": 0}

        def stale_then_real(self, job_id):
            # The first lookup ran before the winner committed.
            calls["n"] += 1
            return None if calls["n"] == 1 else real_get(self, job_id)

        monkeypatch.setattr(job_service.Job, "get_by_id", stale_then_real)
        loser = client.post("/fmtune/api/job/bootstrap", json=BODY)

    assert loser.status_code == 200, loser.text
    assert loser.json()["created"] is False
    assert loser.json()["job_id"] == winner["job_id"]
    assert loser.json()["dataset_id"] == winner["dataset_id"]


def _winner_commits_rows_but_not_the_job(server):
    """Leave the DB as a winner does right before its job insert lands."""
    from api_bridge.services import config_service, dataset_service

    server.db.insert_user("dev@example.com")
    user_id = server.db.get_user("dev@example.com")["id"]
    config_service.Config(server.db).find_or_create(user_id=user_id, **BODY["config"])
    dataset_service.Dataset(server.db).find_or_create(user_id=user_id, **BODY["dataset"])


def test_loser_that_missed_the_winners_dataset_reuses_it(server, monkeypatch):
    """The loser's dataset lookup ran before the winner's dataset insert committed."""
    _winner_commits_rows_but_not_the_job(server)
    real_get_datasets = server.db.get_datasets
    calls = {"n": 0}

    def stale_then_real(user_id, ids=None):
        calls["n"] += 1
        return [] if calls["n"] == 1 else real_get_datasets(user_id, ids)

    monkeypatch.setattr(server.db, "get_datasets", stale_then_real)
    with TestClient(server.app) as client:
        resp = client.post("/fmtune/api/job/bootstrap", json=BODY)

    assert resp.status_code == 200, resp.text
    assert resp.json()["job_id"] == JOB_ID


def test_loser_that_missed_the_winners_config_reuses_it(server, monkeypatch):
    """The loser's config lookup ran before the winner's config insert committed."""
    _winner_commits_rows_but_not_the_job(server)
    real_get_config = server.db.get_config_by_name_and_user
    calls = {"n": 0}

    def stale_then_real(config_name, user_id):
        calls["n"] += 1
        return (
            None if calls["n"] == 1 else real_get_config(config_name=config_name, user_id=user_id)
        )

    monkeypatch.setattr(server.db, "get_config_by_name_and_user", stale_then_real)
    with TestClient(server.app, raise_server_exceptions=False) as client:
        resp = client.post("/fmtune/api/job/bootstrap", json=BODY)

    assert resp.status_code == 200, resp.text
    assert resp.json()["job_id"] == JOB_ID


def test_real_failure_still_raises(server, monkeypatch):
    """With no job row behind the error, the original error propagates."""
    from fastapi import HTTPException

    from api_bridge.services import dataset_service

    def boom(self, **_kw):
        raise HTTPException(status_code=400, detail="Dataset name must be unique")

    monkeypatch.setattr(dataset_service.Dataset, "find_or_create", boom)
    with TestClient(server.app) as client:
        resp = client.post("/fmtune/api/job/bootstrap", json=BODY)
    assert resp.status_code == 400
