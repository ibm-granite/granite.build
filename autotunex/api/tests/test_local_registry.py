# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import json

import models as api
import pytest


class _FakeDB:
    """Async DB stub for LocalRegistry tests."""

    def __init__(self, jobs=None, users=None):
        self._jobs = jobs or {}  # job_id -> job dict
        self._users = users or {}  # user_id -> user dict

    async def get_job(self, id, user_id):
        job = self._jobs.get(id)
        if not job or job.get("user_id") != user_id:
            return None
        return job

    async def get_job_by_id(self, id):
        return self._jobs.get(id)

    async def get_jobs(self, user_id=None):
        return [
            j
            for j in self._jobs.values()
            if user_id is None or j.get("user_id") == user_id
        ]

    async def get_user_by_id(self, user_id):
        return self._users.get(user_id)


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    d = tmp_path / "registry"
    d.mkdir()
    monkeypatch.setenv("AUTOTUNEX_MODELS_DIR", str(d))
    return d


def _make_results(tmp_path, job_id):
    """Create a fake job output dir AUTOTUNE_RESULTS_PATH/output/<job_id>/results."""
    results = tmp_path / "out" / "output" / job_id / "results"
    results.mkdir(parents=True)
    (results / "adapter_model.bin").write_bytes(b"weights")
    (results / "README.md").write_text("# card")
    return tmp_path / "out"


async def test_publish_then_get_model_detail_roundtrip(
    models_dir, tmp_path, monkeypatch
):
    from services.registry.local_backend import LocalRegistry

    results_root = _make_results(tmp_path, "job-1")
    monkeypatch.setenv("AUTOTUNE_RESULTS_PATH", str(results_root))

    db = _FakeDB(
        jobs={"job-1": {"id": "job-1", "user_id": "u1", "model": "granite-3b"}}
    )
    reg = LocalRegistry(db)

    meta = api.DmfMetadata(
        label="cust-bot", variant="lora-r16", type="text-generation", size="3B"
    )
    resp = await reg.publish_model(job_id="job-1", metadata=meta, user_id="u1")
    assert resp["status"] == "Published"

    # model.json was written under <MODELS_DIR>/<label>/<job_id>/
    manifest = models_dir / "cust-bot" / "job-1" / "model.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert data["model_label"] == "cust-bot"
    assert data["revision"] == "job-1"
    assert data["base_model"] == "granite-3b"
    assert data["size"] == "3B"

    detail = reg.get_model_detail(model_label="cust-bot")
    assert detail["revision"] == "job-1"
    assert detail["model_label"] == "cust-bot"


def test_get_model_detail_missing_returns_none(models_dir):
    from services.registry.local_backend import LocalRegistry

    reg = LocalRegistry(_FakeDB())
    assert reg.get_model_detail(model_label="nope") is None


async def _publish(reg, tmp_path, monkeypatch, job_id, label):
    results_root = _make_results(tmp_path, job_id)
    monkeypatch.setenv("AUTOTUNE_RESULTS_PATH", str(results_root))
    meta = api.DmfMetadata(label=label, variant="v", type="text-generation", size="3B")
    job = await reg.db.get_job_by_id(job_id)
    await reg.publish_model(job_id=job_id, metadata=meta, user_id=job["user_id"])


async def test_get_models_filters_by_user_and_enriches(
    models_dir, tmp_path, monkeypatch
):
    from services.registry.local_backend import LocalRegistry

    db = _FakeDB(
        jobs={
            "job-1": {"id": "job-1", "user_id": "u1", "model": "granite-3b"},
            "job-2": {"id": "job-2", "user_id": "u2", "model": "granite-3b"},
        },
        users={"u1": {"email": "a@x.com"}, "u2": {"email": "b@x.com"}},
    )
    reg = LocalRegistry(db)
    await _publish(reg, tmp_path, monkeypatch, "job-1", "model-a")
    await _publish(reg, tmp_path, monkeypatch, "job-2", "model-b")

    out = await reg.get_models(user_id="u1")
    revisions = {m["revision"] for m in out}
    assert revisions == {"job-1"}  # only u1's model
    assert out[0]["user"] == "a@x.com"  # enriched


async def test_get_all_models_returns_all_enriched(models_dir, tmp_path, monkeypatch):
    from services.registry.local_backend import LocalRegistry

    db = _FakeDB(
        jobs={
            "job-1": {"id": "job-1", "user_id": "u1", "model": "granite-3b"},
            "job-2": {"id": "job-2", "user_id": "u2", "model": "granite-3b"},
        },
        users={"u1": {"email": "a@x.com"}, "u2": {"email": "b@x.com"}},
    )
    reg = LocalRegistry(db)
    await _publish(reg, tmp_path, monkeypatch, "job-1", "model-a")
    await _publish(reg, tmp_path, monkeypatch, "job-2", "model-b")

    out = await reg.get_all_models()
    assert {m["revision"] for m in out} == {"job-1", "job-2"}
    assert {m["user"] for m in out} == {"a@x.com", "b@x.com"}


async def test_delete_model_removes_dir(models_dir, tmp_path, monkeypatch):
    from services.registry.local_backend import LocalRegistry

    db = _FakeDB(
        jobs={"job-1": {"id": "job-1", "user_id": "u1", "model": "granite-3b"}}
    )
    reg = LocalRegistry(db)
    await _publish(reg, tmp_path, monkeypatch, "job-1", "model-a")
    assert (models_dir / "model-a" / "job-1").exists()

    ok = await reg.delete_model(job_id="job-1", user_id="u1")
    assert ok is True
    assert not (models_dir / "model-a" / "job-1").exists()


async def test_delete_model_forbidden_for_other_user(models_dir, tmp_path, monkeypatch):
    from fastapi import HTTPException
    from services.registry.local_backend import LocalRegistry

    db = _FakeDB(
        jobs={"job-1": {"id": "job-1", "user_id": "u1", "model": "granite-3b"}}
    )
    reg = LocalRegistry(db)
    await _publish(reg, tmp_path, monkeypatch, "job-1", "model-a")

    with pytest.raises(HTTPException) as ei:
        await reg.delete_model(job_id="job-1", user_id="someone-else")
    assert ei.value.status_code in (400, 404)


def test_search_models_substring_envelope(models_dir, tmp_path, monkeypatch):
    import asyncio

    from services.registry.local_backend import LocalRegistry

    db = _FakeDB(
        jobs={"job-1": {"id": "job-1", "user_id": "u1", "model": "granite-3b"}}
    )
    reg = LocalRegistry(db)
    asyncio.get_event_loop().run_until_complete(
        _publish(reg, tmp_path, monkeypatch, "job-1", "customer-bot")
    )
    res = reg.search_models("customer")
    assert "data" in res
    assert any(m["model_label"] == "customer-bot" for m in res["data"])
    assert reg.search_models("nomatch")["data"] == []


def test_get_model_card_missing_raises_404(models_dir):
    from fastapi import HTTPException
    from services.registry.local_backend import LocalRegistry

    reg = LocalRegistry(_FakeDB())
    with pytest.raises(HTTPException) as ei:
        reg.get_model_card(namespace="n", table="t", model_label="nope", revision="r")
    assert ei.value.status_code == 404


@pytest.mark.asyncio
async def test_local_search_models_includes_namespace(
    models_dir, tmp_path, monkeypatch
):
    from services.registry.local_backend import LocalRegistry

    monkeypatch.setenv("AUTOTUNE_RESULTS_PATH", str(tmp_path / "out"))
    db = _FakeDB(
        jobs={"job-1": {"id": "job-1", "user_id": "u1", "model": "granite-3.0"}},
        users={"u1": {"id": "u1", "email": "a@b.c"}},
    )
    reg = LocalRegistry(db=db)
    _make_results(tmp_path, "job-1")
    meta = api.DmfMetadata(
        label="my-model", variant="v", type="text-generation", size="3B"
    )
    await reg.publish_model(job_id="job-1", metadata=meta, user_id="u1")

    out = reg.search_models("my-model")
    assert out["data"], "expected the published model to match the query"
    item = out["data"][0]
    assert item.get(
        "namespace"
    ), "Local search items must carry a non-empty namespace for the wizard"
