# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

from services import job_service
from services.plugins import Seam, clear_overrides, register_override
from services.registry.base import ModelRegistry


class _FakeRegistry(ModelRegistry):
    def __init__(self, **kw):
        self.calls = []

    async def get_models(self, user_id): ...
    async def get_all_models(self): ...
    async def publish_model(self, job_id, metadata, user_id): ...
    async def delete_model(self, job_id, user_id): ...
    def get_model_detail(
        self, model_label, namespace="base_training", table="model_shared"
    ):
        self.calls.append(("get_model_detail", model_label))
        return {"model_label": model_label, "revision": "r1"}

    def get_checkpoints(self, artifact_url):
        self.calls.append(("get_checkpoints", artifact_url))
        return []

    def pull_all_checkpoint_files(self, artifact_url): ...
    def pull_checkpoint_file(self, artifact_url, file_paths): ...
    def get_model_card(self, namespace, table, model_label, revision): ...
    def search_models(self, query): ...


def test_job_dmf_is_resolved_through_registry_seam(monkeypatch):
    monkeypatch.setenv("AUTOTUNEX_REGISTRY", "local")
    clear_overrides()
    fake = _FakeRegistry()
    register_override(Seam.REGISTRY, "local", lambda **kw: fake)
    try:
        job = job_service.Job(db=None)
        assert job.dmf is fake
        # The monitoring loop's checkpoint/detail ops route through the resolved registry:
        job.dmf.get_checkpoints("uri://ckpt")
        job.dmf.get_model_detail(model_label="m")
        assert ("get_checkpoints", "uri://ckpt") in fake.calls
        assert ("get_model_detail", "m") in fake.calls
    finally:
        clear_overrides()
