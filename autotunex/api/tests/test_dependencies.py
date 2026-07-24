# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import os

import dependencies
from services.plugins import Seam, register_override, clear_overrides
from services.registry.base import ModelRegistry


class _Sentinel(ModelRegistry):
    """Minimal concrete stand-in; methods are never actually called."""

    async def get_models(self, user_id): ...
    async def get_all_models(self): ...
    async def publish_model(self, job_id, metadata, user_id): ...
    async def delete_model(self, job_id, user_id): ...
    def get_model_detail(
        self, model_label, namespace="base_training", table="model_shared"
    ): ...
    def get_checkpoints(self, artifact_url): ...
    def pull_all_checkpoint_files(self, artifact_url): ...
    def pull_checkpoint_file(self, artifact_url, file_paths): ...
    def get_model_card(self, namespace, table, model_label, revision): ...
    def search_models(self, query): ...


def test_get_dmf_service_resolves_through_registry_seam():
    clear_overrides()
    register_override(Seam.REGISTRY, "local", lambda **kw: _Sentinel(**kw))
    os.environ["AUTOTUNEX_REGISTRY"] = (
        "local"  # force the override name regardless of env/lakehouse
    )
    try:
        svc = dependencies.get_dmf_service(database=None)
        assert isinstance(svc, ModelRegistry)
        assert isinstance(svc, _Sentinel)
    finally:
        os.environ.pop("AUTOTUNEX_REGISTRY", None)
        clear_overrides()


def test_get_dmf_service_is_dmf_registry_in_ibm_mode(monkeypatch):
    # IBM mode = lakehouse importable, no env override -> fallback 'dmf' -> DmfRegistry.
    import services.plugins.registry as reg
    from services.registry.dmf_backend import DmfRegistry

    monkeypatch.delenv("AUTOTUNEX_REGISTRY", raising=False)
    monkeypatch.setattr(reg, "_can_import", lambda mod: True)
    svc = dependencies.get_dmf_service(database=None)
    assert isinstance(svc, DmfRegistry)


def test_mcp_get_services_dmf_resolves_through_registry_seam(monkeypatch):
    import mcp_server
    from services.plugins import Seam, register_override, clear_overrides

    # Avoid a real DB connection: stub Database to a harmless object.
    monkeypatch.setattr(mcp_server.db_service, "Database", lambda *a, **k: object())
    monkeypatch.setenv("AUTOTUNEX_REGISTRY", "local")
    clear_overrides()
    register_override(Seam.REGISTRY, "local", lambda **kw: "REGISTRY_SENTINEL")
    try:
        svc = mcp_server._get_services()
        assert svc["dmf"] == "REGISTRY_SENTINEL"
    finally:
        clear_overrides()
