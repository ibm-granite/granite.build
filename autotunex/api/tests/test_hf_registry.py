# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import sys
import types

import pytest
from fastapi import HTTPException


def _install_fake_hf(monkeypatch):
    """Register a minimal fake huggingface_hub so tests need no network/wheel."""
    hub = types.ModuleType("huggingface_hub")

    class _Sibling:
        def __init__(self, name):
            self.rfilename = name

    class _Info:
        def __init__(self):
            self.siblings = [_Sibling("config.json"), _Sibling("model.safetensors")]
            self.id = "org/model"

    class HfApi:
        def model_info(self, repo_id, revision=None):
            return _Info()

        def list_models(self, search=None, limit=None):
            return [
                types.SimpleNamespace(id=f"org/{search}-a"),
                types.SimpleNamespace(id="org/other"),
                types.SimpleNamespace(id="gpt2"),  # slash-free community model id
            ]

    hub.HfApi = HfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


def test_publish_model_raises_501(monkeypatch):
    _install_fake_hf(monkeypatch)
    from services.registry.hf_backend import HuggingFaceRegistry

    reg = HuggingFaceRegistry(db=None)
    with pytest.raises(HTTPException) as ei:
        # publish_model is async; drive it on a loop
        import asyncio

        asyncio.get_event_loop().run_until_complete(
            reg.publish_model(job_id="j", metadata=None, user_id="u")
        )
    assert ei.value.status_code == 501


def test_delete_model_raises_501(monkeypatch):
    _install_fake_hf(monkeypatch)
    import asyncio

    from services.registry.hf_backend import HuggingFaceRegistry

    reg = HuggingFaceRegistry(db=None)
    with pytest.raises(HTTPException) as ei:
        asyncio.get_event_loop().run_until_complete(
            reg.delete_model(job_id="j", user_id="u")
        )
    assert ei.value.status_code == 501


def test_search_models_envelope(monkeypatch):
    _install_fake_hf(monkeypatch)
    from services.registry.hf_backend import HuggingFaceRegistry

    reg = HuggingFaceRegistry(db=None)
    res = reg.search_models("granite")
    assert "data" in res
    assert any(
        "granite" in m.get("model_label", "") or "granite" in m.get("model_id", "")
        for m in res["data"]
    )


def test_get_checkpoints_lists_siblings(monkeypatch):
    _install_fake_hf(monkeypatch)
    from services.registry.hf_backend import HuggingFaceRegistry

    reg = HuggingFaceRegistry(db=None)
    files = reg.get_checkpoints("org/model")
    names = {f["name"] for f in files}
    assert "config.json" in names and "model.safetensors" in names


def test_hf_search_models_includes_wizard_fields(monkeypatch):
    _install_fake_hf(monkeypatch)
    from services.registry.hf_backend import HuggingFaceRegistry

    reg = HuggingFaceRegistry(db=None)
    out = reg.search_models("model")
    assert "data" in out and out["data"], "search must return a non-empty data envelope"
    item = out["data"][0]
    assert item[
        "namespace"
    ], "namespace must be non-empty so the wizard does not skip the item"
    assert item["base_model"]
    assert item["revision"]
    assert item["model_id"] and item["model_label"]

    # slash-free community model ids (e.g. "gpt2") must fall back to the full id
    by_id = {m["model_id"]: m for m in out["data"]}
    assert "gpt2" in by_id, "slash-free model id must be present in results"
    slash_free = by_id["gpt2"]
    assert (
        slash_free["namespace"] == "gpt2"
    ), "namespace must fall back to the full id when there is no slash"
    assert slash_free[
        "namespace"
    ], "namespace must be non-empty so the wizard does not skip the item"
