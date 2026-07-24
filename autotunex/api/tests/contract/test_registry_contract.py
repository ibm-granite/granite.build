# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest

from services.registry.base import ModelRegistry
from services.registry.local_backend import LocalRegistry
from services.registry.hf_backend import HuggingFaceRegistry
from services.registry.dmf_backend import DmfRegistry

BACKENDS = [LocalRegistry, HuggingFaceRegistry, DmfRegistry]


@pytest.mark.parametrize("cls", BACKENDS)
def test_backend_is_concrete_model_registry(cls):
    assert issubclass(cls, ModelRegistry)
    # No abstract methods left unimplemented => instantiable.
    assert not getattr(cls, "__abstractmethods__", frozenset()), (
        f"{cls.__name__} has unimplemented abstract methods: {cls.__abstractmethods__}"
    )


@pytest.mark.parametrize("cls", BACKENDS)
def test_backend_can_instantiate(cls):
    # db=None is acceptable for construction; methods aren't called here.
    instance = cls(db=None)
    assert isinstance(instance, ModelRegistry)


@pytest.mark.parametrize("cls", BACKENDS)
def test_backend_preserves_async_arity(cls):
    assert inspect.iscoroutinefunction(cls.get_models)
    assert inspect.iscoroutinefunction(cls.publish_model)
    assert not inspect.iscoroutinefunction(cls.get_model_detail)


import inspect as _inspect

from services.auth_providers.base import AuthProvider
from services.auth_providers.dev_provider import DevAuthProvider
from services.auth_providers.w3id_provider import W3idAuthProvider
from services.chat_providers.base import ChatProvider
from services.chat_providers.litellm_provider import LiteLLMChatProvider
from services.chat_providers.openai_compat_provider import OpenAICompatChatProvider

AUTH_BACKENDS = [DevAuthProvider, W3idAuthProvider]
CHAT_BACKENDS = [LiteLLMChatProvider, OpenAICompatChatProvider]


@pytest.mark.parametrize("cls", AUTH_BACKENDS)
def test_auth_provider_concrete_and_async(cls):
    assert issubclass(cls, AuthProvider)
    assert not getattr(cls, "__abstractmethods__", frozenset())
    assert _inspect.iscoroutinefunction(cls.get_current_user)


@pytest.mark.parametrize("cls", CHAT_BACKENDS)
def test_chat_provider_concrete(cls):
    assert issubclass(cls, ChatProvider)
    assert not getattr(cls, "__abstractmethods__", frozenset())
    assert hasattr(cls, "build_llm")


from models import ModelInfo

# Required ModelInfo fields every registry backend must emit (files excluded on purpose:
# ModelInfo.files is List[FileInfo], but LocalRegistry stores plain filename strings and we
# deliberately did not harden files into FileInfo — YAGNI, the list table does not consume it).
_REQUIRED_KEYS = (
    "model_id",
    "model_label",
    "base_model",
    "revision",
    "open",
    "product_name",
)

# Representative item mirroring each backend's REAL emitted keys (the shape, not a live call):
#  - Local: publish_model manifest
#  - DMF:   list_models items enriched with `user`
#  - HF:    delegates get_models to Local, so same manifest shape
_REPRESENTATIVE_ITEM = {
    "model_id": "job-1",
    "user": "a@b.c",
    "model_label": "my-model",
    "base_model": "granite-3.0",
    "size": "8B",
    "revision": "job-1",
    "open": False,
    "product_name": "autotunex",
    "files": ["adapter_model.bin"],  # present, but NOT validated below
}


def test_representative_item_validates_required_modelinfo_fields():
    subset = {k: _REPRESENTATIVE_ITEM[k] for k in _REQUIRED_KEYS}
    subset["user"] = _REPRESENTATIVE_ITEM.get("user")  # required-but-Optional
    # Must not raise: proves the required ModelInfo fields are present and typed.
    ModelInfo.model_validate(subset)


def test_missing_required_field_fails_validation():
    subset = {k: _REPRESENTATIVE_ITEM[k] for k in _REQUIRED_KEYS if k != "model_label"}
    subset["user"] = None
    with pytest.raises(Exception):  # pydantic.ValidationError
        ModelInfo.model_validate(subset)
