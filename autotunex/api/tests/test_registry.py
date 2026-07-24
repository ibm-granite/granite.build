# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import services.plugins.registry as reg
from services.plugins import Seam, UnknownProviderError, register_override, resolve


def test_seam_values_are_entry_point_groups():
    assert Seam.RUNNER.value == "autotunex.runners"
    assert Seam.REGISTRY.value == "autotunex.model_registries"
    assert Seam.AUTH.value == "autotunex.auth_providers"
    assert Seam.CHAT.value == "autotunex.chat_providers"
    assert Seam.STORAGE.value == "autotunex.storage_backends"


def test_resolve_uses_override_factory():
    sentinel = object()
    register_override(Seam.RUNNER, "fake", lambda **kw: (sentinel, kw))
    obj, kwargs = resolve(Seam.RUNNER, name="fake", job_id="j1")
    assert obj is sentinel
    assert kwargs == {"job_id": "j1"}


def test_resolve_unknown_name_raises_with_helpful_message():
    try:
        resolve(Seam.RUNNER, name="does-not-exist")
        assert False, "expected UnknownProviderError"
    except UnknownProviderError as exc:
        msg = str(exc)
        assert "does-not-exist" in msg
        assert "autotunex.runners" in msg


def test_fallback_runner_is_gb_when_gb_enabled(monkeypatch):
    monkeypatch.delenv("AUTOTUNEX_RUNNER", raising=False)
    monkeypatch.setattr(reg, "is_gb_enabled", lambda: True)
    assert reg._select_name(Seam.RUNNER, None) == "gb"


def test_fallback_runner_is_local_when_gb_disabled(monkeypatch):
    monkeypatch.delenv("AUTOTUNEX_RUNNER", raising=False)
    monkeypatch.setattr(reg, "is_gb_enabled", lambda: False)
    assert reg._select_name(Seam.RUNNER, None) == "local"


def test_explicit_env_overrides_fallback(monkeypatch):
    monkeypatch.setenv("AUTOTUNEX_RUNNER", "local")
    monkeypatch.setattr(reg, "is_gb_enabled", lambda: True)  # would be gb
    assert reg._select_name(Seam.RUNNER, None) == "local"


def test_explicit_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("AUTOTUNEX_RUNNER", "local")
    assert reg._select_name(Seam.RUNNER, "gb") == "gb"


import ast
from pathlib import Path


def test_fallback_registry_is_dmf_when_lakehouse_importable(monkeypatch):
    monkeypatch.delenv("AUTOTUNEX_REGISTRY", raising=False)
    monkeypatch.setattr(reg, "_can_import", lambda mod: True)
    assert reg._select_name(Seam.REGISTRY, None) == "dmf"


def test_fallback_registry_is_local_when_lakehouse_missing(monkeypatch):
    monkeypatch.delenv("AUTOTUNEX_REGISTRY", raising=False)
    monkeypatch.setattr(reg, "_can_import", lambda mod: False)
    assert reg._select_name(Seam.REGISTRY, None) == "local"


def test_registry_fallback_is_not_coupled_to_gb(monkeypatch):
    # Even with GB enabled, registry selection depends on lakehouse presence, not GB.
    monkeypatch.delenv("AUTOTUNEX_REGISTRY", raising=False)
    monkeypatch.setattr(reg, "is_gb_enabled", lambda: True)
    monkeypatch.setattr(reg, "_can_import", lambda mod: False)
    assert reg._select_name(Seam.REGISTRY, None) == "local"


def test_explicit_registry_env_wins(monkeypatch):
    monkeypatch.setenv("AUTOTUNEX_REGISTRY", "hf")
    monkeypatch.setattr(reg, "_can_import", lambda mod: True)  # would be dmf
    assert reg._select_name(Seam.REGISTRY, None) == "hf"


def test_setup_py_core_has_no_ibm_git_requirements():
    setup_src = Path(__file__).resolve().parent.parent.joinpath("setup.py").read_text()
    tree = ast.parse(setup_src)
    install_requires = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "install_requires":
            install_requires = [
                el.value for el in node.value.elts if isinstance(el, ast.Constant)
            ]
    assert install_requires, "install_requires not found"
    for req in install_requires:
        assert "github.ibm.com" not in req, f"IBM git dep leaked into core: {req}"


def test_setup_py_declares_model_registry_entry_points():
    setup_src = Path(__file__).resolve().parent.parent.joinpath("setup.py").read_text()
    tree = ast.parse(setup_src)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "entry_points":
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant):
                    found[k.value] = [
                        e.value for e in v.elts if isinstance(e, ast.Constant)
                    ]
    assert "autotunex.model_registries" in found
    joined = " ".join(found["autotunex.model_registries"])
    assert "local = services.registry.local_backend:LocalRegistry" in joined
    assert "dmf = services.registry.dmf_backend:DmfRegistry" in joined
    assert "hf = services.registry.hf_backend:HuggingFaceRegistry" in joined


def test_fallback_auth_w3id_requires_all_three_oidc_vars(monkeypatch):
    monkeypatch.delenv("AUTOTUNEX_AUTH", raising=False)
    monkeypatch.setenv("OIDC_CLIENT_ID", "id")
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "secret")
    monkeypatch.setenv("OIDC_SECURITY_ENDPOINT", "https://oidc.example/oauth2")
    assert reg._select_name(Seam.AUTH, None) == "w3id"


def test_fallback_auth_dev_when_any_oidc_var_missing(monkeypatch):
    monkeypatch.delenv("AUTOTUNEX_AUTH", raising=False)
    monkeypatch.setenv("OIDC_CLIENT_ID", "id")
    monkeypatch.delenv("OIDC_CLIENT_SECRET", raising=False)  # missing secret
    monkeypatch.setenv("OIDC_SECURITY_ENDPOINT", "https://oidc.example/oauth2")
    assert reg._select_name(Seam.AUTH, None) == "dev"


def test_fallback_auth_dev_when_no_oidc(monkeypatch):
    monkeypatch.delenv("AUTOTUNEX_AUTH", raising=False)
    monkeypatch.delenv("OIDC_CLIENT_ID", raising=False)
    monkeypatch.delenv("OIDC_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("OIDC_SECURITY_ENDPOINT", raising=False)
    assert reg._select_name(Seam.AUTH, None) == "dev"


def test_explicit_auth_env_wins(monkeypatch):
    monkeypatch.setenv("AUTOTUNEX_AUTH", "dev")
    monkeypatch.setenv("OIDC_CLIENT_ID", "id")
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "secret")
    monkeypatch.setenv("OIDC_SECURITY_ENDPOINT", "https://oidc.example/oauth2")
    assert reg._select_name(Seam.AUTH, None) == "dev"


def test_setup_py_declares_auth_provider_entry_points():
    from pathlib import Path

    setup_src = Path(__file__).resolve().parent.parent.joinpath("setup.py").read_text()
    tree = ast.parse(setup_src)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "entry_points":
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant):
                    found[k.value] = [
                        e.value for e in v.elts if isinstance(e, ast.Constant)
                    ]
    assert "autotunex.auth_providers" in found
    joined = " ".join(found["autotunex.auth_providers"])
    assert "w3id = services.auth_providers.w3id_provider:W3idAuthProvider" in joined
    assert "dev = services.auth_providers.dev_provider:DevAuthProvider" in joined


def test_get_current_user_delegates_to_resolved_provider(monkeypatch):
    import asyncio

    import auth
    from services.plugins import Seam, register_override

    sentinel = object()

    class _FakeProvider:
        async def get_current_user(self, request):
            return sentinel

    register_override(Seam.AUTH, "fake", lambda: _FakeProvider())
    monkeypatch.setenv("AUTOTUNEX_AUTH", "fake")

    class _Req:
        cookies = {}

    result = asyncio.get_event_loop().run_until_complete(auth.get_current_user(_Req()))
    assert result is sentinel


def test_fallback_chat_openai_compat_when_base_url_set(monkeypatch):
    monkeypatch.delenv("AUTOTUNEX_CHAT", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    assert reg._select_name(Seam.CHAT, None) == "openai_compatible"


def test_fallback_chat_litellm_when_no_base_url(monkeypatch):
    monkeypatch.delenv("AUTOTUNEX_CHAT", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert reg._select_name(Seam.CHAT, None) == "litellm"


def test_explicit_chat_env_wins(monkeypatch):
    monkeypatch.setenv("AUTOTUNEX_CHAT", "litellm")
    monkeypatch.setenv(
        "OPENAI_BASE_URL", "http://localhost:11434/v1"
    )  # would be openai_compatible
    assert reg._select_name(Seam.CHAT, None) == "litellm"


def test_setup_py_declares_chat_provider_entry_points():
    from pathlib import Path

    setup_src = Path(__file__).resolve().parent.parent.joinpath("setup.py").read_text()
    tree = ast.parse(setup_src)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "entry_points":
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant):
                    found[k.value] = [
                        e.value for e in v.elts if isinstance(e, ast.Constant)
                    ]
    assert "autotunex.chat_providers" in found
    joined = " ".join(found["autotunex.chat_providers"])
    assert (
        "litellm = services.chat_providers.litellm_provider:LiteLLMChatProvider"
        in joined
    )
    assert (
        "openai_compatible = services.chat_providers.openai_compat_provider:OpenAICompatChatProvider"
        in joined
    )
