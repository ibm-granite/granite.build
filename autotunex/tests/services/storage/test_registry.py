"""get_storage_backend selection by settings and environment.

HuggingFace is chosen only when ``llmb`` resolves and BOTH tokens are set (the GB
token authenticates the CLI, the HF token is the push destination). ``shutil.which``
is patched where a resolvable binary is needed so the tests do not depend on a real
``llmb`` install.
"""

from __future__ import annotations

import pytest

from autotunex.services.storage.fallback import PreviewFallbackStorageBackend
from autotunex.services.storage.huggingface import HuggingFaceStorageBackend
from autotunex.services.storage.local import LocalStorageBackend
from autotunex.services.storage.registry import get_storage_backend
from tests.conftest import make_settings


def _both_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    monkeypatch.setenv("GB_TOKEN", "gb_xxx")


def test_forced_local_is_local(monkeypatch: pytest.MonkeyPatch) -> None:
    _both_tokens(monkeypatch)  # present, but local is forced

    backend = get_storage_backend(make_settings(dataset_storage_backend="local"))

    assert isinstance(backend, LocalStorageBackend)


def test_forced_huggingface_wraps_hf_with_local_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _both_tokens(monkeypatch)  # forced HF validation requires both tokens

    backend = get_storage_backend(make_settings(dataset_storage_backend="huggingface"))

    # The HF backend is wrapped so an empty HF preview falls back to local storage.
    assert isinstance(backend, PreviewFallbackStorageBackend)
    assert isinstance(backend._primary, HuggingFaceStorageBackend)
    assert isinstance(backend._fallback, LocalStorageBackend)


def test_auto_with_llmb_and_both_tokens_wraps_hf_with_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _both_tokens(monkeypatch)
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")

    backend = get_storage_backend(make_settings(dataset_storage_backend="auto"))

    assert isinstance(backend, PreviewFallbackStorageBackend)
    assert isinstance(backend._primary, HuggingFaceStorageBackend)
    assert isinstance(backend._fallback, LocalStorageBackend)


def test_auto_without_tokens_falls_back_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GB_TOKEN", raising=False)
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")

    backend = get_storage_backend(make_settings(dataset_storage_backend="auto"))

    assert isinstance(backend, LocalStorageBackend)


def test_auto_missing_gb_token_falls_back_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    monkeypatch.delenv("GB_TOKEN", raising=False)
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")

    backend = get_storage_backend(make_settings(dataset_storage_backend="auto"))

    assert isinstance(backend, LocalStorageBackend)


def test_huggingface_backend_receives_viewer_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _both_tokens(monkeypatch)
    settings = make_settings(dataset_storage_backend="huggingface")

    backend = get_storage_backend(settings)

    assert isinstance(backend, PreviewFallbackStorageBackend)
    hf = backend._primary
    assert isinstance(hf, HuggingFaceStorageBackend)
    assert hf._hf_preview_enabled == settings.hf_preview_enabled
    assert hf._hf_viewer_base_url == settings.hf_viewer_base_url
    assert hf._hf_viewer_timeout_seconds == settings.hf_viewer_timeout_seconds


def test_huggingface_backend_receives_gb_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _both_tokens(monkeypatch)
    settings = make_settings(dataset_storage_backend="huggingface")

    backend = get_storage_backend(settings)

    assert isinstance(backend, PreviewFallbackStorageBackend)
    hf = backend._primary
    assert isinstance(hf, HuggingFaceStorageBackend)
    assert hf._tags == settings.gb_tags


def test_auto_standalone_bash_uses_local_with_file_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    # llmb + both tokens present would normally pick HF, but standalone can't push.
    _both_tokens(monkeypatch)
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")

    backend = get_storage_backend(
        make_settings(dataset_storage_backend="auto", gb_environment="standalone")
    )

    assert isinstance(backend, LocalStorageBackend)
    assert backend._emit_file_uri is True


def test_auto_standalone_lsf_uses_local_without_file_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    # Remote LSF build: store locally but emit NO local locator (it can't reach it).
    _both_tokens(monkeypatch)
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")

    backend = get_storage_backend(
        make_settings(
            dataset_storage_backend="auto",
            gb_environment="standalone",
            lsf_cluster="my-cluster",
        )
    )

    assert isinstance(backend, LocalStorageBackend)
    assert backend._emit_file_uri is False


def test_forced_local_standalone_bash_emits_file_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = get_storage_backend(
        make_settings(dataset_storage_backend="local", gb_environment="standalone")
    )

    assert isinstance(backend, LocalStorageBackend)
    assert backend._emit_file_uri is True


def test_forced_local_non_standalone_emits_no_file_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = get_storage_backend(make_settings(dataset_storage_backend="local"))

    assert isinstance(backend, LocalStorageBackend)
    assert backend._emit_file_uri is False
