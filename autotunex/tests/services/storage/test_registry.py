"""get_storage_backend selection by settings and environment.

HuggingFace is chosen only when ``llmb`` resolves and BOTH tokens are set (the GB
token authenticates the CLI, the HF token is the push destination). ``shutil.which``
is patched where a resolvable binary is needed so the tests do not depend on a real
``llmb`` install.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import httpx
import pytest

from autotunex.services.storage.fallback import PreviewFallbackStorageBackend
from autotunex.services.storage.huggingface import HuggingFaceStorageBackend
from autotunex.services.storage.local import LocalStorageBackend
from autotunex.services.storage.registry import get_storage_backend
from tests.conftest import make_settings


def _both_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    monkeypatch.setenv("GB_TOKEN", "gb_xxx")


# --- Local-resolved backends keep an HF preview fallback -----------------------
#
# Choosing local storage is a *write*-path decision: `llmb artifact push` is
# disabled under GB_ENVIRONMENT=STANDALONE, and `auto` degrades when llmb or the
# tokens are missing. Reading a preview instead goes over the HF dataset-viewer
# HTTP API, which needs neither llmb nor the GB token. So a dataset row that
# already carries an `hf://` locator (pushed before the switch to standalone, or
# written by another deployment sharing the database) must still preview: local is
# the primary (it owns persist/delete and the file:// locator) with HF as a
# preview-only fallback.

HF_HOSTED_ID = UUID("10d94a61-0000-4000-8000-000000000000")
HF_HOSTED_URL = f"hf://huggingface.co/datasets/example-org/eli5-test_{str(HF_HOSTED_ID)[:8]}"


def _local_primary(backend: object, *, emit_file_uri: bool) -> None:
    """Assert ``backend`` is local-primary with an HF preview fallback."""
    assert isinstance(backend, PreviewFallbackStorageBackend)
    primary = backend._primary
    assert isinstance(primary, LocalStorageBackend)
    assert primary._emit_file_uri is emit_file_uri
    assert isinstance(backend._fallback, HuggingFaceStorageBackend)


def test_forced_local_writes_locally_and_previews_from_hf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `local` pins where files are *written*; it does not blind the preview to a
    # dataset whose files live on HuggingFace.
    _both_tokens(monkeypatch)  # present, but local is forced

    backend = get_storage_backend(make_settings(dataset_storage_backend="local"))

    _local_primary(backend, emit_file_uri=False)


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

    _local_primary(backend, emit_file_uri=False)


def test_auto_missing_gb_token_falls_back_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    monkeypatch.delenv("GB_TOKEN", raising=False)
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")

    backend = get_storage_backend(make_settings(dataset_storage_backend="auto"))

    # The HF token alone cannot push, but it is all the viewer read needs.
    _local_primary(backend, emit_file_uri=False)


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

    _local_primary(backend, emit_file_uri=True)


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

    _local_primary(backend, emit_file_uri=False)


def test_forced_local_standalone_bash_emits_file_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = get_storage_backend(
        make_settings(dataset_storage_backend="local", gb_environment="standalone")
    )

    _local_primary(backend, emit_file_uri=True)


def test_forced_local_non_standalone_emits_no_file_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = get_storage_backend(make_settings(dataset_storage_backend="local"))

    _local_primary(backend, emit_file_uri=False)


async def test_auto_standalone_previews_an_hf_hosted_dataset_from_the_viewer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The reported bug, end to end: BASH runner + ``auto`` + an ``hf://`` dataset.

    Nothing for this dataset is on local disk (the HF push stages into a temp dir
    and never populates ``dataset_storage_dir``), so the local primary yields an
    empty preview and the HF fallback must serve the rows.
    """
    monkeypatch.setenv("HF_TOKEN", "hf_tok")
    monkeypatch.setenv("GB_TOKEN", "gb_tok")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/splits":
            return httpx.Response(200, json={"splits": [{"config": "default", "split": "train"}]})
        return httpx.Response(200, json={"rows": [{"row": {"text": "hello"}}]})

    # Bind the real class first: patching the attribute on the shared `httpx`
    # module rebinds it for this module too, so a lambda calling
    # `httpx.AsyncClient` would recurse into itself.
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "autotunex.services.storage.huggingface.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    backend = get_storage_backend(
        make_settings(
            dataset_storage_backend="auto",
            gb_environment="standalone",
            dataset_storage_dir=tmp_path,
        )
    )
    preview = await backend.preview(
        dataset_id=HF_HOSTED_ID,
        name="eli5-test",
        data_format="jsonl",
        artifact_url=HF_HOSTED_URL,
        rows=10,
    )

    assert preview.train == [{"text": "hello"}]


@pytest.mark.parametrize(
    ("backend", "gb_environment", "lsf_cluster", "llmb"),
    [
        ("local", "standalone", None, True),
        ("local", "prod", None, True),
        ("huggingface", "prod", None, True),
        ("huggingface", "standalone", "lsf-a", True),
        ("auto", "standalone", None, True),
        ("auto", "standalone", "lsf-a", True),
        ("auto", "prod", None, True),
        ("auto", "prod", None, False),
    ],
)
def test_backend_matches_resolved_dataset_storage(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    gb_environment: str,
    lsf_cluster: str | None,
    llmb: bool,
) -> None:
    """The backend built is the one Settings.resolved_dataset_storage names.

    hf_import_available gates on that property, so a drift here would open import
    where the launch cannot read the result.
    """
    _both_tokens(monkeypatch)
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}" if llmb else None)
    settings = make_settings(
        dataset_storage_backend=backend,  # type: ignore[arg-type]
        gb_environment=gb_environment,
        lsf_cluster=lsf_cluster,
    )

    built = get_storage_backend(settings)

    expected = settings.resolved_dataset_storage
    assert isinstance(built, PreviewFallbackStorageBackend)
    if expected == "huggingface":
        assert isinstance(built._primary, HuggingFaceStorageBackend)
    else:
        _local_primary(built, emit_file_uri=expected == "local_file_uri")
