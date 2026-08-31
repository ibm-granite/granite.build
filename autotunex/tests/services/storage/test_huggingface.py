"""HuggingFaceStorageBackend: llmb auth + artifact push, subprocess faked.

``subprocess.run`` is faked so no real ``llmb`` is needed; the train/validation
files are staged for real into a temp directory, which the fake inspects before
the backend cleans it up. Token env vars are test-only names, set or cleared
explicitly per test so an ambient ``GB_TOKEN``/``HF_TOKEN`` cannot change results.
"""

from __future__ import annotations

import subprocess
import traceback
from pathlib import Path
from uuid import UUID

import httpx
import pytest

from autotunex.core.exceptions import DatasetPushFailedError, DatasetPushTimeoutError
from autotunex.models.dataset import DatasetPreview
from autotunex.services.storage.huggingface import HuggingFaceStorageBackend

DATASET_ID = UUID("0f55c1a6-92b5-4237-b0cd-ac05bc8381ce")
GB_ENV = "AUTOTUNEX_STORAGE_TEST_GB"
HF_ENV = "AUTOTUNEX_STORAGE_TEST_HF"
PUSH_STDOUT = (
    "pushed artifact 11111111-1111-1111-1111-111111111111 "
    "to hf://huggingface.co/datasets/ibm-research/finance_0f55c1a6"
)


def _backend(
    *,
    hf_preview_enabled: bool = True,
    tags: str = "autotunex",
    http_client: httpx.AsyncClient | None = None,
    push_timeout_seconds: float = 30.0,
) -> HuggingFaceStorageBackend:
    return HuggingFaceStorageBackend(
        llmb_command="llmb",
        hf_token_env=HF_ENV,
        gb_token_env=GB_ENV,
        hf_namespace=None,
        hf_preview_enabled=hf_preview_enabled,
        hf_viewer_base_url="https://viewer.test",
        hf_viewer_timeout_seconds=2.5,
        tags=tags,
        push_timeout_seconds=push_timeout_seconds,
        http_client=http_client,
    )


def _staged(tmp_path: Path) -> tuple[Path, Path]:
    """Two source files with non-canonical names, beside an unrelated leftover."""
    staging = tmp_path / "staging"
    staging.mkdir()
    train = staging / "train_split.jsonl"
    train.write_text('{"a": 1}\n')
    validation = staging / "validation_split.jsonl"
    validation.write_text('{"a": 2}\n')
    (staging / "leftover_original.jsonl").write_text("junk\n")  # must NOT be pushed
    return train, validation


async def test_persist_authenticates_then_pushes_and_parses_refs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(GB_ENV, "gb-secret")
    monkeypatch.setenv(HF_ENV, "hf-secret")
    train, validation = _staged(tmp_path)
    calls: list[list[str]] = []
    pushed_names: list[str] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[1:3] == ["artifact", "push"]:
            from_local = Path(args[args.index("--from-local") + 1])
            pushed_names.extend(sorted(p.name for p in from_local.iterdir()))
        return subprocess.CompletedProcess(args, 0, stdout=PUSH_STDOUT, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    artifact_id, artifact_url = await _backend().persist(
        dataset_id=DATASET_ID,
        name="finance",
        data_format="jsonl",
        train=train,
        validation=validation,
    )

    # auth first, then the HF push.
    assert calls[0] == ["llmb", "auth", "login", "--token", "gb-secret"]
    push = calls[1]
    assert push[1:3] == ["artifact", "push"]
    assert push[push.index("--type") + 1] == "dataset"
    assert push[push.index("--store") + 1] == "hf"
    assert push[push.index("--tag") + 1] == "autotunex"
    assert "--certify-no-restrictions" in push
    assert push[push.index("--artifact-name") + 1] == "finance_0f55c1a6"
    # only the two canonical files are staged for the push — not the leftover.
    assert pushed_names == ["finance_train.jsonl", "finance_validation.jsonl"]
    assert artifact_id == UUID("11111111-1111-1111-1111-111111111111")
    assert artifact_url == "hf://huggingface.co/datasets/ibm-research/finance_0f55c1a6"


async def test_persist_passes_the_configured_tag_to_the_push(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(GB_ENV, "gb-secret")
    monkeypatch.setenv(HF_ENV, "hf-secret")
    train, validation = _staged(tmp_path)
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=PUSH_STDOUT, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # A comma-separated value is forwarded verbatim; llmb splits it, not us.
    await _backend(tags="autotunex,dataset").persist(
        dataset_id=DATASET_ID,
        name="finance",
        data_format="jsonl",
        train=train,
        validation=validation,
    )

    push = calls[1]
    assert push[push.index("--tag") + 1] == "autotunex,dataset"


async def test_persist_omits_tag_when_none_is_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(GB_ENV, "gb-secret")
    monkeypatch.setenv(HF_ENV, "hf-secret")
    train, validation = _staged(tmp_path)
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=PUSH_STDOUT, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    await _backend(tags="   ").persist(  # whitespace-only == unset
        dataset_id=DATASET_ID,
        name="finance",
        data_format="jsonl",
        train=train,
        validation=validation,
    )

    push = calls[1]
    assert "--tag" not in push
    assert "--tags" not in push  # the old hardcoded flag is gone too


async def test_persist_without_validation_pushes_only_the_train_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(GB_ENV, "gb-secret")
    monkeypatch.setenv(HF_ENV, "hf-secret")
    train, _ = _staged(tmp_path)
    pushed_names: list[str] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[1:3] == ["artifact", "push"]:
            from_local = Path(args[args.index("--from-local") + 1])
            pushed_names.extend(sorted(p.name for p in from_local.iterdir()))
        return subprocess.CompletedProcess(args, 0, stdout=PUSH_STDOUT, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    await _backend().persist(
        dataset_id=DATASET_ID, name="finance", data_format="jsonl", train=train, validation=None
    )

    assert pushed_names == ["finance_train.jsonl"]


async def test_persist_skips_auth_when_gb_token_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(GB_ENV, raising=False)
    monkeypatch.setenv(HF_ENV, "hf-secret")
    train, validation = _staged(tmp_path)
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=PUSH_STDOUT, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    await _backend().persist(
        dataset_id=DATASET_ID,
        name="finance",
        data_format="jsonl",
        train=train,
        validation=validation,
    )

    assert all(call[1:3] != ["auth", "login"] for call in calls)
    assert calls[0][1:3] == ["artifact", "push"]


async def test_persist_auth_failure_raises_without_leaking_the_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(GB_ENV, "gb-secret")
    monkeypatch.setenv(HF_ENV, "hf-secret")
    train, validation = _staged(tmp_path)

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[1:3] == ["auth", "login"]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="invalid token")
        return subprocess.CompletedProcess(args, 0, stdout=PUSH_STDOUT, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(DatasetPushFailedError) as exc_info:
        await _backend().persist(
            dataset_id=DATASET_ID,
            name="finance",
            data_format="jsonl",
            train=train,
            validation=validation,
        )

    message = str(exc_info.value)
    assert "invalid token" not in message  # llmb's raw stderr is never surfaced
    assert "gb-secret" not in message  # the token is never in the error


async def test_auth_login_timeout_does_not_leak_the_token_into_the_exception_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ``TimeoutExpired`` on ``llmb auth login`` must not smuggle the GB token.

    ``subprocess.run`` attaches its full argv (``--token <token>`` included) to
    ``TimeoutExpired.cmd``, and that argv is rendered by ``str()``/``repr()`` of
    the exception. If the handler re-raises ``from exc`` (or lets a bare
    ``raise`` set ``__context__``), the token-bearing ``TimeoutExpired`` rides
    along as ``__cause__``/``__context__`` and would be printed in full by the
    runner's ``logger.exception(...)`` — leaking a live credential to
    application logs. This asserts the token is absent from the exception's
    *entire* rendered chain, the same rendering ``logger.exception`` produces.
    """
    token = "gb-SUPERSECRET-do-not-leak"
    monkeypatch.setenv(GB_ENV, token)
    train, _ = _staged(tmp_path)

    def fake_run(*args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        # subprocess.run puts the full argv (incl. the token) on TimeoutExpired.cmd.
        raise subprocess.TimeoutExpired(
            cmd=["llmb", "auth", "login", "--token", token], timeout=1.0
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(DatasetPushTimeoutError) as exc_info:
        await _backend(push_timeout_seconds=1.0).persist(
            dataset_id=DATASET_ID, name="finance", data_format="jsonl", train=train, validation=None
        )

    rendered = "".join(
        traceback.format_exception(
            type(exc_info.value), exc_info.value, exc_info.value.__traceback__
        )
    )
    assert token not in rendered
    assert exc_info.value.__cause__ is None
    # `from None` does NOT clear `__context__` (CPython still records the
    # in-flight TimeoutExpired there — verified against 3.11/3.12 semantics);
    # it sets `__suppress_context__ = True`, which is what makes
    # `traceback.format_exception`/`logger.exception` skip rendering that
    # context at all. That suppression is the actual security-relevant
    # invariant here, not `__context__` being unset.
    assert exc_info.value.__suppress_context__ is True


async def test_persist_raises_when_the_push_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(GB_ENV, "gb-secret")
    monkeypatch.setenv(HF_ENV, "hf-secret")
    train, validation = _staged(tmp_path)

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[1:3] == ["artifact", "push"]:
            raise subprocess.CalledProcessError(1, args, stderr="boom")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(DatasetPushFailedError):
        await _backend().persist(
            dataset_id=DATASET_ID,
            name="finance",
            data_format="jsonl",
            train=train,
            validation=validation,
        )


async def test_persist_push_timeout_maps_to_push_timeout_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(GB_ENV, raising=False)  # skip auth; go straight to push
    train, _ = _staged(tmp_path)

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="llmb", timeout=1.0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(DatasetPushTimeoutError):
        await _backend(push_timeout_seconds=1.0).persist(
            dataset_id=DATASET_ID, name="finance", data_format="jsonl", train=train, validation=None
        )


async def test_persist_push_nonzero_exit_maps_to_push_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(GB_ENV, raising=False)
    train, _ = _staged(tmp_path)

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(returncode=2, cmd="llmb", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(DatasetPushFailedError):
        await _backend().persist(
            dataset_id=DATASET_ID, name="finance", data_format="jsonl", train=train, validation=None
        )


async def test_persist_push_missing_binary_maps_to_push_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(GB_ENV, raising=False)
    train, _ = _staged(tmp_path)

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("llmb")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(DatasetPushFailedError):
        await _backend().persist(
            dataset_id=DATASET_ID, name="finance", data_format="jsonl", train=train, validation=None
        )


async def test_persist_unparseable_push_output_maps_to_push_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(GB_ENV, raising=False)
    train, _ = _staged(tmp_path)

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="done, no refs", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(DatasetPushFailedError):
        await _backend().persist(
            dataset_id=DATASET_ID, name="finance", data_format="jsonl", train=train, validation=None
        )


VALID_URL = "hf://huggingface.co/datasets/ibm-research/finance_0f55c1a6"


def _viewer_client(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


async def test_preview_returns_rows_for_both_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HF_ENV, "hf_tok")

    def handler(request: httpx.Request) -> httpx.Response:
        split = request.url.params["split"]
        return httpx.Response(200, json={"rows": [{"row": {"split": split}}]})

    async with _viewer_client(handler) as client:
        preview = await _backend(http_client=client).preview(
            dataset_id=DATASET_ID,
            name="finance",
            data_format="jsonl",
            artifact_url=VALID_URL,
            rows=10,
        )

    assert preview.train == [{"split": "train"}]
    assert preview.validation == [{"split": "validation"}]


async def test_preview_uses_and_closes_its_own_client_when_none_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production path: no ``http_client`` injected, so ``preview`` opens and closes one.

    ``httpx.AsyncClient`` is patched at the name the backend module calls it by
    (``autotunex.services.storage.huggingface.httpx.AsyncClient``) with a factory
    that hands back a client wired to a ``MockTransport`` handler, wrapping
    ``aclose`` to record that it ran. If the ``finally: await client.aclose()``
    in ``_preview_from_viewer`` were ever removed, ``closed`` would stay empty
    and the final assertion would fail even though the rows still come back.
    """
    monkeypatch.setenv(HF_ENV, "hf_tok")
    closed: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:
        split = request.url.params["split"]
        return httpx.Response(200, json={"rows": [{"row": {"split": split}}]})

    class _OwnedClient(httpx.AsyncClient):
        async def aclose(self) -> None:
            closed.append(True)
            await super().aclose()

    def factory(*, timeout: float) -> httpx.AsyncClient:
        assert timeout == 2.5  # the backend's configured hf_viewer_timeout_seconds
        return _OwnedClient(transport=httpx.MockTransport(handler), timeout=timeout)

    monkeypatch.setattr("autotunex.services.storage.huggingface.httpx.AsyncClient", factory)

    preview = await _backend().preview(
        dataset_id=DATASET_ID,
        name="finance",
        data_format="jsonl",
        artifact_url=VALID_URL,
        rows=10,
    )

    assert preview.train == [{"split": "train"}]
    assert preview.validation == [{"split": "validation"}]
    assert closed == [True]


async def test_preview_partial_when_validation_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(HF_ENV, "hf_tok")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["split"] == "train":
            return httpx.Response(200, json={"rows": [{"row": {"a": 1}}]})
        return httpx.Response(404, json={"error": "no validation split"})

    async with _viewer_client(handler) as client:
        preview = await _backend(http_client=client).preview(
            dataset_id=DATASET_ID,
            name="finance",
            data_format="jsonl",
            artifact_url=VALID_URL,
            rows=10,
        )

    assert preview.train == [{"a": 1}]
    assert preview.validation == []


async def test_preview_empty_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HF_ENV, "hf_tok")
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"rows": [{"row": {"a": 1}}]})

    async with _viewer_client(handler) as client:
        preview = await _backend(hf_preview_enabled=False, http_client=client).preview(
            dataset_id=DATASET_ID,
            name="finance",
            data_format="jsonl",
            artifact_url=VALID_URL,
            rows=10,
        )

    assert preview == DatasetPreview(train=[], validation=[])
    assert called is False


async def test_preview_empty_when_artifact_url_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HF_ENV, "hf_tok")

    preview = await _backend().preview(
        dataset_id=DATASET_ID,
        name="finance",
        data_format="jsonl",
        artifact_url=None,
        rows=10,
    )

    assert preview == DatasetPreview(train=[], validation=[])


async def test_preview_empty_when_repo_belongs_to_other_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(HF_ENV, "hf_tok")

    preview = await _backend().preview(
        dataset_id=DATASET_ID,
        name="finance",
        data_format="jsonl",
        artifact_url="hf://huggingface.co/datasets/ibm-research/finance_deadbeef",
        rows=10,
    )

    assert preview == DatasetPreview(train=[], validation=[])


async def test_preview_empty_when_token_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(HF_ENV, raising=False)

    preview = await _backend().preview(
        dataset_id=DATASET_ID,
        name="finance",
        data_format="jsonl",
        artifact_url=VALID_URL,
        rows=10,
    )

    assert preview == DatasetPreview(train=[], validation=[])


async def test_preview_never_raises_on_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(HF_ENV, "hf_tok")

    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("unexpected")

    async with _viewer_client(handler) as client:
        preview = await _backend(http_client=client).preview(
            dataset_id=DATASET_ID,
            name="finance",
            data_format="jsonl",
            artifact_url=VALID_URL,
            rows=10,
        )

    assert preview == DatasetPreview(train=[], validation=[])
