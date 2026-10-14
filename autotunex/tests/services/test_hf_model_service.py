# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""HfModelService: the allowlist decides where the server's HF token may go."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from autotunex.core.exceptions import HfHubUnreachableError, HfModelNotFoundError
from autotunex.services.hf_model_service import HfModelService
from tests.conftest import make_settings

Handler = Callable[[httpx.Request], httpx.Response]


def _service(handler: Handler, namespaces: list[str]) -> HfModelService:
    settings = make_settings().model_copy(update={"hf_import_namespaces": namespaces})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HfModelService(client=client, settings=settings)


def _hub(scoped: list[str], public: list[str]) -> Handler:
    """Answer a pinned (``author=``) search with ``scoped`` (as private hits).

    Any other search is answered with ``public``.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if "author" in request.url.params:
            assert request.headers["authorization"] == "Bearer tok"
            return httpx.Response(200, json=[{"id": i, "private": True} for i in scoped])
        assert "authorization" not in request.headers
        return httpx.Response(200, json=[{"id": i} for i in public])

    return handler


async def test_search_puts_allowlisted_hits_first_and_drops_foreign_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "tok")
    service = _service(
        _hub(
            scoped=["example-org/private-a", "someone/else", "example-org/shared"],
            public=["example-org/shared", "pub/x"],
        ),
        ["example-org"],
    )

    ids = await service.search(query="a", limit=10)

    assert ids == ["example-org/private-a", "example-org/shared", "pub/x"]


async def test_search_drops_public_hits_from_a_pinned_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pinned query returns a whole namespace; only its private hits should survive."""
    monkeypatch.setenv("HF_TOKEN", "tok")

    def handler(request: httpx.Request) -> httpx.Response:
        if "author" in request.url.params:
            return httpx.Response(
                200,
                json=[
                    {"id": "example-org/pub", "private": False},
                    {"id": "example-org/priv", "private": True},
                ],
            )
        return httpx.Response(200, json=[{"id": "ibm-granite/a"}])

    ids = await _service(handler, ["example-org"]).search(query="granite", limit=10)

    assert ids == ["example-org/priv", "ibm-granite/a"]


async def test_search_finds_a_private_hit_ranked_below_limit_public_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pinned query's page holds public repos too, so ``limit`` alone can crowd out privates."""
    monkeypatch.setenv("HF_TOKEN", "tok")
    namespace = [{"id": f"example-org/pub-{n}", "private": False} for n in range(25)]
    namespace.append({"id": "example-org/priv", "private": True})

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["limit"])
        if "author" in request.url.params:
            return httpx.Response(200, json=namespace[:page])
        return httpx.Response(200, json=[{"id": "pub/x"}])

    ids = await _service(handler, ["example-org"]).search(query="granite", limit=20)

    assert ids[0] == "example-org/priv"


async def test_search_truncates_to_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "tok")
    service = _service(_hub(scoped=["example-org/a"], public=["p/b", "p/c"]), ["example-org"])

    assert await service.search(query="a", limit=2) == ["example-org/a", "p/b"]


async def test_a_failed_tokened_search_still_returns_public_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "tok")

    def handler(request: httpx.Request) -> httpx.Response:
        if "author" in request.url.params:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json=[{"id": "pub/x"}])

    assert await _service(handler, ["example-org"]).search(query="x", limit=5) == ["pub/x"]


async def test_a_failed_public_search_is_hub_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "tok")

    def handler(request: httpx.Request) -> httpx.Response:
        if "author" in request.url.params:
            return httpx.Response(200, json=[{"id": "example-org/a"}])
        return httpx.Response(502, text="bad gateway")

    with pytest.raises(HfHubUnreachableError):
        await _service(handler, ["example-org"]).search(query="x", limit=5)


@pytest.mark.parametrize(("token", "namespaces"), [(None, ["example-org"]), ("tok", [])])
async def test_search_is_one_anonymous_call_without_a_token_or_an_allowlist(
    monkeypatch: pytest.MonkeyPatch, token: str | None, namespaces: list[str]
) -> None:
    if token is None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HF_TOKEN", token)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[{"id": "pub/x"}])

    await _service(handler, namespaces).search(query="x", limit=5)

    assert len(seen) == 1
    assert "author" not in seen[0].url.params
    assert "authorization" not in seen[0].headers


@pytest.mark.parametrize(
    ("repo_id", "expect_token"), [("example-org/m", True), ("elsewhere/m", False)]
)
async def test_card_sends_the_token_only_to_allowlisted_repos(
    monkeypatch: pytest.MonkeyPatch, repo_id: str, expect_token: bool
) -> None:
    monkeypatch.setenv("HF_TOKEN", "tok")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text="# card")

    text = await _service(handler, ["example-org"]).card(repo_id=repo_id)

    assert text == "# card"
    assert ("authorization" in seen[0].headers) is expect_token


async def test_card_not_found_is_hf_model_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "tok")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="no")

    with pytest.raises(HfModelNotFoundError):
        await _service(handler, ["example-org"]).card(repo_id="example-org/m")


async def test_card_outage_is_hub_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "tok")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    with pytest.raises(HfHubUnreachableError):
        await _service(handler, ["example-org"]).card(repo_id="example-org/m")
