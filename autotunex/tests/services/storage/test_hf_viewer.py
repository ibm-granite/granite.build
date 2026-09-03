"""hf_viewer: owner/repo derivation and the datasets-server /rows fetch.

The viewer HTTP is driven by ``httpx.MockTransport`` so no network is touched;
each handler asserts the request the client built and returns a canned response.
"""

from __future__ import annotations

import httpx
import pytest

from autotunex.services.storage import hf_viewer

BASE = "https://datasets-server.huggingface.co"


@pytest.mark.parametrize(
    ("artifact_url", "expected"),
    [
        ("hf://owner/repo", "owner/repo"),
        ("hf://huggingface.co/owner/repo", "owner/repo"),
        ("hf://huggingface.co/datasets/owner/repo", "owner/repo"),
        ("  hf://owner/repo  ", "owner/repo"),
        ("lh://owner/repo", None),
        ("https://huggingface.co/owner/repo", None),
        ("hf://only-one", None),
        (None, None),
        ("", None),
    ],
)
def test_repo_id_from_artifact_url(artifact_url: str | None, expected: str | None) -> None:
    assert hf_viewer.repo_id_from_artifact_url(artifact_url) == expected


def _client(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


async def test_fetch_rows_returns_normalized_records() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rows"
        params = request.url.params
        assert params["dataset"] == "owner/repo"
        assert params["config"] == "default"
        assert params["split"] == "train"
        assert params["offset"] == "0"
        assert params["length"] == "10"
        assert request.headers["authorization"] == "Bearer tok"
        return httpx.Response(200, json={"rows": [{"row": {"a": 1}}, {"row": {"a": 2}}]})

    async with _client(handler) as client:
        records = await hf_viewer.fetch_rows(
            client, base_url=BASE, repo_id="owner/repo", split="train", limit=10, token="tok"
        )

    assert records == [{"a": 1}, {"a": 2}]


async def test_fetch_rows_clamps_length_to_viewer_cap() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["length"] == "100"
        return httpx.Response(200, json={"rows": [{"row": {"a": 1}}]})

    async with _client(handler) as client:
        await hf_viewer.fetch_rows(
            client, base_url=BASE, repo_id="o/r", split="train", limit=500, token="t"
        )


async def test_fetch_rows_strips_a_trailing_slash_from_base_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rows"
        return httpx.Response(200, json={"rows": [{"row": {"a": 1}}]})

    async with _client(handler) as client:
        await hf_viewer.fetch_rows(
            client,
            base_url="https://datasets-server.huggingface.co/",
            repo_id="o/r",
            split="train",
            limit=10,
            token="t",
        )


async def test_fetch_rows_omits_auth_header_without_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"rows": [{"row": {"a": 1}}]})

    async with _client(handler) as client:
        await hf_viewer.fetch_rows(
            client, base_url=BASE, repo_id="o/r", split="train", limit=10, token=None
        )


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(404, json={"error": "not found"}),
        httpx.Response(500, text="boom"),
        httpx.Response(200, json={"rows": []}),
        httpx.Response(200, json={"nope": 1}),
        httpx.Response(200, json={"rows": [{"nope": 1}]}),
        httpx.Response(200, text="not json"),
    ],
)
async def test_fetch_rows_raises_unavailable(response: httpx.Response) -> None:
    async with _client(lambda request: response) as client:
        with pytest.raises(hf_viewer.HFViewerUnavailable):
            await hf_viewer.fetch_rows(
                client, base_url=BASE, repo_id="o/r", split="train", limit=10, token="t"
            )


async def test_fetch_rows_wraps_transport_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    async with _client(handler) as client:
        with pytest.raises(hf_viewer.HFViewerUnavailable):
            await hf_viewer.fetch_rows(
                client, base_url=BASE, repo_id="o/r", split="train", limit=10, token="t"
            )


async def test_discover_config_returns_config_matching_preferred_split() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/splits"
        assert request.url.params["dataset"] == "owner/repo"
        assert "split" not in request.url.params  # /splits takes only the dataset
        assert request.headers["authorization"] == "Bearer tok"
        return httpx.Response(
            200,
            json={
                "splits": [
                    {"dataset": "owner/repo", "config": "alt", "split": "test"},
                    {"dataset": "owner/repo", "config": "main", "split": "train"},
                ],
                "pending": [],
                "failed": [],
            },
        )

    async with _client(handler) as client:
        config = await hf_viewer.discover_config(
            client, base_url=BASE, repo_id="owner/repo", token="tok"
        )

    assert config == "main"


async def test_discover_config_falls_back_to_first_entry_without_preferred_split() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"splits": [{"config": "only", "split": "test"}]})

    async with _client(handler) as client:
        config = await hf_viewer.discover_config(client, base_url=BASE, repo_id="o/r", token="t")

    assert config == "only"


async def test_discover_config_omits_auth_header_without_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"splits": [{"config": "default", "split": "train"}]})

    async with _client(handler) as client:
        await hf_viewer.discover_config(client, base_url=BASE, repo_id="o/r", token=None)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(404, json={"error": "not found"}),
        httpx.Response(200, json={"splits": []}),
        httpx.Response(200, json={"nope": 1}),
        httpx.Response(200, json={"splits": [{"split": "train"}]}),  # entry has no config
        httpx.Response(200, text="not json"),
    ],
)
async def test_discover_config_raises_unavailable(response: httpx.Response) -> None:
    async with _client(lambda request: response) as client:
        with pytest.raises(hf_viewer.HFViewerUnavailable):
            await hf_viewer.discover_config(client, base_url=BASE, repo_id="o/r", token="t")


async def test_discover_config_wraps_transport_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    async with _client(handler) as client:
        with pytest.raises(hf_viewer.HFViewerUnavailable):
            await hf_viewer.discover_config(client, base_url=BASE, repo_id="o/r", token="t")
