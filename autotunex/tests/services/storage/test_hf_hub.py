# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""hf_hub: Hub REST reads driven by httpx.MockTransport — no network touched."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from autotunex.services.storage import hf_hub

BASE = "https://huggingface.co"
REVISION = "a" * 40


def _client(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("repo_id", "namespaces", "expected"),
    [
        ("ibm-granite/x", ["ibm-granite"], "tok"),
        ("tatsu-lab/alpaca", ["ibm-granite"], None),
        ("ibm-granite/x", [], None),
        ("no-slash", ["ibm-granite"], None),
        ("ibm-granite/../elsewhere/x", ["ibm-granite"], None),
        ("ibm-granite/x/extra", ["ibm-granite"], None),
        ("IBM-Granite/x", ["ibm-granite"], None),
    ],
)
def test_token_is_sent_only_for_allowlisted_namespaces(
    repo_id: str, namespaces: list[str], expected: str | None
) -> None:
    assert hf_hub.token_for(repo_id, token="tok", namespaces=namespaces) == expected


async def test_list_parquet_files_returns_config_split_urls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/api/datasets/tatsu-lab/alpaca/revision/{REVISION}"
        assert "authorization" not in request.headers
        return httpx.Response(
            200, json={"sha": REVISION, "siblings": [{"rfilename": "default/train/0000.parquet"}]}
        )

    async with _client(handler) as client:
        result = await hf_hub.list_parquet_files(
            client, base_url=BASE, repo_id="tatsu-lab/alpaca", revision=REVISION, token=None
        )

    assert result == {
        "default": {
            "train": [
                f"{BASE}/datasets/tatsu-lab/alpaca/resolve/{REVISION}/default/train/0000.parquet"
            ]
        }
    }


async def test_list_parquet_files_sends_the_bearer_token_when_given() -> None:
    """The counterpart to the ``token=None`` assertion above.

    Every other test in this file passes ``token=None``, so nothing previously
    asserted the header IS emitted when a token is supplied.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer tok"
        return httpx.Response(
            200, json={"sha": REVISION, "siblings": [{"rfilename": "default/train/0000.parquet"}]}
        )

    async with _client(handler) as client:
        await hf_hub.list_parquet_files(
            client, base_url=BASE, repo_id="tatsu-lab/alpaca", revision=REVISION, token="tok"
        )


async def test_resolve_revision_400_means_not_converted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "not supported"})

    async with _client(handler) as client:
        with pytest.raises(hf_hub.HfDatasetNotConverted):
            await hf_hub.resolve_revision(client, base_url=BASE, repo_id="o/r", token=None)


async def test_list_parquet_files_empty_payload_means_no_tabular_data() -> None:
    """Real case: world-igr-plum/regions and yuiseki/osm-wiki both return 200 {}."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"sha": REVISION, "siblings": []})

    async with _client(handler) as client:
        with pytest.raises(hf_hub.HfNoTabularData):
            await hf_hub.list_parquet_files(
                client, base_url=BASE, repo_id="o/r", revision=REVISION, token=None
            )


async def test_list_parquet_files_transport_error_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    async with _client(handler) as client:
        with pytest.raises(hf_hub.HfHubUnavailable):
            await hf_hub.list_parquet_files(
                client, base_url=BASE, repo_id="o/r", revision=REVISION, token=None
            )


async def test_resolve_revision_returns_the_sha() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/datasets/o/r/revision/refs/convert/parquet"
        return httpx.Response(200, json={"sha": REVISION})

    async with _client(handler) as client:
        assert (
            await hf_hub.resolve_revision(client, base_url=BASE, repo_id="o/r", token=None)
            == REVISION
        )


@pytest.mark.parametrize("status", [400, 404])
async def test_missing_conversion_does_not_fall_back_to_main(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/revision/refs/convert/parquet")
        return httpx.Response(status)

    async with _client(handler) as client:
        with pytest.raises(hf_hub.HfDatasetNotConverted):
            await hf_hub.resolve_revision(client, base_url=BASE, repo_id="o/r", token=None)


async def test_metadata_for_another_revision_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "sha": "b" * 40,
                "siblings": [{"rfilename": "default/train/0000.parquet"}],
            },
        )

    async with _client(handler) as client:
        with pytest.raises(hf_hub.HfHubUnavailable, match="selected revision"):
            await hf_hub.list_parquet_files(
                client,
                base_url=BASE,
                repo_id="o/r",
                revision=REVISION,
                token=None,
            )


async def test_search_datasets_returns_ids() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/datasets"
        assert request.url.params["search"] == "alpaca"
        assert request.url.params["limit"] == "5"
        return httpx.Response(200, json=[{"id": "tatsu-lab/alpaca"}, {"id": "x/y"}, {}])

    async with _client(handler) as client:
        ids = await hf_hub.search_datasets(
            client, base_url=BASE, query="alpaca", limit=5, token=None
        )

    assert ids == ["tatsu-lab/alpaca", "x/y"]


async def test_head_total_bytes_sums_content_length() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "HEAD"
        return httpx.Response(200, headers={"content-length": "100"})

    async with _client(handler) as client:
        total = await hf_hub.head_total_bytes(
            client, urls=["https://cdn/0.parquet", "https://cdn/1.parquet"], token=None
        )

    assert total == 200


async def test_head_total_bytes_lets_every_sibling_settle_before_it_raises() -> None:
    """A bare gather abandons the siblings of the HEAD that failed.

    ``asyncio.gather`` propagates the first exception immediately and explicitly
    does *not* cancel the rest, so they keep running against a pool the caller has
    already stopped waiting on and their own failures are never retrieved. Waiting
    for all of them keeps the raised type the same, which is what ``build_plan``
    matches on.
    """
    settled = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal settled
        if request.url.path == "/bad.parquet":
            settled += 1
            return httpx.Response(500)
        # Long enough that the fast failure above lands first.
        await asyncio.sleep(0.05)
        settled += 1
        return httpx.Response(200, headers={"content-length": "1"})

    urls = ["https://cdn/bad.parquet"] + [f"https://cdn/{n}.parquet" for n in range(7)]
    async with _client(handler) as client:
        with pytest.raises(hf_hub.HfHubUnavailable):
            await hf_hub.head_total_bytes(client, urls=urls, token=None)
        at_raise = settled

    assert at_raise == len(urls), (
        f"{len(urls) - at_raise} HEAD(s) were still in flight when the gate gave up"
    )


async def test_head_total_bytes_bounds_its_concurrency() -> None:
    """The size gate must not fan out one HEAD per shard all at once.

    ``urls`` is every parquet shard of the selected split(s) -- thousands for a
    many-shard repo -- and this runs on the app-wide client whose pool (100
    connections by default) every other outbound call shares. An unbounded gather
    exhausts it, so most of these HEADs time out queued and everything else in the
    process starves alongside them, reachable from ``POST /hf/preview``.
    """
    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        # Long enough that every HEAD the driver is willing to start overlaps here.
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, headers={"content-length": "1"})

    urls = [f"https://cdn/{n}.parquet" for n in range(50)]
    async with _client(handler) as client:
        total = await hf_hub.head_total_bytes(client, urls=urls, token=None)

    assert total == 50
    assert peak <= hf_hub._HEAD_CONCURRENCY
    # Non-vacuous even if the cap is later raised: the point is that it does not
    # fan out over every shard at once.
    assert peak < len(urls)


async def test_resolve_revision_follows_the_hubs_canonical_id_redirect() -> None:
    """A bare canonical id redirects to its namespaced form, and that must resolve.

    Verified against the live Hub on 2026-09-23:
    ``/api/datasets/imdb/revision/refs%2Fconvert%2Fparquet`` answers **307** to
    ``/api/datasets/stanfordnlp/imdb/...``. Unfollowed, that became
    ``HfHubUnavailable``, which :meth:`HfImportService.splits` reports as "import
    is not available in this deployment" -- for a dataset that imports fine. The
    picker never produces a bare id (search returns ``stanfordnlp/imdb``), so this
    is the hand-typed and direct-API path.
    """
    canonical = f"{BASE}/api/datasets/stanfordnlp/imdb/revision/refs%2Fconvert%2Fparquet"

    def handler(request: httpx.Request) -> httpx.Response:
        if "/datasets/imdb/" in str(request.url):
            return httpx.Response(307, headers={"Location": canonical})
        return httpx.Response(200, json={"sha": REVISION})

    async with _client(handler) as client:
        sha = await hf_hub.resolve_revision(client, base_url=BASE, repo_id="imdb", token=None)

    assert sha == REVISION


async def test_a_tokened_request_does_not_follow_a_redirect() -> None:
    """Following a redirect must never widen where the server's token is sent.

    httpx strips ``Authorization`` only *cross-origin*, and a Hub rename redirects
    huggingface.co -> huggingface.co, so a tokened request that followed its
    redirect could hand the token to a namespace outside
    ``hf_import_namespaces``. The allowlist is the whole security control for that
    token, so a tokened request stays unfollowed and fails loudly instead.
    """
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(307, headers={"Location": f"{BASE}/api/datasets/elsewhere/x"})

    async with _client(handler) as client:
        with pytest.raises(hf_hub.HfHubUnavailable):
            await hf_hub.resolve_revision(client, base_url=BASE, repo_id="allowed/x", token="tok")

    assert len(seen) == 1


async def test_search_models_returns_ids_and_skips_malformed_entries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/models"
        assert request.url.params["search"] == "granite"
        assert request.url.params["limit"] == "5"
        assert "author" not in request.url.params
        assert "authorization" not in request.headers
        return httpx.Response(200, json=[{"id": "ibm-granite/a"}, {}, {"id": 3}, {"id": "o/b"}])

    async with _client(handler) as client:
        ids = await hf_hub.search_models(
            client, base_url=BASE, query="granite", limit=5, token=None
        )

    assert ids == ["ibm-granite/a", "o/b"]


async def test_search_models_pins_author_and_sends_the_token_when_given() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["author"] == "example-org"
        assert request.headers["authorization"] == "Bearer tok"
        return httpx.Response(200, json=[{"id": "example-org/m"}])

    async with _client(handler) as client:
        ids = await hf_hub.search_models(
            client, base_url=BASE, query="m", limit=5, token="tok", author="example-org"
        )

    assert ids == ["example-org/m"]


async def test_search_models_private_only_keeps_only_private_hits() -> None:
    """The Hub's model-list entries carry a boolean ``private``; keep only ``True``."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"id": "o/pub", "private": False},
                {"id": "o/unmarked"},
                {"id": "o/priv", "private": True},
            ],
        )

    async with _client(handler) as client:
        ids = await hf_hub.search_models(
            client, base_url=BASE, query="m", limit=5, token=None, private_only=True
        )

    assert ids == ["o/priv"]


async def test_search_models_default_keeps_all_regardless_of_private() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"id": "o/pub", "private": False},
                {"id": "o/unmarked"},
                {"id": "o/priv", "private": True},
            ],
        )

    async with _client(handler) as client:
        ids = await hf_hub.search_models(client, base_url=BASE, query="m", limit=5, token=None)

    assert ids == ["o/pub", "o/unmarked", "o/priv"]


async def test_search_models_non_list_payload_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "nope"})

    async with _client(handler) as client:
        with pytest.raises(hf_hub.HfHubUnavailable):
            await hf_hub.search_models(client, base_url=BASE, query="m", limit=5, token=None)


async def test_fetch_model_card_returns_the_readme_text_with_the_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/example-org/m/raw/main/README.md"
        assert request.headers["authorization"] == "Bearer tok"
        return httpx.Response(200, text="# Card\n")

    async with _client(handler) as client:
        text = await hf_hub.fetch_model_card(
            client, base_url=BASE, repo_id="example-org/m", token="tok"
        )

    assert text == "# Card\n"


@pytest.mark.parametrize("status", [401, 404])
async def test_fetch_model_card_401_and_404_mean_not_found(status: int) -> None:
    """The Hub answers 401 for a private repo the caller may not read — same as absent."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="no")

    async with _client(handler) as client:
        with pytest.raises(hf_hub.HfModelNotFound):
            await hf_hub.fetch_model_card(client, base_url=BASE, repo_id="o/m", token=None)


async def test_fetch_model_card_other_errors_are_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="bad gateway")

    async with _client(handler) as client:
        with pytest.raises(hf_hub.HfHubUnavailable):
            await hf_hub.fetch_model_card(client, base_url=BASE, repo_id="o/m", token=None)


async def test_fetch_model_card_does_not_follow_a_tokened_redirect() -> None:
    """Unfollowed, a rename or case redirect is not-found rather than an outage."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(307, headers={"Location": f"{BASE}/elsewhere/x/raw/main/README.md"})

    async with _client(handler) as client:
        with pytest.raises(hf_hub.HfModelNotFound):
            await hf_hub.fetch_model_card(
                client, base_url=BASE, repo_id="example-org/m", token="tok"
            )

    assert len(seen) == 1


async def test_fetch_model_card_follows_an_anonymous_redirect() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/old/"):
            return httpx.Response(307, headers={"Location": f"{BASE}/new/m/raw/main/README.md"})
        return httpx.Response(200, text="moved")

    async with _client(handler) as client:
        text = await hf_hub.fetch_model_card(client, base_url=BASE, repo_id="old/m", token=None)

    assert text == "moved"
