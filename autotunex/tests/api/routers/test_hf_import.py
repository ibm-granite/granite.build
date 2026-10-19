# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace import endpoints, end to end over HTTP.

The Hub and the viewer are both driven by ``httpx.MockTransport``, through the
``get_hf_http_client`` and ``get_hf_viewer_http_client`` dependencies
respectively, so no test touches the network.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from uuid import UUID

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from autotunex.api.deps import (
    get_background_session_factory,
    get_dataset_runner,
    get_hf_http_client,
    get_hf_viewer_http_client,
)
from autotunex.core.config import get_settings
from autotunex.services import hf_import_service
from autotunex.services.dataset_runner import NoOpDatasetUploadRunner
from tests.conftest import API, make_settings

REVISION = "a" * 40

PROBLEM_JSON = "application/problem+json"

Handler = Callable[[httpx.Request], httpx.Response]


def _parquet_bytes(rows: int) -> bytes:
    """A minimal in-memory parquet with the columns a mapping would target.

    Mirrors ``tests/services/test_hf_import_staging.py``'s helper of the same
    name/shape.
    """
    table = pa.table(
        {
            "instruction": [f"q{n}" for n in range(rows)],
            "response": [f"a{n}" for n in range(rows)],
        }
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return bytes(sink.getvalue().to_pybytes())


@pytest.fixture
def hf_transport(app: FastAPI) -> Callable[[Handler], None]:
    """Install a MockTransport-backed client for both the Hub and viewer dependencies."""

    def _install(handler: Handler) -> None:
        def _client() -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        app.dependency_overrides[get_hf_http_client] = _client
        app.dependency_overrides[get_hf_viewer_http_client] = _client

    return _install


@pytest.fixture(autouse=True)
def _background_import_wiring(app: FastAPI, engine: AsyncEngine) -> NoOpDatasetUploadRunner:
    """Route the import background path to the test DB and a no-op runner.

    Every test in this file resolves ``get_hf_import_service``, which depends
    on both ``get_background_session_factory`` and ``get_dataset_runner``.
    Without this override, all ten tests would construct the real,
    ``@lru_cache``d ``_shared_upload_runner`` (violating its own
    process-wide-singleton docstring) and the background task would write
    through the real process-wide session factory instead of this test's own
    in-memory ``engine`` — mirrors ``test_datasets.py``'s ``local_storage``
    fixture, made autouse because every test here shares the exposure.
    """
    app.dependency_overrides[get_background_session_factory] = lambda: async_sessionmaker(
        bind=engine, expire_on_commit=False
    )
    runner = NoOpDatasetUploadRunner()
    app.dependency_overrides[get_dataset_runner] = lambda: runner
    return runner


def _gate(
    app: FastAPI, tmp_path: Path, *, available: bool, dataset_storage_dir: Path | None = None
) -> None:
    """Open or close the availability gate via the deployment shape.

    ``dataset_storage_dir`` defaults to a path under the caller's own ``tmp_path``,
    so every test here is isolated from the repo's real ``artifacts/datasets`` by
    default — opting OUT of isolation now takes an explicit
    ``dataset_storage_dir=`` override (nothing currently needs one), rather than
    opting in via a ``tmp_path=`` that is easy to forget to pass.
    """
    app.dependency_overrides[get_settings] = lambda: make_settings(
        gb_environment="standalone",
        lsf_cluster=None if available else "a-cluster",
        dataset_storage_dir=dataset_storage_dir
        if dataset_storage_dir is not None
        else tmp_path / "datasets",
    )


def _hub(sha: str = REVISION, branch: dict[str, dict[str, list[str]]] | None = None) -> Handler:
    payload = {"default": {"train": ["https://cdn/0.parquet"]}} if branch is None else branch
    parquet = _parquet_bytes(5)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/datasets/o/r/revision/refs/convert/parquet":
            return httpx.Response(200, json={"sha": sha})
        if request.url.path == f"/api/datasets/o/r/revision/{REVISION}":
            return httpx.Response(
                200,
                json={
                    "sha": sha,
                    "siblings": [
                        {"rfilename": f"{config}/{split}/{index:04}.parquet"}
                        for config, splits in payload.items()
                        for split, urls in splits.items()
                        for index, _ in enumerate(urls)
                    ],
                },
            )
        # The viewer (datasets-server.huggingface.co) is a second host behind the
        # same MockTransport handler, per the brief's Step 2 guidance — the preview
        # test's ``/rows`` call lands here rather than in either branch above.
        if request.url.host == "datasets-server.huggingface.co":
            return httpx.Response(
                200, json={"rows": [{"row": {"instruction": "q", "response": "a"}}]}
            )
        # The remaining traffic is the CDN URL from `payload` above, hit twice for
        # two different purposes: a HEAD from `hf_hub.head_total_bytes` (the plan's
        # size gate, which only reads `content-length`) and a GET from
        # `_stage_split` (the real staging download, which needs a real parquet
        # body to parse).
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(parquet))})
        return httpx.Response(200, content=parquet)

    return handler


async def test_splits_is_refused_when_the_gate_is_closed(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
) -> None:
    """A deployment that cannot use an imported dataset must refuse up front."""
    _gate(app, tmp_path, available=False)
    hf_transport(_hub())

    response = await client.get(f"{API}/datasets/hf/splits", params={"repo_id": "o/r"})

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_splits_returns_configs_and_a_pinned_revision(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
) -> None:
    _gate(app, tmp_path, available=True)
    hf_transport(_hub(sha=REVISION))

    response = await client.get(f"{API}/datasets/hf/splits", params={"repo_id": "o/r"})

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["revision"] == REVISION
    assert body["configs"] == {"default": ["train"]}


async def test_splits_refuses_a_repo_id_that_escapes_the_hub_path(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
) -> None:
    """``repo_id`` is interpolated into the Hub URL path, so it must be patterned.

    ``hf_hub`` builds ``{base_url}/api/datasets/{repo_id}`` and ``httpx`` removes
    dot segments per RFC 3986, so an unconstrained value aims the request at any
    path under ``hf_hub_base_url`` -- request forgery once that setting points at a
    mirror or an internal double, which it is documented as supporting. The two
    request *bodies* have always used ``HfRepoId`` for this; the query parameter
    validated length only, which ``../../`` satisfies.
    """
    _gate(app, tmp_path, available=True)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"sha": "deadbeef"})

    hf_transport(handler)

    response = await client.get(
        f"{API}/datasets/hf/splits", params={"repo_id": "../../internal-admin"}
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert seen == [], f"a rejected repo_id still reached the Hub: {seen}"


async def test_a_400_from_the_parquet_branch_says_not_converted_yet(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "not supported"})

    _gate(app, tmp_path, available=True)
    hf_transport(handler)

    response = await client.get(f"{API}/datasets/hf/splits", params={"repo_id": "o/r"})

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert "has not converted" in response.text


async def test_an_empty_parquet_payload_says_no_tabular_data(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
) -> None:
    """Real case: world-igr-plum/regions answers 200 with an empty object."""
    _gate(app, tmp_path, available=True)
    hf_transport(_hub(branch={}))

    response = await client.get(f"{API}/datasets/hf/splits", params={"repo_id": "o/r"})

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert "No tabular data" in response.text


async def test_preview_reports_raw_rows_mapped_rows_and_survivors(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
) -> None:
    """The mapped sample and its survival count are what stop a bad mapping."""
    _gate(app, tmp_path, available=True)
    hf_transport(_hub())

    response = await client.post(
        f"{API}/datasets/hf/preview",
        json={
            "repo_id": "o/r",
            "revision": REVISION,
            "config": "default",
            "train_split": "train",
            "column_mapping": {"input": "instruction", "output": "response"},
        },
    )

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["sampled"] >= 1
    assert set(body["mapped_rows"][0]) <= {"input", "output"}
    assert 0 <= body["survived"] <= body["sampled"]


async def test_preview_counts_survivors_for_a_partly_blank_mapping(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
) -> None:
    """A target still awaiting its source must not zero out the survival count.

    ``survival_count`` documents ``survived == 0`` as "the mapping is simply wrong
    and the caller blocks", and a blank source can never produce its target -- so
    requiring every target made the wizard block on its own intermediate state,
    while the import path accepts that same mapping
    (``test_stage_and_submit_accepts_a_partly_blank_mapping``). ``hf_import``'s
    module docstring promises the preview computes what the import computes.
    """
    _gate(app, tmp_path, available=True)
    hf_transport(_hub())

    response = await client.post(
        f"{API}/datasets/hf/preview",
        json={
            "repo_id": "o/r",
            "revision": REVISION,
            "config": "default",
            "train_split": "train",
            "column_mapping": {"input": "instruction", "output": ""},
        },
    )

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["sampled"] >= 1
    assert body["survived"] == body["sampled"]
    assert list(body["mapped_rows"][0]) == ["input"]


async def test_preview_reports_no_survivors_when_the_mapping_names_nothing(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
) -> None:
    """The all-blank mapping is the one the import *does* refuse, so it blocks here.

    Deriving the required targets from the non-blank sources leaves an empty
    required set for this input, and ``all()`` over nothing is true -- which would
    report every row as intact for a mapping ``_reject_unmapped_sources`` rejects.
    """
    _gate(app, tmp_path, available=True)
    hf_transport(_hub())

    response = await client.post(
        f"{API}/datasets/hf/preview",
        json={
            "repo_id": "o/r",
            "revision": REVISION,
            "config": "default",
            "train_split": "train",
            "column_mapping": {"input": "", "output": ""},
        },
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["survived"] == 0


async def test_preview_asks_the_viewer_for_the_unpartitioned_split_name(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
) -> None:
    """A split HF sharded into ``train-part0`` is still just ``train`` to the viewer.

    Concrete case: ``HuggingFaceFW/fineweb`` stores ``default/train-part0``,
    ``-part1`` and ``-part2`` on the converted branch, while the viewer's
    ``/splits`` reports one split named ``train``. Sending the picker's own name
    would fail for exactly the largest datasets.
    """
    asked: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/datasets/o/r/revision/refs/convert/parquet":
            return httpx.Response(200, json={"sha": REVISION})
        if request.url.path == f"/api/datasets/o/r/revision/{REVISION}":
            return httpx.Response(
                200,
                json={
                    "sha": REVISION,
                    "siblings": [{"rfilename": "default/train-part0/0000.parquet"}],
                },
            )
        if request.url.path == "/rows":
            asked.append(request.url.params.get("split"))
            return httpx.Response(200, json={"rows": [{"row": {"instruction": "q"}}]})
        return httpx.Response(200, headers={"content-length": "1024"})

    _gate(app, tmp_path, available=True)
    hf_transport(handler)

    response = await client.post(
        f"{API}/datasets/hf/preview",
        json={
            "repo_id": "o/r",
            "revision": REVISION,
            "config": "default",
            "train_split": "train-part0",
            "column_mapping": {"input": "instruction"},
        },
    )

    assert response.status_code == HTTPStatus.OK
    assert asked == ["train"]


async def test_preview_says_preview_is_unavailable_not_that_import_is(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
) -> None:
    """The viewer cannot serve every dataset, and that must not read as "no import".

    Concrete case: ``HuggingFaceFW/fineweb`` answers 501 ("Job manager crashed
    while running this job"). Importing it works regardless, so the message names
    preview -- reporting the deployment-level "import is not available" instead
    would tell a caller to abandon a dataset they can in fact import.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/datasets/o/r/revision/refs/convert/parquet":
            return httpx.Response(200, json={"sha": REVISION})
        if request.url.path == f"/api/datasets/o/r/revision/{REVISION}":
            return httpx.Response(
                200,
                json={"sha": REVISION, "siblings": [{"rfilename": "default/train/0000.parquet"}]},
            )
        if request.url.path == "/rows":
            return httpx.Response(501, json={"error": "Job manager crashed while running"})
        return httpx.Response(200, headers={"content-length": "1024"})

    _gate(app, tmp_path, available=True)
    hf_transport(handler)

    response = await client.post(
        f"{API}/datasets/hf/preview",
        json={
            "repo_id": "o/r",
            "revision": REVISION,
            "config": "default",
            "train_split": "train",
            "column_mapping": {"input": "instruction"},
        },
    )

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.headers["content-type"].startswith(PROBLEM_JSON)
    assert "Preview is unavailable for o/r" in response.json()["detail"]


async def test_preview_columns_are_the_union_of_ragged_row_keys(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
) -> None:
    """A column absent from row 0 must still appear in the picker.

    Concrete case this guards: ``vicgalle/alpaca-gpt4`` has a column empty (and,
    server-side, sometimes entirely absent from the row dict) in a majority of
    rows -- deriving ``columns`` from ``raw_rows[0]`` alone would silently drop
    it from the preview and make it unselectable. The viewer omits the key
    outright, which is why the rows below are ragged rather than null-filled: a
    parquet sample could not reproduce it, since a parquet schema is fixed and
    yields ``None`` for a missing value instead of dropping the key.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/datasets/o/r/revision/refs/convert/parquet":
            return httpx.Response(200, json={"sha": REVISION})
        if request.url.path == f"/api/datasets/o/r/revision/{REVISION}":
            return httpx.Response(
                200,
                json={"sha": REVISION, "siblings": [{"rfilename": "default/train/0000.parquet"}]},
            )
        if request.url.path == "/rows":
            return httpx.Response(
                200,
                json={
                    "rows": [
                        {"row": {"instruction": "q0", "response": "a0"}},
                        {"row": {"instruction": "q1", "note": "extra"}},
                    ]
                },
            )
        return httpx.Response(200, headers={"content-length": "1024"})

    _gate(app, tmp_path, available=True)
    hf_transport(handler)

    response = await client.post(
        f"{API}/datasets/hf/preview",
        json={
            "repo_id": "o/r",
            "revision": REVISION,
            "config": "default",
            "train_split": "train",
            "column_mapping": {"input": "instruction"},
        },
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["columns"] == ["instruction", "response", "note"]


# The two routes below are not covered by the brief's given test file; added to
# exercise the remaining endpoints (``search`` and ``import``) end to end.


async def test_search_is_refused_when_the_gate_is_closed(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
) -> None:
    _gate(app, tmp_path, available=False)
    hf_transport(_hub())

    response = await client.get(f"{API}/datasets/hf/search", params={"query": "alpaca"})

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_search_returns_matching_repo_ids(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/datasets"
        assert request.url.params["search"] == "alpaca"
        return httpx.Response(200, json=[{"id": "tatsu-lab/alpaca"}, {"id": "o/r"}])

    _gate(app, tmp_path, available=True)
    hf_transport(handler)

    response = await client.get(f"{API}/datasets/hf/search", params={"query": "alpaca"})

    assert response.status_code == HTTPStatus.OK
    assert response.json() == ["tatsu-lab/alpaca", "o/r"]


async def test_import_returns_202_and_importing_status(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
    _background_import_wiring: NoOpDatasetUploadRunner,
) -> None:
    _gate(app, tmp_path, available=True)
    hf_transport(_hub())

    response = await client.post(
        f"{API}/datasets/hf/import",
        json={
            "name": "alpaca-import",
            "repo_id": "o/r",
            "revision": REVISION,
            "config": "default",
            "train_split": "train",
            "column_mapping": {"input": "instruction", "output": "response"},
        },
    )

    assert response.status_code == HTTPStatus.ACCEPTED
    body = response.json()
    assert body["status"] == "importing"
    assert body["name"] == "alpaca-import"

    # The 202 only proves the request was accepted; the actual staging fetch and
    # runner hand-off happen in a background task. Poll for that background work
    # to reach the (stubbed) runner rather than asserting only the immediate
    # response — mirrors the plain-counter poll in
    # ``test_dataset_runner.test_process_bounds_concurrency_to_the_configured_limit``.
    # Bounded (mirrors ``local.test_runner._wait_until``): wiring that never
    # reaches submit() must fail the test, not hang the suite.
    dataset_id = UUID(body["id"])
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 2.0
    while not _background_import_wiring.submitted:
        if loop.time() >= deadline:
            raise AssertionError("background import never reached the runner before the timeout")
        await asyncio.sleep(0)
    assert _background_import_wiring.submitted == [dataset_id]

    # The completed import's provenance must actually be readable back, not just
    # written — dataset_to_read is the only DatasetRead construction site, and a
    # field missing there reads as null regardless of what the row stores.
    read_back = await client.get(f"{API}/datasets/{dataset_id}")
    assert read_back.json()["hf_repo_id"] == "o/r"


async def test_import_reads_the_picker_snapshot_while_preview_samples_the_viewer(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
    _background_import_wiring: NoOpDatasetUploadRunner,
) -> None:
    _gate(app, tmp_path, available=True)
    current = REVISION
    downloads: list[str] = []
    viewer_reads: list[str] = []
    branch_reads = 0
    payload = _parquet_bytes(3)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal branch_reads
        if request.url.path.endswith("/revision/refs/convert/parquet"):
            branch_reads += 1
            return httpx.Response(200, json={"sha": current})
        if request.url.path == f"/api/datasets/o/r/revision/{REVISION}":
            return httpx.Response(
                200,
                json={
                    "sha": REVISION,
                    "siblings": [{"rfilename": "default/train/0000.parquet"}],
                },
            )
        if request.url.path == "/rows":
            viewer_reads.append(str(request.url))
            return httpx.Response(
                200, json={"rows": [{"row": {"instruction": f"q{i}"}} for i in range(3)]}
            )
        assert request.url.path == f"/datasets/o/r/resolve/{REVISION}/default/train/0000.parquet"
        if request.method == "GET":
            downloads.append(request.url.path)
        return httpx.Response(200, content=payload)

    hf_transport(handler)
    picker = await client.get(f"{API}/datasets/hf/splits", params={"repo_id": "o/r"})
    assert picker.status_code == HTTPStatus.OK
    current = "b" * 40
    selection = {
        "repo_id": "o/r",
        "revision": picker.json()["revision"],
        "config": "default",
        "train_split": "train",
        "column_mapping": {"input": "instruction"},
    }

    preview = await client.post(f"{API}/datasets/hf/preview", json=selection)
    assert preview.status_code == HTTPStatus.OK
    # Preview reports the revision an import will read, while sampling the viewer,
    # which serves the latest conversion and takes no revision parameter -- so it
    # downloads no shard and stages nothing.
    assert preview.json()["revision"] == REVISION
    assert preview.json()["mapped_rows"] == [{"input": f"q{i}"} for i in range(3)]
    assert len(viewer_reads) == 1
    assert downloads == []
    assert not list((tmp_path / "datasets" / ".staging").glob("hf-preview-*"))
    current = "c" * 40
    imported = await client.post(
        f"{API}/datasets/hf/import",
        json={**selection, "name": "pinned-import"},
    )
    assert imported.status_code == HTTPStatus.ACCEPTED
    async with asyncio.timeout(2):
        await asyncio.gather(*hf_import_service._background_tasks)
    assert _background_import_wiring.submitted == [UUID(imported.json()["id"])]
    dataset = (await client.get(f"{API}/datasets/{imported.json()['id']}")).json()

    assert dataset["hf_revision"] == REVISION
    assert dataset["hf_provenance"]["train_retained_rows"] == 3
    assert dataset["hf_provenance"]["train_truncated"] is False
    # One GET, by the import alone, and against the revision the picker pinned --
    # not the "b"/"c" shas the branch moved to in between.
    assert downloads == [f"/datasets/o/r/resolve/{REVISION}/default/train/0000.parquet"]
    assert branch_reads == 1


async def test_import_is_refused_when_the_gate_is_closed(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
) -> None:
    _gate(app, tmp_path, available=False)
    hf_transport(_hub())

    response = await client.post(
        f"{API}/datasets/hf/import",
        json={
            "name": "alpaca-import",
            "repo_id": "o/r",
            "revision": REVISION,
            "config": "default",
            "train_split": "train",
            "column_mapping": {"input": "instruction", "output": "response"},
        },
    )

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_import_too_large_is_422(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
) -> None:
    """The remote dataset's size, not the request body's, gates this — 422 either way."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/datasets/o/r/revision/refs/convert/parquet":
            return httpx.Response(200, json={"sha": REVISION})
        if request.url.path == f"/api/datasets/o/r/revision/{REVISION}":
            return httpx.Response(
                200,
                json={"sha": REVISION, "siblings": [{"rfilename": "default/train/0000.parquet"}]},
            )
        return httpx.Response(200, headers={"content-length": str(10 * 1024**3)})

    _gate(app, tmp_path, available=True)
    hf_transport(handler)

    response = await client.post(
        f"{API}/datasets/hf/import",
        json={
            "name": "too-big",
            "repo_id": "o/r",
            "revision": REVISION,
            "config": "default",
            "train_split": "train",
            "column_mapping": {"input": "instruction", "output": "response"},
        },
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_import_rejects_a_path_traversal_name(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
) -> None:
    """``name`` becomes a filesystem path segment in staging.

    Traversal must 422 before any fetch, matching ``DatasetCreate``'s own ``name``
    validation.
    """
    _gate(app, tmp_path, available=True)
    hf_transport(_hub())

    response = await client.post(
        f"{API}/datasets/hf/import",
        json={
            "name": "../../../../tmp/pwn",
            "repo_id": "o/r",
            "revision": REVISION,
            "config": "default",
            "train_split": "train",
            "column_mapping": {"input": "instruction", "output": "response"},
        },
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_splits_says_the_hub_is_unreachable_not_that_import_is_off(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
) -> None:
    """A Hub-side outage must not read as "this deployment cannot import".

    ``HfImportUnavailableError``'s message is what ``_require_hf_import`` returns
    when the *operator* switched import off. Reusing it for a 502 from
    huggingface.co sends a caller in a correctly-configured deployment to abandon
    the dataset, and the operator to debug ``AUTOTUNEX_HF_IMPORT_*`` for a problem
    that does not exist. Same distinction the preview path already draws above.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(HTTPStatus.BAD_GATEWAY, text="bad gateway")

    _gate(app, tmp_path, available=True)
    hf_transport(handler)

    response = await client.get(f"{API}/datasets/hf/splits", params={"repo_id": "o/r"})

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.headers["content-type"].startswith(PROBLEM_JSON)
    detail = response.json()["detail"]
    assert "not available in this deployment" not in detail
    assert "could not be reached" in detail


async def test_search_is_served_on_a_cluster_that_pushes_to_huggingface(
    app: FastAPI,
    client: AsyncClient,
    hf_transport: Callable[[Handler], None],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cluster case: storage pushes to HF, so the route is not refused."""
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    monkeypatch.setenv("GB_TOKEN", "gb_xxx")
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")
    app.dependency_overrides[get_settings] = lambda: make_settings(
        gb_environment="prod", dataset_storage_dir=tmp_path / "datasets"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/datasets"
        return httpx.Response(200, json=[{"id": "o/r"}])

    hf_transport(handler)

    response = await client.get(f"{API}/datasets/hf/search", params={"query": "alpaca"})

    assert response.status_code == HTTPStatus.OK
    assert response.json() == ["o/r"]
