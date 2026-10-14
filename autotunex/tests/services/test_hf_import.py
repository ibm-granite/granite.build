# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""hf_import: plan construction, the size gate, and mapped-preview arithmetic."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from autotunex.services import hf_import

BASE = "https://huggingface.co"


async def test_build_plan_uses_the_selected_snapshot_after_the_branch_moves() -> None:
    revision = "a" * 40
    pinned = f"{BASE}/datasets/o/r/resolve/{revision}/default/train/0000.parquet"
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == f"/api/datasets/o/r/revision/{revision}":
            return httpx.Response(
                200,
                json={
                    "sha": revision,
                    "siblings": [{"rfilename": "default/train/0000.parquet"}],
                },
            )
        if request.method == "HEAD" and str(request.url) == pinned:
            return httpx.Response(200, headers={"content-length": "10"})
        pytest.fail(f"Request escaped the selected snapshot: {request.url}")

    async with _client(handler) as client:
        plan = await hf_import.build_plan(
            client,
            base_url=BASE,
            repo_id="o/r",
            revision=revision,
            config="default",
            train_split="train",
            validation_split=None,
            token=None,
            max_bytes=1_000,
        )

    assert plan.revision == revision
    assert plan.train_urls == [pinned]
    assert len(seen) == 2


def _client(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def test_apply_mapping_projects_and_renames() -> None:
    rows = [{"instruction": "hi", "response": "yo", "extra": 1}]
    mapped = hf_import.apply_mapping(rows, {"input": "instruction", "output": "response"})
    assert mapped == [{"input": "hi", "output": "yo"}]


def test_apply_mapping_drops_absent_and_blank_sources_like_remap_records() -> None:
    rows = [{"instruction": "hi"}]
    mapped = hf_import.apply_mapping(rows, {"input": "instruction", "output": "nope"})
    assert mapped == [{"input": "hi"}]


def test_survival_count_requires_every_required_column() -> None:
    rows = [
        {"instruction": "a", "response": "b"},
        {"instruction": "c"},
        {"instruction": "d", "response": ""},
    ]
    mapping = {"input": "instruction", "output": "response"}
    assert hf_import.survival_count(rows, mapping, ["input", "output"]) == 1


def test_survival_count_is_zero_when_the_source_column_is_absent() -> None:
    rows = [{"a": 1}, {"a": 2}]
    assert hf_import.survival_count(rows, {"input": "missing"}, ["input"]) == 0


def test_survival_count_preserves_falsy_values() -> None:
    """Rows with legitimate falsy values (0, False) must survive, not be miscounted as casualties.

    A dataset whose output column legitimately contains 0 or False must not be
    reported to the user as having zero survivors.
    """
    rows: list[dict[str, Any]] = [
        {"instruction": "x", "response": 0},
        {"instruction": "y", "response": False},
        {"instruction": "z"},
    ]
    mapping = {"input": "instruction", "output": "response"}
    assert hf_import.survival_count(rows, mapping, ["input", "output"]) == 2


REVISION = "a" * 40


def _hub(request: httpx.Request, paths: list[str], size: int = 10) -> httpx.Response:
    if request.url.path == "/api/datasets/o/r/revision/refs/convert/parquet":
        return httpx.Response(200, json={"sha": REVISION})
    if request.url.path == f"/api/datasets/o/r/revision/{REVISION}":
        return httpx.Response(
            200,
            json={
                "sha": REVISION,
                "siblings": [{"rfilename": path} for path in paths],
            },
        )
    assert request.method == "HEAD"
    assert request.url.path.startswith(f"/datasets/o/r/resolve/{REVISION}/")
    return httpx.Response(200, headers={"content-length": str(size)})


async def test_build_plan_pins_revision_and_selects_urls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _hub(request, ["default/train/t.parquet", "default/validation/v.parquet"])

    async with _client(handler) as client:
        plan = await hf_import.build_plan(
            client,
            base_url=BASE,
            repo_id="o/r",
            config="default",
            train_split="train",
            validation_split="validation",
            token=None,
            max_bytes=1_000,
        )

    assert plan.revision == REVISION
    assert plan.train_urls == [f"{BASE}/datasets/o/r/resolve/{REVISION}/default/train/t.parquet"]
    assert plan.validation_split == "validation"
    assert plan.gated_bytes == 20


async def test_build_plan_gates_on_the_first_shard_not_the_split_total() -> None:
    heads = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal heads
        heads += request.method == "HEAD"
        return _hub(request, [f"default/train/{n:04}.parquet" for n in range(400)], size=100)

    async with _client(handler) as client:
        plan = await hf_import.build_plan(
            client,
            base_url=BASE,
            repo_id="o/r",
            config="default",
            train_split="train",
            validation_split=None,
            token=None,
            max_bytes=1_000,
        )

    assert plan.gated_bytes == 100
    assert heads == 1
    assert len(plan.train_urls) == 400


async def test_build_plan_gates_the_first_shard_of_each_selected_split() -> None:
    sized: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            sized.append(request.url.path)
        return _hub(
            request,
            [
                "default/train/t0.parquet",
                "default/train/t1.parquet",
                "default/validation/v0.parquet",
                "default/validation/v1.parquet",
            ],
        )

    async with _client(handler) as client:
        plan = await hf_import.build_plan(
            client,
            base_url=BASE,
            repo_id="o/r",
            config="default",
            train_split="train",
            validation_split="validation",
            token=None,
            max_bytes=1_000,
        )

    assert sorted(sized) == [
        f"/datasets/o/r/resolve/{REVISION}/default/train/t0.parquet",
        f"/datasets/o/r/resolve/{REVISION}/default/validation/v0.parquet",
    ]
    assert plan.gated_bytes == 20


async def test_build_plan_refuses_above_the_byte_ceiling() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _hub(request, ["default/train/t.parquet"], size=5_000)

    async with _client(handler) as client:
        with pytest.raises(hf_import.HfImportTooLarge):
            await hf_import.build_plan(
                client,
                base_url=BASE,
                repo_id="o/r",
                config="default",
                train_split="train",
                validation_split=None,
                token=None,
                max_bytes=1_000,
            )


async def test_build_plan_rejects_an_unknown_split() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _hub(request, ["default/train/t.parquet"])

    async with _client(handler) as client:
        with pytest.raises(ValueError, match="split"):
            await hf_import.build_plan(
                client,
                base_url=BASE,
                repo_id="o/r",
                config="default",
                train_split="nonexistent",
                validation_split=None,
                token=None,
                max_bytes=1_000,
            )
