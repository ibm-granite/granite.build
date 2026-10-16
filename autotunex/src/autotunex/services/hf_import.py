# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Turn a HuggingFace repo selection into a concrete, size-gated import plan.

Sample-first by design: the caller previews and maps against ~100 rows before
importing. The mapping shown to the user must be computed from the same snapshot
and the same way the import will compute it — see
:func:`apply_mapping`, which mirrors ``datasets_io.remap_records`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from autotunex.services.storage import hf_hub


class HfImportTooLarge(Exception):  # noqa: N818
    """The gated parquet shards exceed the configured byte ceiling.

    Raised on the first shard of each selected split, not on the selection's total
    -- see :func:`build_plan` for why.
    """

    def __init__(self, total_bytes: int, max_bytes: int) -> None:
        super().__init__(f"{total_bytes} bytes exceeds the {max_bytes}-byte import limit")
        self.total_bytes = total_bytes
        self.max_bytes = max_bytes


@dataclass(frozen=True)
class HfImportPlan:
    """Everything the runner needs to fetch and materialize one dataset."""

    repo_id: str
    revision: str
    config: str
    train_split: str
    validation_split: str | None
    train_urls: list[str]
    validation_urls: list[str]
    gated_bytes: int
    """What the size gate actually measured: the first shard of each selected split.

    Not the selection's total size -- see :func:`build_plan` for why summing every
    shard was the wrong question. Informational only; nothing reads it but the
    refusal message.
    """


def apply_mapping(rows: list[dict[str, Any]], mapping: dict[str, str]) -> list[dict[str, Any]]:
    """Project ``rows`` through ``mapping`` (target name -> source column).

    Deliberately identical to ``datasets_io.remap_records``' projection, including
    its tolerance: a target whose source is blank or absent from the row is
    skipped rather than raising. A preview that disagreed with the writer would be
    worse than no preview.
    """
    return [
        {target: row[source] for target, source in mapping.items() if source and source in row}
        for row in rows
    ]


def survival_count(rows: list[dict[str, Any]], mapping: dict[str, str], required: list[str]) -> int:
    """How many of ``rows`` yield a non-empty value for every required target.

    This is the number shown to the user. Zero means the mapping is simply wrong
    and the caller blocks; anything else is a warning, because real Hub datasets
    are legitimately ragged and an 85%-clean dataset may be exactly what someone
    wants.
    """
    mapped = apply_mapping(rows, mapping)
    return sum(1 for row in mapped if all(row.get(target) not in (None, "") for target in required))


async def build_plan(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    repo_id: str,
    config: str,
    train_split: str,
    validation_split: str | None,
    token: str | None,
    max_bytes: int,
    revision: str | None = None,
) -> HfImportPlan:
    """Pick immutable split URLs and size-gate the first shards.

    A supplied revision is never re-resolved; without one, pin the current
    parquet conversion commit. API preview/import requests require the picker
    revision, so a later branch update cannot change an approved selection.

    Raises:
        ValueError: ``config`` or either split is absent from the parquet branch.
        HfImportTooLarge: the first shard of a selected split already exceeds
            ``max_bytes`` (see the gate's rationale in the body).
        hf_hub.HfDatasetNotConverted / HfNoTabularData / HfHubUnavailable: as raised
            by :func:`hf_hub.resolve_revision`, :func:`hf_hub.list_parquet_files`,
            or :func:`hf_hub.head_total_bytes`.
    """
    if revision is None:
        revision = await hf_hub.resolve_revision(
            client, base_url=base_url, repo_id=repo_id, token=token
        )
    branch = await hf_hub.list_parquet_files(
        client, base_url=base_url, repo_id=repo_id, revision=revision, token=token
    )
    if config not in branch:
        raise ValueError(f"config {config!r} not in {sorted(branch)}")
    splits = branch[config]
    if train_split not in splits:
        raise ValueError(f"split {train_split!r} not in {sorted(splits)}")
    if validation_split is not None and validation_split not in splits:
        raise ValueError(f"split {validation_split!r} not in {sorted(splits)}")

    train_urls = splits[train_split]
    validation_urls = splits[validation_split] if validation_split else []
    # Gate on the first shard of each selected split rather than summing every
    # shard. `hf_import_service._stage_split` stops downloading the moment
    # `hf_import_max_rows` is reached -- often after shard 0 -- so a split's full
    # size is not what the import fetches, and refusing on it contradicted
    # `hf_import_max_rows`'s own contract ("Not a refusal: a larger dataset imports
    # its first N rows"). It also cost one HEAD per shard on the app-wide pool, which
    # a shard-count cap could only bound by under-counting, making the gate
    # non-monotonic: a 20 GiB/600-shard repo passed while a 6 GiB/400-shard one was
    # refused.
    #
    # What survives is the case truncation cannot rescue: `_stage_split` writes a
    # whole shard to disk before any row cap applies, so a single shard already over
    # the budget is impossible regardless of `max_rows`. Both splits share one
    # budget, so their first shards are summed. Anything larger that slips through is
    # still refused by that function's inline byte cap, which is the enforced
    # boundary -- this gate is only the cheap, instant refusal in front of it.
    gated_urls = [urls[0] for urls in (train_urls, validation_urls) if urls]
    gated_bytes = await hf_hub.head_total_bytes(client, urls=gated_urls, token=token)
    if gated_bytes > max_bytes:
        raise HfImportTooLarge(gated_bytes, max_bytes)

    return HfImportPlan(
        repo_id=repo_id,
        revision=revision,
        config=config,
        train_split=train_split,
        validation_split=validation_split,
        train_urls=train_urls,
        validation_urls=validation_urls,
        gated_bytes=gated_bytes,
    )
