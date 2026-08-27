# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace dataset-viewer client — bounded, dependency-free row reads.

Ported from the 2025 app's ``api/services/storage/hf_viewer.py``. This module
knows nothing about FastAPI, the database, or ``Settings``: it takes primitives
and returns plain row dicts, so it is trivially unit-testable with
``httpx.MockTransport``.

The viewer is the public ``datasets-server.huggingface.co`` service. It lets us
preview a dataset that lives only in HuggingFace (an ``llmb artifact push``
retains no local copy), which is exactly the "preview even though the file is not
in local storage" case.
"""

from __future__ import annotations

from typing import Any

import httpx

from autotunex.core.logging import get_logger

logger = get_logger(__name__)

_HF_SCHEME = "hf://"
_HF_HOST = "huggingface.co"
_VIEWER_ROW_CAP = 100  # the viewer rejects length > 100 with a 422


class HFViewerUnavailable(Exception):  # noqa: N818 (name is a fixed cross-task interface)
    """The dataset viewer cannot serve rows right now.

    Raised for a transport error, a non-200 status, a malformed body, or a 200
    that carries no usable rows (e.g. the viewer is still precomputing the
    dataset). Never escapes the storage layer — the backend degrades it to an
    empty preview.
    """


def repo_id_from_artifact_url(artifact_url: str | None) -> str | None:
    """Derive an ``owner/repo`` HuggingFace dataset id from a stored locator.

    Only ``hf://`` locators are accepted. A leading ``huggingface.co`` host and an
    intervening ``datasets`` segment are tolerated; the last two path segments are
    taken as ``owner/repo``. Returns ``None`` for a missing value, a non-``hf://``
    scheme, or fewer than two segments — the caller then skips the viewer.
    """
    if not artifact_url:
        return None
    url = artifact_url.strip()
    if not url.startswith(_HF_SCHEME):
        return None
    segments = [s for s in url[len(_HF_SCHEME) :].split("/") if s]
    if segments and segments[0] == _HF_HOST:
        segments = segments[1:]
    if len(segments) < 2:
        return None
    return "/".join(segments[-2:])


async def fetch_rows(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    repo_id: str,
    split: str,
    limit: int,
    token: str | None,
    config: str = "default",
) -> list[dict[str, Any]]:
    """Fetch up to ``limit`` rows of ``split`` from the HF dataset viewer.

    Calls ``GET {base_url}/rows``. ``length`` is clamped to the viewer's 100-row
    page cap. A bearer token is sent only when one is provided (public datasets
    need none; the app's pushed repos are private, so one is required in
    practice). Raises :class:`HFViewerUnavailable` on any failure — including a
    200 whose body carries no usable rows.
    """
    params: dict[str, str | int] = {
        "dataset": repo_id,
        "config": config,
        "split": split,
        "offset": 0,
        "length": min(limit, _VIEWER_ROW_CAP),
    }
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    where = f"{repo_id}#{config}/{split}"
    try:
        response = await client.get(f"{base_url.rstrip('/')}/rows", params=params, headers=headers)
    except httpx.HTTPError as exc:
        raise HFViewerUnavailable(f"{where}: transport error: {exc}") from exc

    if response.status_code != 200:
        raise HFViewerUnavailable(f"{where}: HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise HFViewerUnavailable(f"{where}: malformed JSON") from exc

    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise HFViewerUnavailable(f"{where}: response carried no rows")

    truncated = sum(1 for r in rows if isinstance(r, dict) and r.get("truncated_cells"))
    if truncated:
        logger.debug("HF viewer truncated cells in %d row(s) of %s", truncated, where)

    records = [r["row"] for r in rows if isinstance(r, dict) and isinstance(r.get("row"), dict)]
    if not records:
        raise HFViewerUnavailable(f"{where}: no usable row objects in response")
    return records
