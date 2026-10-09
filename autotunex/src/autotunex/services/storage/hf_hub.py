# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace Hub REST client — dataset/model search, model cards, and parquet-branch reads.

A sibling of ``hf_viewer`` and deliberately the same shape: it knows nothing about
FastAPI, the database, or ``Settings``, takes primitives, returns plain data, and is
unit-tested with ``httpx.MockTransport``. No ``huggingface_hub`` dependency, matching
the convention stated in ``services/storage/artifacts.py``.

Where ``hf_viewer`` talks to the dataset *viewer* (``datasets-server``), this module
talks to the Hub itself and pins HF's auto-converted **parquet branch** to a commit
before enumerating or downloading its files.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from typing import Any
from urllib.parse import quote

import httpx

from autotunex.core.logging import get_logger

logger = get_logger(__name__)


class HfHubUnavailable(Exception):  # noqa: N818 (matches HFViewerUnavailable)
    """The Hub could not be reached, or answered in a way we cannot use."""


class HfDatasetNotConverted(Exception):  # noqa: N818
    """The dataset has no auto-converted parquet branch (Hub answers 400/404).

    Distinct from :class:`HfNoTabularData` because it is usually transient — HF
    converts new uploads on a delay — so the user should be told to retry later.
    """


class HfNoTabularData(Exception):  # noqa: N818
    """The parquet branch exists but lists no configs/splits (a 200 with ``{}``).

    Observed on ``world-igr-plum/regions`` and ``yuiseki/osm-wiki``: the repo holds
    no tabular data, so waiting will not help.
    """


class HfModelNotFound(Exception):  # noqa: N818 (matches HfHubUnavailable)
    """The Hub answered 401/404 for a model repo, or a redirect a tokened read won't follow.

    One exception for both: the Hub answers 401 for a private repo the caller's
    token may not read, and a caller must not learn such a repo exists.
    """


_OWNER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
"""Exactly ``owner/name``: a third segment or a ``..`` never reaches the owner check."""


def is_allowlisted(repo_id: str, namespaces: Sequence[str]) -> bool:
    """Whether ``repo_id`` is exactly ``owner/name`` with ``owner`` in ``namespaces``.

    The one owner rule behind every use of the allowlist: where the token is sent
    (:func:`token_for`), which search hits are kept, and which base models a build
    binds with the space's credentials. Case-sensitive on purpose: the Hub answers
    a non-canonical case with a 307, which a tokened read never follows, so
    matching ``IBM-Org/x`` would only swap a working anonymous read for a failure.
    """
    return _OWNER_NAME.fullmatch(repo_id) is not None and repo_id.split("/")[0] in namespaces


def token_for(repo_id: str, *, token: str | None, namespaces: list[str]) -> str | None:
    """Return the token only if ``repo_id``'s namespace is allowlisted.

    A shared server token is otherwise a universal read key. An empty allowlist
    means public-only. A repo id with no ``owner/`` prefix is never allowlisted.
    """
    if not token:
        return None
    return token if is_allowlisted(repo_id, namespaces) else None


def _headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


async def _get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    token: str | None,
    params: dict[str, Any] | None = None,
    parquet_branch: bool = False,
) -> Any:  # noqa: ANN401
    """GET and decode JSON, mapping every failure onto :class:`HfHubUnavailable`.

    Redirects are followed only for an *anonymous* read. The Hub answers 307 for a
    bare canonical id (``imdb`` -> ``stanfordnlp/imdb``), and leaving that
    unfollowed reported a perfectly importable dataset as unavailable. But httpx
    strips ``Authorization`` only *cross-origin*, and a Hub rename redirects
    huggingface.co -> huggingface.co, so following a *tokened* request could hand
    the server token to a namespace outside ``hf_import_namespaces`` -- and that
    allowlist is the whole control on where this token is sent (:func:`token_for`).
    A bare canonical id never carries a token (``token_for`` requires a namespace),
    which is exactly the case that needs following; a tokened redirect stays
    unfollowed and surfaces as :class:`HfHubUnavailable`.
    """
    try:
        response = await client.get(
            url, params=params, headers=_headers(token), follow_redirects=token is None
        )
    except httpx.HTTPError as exc:
        raise HfHubUnavailable(f"{url}: transport error: {exc}") from exc
    if parquet_branch and response.status_code in (400, 404):
        raise HfDatasetNotConverted(url)
    if response.status_code != 200:
        raise HfHubUnavailable(f"{url}: HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        raise HfHubUnavailable(f"{url}: malformed JSON") from exc


async def search_datasets(
    client: httpx.AsyncClient, *, base_url: str, query: str, limit: int, token: str | None
) -> list[str]:
    """Return dataset ids matching ``query``, most relevant first.

    Entries without a string ``id`` are skipped rather than failing the search.
    """
    payload = await _get_json(
        client,
        f"{base_url.rstrip('/')}/api/datasets",
        token=token,
        params={"search": query, "limit": limit},
    )
    if not isinstance(payload, list):
        raise HfHubUnavailable("search: expected a JSON array")
    return [e["id"] for e in payload if isinstance(e, dict) and isinstance(e.get("id"), str)]


async def search_models(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    query: str,
    limit: int,
    token: str | None,
    author: str | None = None,
    private_only: bool = False,
) -> list[str]:
    """Return model ids matching ``query``, most relevant first.

    ``author`` pins the search to one namespace — the only way a tokened search
    stays inside the allowlist, since a free-text query has no ``repo_id`` for
    :func:`token_for` to check. Entries without a string ``id`` are skipped.
    ``private_only`` keeps only entries whose ``private`` field (a boolean the
    Hub's model-list entries carry) is ``True``, dropping public and unmarked
    entries.
    """
    params: dict[str, Any] = {"search": query, "limit": limit}
    if author is not None:
        params["author"] = author
    payload = await _get_json(
        client, f"{base_url.rstrip('/')}/api/models", token=token, params=params
    )
    if not isinstance(payload, list):
        raise HfHubUnavailable("model search: expected a JSON array")
    entries = [e for e in payload if isinstance(e, dict)]
    if private_only:
        entries = [e for e in entries if e.get("private") is True]
    return [e["id"] for e in entries if isinstance(e.get("id"), str)]


async def fetch_model_card(
    client: httpx.AsyncClient, *, base_url: str, repo_id: str, token: str | None
) -> str:
    """Return ``repo_id``'s ``README.md`` from ``main`` as text.

    Same redirect rule as :func:`_get_json`: only an anonymous read follows one,
    so the token can never be carried to a namespace outside the allowlist. An
    unfollowed redirect (a renamed repo, or a non-canonical case) is not-found
    rather than an outage: the repo is not at the id the caller gave.

    Raises:
        HfModelNotFound: the Hub answered 401 or 404, or a redirect to a tokened read.
        HfHubUnavailable: anything else.
    """
    url = f"{base_url.rstrip('/')}/{repo_id}/raw/main/README.md"
    try:
        response = await client.get(url, headers=_headers(token), follow_redirects=token is None)
    except httpx.HTTPError as exc:
        raise HfHubUnavailable(f"{url}: transport error: {exc}") from exc
    if response.status_code in (401, 404) or response.is_redirect:
        raise HfModelNotFound(repo_id)
    if response.status_code != 200:
        raise HfHubUnavailable(f"{url}: HTTP {response.status_code}")
    return response.text


async def resolve_revision(
    client: httpx.AsyncClient, *, base_url: str, repo_id: str, token: str | None
) -> str:
    """Resolve the converted parquet branch to its immutable commit SHA.

    Converted files live on ``refs/convert/parquet``, not ``main``. Its SHA is
    the revision consumers must carry from the picker through preview/import.
    """
    payload = await _get_json(
        client,
        f"{base_url.rstrip('/')}/api/datasets/{repo_id}/revision/refs%2Fconvert%2Fparquet",
        token=token,
        parquet_branch=True,
    )
    sha = payload.get("sha") if isinstance(payload, dict) else None
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise HfHubUnavailable(f"{repo_id}: no commit sha in repo metadata")
    return sha


async def list_parquet_files(
    client: httpx.AsyncClient, *, base_url: str, repo_id: str, revision: str, token: str | None
) -> dict[str, dict[str, list[str]]]:
    """Enumerate parquet files at one commit and construct immutable download URLs.

    The Hub's live ``/parquet`` listing is not revision-scoped. Dataset metadata
    at a commit includes all file paths in ``siblings``; the converted branch
    stores shards as ``<config>/<split>/<shard>.parquet``. Both enumeration and
    download use the selected commit, even if conversion advances meanwhile.

    Raises:
        HfNoTabularData: HTTP 200 with no usable config/split entries.
        HfHubUnavailable: anything else.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full commit SHA")
    base = base_url.rstrip("/")
    url = f"{base}/api/datasets/{repo_id}/revision/{revision}"
    payload = await _get_json(client, url, token=token)
    if not isinstance(payload, dict) or payload.get("sha") != revision:
        raise HfHubUnavailable(f"{repo_id}: metadata does not match the selected revision")
    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        raise HfHubUnavailable(f"{repo_id}: missing file listing in repo metadata")
    result: dict[str, dict[str, list[str]]] = {}
    paths = sorted(
        {
            entry["rfilename"]
            for entry in siblings
            if isinstance(entry, dict) and isinstance(entry.get("rfilename"), str)
        }
    )
    for path in paths:
        parts = path.split("/")
        if len(parts) != 3 or not path.endswith(".parquet"):
            continue
        if any(part in ("", ".", "..") for part in parts):
            raise HfHubUnavailable(f"{repo_id}: invalid parquet path")
        config, split, _ = parts
        download = f"{base}/datasets/{repo_id}/resolve/{revision}/{quote(path, safe='/')}"
        result.setdefault(config, {}).setdefault(split, []).append(download)

    if not result:
        raise HfNoTabularData(repo_id)
    return result


_HEAD_CONCURRENCY = 8
"""Concurrent HEAD requests allowed per size gate; see :func:`head_total_bytes`."""


async def head_total_bytes(client: httpx.AsyncClient, *, urls: list[str], token: str | None) -> int:
    """Total ``content-length`` across ``urls``, via bounded-concurrency HEAD requests.

    The CDN answers HEAD with a length and advertises ``accept-ranges: bytes``, so
    the size gate costs no download. A URL whose length is missing or unparseable
    contributes 0 rather than failing the gate — an under-estimate refuses less,
    and ``hf_import_service._stage_split`` counts the real bytes inline as it
    downloads, so an under-estimate here cannot become an unbounded fetch.

    Concurrency is capped at :data:`_HEAD_CONCURRENCY` rather than fanning out over
    every URL at once, since this runs on the app-wide client whose connection pool
    every other outbound call (OIDC JWKS refresh, the LLM gateway, reconcile)
    shares: an unbounded gather there exhausts that pool, times out most of its own
    requests, and starves the rest of the app for the duration. ``build_plan``, the
    only caller, now passes one URL per selected split rather than every shard, so
    the bound is defence in depth rather than the thing standing between a
    many-shard repo and the pool.

    ``return_exceptions`` keeps one failing HEAD from abandoning its siblings
    mid-flight: a bare gather propagates the first failure and leaves the rest
    running against a pool the caller has already stopped waiting on, with their
    own exceptions never retrieved. The first failure is re-raised once all of
    them have settled, so the type callers match on is unchanged.
    """
    semaphore = asyncio.Semaphore(_HEAD_CONCURRENCY)

    async def one(url: str) -> int:
        async with semaphore:
            try:
                response = await client.head(url, headers=_headers(token), follow_redirects=True)
            except httpx.HTTPError as exc:
                raise HfHubUnavailable(f"{url}: HEAD transport error: {exc}") from exc
        if response.status_code != 200:
            raise HfHubUnavailable(f"{url}: HEAD HTTP {response.status_code}")
        try:
            return int(response.headers.get("content-length", "0"))
        except ValueError:
            logger.debug("Unparseable content-length for %s", url)
            return 0

    results = await asyncio.gather(*(one(u) for u in urls), return_exceptions=True)
    total = 0
    failure: BaseException | None = None
    for result in results:
        if isinstance(result, BaseException):
            failure = failure if failure is not None else result
        else:
            total += result
    if failure is not None:
        raise failure
    return total
