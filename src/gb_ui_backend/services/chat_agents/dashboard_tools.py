"""Framework-agnostic dashboard data tools — parallel to ui_actions.py.

None of these touch gbmcp or its allowlist; they wrap gb_ui_backend's own
existing services directly (GbserverSource, cloud_logs, gbd_meta/AI analysis,
local docs, gbserver's REST artifacts API). All ten are read-only — there is
no mutation path through any function in this module.

Each function raises DashboardToolError (never a bare exception) when it
can't run — the wrapper in tool_registry.py's build_dashboard_tools() turns
that into a tool error result the model can react to, rather than letting an
unrelated exception surface.
"""

from __future__ import annotations

import asyncio
import difflib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import httpx
import regex
from sqlalchemy import select

from gb_ui_backend.config import get_config
from gb_ui_backend.services.cloud_logs import get_cloud_logs_client
from gb_ui_backend.services.db_schema import GbdMeta, _get_session_factory
from gb_ui_backend.services.gbserver_source import get_gbserver_source


class DashboardToolError(RuntimeError):
    """Raised when a tool can't run (not configured, not found, bad input)."""


# ── search_docs ──────────────────────────────────────────────────────────────
# Mirrors the curated reading-path index in docs/README.md and the gb-docs
# skill (.claude/skills/gb-docs/SKILL.md) — same topic keys, not a second index.

DOC_TOPICS: dict[str, str] = {
    "getting_started": "getting-started.md",
    "builds_overview": "builds/README.md",
    "build_yaml_reference": "builds/build-yaml-reference.md",
    "gb_cli_reference": "cli/gb-cli-reference.md",
    "templates": "templates/README.md",
    "steps": "steps/README.md",
    "monitoring_and_artifact_events": "steps/monitoring-and-artifact-events.md",
    "hf_push": "builds/hf-push.md",
    "bring_your_own_step": "steps/bring-your-own-step.md",
    "custom_code_steps": "steps/custom-code-steps.md",
    "bring_your_own_image": "steps/bring-your-own-image.md",
    "faq": "help/faq.md",
    "glossary": "glossary.md",
    "demos": "demos/README.md",
    "build_retry": "builds/build-retry.md",
    "target_reuse": "builds/target-reuse.md",
    "step_retry": "builds/step-retry-configuration.md",
    "retry_overview": "builds/retry.md",
    "lineage": "builds/lineage.md",
    "event_notifications": "builds/event-notifications.md",
    "gbserver_cli_reference": "cli/gbserver-cli-reference.md",
    "configuration": "configuration/README.md",
    "spaces": "spaces/README.md",
    "environments": "environments/README.md",
    "step_resolution": "environments/step-resolution.md",
    "asset_stores": "asset-stores/README.md",
    "secrets": "secrets/README.md",
    "rest_api": "rest-api/README.md",
    "troubleshooting": "help/troubleshooting.md",
    "architecture": "architecture/README.md",
    "architecture_diagram": "architecture/arch-diagram.md",
    "environment_classes": "architecture/environment-classes.md",
    "testing": "testing/README.md",
}

_MAX_DOC_CHARS = 20_000  # guard against the one ~32KB outlier (demos/granite4_nano.md)


def _docs_root() -> Optional[Path]:
    """Resolve docs/ the same way constants.py's find_configurations_root()
    resolves configurations/ — repo-root relative. Editable-install only, same
    as that function; fine, since this only ever runs from a checkout."""
    candidates = (Path.cwd() / "docs", Path(__file__).resolve().parents[4] / "docs")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def search_docs(topic: str) -> str:
    root = _docs_root()
    if root is None:
        raise DashboardToolError(
            "Local docs/ directory not found — only available when running from a granite.build checkout."
        )

    rel_path = DOC_TOPICS.get(topic)
    target = root / rel_path if rel_path else None
    if target is None:
        # Fuzzy fallback: substring match against every doc's filename.
        needle = topic.lower().replace(" ", "-").replace("_", "-")
        matches = [p for p in sorted(root.rglob("*.md")) if needle in p.stem.lower()]
        target = matches[0] if matches else None

    if target is None or not target.is_file():
        raise DashboardToolError(
            f"No doc found for topic {topic!r}. Known topics: {', '.join(sorted(DOC_TOPICS))}"
        )

    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > _MAX_DOC_CHARS:
        text = text[:_MAX_DOC_CHARS] + "\n\n...[truncated]"
    return text


# ── search_builds ─────────────────────────────────────────────────────────────


async def search_builds(
    query: Optional[str] = None,
    status: Optional[str] = None,
    user: Optional[str] = None,
    space_name: Optional[str] = None,
    days_back: int = 30,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    source = get_gbserver_source()
    if source is None:
        raise DashboardToolError(
            "gbserver's build database isn't connected (GB_UI_GBSERVER_DB_URL not set)."
        )

    builds = await source.list_builds(
        days_back=days_back,
        date_from=date_from,
        date_to=date_to,
        status=status,
        space_name=space_name,
        username=user,
        limit=min(limit, 200),
    )
    if query:
        q = query.lower()
        builds = [b for b in builds if q in (b.get("name") or "").lower()]
    return builds


# ── search_build_yaml ──────────────────────────────────────────────────────────

_MAX_YAML_SCAN_LIMIT = 200  # mirrors search_builds' own cap
_REGEX_MATCH_TIMEOUT_SECONDS = 2.0


async def search_build_yaml(
    pattern: str, days_back: int = 5, limit: int = 200
) -> dict[str, Any]:
    source = get_gbserver_source()
    if source is None:
        raise DashboardToolError(
            "gbserver's build database isn't connected (GB_UI_GBSERVER_DB_URL not set)."
        )
    try:
        compiled_pattern = regex.compile(pattern, regex.IGNORECASE)
    except regex.error as exc:
        raise DashboardToolError(f"Invalid regex pattern: {exc}") from exc

    # pattern is model-chosen, not developer-written — clamp like
    # search_builds does rather than trusting an arbitrary limit.
    limit = min(max(limit, 1), _MAX_YAML_SCAN_LIMIT)

    # No indexed search exists — build.yaml lives inside a zipped archive
    # column, so this is fetch-and-unzip-per-build, O(N) bounded by limit/days_back.
    builds, warning = await source.list_builds_for_dp_scan(
        days_back=days_back, limit=limit
    )

    matches = []
    timed_out_for = []
    for b in builds:
        yaml_content = b.get("yaml_content")
        if not yaml_content:
            continue
        try:
            # `timeout=` here is a real, hard bound enforced inside the
            # `regex` module's own matching loop (it checks elapsed time
            # during backtracking and raises TimeoutError itself) — unlike
            # stdlib `re`, which never releases the GIL during a match, so
            # asyncio.wait_for's timeout couldn't actually preempt a
            # catastrophically backtracking pattern (it would just block the
            # whole single-threaded server, for every session, until the
            # match finally returned on its own — which could be effectively
            # forever). asyncio.to_thread + the outer wait_for stay as a
            # belt-and-suspenders backstop and to keep this off the event
            # loop for the duration of a match; the worker thread itself
            # can't be killed on timeout, but that's an acceptable one-off
            # cost bounded by `limit`, not a growing leak.
            found = await asyncio.wait_for(
                asyncio.to_thread(
                    compiled_pattern.search,
                    yaml_content,
                    timeout=_REGEX_MATCH_TIMEOUT_SECONDS,
                ),
                timeout=_REGEX_MATCH_TIMEOUT_SECONDS + 1,
            )
        except TimeoutError:
            # Covers both: regex's own internal timeout (the expected case
            # — see the comment above) and asyncio.wait_for's outer one
            # (the backstop) — asyncio.TimeoutError has been an alias for
            # the builtin TimeoutError since Python 3.11, so one except
            # clause catches either.
            timed_out_for.append(b["uuid"])
            continue
        if found:
            matches.append(
                {
                    "uuid": b["uuid"],
                    "name": b["name"],
                    "status": b["status"],
                    "username": b.get("username"),
                }
            )

    result: dict[str, Any] = {"matches": matches, "scanned": len(builds)}
    if warning:
        result["warning"] = warning
    if timed_out_for:
        result["timed_out_for"] = timed_out_for
    return result


# ── search_build_errors ────────────────────────────────────────────────────────

# Shared with compare_builds below — both take a model-supplied days_back
# with no schema-level minimum/maximum, so an arbitrarily large value must
# be clamped here rather than trusted (an unbounded window on
# search_build_errors materializes the whole matching table into Python
# objects in one request; on compare_builds it widens an
# archive-fetch-and-unzip scan over that many days' builds).
_MAX_DAYS_BACK = 365
_MAX_SEARCH_ERRORS_LIMIT = 200  # mirrors search_builds' own cap


async def search_build_errors(
    query: str, days_back: int = 7, limit: int = 200
) -> list[dict[str, Any]]:
    config = get_config()
    if not config.db_enabled:
        raise DashboardToolError(
            "AI analysis database isn't configured (GB_UI_DATABASE_URL not set)."
        )

    days_back = min(max(days_back, 1), _MAX_DAYS_BACK)
    limit = min(max(limit, 1), _MAX_SEARCH_ERRORS_LIMIT)

    since = datetime.now(timezone.utc) - timedelta(days=days_back)
    q = query.lower()
    async with _get_session_factory()() as session:
        stmt = (
            select(GbdMeta)
            .where(GbdMeta.created_at >= since)
            .order_by(GbdMeta.created_at.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()

    return [
        {
            "build_id": str(m.build_id),
            "summary": m.summary,
            "root_cause": m.root_cause,
            "error_category_1": m.error_category_1,
            "created_at": m.created_at.isoformat(),
        }
        for m in rows
        if q in (m.summary or "").lower() or q in (m.root_cause or "").lower()
    ]


# ── get_ai_analysis ────────────────────────────────────────────────────────────


async def get_ai_analysis(build_id: str) -> list[dict[str, Any]]:
    config = get_config()
    if not config.db_enabled:
        raise DashboardToolError(
            "AI analysis database isn't configured (GB_UI_DATABASE_URL not set)."
        )
    try:
        build_uuid = UUID(build_id)
    except ValueError as exc:
        raise DashboardToolError(f"Invalid build ID: {build_id!r}") from exc

    async with _get_session_factory()() as session:
        stmt = (
            select(GbdMeta)
            .where(GbdMeta.build_id == build_uuid)
            .order_by(GbdMeta.created_at.desc())
            .limit(10)
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()

    return [
        {
            "summary": m.summary,
            "root_cause": m.root_cause,
            "suggested_action": m.suggested_action,
            "confidence": m.confidence,
            "error_category_1": m.error_category_1,
            "error_category_2": m.error_category_2,
            "created_at": m.created_at.isoformat(),
        }
        for m in rows
    ]


# ── search_build_logs ──────────────────────────────────────────────────────────


async def search_build_logs(
    build_id: str, search: Optional[str] = None, tail: int = 500
) -> list[str]:
    config = get_config()
    if not (config.cloud_logs_url and config.cloud_logs_api_key):
        raise DashboardToolError(
            "Cloud logs aren't configured (GB_UI_CLOUD_LOGS_URL/API_KEY not set)."
        )

    # Clamped separately from the slice below: the API's page_size wants at
    # least 1, but the slice must preserve tail=0 meaning "return nothing"
    # — `lines[-0:]` is `lines[0:]` (the whole list, since `-0 == 0`), which
    # is why this used the raw, unclamped `tail` for the slice before.
    page_size = min(max(tail, 1), 2000)
    clamped_tail = min(max(tail, 0), 2000)
    client = get_cloud_logs_client(config.cloud_logs_url, config.cloud_logs_api_key)
    response = await client.query_logs(build_id=build_id, page_size=page_size)
    lines = client.parse_logs(response)
    if search:
        needle = search.lower()
        lines = [line for line in lines if needle in line.lower()]
    return lines[-clamped_tail:] if clamped_tail else []


# ── compare_builds ─────────────────────────────────────────────────────────────


async def compare_builds(build_ids: list[str], days_back: int = 30) -> dict[str, Any]:
    if len(build_ids) < 2:
        raise DashboardToolError("Need at least two build IDs to compare.")
    source = get_gbserver_source()
    if source is None:
        raise DashboardToolError(
            "gbserver's build database isn't connected (GB_UI_GBSERVER_DB_URL not set)."
        )

    days_back = min(max(days_back, 1), _MAX_DAYS_BACK)

    # Concurrent, not sequential — these are independent network/DB round
    # trips, so N of them cost roughly one round trip's latency instead of
    # N. Checked back in list order afterward so the error for multiple
    # missing builds still names the first one, matching the old sequential
    # behavior exactly.
    fetched = await asyncio.gather(*(source.get_build(bid) for bid in build_ids))
    builds: dict[str, dict[str, Any]] = {}
    for bid, b in zip(build_ids, fetched):
        if b is None:
            raise DashboardToolError(f"Build {bid} not found.")
        builds[bid] = b

    # GbserverSource has no single-build YAML fetch — list_builds_for_dp_scan's
    # archive-extraction is the only path that decodes YAML at all, so this
    # widens the scan to days_back and matches by ID. Builds older than that
    # window won't have YAML available here; the structured-field diff below
    # still works regardless.
    scanned, _warning = await source.list_builds_for_dp_scan(
        days_back=days_back, limit=5000
    )
    yaml_by_id = {b["uuid"]: b.get("yaml_content") for b in scanned}

    fields = ["name", "space_name", "username", "status"]
    comparison = {f: {bid: builds[bid].get(f) for bid in build_ids} for f in fields}

    ids_with_yaml = [bid for bid in build_ids if yaml_by_id.get(bid)]
    yaml_diffs: dict[str, str] = {}
    for i in range(len(ids_with_yaml)):
        for j in range(i + 1, len(ids_with_yaml)):
            id_a, id_b = ids_with_yaml[i], ids_with_yaml[j]
            diff = difflib.unified_diff(
                (yaml_by_id[id_a] or "").splitlines(),
                (yaml_by_id[id_b] or "").splitlines(),
                fromfile=id_a,
                tofile=id_b,
                lineterm="",
            )
            yaml_diffs[f"{id_a}_vs_{id_b}"] = "\n".join(diff) or "(identical)"

    result: dict[str, Any] = {"fields": comparison, "yaml_diffs": yaml_diffs}
    missing_yaml = [bid for bid in build_ids if bid not in ids_with_yaml]
    if missing_yaml:
        result["yaml_unavailable_for"] = missing_yaml
    return result


# ── wait_for_build ─────────────────────────────────────────────────────────────

_TERMINAL_STATUSES = {"success", "failed", "cancelled", "invalid", "error"}


async def wait_for_build(
    build_id: str, timeout_minutes: int = 15, poll_interval_seconds: int = 15
) -> dict[str, Any]:
    # Capped well below gb_dashboard's 360 min — this runs inside a chat
    # turn's async context, don't let one call hold a session open for hours.
    timeout_minutes = min(max(timeout_minutes, 1), 30)
    poll_interval_seconds = max(poll_interval_seconds, 10)
    source = get_gbserver_source()
    if source is None:
        raise DashboardToolError(
            "gbserver's build database isn't connected (GB_UI_GBSERVER_DB_URL not set)."
        )

    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_minutes * 60
    while True:
        build = await source.get_build(build_id)
        if build is None:
            raise DashboardToolError(f"Build {build_id} not found.")
        if build["status"] in _TERMINAL_STATUSES:
            return {"build_id": build_id, "status": build["status"], "timed_out": False}
        if loop.time() >= deadline:
            return {"build_id": build_id, "status": build["status"], "timed_out": True}
        await asyncio.sleep(poll_interval_seconds)


# ── list_artifacts / describe_artifact ────────────────────────────────────────
# No existing Python wrapper for gbserver REST calls to copy — modeled on
# ai_prompts.py's per-call `async with httpx.AsyncClient(...)` shape, but
# caching the client across calls the way cloud_logs.py's CloudLogsClient
# does, rather than opening a fresh one (and its connection pool) per call.
# No auth header needed, per frontend/api/gbserver.ts's own usage of these
# endpoints.

_artifacts_http_client: Optional[httpx.AsyncClient] = None


def _get_artifacts_http_client() -> httpx.AsyncClient:
    global _artifacts_http_client
    if _artifacts_http_client is None:
        _artifacts_http_client = httpx.AsyncClient(timeout=30.0)
    return _artifacts_http_client


async def list_artifacts(
    build_id: Optional[str] = None,
    space_name: Optional[str] = None,
    tag: Optional[str] = None,
    username: Optional[str] = None,
) -> list[dict[str, Any]]:
    config = get_config()
    params: dict[str, Any] = {}
    if build_id:
        params["build_id"] = build_id
    if space_name:
        params["space_name"] = space_name
    if username:
        params["username"] = username
    if tag:
        params["tag"] = tag

    client = _get_artifacts_http_client()
    resp = await client.get(f"{config.gbserver_url}/api/v1/artifacts/", params=params)
    resp.raise_for_status()
    data = resp.json()
    return data.get("artifacts", [])


async def describe_artifact(artifact_id: str) -> dict[str, Any]:
    config = get_config()
    client = _get_artifacts_http_client()
    resp = await client.get(f"{config.gbserver_url}/api/v1/artifacts/{artifact_id}")
    if resp.status_code == 404:
        raise DashboardToolError(f"Artifact {artifact_id} not found.")
    resp.raise_for_status()
    data = resp.json()
    return data.get("artifact", data)
