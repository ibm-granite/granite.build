"""Utility for resolving full build UUIDs from partial or full IDs."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gbmcp.services.build_cache.build_cache import BuildCache

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def resolve_build_id(
    partial_id: str,
    cache: "BuildCache | None",
) -> tuple[str, str | None]:
    """Resolve a full build UUID from a full or partial build ID.

    Searches the in-memory build cache for a build whose UUID exactly matches
    or starts with the provided string (case-insensitive). Full 36-char UUIDs
    are returned immediately without a cache lookup. If multiple builds share
    a prefix, the first match in cache order is returned.

    Always returns an ID to use (never None) — either the resolved UUID or the
    original input as a fallback. The second element is a warning hint when the
    ID could not be resolved from cache; callers that require cache resolution
    (e.g. cache-only tools) should treat a non-None hint as an error, while
    callers that can pass IDs through to a downstream API should ignore it.

    Args:
        partial_id: Full or partial build UUID to resolve.
        cache: The BuildCache instance to search, or None if not initialised.

    Returns:
        A tuple (id_to_use, hint). On success, id_to_use is the resolved full
        UUID and hint is None. On cache miss, id_to_use is the original stripped
        input (passthrough) and hint is an informational message.
    """
    stripped = partial_id.strip()

    # Already a full UUID — no cache lookup required.
    if _UUID_RE.match(stripped):
        return stripped, None

    if cache is None or cache._cache is None:
        return stripped, (
            f"Build ID '{stripped}' could not be resolved against the build cache "
            "(cache not available). If this is a partial ID, retry with the full UUID."
        )

    needle = stripped.lower()
    for build in cache._cache.builds:
        if build.uuid.lower() == needle or build.uuid.lower().startswith(needle):
            return build.uuid, None

    return stripped, (
        f"Build ID '{stripped}' not found in build cache. "
        "If this is a partial ID, retry with the full UUID. "
        "If the build is recent, the cache may not have refreshed yet."
    )
