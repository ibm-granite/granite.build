"""Utility for normalizing build IDs."""

from __future__ import annotations


def resolve_build_id(partial_id: str) -> str:
    """Normalize a build ID by stripping surrounding whitespace.

    Partial-ID resolution is not available in standalone: pass a full build
    UUID. Use build_list to find the full UUID of a build.

    Args:
        partial_id: The build ID to normalize.

    Returns:
        The stripped build ID.
    """
    return partial_id.strip()
