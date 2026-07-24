"""Opt-in gate for AutoTuneX bridge logging.

Bridge logging is OFF by default. It is enabled only when the user passes a
non-empty ``--autotunex_server_url``. Keeping this decision in a tiny importable
helper lets it be unit-tested without executing main.py (which triggers Ray).
"""

from typing import Optional, Tuple


def resolve_bridge_settings(
    autotunex_server_url: Optional[str],
) -> Tuple[bool, Optional[str]]:
    """Return ``(bridge_enabled, base_url)``.

    The bridge is enabled only when ``autotunex_server_url`` is a non-empty
    string. An absent (``None``) or empty value means the run executes fully
    offline with no bridge calls.
    """
    if autotunex_server_url:
        return True, autotunex_server_url
    return False, None
