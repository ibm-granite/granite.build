"""Test mode configuration for mock vs. live test execution.

Controls which external services are mocked vs. live-connected per test.

Configuration precedence (highest to lowest):
    1. @pytest.mark.live("service1", "service2") on the test
    2. Per-service env var: GBTEST_LIVE_STORAGE=true, GBTEST_LIVE_GITHUB=true, etc.
    3. Global mode: GBTEST_MODE=live (default) | mock
"""

import os

_DEFAULT_MODE = "live"

SERVICES = [
    "storage",
    "github",
    "lakehouse",
    "kubernetes",
    "hf",
    "messaging",
    "secrets",
    "time",
]


def get_test_mode() -> str:
    """Return the current test mode ('live' or 'mock')."""
    return os.environ.get("GBTEST_MODE", _DEFAULT_MODE).lower()


def is_mock_mode() -> bool:
    """Return True if the global test mode is 'mock'."""
    return get_test_mode() != "live"


def is_live(service: str) -> bool:
    """Return True if the given service should use real external connections.

    Checks per-service env var, then global mode. Does NOT check markers
    (markers require a pytest request object — use should_use_live for that).
    """
    env_key = f"GBTEST_LIVE_{service.upper()}"
    if os.environ.get(env_key, "").lower() == "true":
        return True
    return get_test_mode() == "live"


def should_use_live(request, service: str) -> bool:
    """Check markers, then per-service env var, then global mode.

    Args:
        request: The pytest request fixture (provides access to markers).
        service: One of SERVICES.
    """
    for mark in request.node.iter_markers("live"):
        if service in mark.args:
            return True

    env_key = f"GBTEST_LIVE_{service.upper()}"
    if os.environ.get(env_key, "").lower() == "true":
        return True

    return get_test_mode() == "live"
