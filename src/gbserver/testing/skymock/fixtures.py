"""Reusable pytest fixtures for testing Skypilot with SkyMock."""

import asyncio
from contextlib import contextmanager
from unittest.mock import patch

from gbserver.environment.skypilot import Skypilot
from gbserver.testing.skymock.mock_sky import MockSky
from gbserver.testing.skymock.scenario import Scenario
from gbserver.types.environmentconfig import EnvironmentConfig


@contextmanager
def patched_skypilot(mock_sky: MockSky):
    """Context manager that patches sky module and HAS_SKYPILOT in skypilot.py."""
    with (
        patch("gbserver.environment.skypilot.sky", mock_sky),
        patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
    ):
        yield mock_sky


def make_skypilot_env(
    scenario: Scenario,
    cloud: str = "aws",
    idle_minutes: int = 0,
) -> tuple[Skypilot, asyncio.Queue, MockSky]:
    """Create a Skypilot environment configured for testing with SkyMock.

    Returns (env, event_q, mock_sky) tuple.
    """
    event_q: asyncio.Queue = asyncio.Queue()
    config = EnvironmentConfig(
        name=f"test-{cloud}",
        type="Skypilot",
        config={
            "default_cloud": cloud,
            "idle_minutes_to_autostop": idle_minutes,
        },
    )
    mock_sky = MockSky(default_scenario=scenario)
    env = Skypilot(event_q=event_q, environment_config=config)
    return env, event_q, mock_sky
