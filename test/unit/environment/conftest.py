# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the environment unit tests."""

import pytest


@pytest.fixture
def real_apply_cloud_config():
    """Opt a test out of :func:`_no_cloud_config_writes` (it patches what it needs)."""


@pytest.fixture(autouse=True)
def _no_cloud_config_writes(request, monkeypatch):
    """Keep unit-test launches from rewriting the developer's ~/.sky/config.yaml.

    Every Skypilot launch applies its env's cloud_config last-writer-wins, which
    writes the real home directory's SkyPilot config and reloads the client. A
    unit test that launches (with sky mocked) must not do either, so this makes
    ``_apply_cloud_config`` a no-op unless the test requests
    ``real_apply_cloud_config``.
    """
    if "real_apply_cloud_config" in request.fixturenames:
        return
    monkeypatch.setattr(
        "gbserver.environment.skypilot._apply_cloud_config",
        lambda env_name, cloud_config: None,
    )
