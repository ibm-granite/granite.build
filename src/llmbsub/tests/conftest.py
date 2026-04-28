"""
Pytest configuration for llmbsub input upload tests.

This file registers custom markers and provides shared fixtures.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpfs: marks tests that require GPFS filesystem (run on BlueVela)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running (deselect with '-m \"not slow\"')"
    )
