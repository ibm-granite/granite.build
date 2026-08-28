"""Smoke test: the package imports and reports a version."""

from __future__ import annotations

import autotunex


def test_package_exposes_a_version() -> None:
    assert autotunex.__version__
