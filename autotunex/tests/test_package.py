"""Smoke test: the package imports and reports a version."""

from __future__ import annotations

import tomllib
from pathlib import Path

import autotunex

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_package_exposes_a_version() -> None:
    assert autotunex.__version__


def test_package_version_is_read_from_dunder_version_not_declared_twice() -> None:
    # `/health` reports `__version__`, so a static `project.version` beside it lets a
    # bump update the wheel and silently leave the running service on the old number.
    with _PYPROJECT.open("rb") as handle:
        config = tomllib.load(handle)

    project = config["project"]
    hatch_version = config["tool"]["hatch"]["version"]

    assert "version" not in project
    assert "version" in project["dynamic"]
    assert hatch_version["path"] == "src/autotunex/__init__.py"
