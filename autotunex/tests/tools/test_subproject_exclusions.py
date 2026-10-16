"""The self-contained sibling subprojects under ``src/`` stay out of the root toolchain.

``src/api-bridge/`` and ``src/fm-tune/`` each carry their own ruff/mypy config and are
deliberately not held to the main service's ruff and mypy bars; ``src/ux/`` holds no
Python at all and is linted separately (prettier + eslint) instead. These assertions
exist so a future edit to ``pyproject.toml`` cannot silently pull a vendored tree into
the strict run — which would turn an unrelated upstream change into a red build.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _config() -> dict[str, object]:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def test_ruff_excludes_the_vendored_fm_tune_tree() -> None:
    config = _config()

    excluded = config["tool"]["ruff"]["extend-exclude"]  # type: ignore[index]

    assert "src/fm-tune" in excluded


def test_mypy_excludes_the_vendored_fm_tune_tree() -> None:
    config = _config()

    excluded = config["tool"]["mypy"]["exclude"]  # type: ignore[index]

    assert "src/fm-tune" in excluded


def test_the_autotune_import_override_is_retained() -> None:
    config = _config()

    overrides = config["tool"]["mypy"]["overrides"]  # type: ignore[index]
    autotune = [o for o in overrides if "autotune" in o["module"]]

    assert autotune, "the autotune ignore_missing_imports override must survive vendoring"
    assert autotune[0]["ignore_missing_imports"] is True


def test_the_granite_build_extra_no_longer_points_at_an_internal_host() -> None:
    config = _config()

    extra = config["project"]["optional-dependencies"]["granite-build"]  # type: ignore[index]

    assert not any("ibm.com" in dep for dep in extra), (
        "fm-tune is vendored at src/fm-tune and installed from the local path; "
        "the extra must not fetch it from an internal host"
    )
