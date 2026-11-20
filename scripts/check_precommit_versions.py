#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Check that the pre-commit hook revs match the pins in the ``dev`` extra.

pre-commit installs its own copies of black and isort, in isolated environments, at
the ``rev`` given in ``.pre-commit-config.yaml``. CI runs the copies installed from the
``dev`` extra of ``pyproject.toml``. If the two drift apart, the git hook can write
formatting that CI then rejects, so each hook ``rev`` must equal the matching
``<package>==<version>`` pin.

Usage::

    scripts/check_precommit_versions.py

Exit status is 1 when a hook rev and the ``dev`` pin differ, or when either is missing.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# pre-commit hook repo URL -> package name pinned in the ``dev`` extra.
HOOK_PACKAGES = {
    "https://github.com/psf/black": "black",
    "https://github.com/pycqa/isort": "isort",
}


def hook_revs() -> dict[str, str]:
    """Return {package: rev} for the hook repos in HOOK_PACKAGES."""
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text())
    revs = {}
    for repo in config["repos"]:
        package = HOOK_PACKAGES.get(repo["repo"])
        if package:
            revs[package] = str(repo["rev"]).removeprefix("v")
    return revs


def dev_pins() -> dict[str, str]:
    """Return {package: version} for the ``package==version`` entries of the dev extra."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    pins = {}
    for requirement in pyproject["project"]["optional-dependencies"]["dev"]:
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==(\S+)", requirement.strip())
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def main() -> int:
    revs, pins = hook_revs(), dev_pins()
    errors = []
    for package in HOOK_PACKAGES.values():
        rev, pin = revs.get(package), pins.get(package)
        if rev is None:
            errors.append(f"{package}: no hook rev found in .pre-commit-config.yaml")
        elif pin is None:
            errors.append(f"{package}: no '{package}==<version>' pin in the dev extra")
        elif rev != pin:
            errors.append(
                f"{package}: hook rev {rev} (.pre-commit-config.yaml) != "
                f"dev pin {pin} (pyproject.toml)"
            )
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        print(
            "Bump the hook rev and the dev pin together (see CONTRIBUTING.md, "
            '"Pre-commit hooks").',
            file=sys.stderr,
        )
        return 1
    print(
        "pre-commit hook revs match the dev pins: "
        + ", ".join(f"{p} {revs[p]}" for p in HOOK_PACKAGES.values())
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
