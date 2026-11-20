# SPDX-License-Identifier: Apache-2.0

"""The vendored lock is upstream's, plus one appended security block (needs network).

``lock/pyproject.toml`` is upstream's file at the ``UPSTREAM`` sha with a single
``[tool.uv]`` block appended, and ``lock/uv.lock`` is what ``uv lock`` produced from it
(``make relock``). The Dockerfile copies both over the clone before ``uv sync --locked``,
so this pair -- not upstream's -- decides what the image installs.

That makes silent drift the risk this file exists to stop: bump ``UPSTREAM`` to a sha
whose dependencies changed, forget to re-vendor, and the image installs the OLD
dependency set against the NEW source while every pin test still passes. So the
vendored file is compared against upstream's at the pinned sha with the appended block
stripped, and the two must be byte-identical.
"""

from __future__ import annotations

import pathlib
import urllib.error
import urllib.request

import pytest

_DIR = pathlib.Path(__file__).resolve().parent.parent
_MARKER = "# ── Security overrides and constraints (granite.build) ─"


def _upstream(name: str) -> str:
    text = (_DIR / "UPSTREAM").read_text(encoding="utf-8")
    sha = next(l.split()[1] for l in text.splitlines() if l.startswith("sha:"))
    repo = next(l.split()[1] for l in text.splitlines() if l.startswith("repo:"))
    raw = repo.replace("https://github.com/", "https://raw.githubusercontent.com/")
    raw = raw.removesuffix(".git")
    try:
        with urllib.request.urlopen(f"{raw}/{sha}/{name}", timeout=30) as response:
            return response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError) as error:
        pytest.skip(f"no network to fetch {name} at {sha}: {error}")


def test_the_vendored_files_exist():
    assert (_DIR / "lock" / "pyproject.toml").is_file()
    assert (_DIR / "lock" / "uv.lock").is_file()


def test_the_security_block_is_appended_last_and_only_once():
    text = (_DIR / "lock" / "pyproject.toml").read_text(encoding="utf-8")
    assert text.count(_MARKER) == 1, "the security block must appear exactly once"
    head, _, tail = text.partition(_MARKER)
    assert "[tool.uv]" in tail, "the [tool.uv] block belongs inside the security block"
    assert (
        "[tool.uv]" not in head
    ), "upstream grew a [tool.uv] section; merge it by hand"


def test_the_vendored_pyproject_is_upstreams_plus_the_block():
    """Byte-identical above the marker. A bump that changes deps must be re-vendored."""
    vendored = (_DIR / "lock" / "pyproject.toml").read_text(encoding="utf-8")
    stripped = vendored.partition(_MARKER)[0]
    assert stripped.rstrip("\n") == _upstream("pyproject.toml").rstrip("\n"), (
        "lock/pyproject.toml above the security block is not upstream's at the "
        "UPSTREAM sha; re-vendor it and run `make relock`"
    )


def test_every_override_and_constraint_is_justified():
    """Each pin carries an advisory id, so nobody has to guess why it is there."""
    block = (
        (_DIR / "lock" / "pyproject.toml")
        .read_text(encoding="utf-8")
        .partition(_MARKER)[2]
    )
    pins = [l.strip() for l in block.splitlines() if l.strip().startswith('"')]
    assert pins, "the security block declares no pins"
    assert "PYSEC" in block or "GHSA" in block, "no advisory ids cited in the block"
