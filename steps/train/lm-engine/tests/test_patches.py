# SPDX-License-Identifier: Apache-2.0
"""The lm-engine patches apply, in order, at the pinned UPSTREAM sha (needs network)."""

import pathlib
import re
import shutil
import subprocess

import pytest

_DIR = pathlib.Path(__file__).resolve().parents[1]


def test_upstream_pins_a_full_sha():
    text = (_DIR / "UPSTREAM").read_text()
    assert re.search(
        r"^sha: [0-9a-f]{40}$", text, re.M
    ), "UPSTREAM must pin a full 40-char sha"


def test_patches_are_numbered_and_listed():
    names = sorted(p.name for p in (_DIR / "patches").glob("*.patch"))
    assert [n[:4] for n in names] == ["0001", "0002", "0003", "0004"]
    text = (_DIR / "UPSTREAM").read_text()
    for n in names:
        assert n[:4] in text, f"{n} not explained in UPSTREAM"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_patches_apply():
    try:
        subprocess.run(
            ["git", "ls-remote", "https://github.com/open-lm-engine/lm-engine", "HEAD"],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except Exception:
        pytest.skip("no network to github.com")
    subprocess.run([str(_DIR / "scripts" / "check-patches.sh")], check=True)
