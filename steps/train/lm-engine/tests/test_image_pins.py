# SPDX-License-Identifier: Apache-2.0
"""Dockerfile.gpu pins every external input, and its lm-engine sha matches UPSTREAM.

Offline and cheap. Dockerfile.gpu also fails its own build when LME_SHA has drifted
from UPSTREAM, but that costs a ~10 GB build to find out; these catch it in a second.
"""

import pathlib
import re

_DIR = pathlib.Path(__file__).resolve().parents[1]
_DOCKERFILE = (_DIR / "Dockerfile.gpu").read_text()
_UPSTREAM = (_DIR / "UPSTREAM").read_text()


def _arg(name: str) -> str:
    match = re.search(rf"^ARG {name}=(\S+)$", _DOCKERFILE, re.M)
    assert match, f"Dockerfile.gpu does not declare ARG {name}"
    return match.group(1)


def test_lme_sha_matches_upstream():
    upstream_sha = re.search(r"^sha: ([0-9a-f]{40})$", _UPSTREAM, re.M)
    assert upstream_sha, "UPSTREAM must pin a full 40-char sha"
    assert _arg("LME_SHA") == upstream_sha.group(1), (
        "Dockerfile.gpu's LME_SHA has drifted from UPSTREAM's sha; UPSTREAM is the "
        "source of truth and documents the patches that apply at that sha"
    )


def test_lme_repo_matches_upstream():
    upstream_repo = re.search(r"^repo: (\S+)$", _UPSTREAM, re.M).group(1)
    assert _arg("LME_REPO").removesuffix(".git") == upstream_repo.removesuffix(".git")


def test_base_image_pinned_by_digest():
    assert re.search(
        r"^FROM ubuntu:22\.04@sha256:[0-9a-f]{64}$", _DOCKERFILE, re.M
    ), "FROM must pin the base image by @sha256: digest, not by tag alone"


def test_flash_attn_pinned_by_url_and_sha256():
    url = _arg("FLASH_ATTN_WHL")
    assert url.startswith("https://github.com/Dao-AILab/flash-attention/releases/")
    # the wheel's ABI tag is what ties the image to the torch and python pins
    assert "cu12torch2.10" in url and "cp312-cp312" in url
    assert re.fullmatch(r"[0-9a-f]{64}", _arg("FLASH_ATTN_SHA256"))
    assert "sha256sum -c -" in _DOCKERFILE, "the wheel's sha256 must be verified"


def test_uv_and_python_pinned_to_exact_versions():
    assert re.fullmatch(r"\d+\.\d+\.\d+", _arg("UV_VERSION"))
    # not just "3.12": the flash-attn wheel is cp312-only and the patch release must
    # not float between builds
    assert re.fullmatch(r"3\.12\.\d+", _arg("PYTHON_VERSION"))
    assert "COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION}" in _DOCKERFILE


def test_lm_engine_is_not_pip_installed():
    # tools/ (the export step's model_conversion) is not part of the wheel, so the
    # source tree has to stay on PYTHONPATH instead
    assert "--no-install-project" in _DOCKERFILE
    assert "PYTHONPATH=/app/src" in _DOCKERFILE
    assert "uv sync --locked" in _DOCKERFILE, "the committed uv.lock must be honoured"


def test_check_image_script_is_copied_into_the_image():
    assert "COPY scripts/check-image.sh /app/check-image.sh" in _DOCKERFILE
    script = _DIR / "scripts" / "check-image.sh"
    assert script.stat().st_mode & 0o111, "check-image.sh must be executable"


def test_einops_pinned():
    # installed outside the lock (flash-attn needs it), so it needs its own pin
    assert re.fullmatch(r"\d+\.\d+\.\d+", _arg("EINOPS_VERSION"))
    assert "einops==$EINOPS_VERSION" in _DOCKERFILE
