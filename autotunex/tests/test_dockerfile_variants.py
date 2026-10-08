"""The root ``Dockerfile``'s device stages stay wired the way a plain build expects.

A plain ``docker build .`` must keep producing the CPU image. Which base image the
runtime stage sits on is chosen by a global ``ARG DEVICE``, so flipping that one
default would silently turn every default build into the far larger CUDA image.
Nothing in CI builds the container, so that mistake would ship unnoticed.

These read the Dockerfile as text on purpose: building it needs a daemon, a
multi-GB pull, and — for the CUDA base — linux/amd64 hardware, none of which the
unit suite has. The trade is deliberate; the real build verification is manual and
recorded in ``docs/superpowers/specs/2026-09-16-cuda-container-image-design.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

_DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"


def _dockerfile() -> str:
    """Return the root Dockerfile's text."""
    return _DOCKERFILE.read_text(encoding="utf-8")


def _stage_body(content: str, name: str) -> str:
    """Return the instructions between ``AS <name>`` and the next ``FROM``.

    Args:
        content: The whole Dockerfile.
        name: A build-stage name, as it appears after ``AS``.

    Returns:
        The stage's instructions, excluding its own ``FROM`` line.
    """
    after = content.split(f" AS {name}\n", 1)[1]
    return after.split("\nFROM ", 1)[0]


def test_the_device_build_arg_defaults_to_cpu() -> None:
    match = re.search(r"^ARG DEVICE=(\S+)", _dockerfile(), re.MULTILINE)

    assert match is not None, "the Dockerfile no longer declares a global `ARG DEVICE`"
    assert match.group(1) == "cpu"


def test_both_documented_device_values_name_a_real_stage() -> None:
    stages = set(re.findall(r"^FROM \S+ AS (\S+)", _dockerfile(), re.MULTILINE))

    assert {"cpu", "cuda"} <= stages


def test_the_cpu_stage_selects_the_lean_core_extras() -> None:
    body = _stage_body(_dockerfile(), "cpu")

    assert "FM_TUNE_EXTRAS=core" in body


def test_the_cuda_stage_selects_the_full_extras() -> None:
    body = _stage_body(_dockerfile(), "cuda")

    assert "FM_TUNE_EXTRAS=full" in body


def test_the_cpu_stage_requests_no_flash_attn_wheel() -> None:
    body = _stage_body(_dockerfile(), "cpu")

    assert re.search(r"FLASH_ATTN_WHEEL=\s*$", body, re.MULTILINE) is not None


def test_the_cuda_stage_pins_a_cp312_flash_attn_wheel() -> None:
    body = _stage_body(_dockerfile(), "cuda")

    assert "flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl" in body
