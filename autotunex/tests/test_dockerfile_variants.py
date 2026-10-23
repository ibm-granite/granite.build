"""The root ``Dockerfile``'s device stages stay wired the way a plain build expects.

A plain ``docker build .`` must keep producing the CPU image. Which base image the
runtime stage sits on is chosen by a global ``ARG DEVICE``, so flipping that one
default would silently turn every default build into the far larger CUDA image.
Nothing in CI builds the container, so that mistake would ship unnoticed.

These read the Dockerfile as text on purpose: building it needs a daemon, a
multi-GB pull, and — for the CUDA base — linux/amd64 hardware, none of which the
unit suite has. The trade is deliberate; the real build verification is manual.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"
_FM_TUNE_PYPROJECT = Path(__file__).resolve().parents[1] / "src" / "fm-tune" / "pyproject.toml"


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


def test_neither_stage_sets_a_prebuilt_flash_attn_wheel() -> None:
    # fm-tune no longer pins a flash-attn wheel: there is none for torch 2.11 + CUDA 13,
    # so flash-attn is compiled from source by the uv install, not fetched.
    assert "FLASH_ATTN_WHEEL" not in _dockerfile()


def test_the_cuda_base_is_a_cuda_13_devel_image() -> None:
    # torch 2.11's PyPI wheels are built for CUDA 13.0 and flash-attn is compiled with the
    # image's nvcc, so the base must be CUDA 13 and a -devel tag (which ships nvcc).
    match = re.search(r"^ARG CUDA_BASE=(\S+)", _dockerfile(), re.MULTILINE)

    assert match is not None
    assert re.fullmatch(r"nvidia/cuda:13\.\d+\.\d+-devel-ubuntu24\.04", match.group(1))


def test_the_extras_install_goes_through_uv_from_the_fm_tune_directory() -> None:
    # pip ignores [tool.uv.extra-build-dependencies], where fm-tune declares how to build
    # flash-attn / mamba-ssm against the runtime torch; uv reads it only from the project dir.
    run = _dockerfile().split('RUN pip install --no-cache-dir "./src/fm-tune[core]"', 1)[1]
    run = run.split("\n\n", 1)[0]

    assert "cd src/fm-tune" in run
    assert re.search(r"uv pip install .*\[\$FM_TUNE_EXTRAS\]", run) is not None


def test_the_cuda_stage_extras_requirements_are_declared_for_uv_builds() -> None:
    pyproject = tomllib.loads(_FM_TUNE_PYPROJECT.read_text(encoding="utf-8"))
    uv_config = pyproject["tool"]["uv"]
    full = pyproject["project"]["optional-dependencies"]["full"]

    assert "flash-attn" in uv_config["extra-build-dependencies"]
    assert any(dep.startswith("flash-attn==") for dep in full)
    assert "sources" not in uv_config, "a pinned wheel URL is back; the Dockerfile could use it"
