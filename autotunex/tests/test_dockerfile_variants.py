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


def _fm_tune_install_run() -> str:
    """Return the ``RUN`` that installs fm-tune, from after its ``[core]`` install."""
    run = _dockerfile().split('RUN pip install --no-cache-dir "./src/fm-tune[core]"', 1)[1]
    return run.split("\n\n", 1)[0]


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
    # fm-tune no longer pins a flash-attn wheel in pyproject: the prebuilt kernel wheels
    # live in docker/cuda-kernel-wheels.txt, behind the CUDA_KERNEL_WHEELS switch.
    assert "FLASH_ATTN_WHEEL" not in _dockerfile()


def test_the_cuda_base_is_a_cuda_13_devel_image() -> None:
    # torch 2.11's PyPI wheels are built for CUDA 13.0, and with the prebuilt kernel wheels
    # turned off the kernels compile with the image's nvcc, so the base must be a CUDA 13
    # -devel tag (which ships nvcc).
    match = re.search(r"^ARG CUDA_BASE=(\S+)", _dockerfile(), re.MULTILINE)

    assert match is not None
    assert re.fullmatch(r"nvidia/cuda:13\.\d+\.\d+-devel-ubuntu24\.04", match.group(1))


def test_the_extras_install_goes_through_uv_from_the_fm_tune_directory() -> None:
    # pip ignores [tool.uv.extra-build-dependencies], where fm-tune declares how to build
    # flash-attn / mamba-ssm against the runtime torch; uv reads it only from the project dir.
    run = _fm_tune_install_run()

    assert "cd src/fm-tune" in run
    assert re.search(r"uv pip install .*\[\$FM_TUNE_EXTRAS\]", run) is not None


def test_the_cuda_stage_extras_requirements_are_declared_for_uv_builds() -> None:
    pyproject = tomllib.loads(_FM_TUNE_PYPROJECT.read_text(encoding="utf-8"))
    uv_config = pyproject["tool"]["uv"]
    full = pyproject["project"]["optional-dependencies"]["full"]

    assert "flash-attn" in uv_config["extra-build-dependencies"]
    assert any(dep.startswith("flash-attn==") for dep in full)
    assert "sources" not in uv_config, "a pinned wheel URL is back; the Dockerfile could use it"


_CUDA_WHEELS = Path(__file__).resolve().parents[1] / "docker" / "cuda-kernel-wheels.txt"
_WHEEL_LINE = re.compile(
    r"^(?P<name>[\w.-]+) @ (?P<url>https://\S+/(?P<file>[^/\s]+\.whl))"
    r" --hash=sha256:(?P<sha>[0-9a-f]{64})$"
)


def _pinned_wheels() -> list[re.Match[str]]:
    """Return the match for every requirement line in the CUDA wheels file."""
    lines = [
        line.strip()
        for line in _CUDA_WHEELS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    matches = [_WHEEL_LINE.match(line) for line in lines]
    assert all(matches), (
        f"a line in {_CUDA_WHEELS.name} is not 'name @ https-url.whl --hash=sha256:<64 hex>'"
    )
    return [m for m in matches if m is not None]


def test_every_cuda_wheel_is_pinned_by_sha256() -> None:
    wheels = _pinned_wheels()

    assert {m["name"] for m in wheels} == {"flash-attn", "mamba-ssm", "causal-conv1d"}


def test_the_cuda_wheel_versions_match_fm_tunes_full_extra() -> None:
    full = tomllib.loads(_FM_TUNE_PYPROJECT.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]["full"]
    pinned = {
        re.split(r"[=;<>\[ ]", dep, maxsplit=1)[0]: dep.split("==")[1].split(";")[0].strip()
        for dep in full
        if "==" in dep
    }

    for wheel in _pinned_wheels():
        filename_version = wheel["file"].split("-")[1]
        name, expected = wheel["name"], pinned[wheel["name"]]
        assert filename_version == expected, (
            f"{name}: wheel {filename_version}, [full] pins {expected}"
        )


def test_the_cuda_wheels_are_cp312_linux_x86_64_like_the_image() -> None:
    for wheel in _pinned_wheels():
        assert wheel["file"].endswith("-cp312-cp312-linux_x86_64.whl"), wheel["file"]


def test_the_cuda_wheels_all_come_from_one_release() -> None:
    releases = {m["url"].rsplit("/", 1)[0] for m in _pinned_wheels()}

    assert len(releases) == 1, f"the wheels are spread over several releases: {sorted(releases)}"
    assert "cu130-torch2" in next(iter(releases)), (
        "the release tag should name the CUDA and torch it was built for"
    )


def test_the_cuda_image_installs_the_wheels_hash_checked_without_deps() -> None:
    body = _fm_tune_install_run()

    assert "--require-hashes" in body
    assert "--no-deps" in body
    assert "-r /tmp/cuda-kernel-wheels.txt" in body
    assert "COPY docker/cuda-kernel-wheels.txt /tmp/cuda-kernel-wheels.txt" in _dockerfile()


def test_the_wheels_install_after_core_so_they_bind_to_its_torch() -> None:
    body = _fm_tune_install_run()

    assert body.index("--require-hashes") < body.index("uv pip install")


def test_the_cpu_image_never_installs_the_cuda_wheels() -> None:
    # The wheel install is guarded on FM_TUNE_EXTRAS != core, which only the CUDA stage changes.
    assert "FM_TUNE_EXTRAS=core" in _stage_body(_dockerfile(), "cpu")
    body = _fm_tune_install_run()
    assert '[ "$FM_TUNE_EXTRAS" != "core" ] && [ -n "$CUDA_KERNEL_WHEELS" ]' in body


def test_the_cuda_kernel_wheels_switch_defaults_on() -> None:
    # Empty turns the prebuilt wheels off, so the default must be a non-empty value.
    match = re.search(r"^ARG CUDA_KERNEL_WHEELS=(\S*)$", _dockerfile(), re.MULTILINE)

    assert match is not None
    assert match.group(1)
