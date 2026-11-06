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
_BASE_DOCKERFILE = Path(__file__).resolve().parents[1] / "docker" / "cuda-base" / "Dockerfile"
_FM_TUNE_PYPROJECT = Path(__file__).resolve().parents[1] / "src" / "fm-tune" / "pyproject.toml"
_KERNELS = ("flash-attn", "mamba-ssm", "causal-conv1d")


def _dockerfile() -> str:
    """Return the root Dockerfile's text."""
    return _DOCKERFILE.read_text(encoding="utf-8")


def _base_dockerfile() -> str:
    """Return the CUDA kernel base's Dockerfile text."""
    return _BASE_DOCKERFILE.read_text(encoding="utf-8")


def _instructions(content: str) -> str:
    """Return ``content`` without its comment lines."""
    return "\n".join(line for line in content.splitlines() if not line.lstrip().startswith("#"))


def _fm_tune_pins() -> dict[str, str]:
    """Return ``name -> version`` for every ``==`` pin in fm-tune's [core] and [full] extras."""
    extras = tomllib.loads(_FM_TUNE_PYPROJECT.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    return {
        re.split(r"[=;<>\[ ]", dep, maxsplit=1)[0]: dep.split("==")[1].split(";")[0].strip()
        for dep in extras["core"] + extras["full"]
        if "==" in dep
    }


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


def test_no_dockerfile_installs_a_prebuilt_kernel_wheel() -> None:
    # flash-attn, mamba-ssm and causal-conv1d compile from source in the kernel base, so
    # the default build runs only upstream code built by a Dockerfile anyone can review.
    for dockerfile in (_dockerfile(), _base_dockerfile()):
        assert "FLASH_ATTN_WHEEL" not in dockerfile
        assert ".whl" not in dockerfile
        assert "--require-hashes" not in dockerfile
    assert not (_DOCKERFILE.parent / "docker" / "cuda-kernel-wheels.txt").exists()


def test_the_cuda_stage_starts_from_the_digest_pinned_kernel_base() -> None:
    # The published kernel base, pinned by digest so its tag cannot move under this build.
    match = re.search(r"^ARG CUDA_KERNEL_BASE=(\S+)", _dockerfile(), re.MULTILINE)

    assert match is not None
    assert re.fullmatch(
        r"quay\.io/granitebuild/autotunex/cuda-base:\S+@sha256:[0-9a-f]{64}", match.group(1)
    )
    assert re.search(r"^FROM \$\{CUDA_KERNEL_BASE\} AS cuda$", _dockerfile(), re.MULTILINE)


def test_the_kernel_base_digest_is_a_published_one() -> None:
    # A placeholder digest (one repeated character) is not a pullable image: every
    # DEVICE=cuda build would fail at FROM. Pin the digest make publish-base-image prints.
    match = re.search(
        r"^ARG CUDA_KERNEL_BASE=\S+@sha256:([0-9a-f]{64})$", _dockerfile(), re.MULTILINE
    )

    assert match is not None
    assert len(set(match.group(1))) > 1, "CUDA_KERNEL_BASE still pins a placeholder digest"


def test_the_kernel_base_tag_is_the_one_the_makefile_publishes() -> None:
    makefile = (_DOCKERFILE.parent / "Makefile").read_text(encoding="utf-8")
    version = re.search(r"^CUDA_BASE_IMAGE_VERSION \?= (\S+)$", makefile, re.MULTILINE)
    tag = re.search(r"^ARG CUDA_KERNEL_BASE=[^:]+:([^@]+)@", _dockerfile(), re.MULTILINE)

    assert version is not None and tag is not None
    assert tag.group(1) == version.group(1)


def test_the_kernel_base_is_a_digest_pinned_cuda_13_devel_image() -> None:
    # torch 2.11's PyPI wheels are built for CUDA 13.0 and the kernels compile with the
    # image's nvcc, so the base must be a CUDA 13 -devel tag (which ships nvcc). The digest
    # keeps a published image reproducible if the tag is re-pushed upstream.
    match = re.search(r"^ARG CUDA_BASE=(\S+)", _base_dockerfile(), re.MULTILINE)

    assert match is not None
    assert re.fullmatch(
        r"nvidia/cuda:13\.\d+\.\d+-devel-ubuntu24\.04@sha256:[0-9a-f]{64}", match.group(1)
    )


def test_the_kernel_base_pins_match_fm_tune() -> None:
    # The app build installs fm-tune's extras on top of the base and refuses to compile
    # the kernels, so a pin the base does not match would fail that build.
    install = _instructions(_base_dockerfile())
    pins = re.findall(r'"([a-z0-9-]+)==([^"]+)"', install)
    expected = _fm_tune_pins()

    assert {name for name, _ in pins} == {"torch", *_KERNELS}
    for name, version in pins:  # every occurrence: torch is named in both installs
        assert version == expected[name], f"{name}: base {version}, fm-tune {expected[name]}"


def test_the_kernel_install_names_torch_for_match_runtime() -> None:
    # With --no-deps, uv's match-runtime takes the build-time torch only from the packages
    # named in the same install; without torch there, the kernels' builds fail.
    install = _instructions(_base_dockerfile())
    kernel_install = install.split("--no-deps", 1)[1].split("&&", 1)[0]

    for name in ("torch", *_KERNELS):
        assert f'"{name}==' in kernel_install, f"{name} is not in the --no-deps kernel install"


def test_the_kernel_base_records_the_torch_it_compiled_against() -> None:
    # The app build compares this with the installed torch and refuses a mismatch.
    match = re.search(r"^ENV AUTOTUNEX_KERNEL_BASE_TORCH=(\S+)$", _base_dockerfile(), re.MULTILINE)

    assert match is not None
    assert match.group(1) == _fm_tune_pins()["torch"]
    assert "AUTOTUNEX_KERNEL_BASE_TORCH" in _fm_tune_install_run()


def test_the_kernel_base_builds_from_fm_tunes_uv_recipe() -> None:
    # uv reads [tool.uv.extra-build-dependencies] (torch matched to the runtime env) and
    # [tool.uv.extra-build-variables] (arch list, MAX_JOBS) only from the pyproject.toml in
    # its working directory; pip does not read them at all.
    install = _instructions(_base_dockerfile())

    assert "COPY src/fm-tune/pyproject.toml /opt/kernel-build/pyproject.toml" in install
    assert "cd /opt/kernel-build" in install
    assert install.index("cd /opt/kernel-build") < install.index("uv pip install")


def test_the_app_build_refuses_to_compile_the_kernels() -> None:
    # The kernels come from the base; a stale base must fail fast, not compile for ~35 min.
    run = _fm_tune_install_run()

    for kernel in _KERNELS:
        assert f"--only-binary {kernel}" in run


def test_the_extras_install_goes_through_uv_from_the_fm_tune_directory() -> None:
    # pip ignores [tool.uv.extra-build-dependencies], where fm-tune declares how to build
    # flash-attn / mamba-ssm against the runtime torch; uv reads it only from the project dir.
    run = _fm_tune_install_run()

    assert "cd src/fm-tune" in run
    assert re.search(r"uv pip install .*\[\$FM_TUNE_EXTRAS\]", run, re.DOTALL) is not None


def test_the_cuda_stage_extras_requirements_are_declared_for_uv_builds() -> None:
    pyproject = tomllib.loads(_FM_TUNE_PYPROJECT.read_text(encoding="utf-8"))
    uv_config = pyproject["tool"]["uv"]
    full = pyproject["project"]["optional-dependencies"]["full"]

    assert "flash-attn" in uv_config["extra-build-dependencies"]
    assert any(dep.startswith("flash-attn==") for dep in full)
    assert "sources" not in uv_config, "a pinned wheel URL is back; the Dockerfile could use it"


def test_the_cpu_image_never_installs_the_full_extras() -> None:
    # The extras install is guarded on FM_TUNE_EXTRAS != core, which only the CUDA stage changes.
    assert "FM_TUNE_EXTRAS=core" in _stage_body(_dockerfile(), "cpu")
    assert '[ "$FM_TUNE_EXTRAS" != "core" ]' in _fm_tune_install_run()


def test_every_copy_source_exists_in_the_build_context() -> None:
    # Both Dockerfiles build with autotunex/ as the context, so a COPY of a file that only
    # exists at the repo root (LICENSE, removed from autotunex/ in favour of the root one)
    # fails the build.
    root = _DOCKERFILE.parent
    for dockerfile in (_dockerfile(), _base_dockerfile()):
        for line in dockerfile.splitlines():
            if not line.startswith("COPY ") or "--from=" in line:
                continue
            for source in line.split()[1:-1]:
                assert (root / source).exists(), f"{line!r}: {source} is not in the build context"


def test_the_kernel_base_installs_the_python_headers_the_kernels_compile_against() -> None:
    # flash-attn, mamba-ssm and causal-conv1d include <Python.h> through torch's headers;
    # without python3.12-dev their C++ step fails with "Python.h: No such file or directory".
    install = (
        _instructions(_base_dockerfile()).split("apt-get install", 1)[1].split("&& rm -rf", 1)[0]
    )

    assert "python3.12-dev" in install.split()


def test_the_kernel_base_installs_the_kernels_runtime_dependencies_from_fm_tunes_lock() -> None:
    # The kernels compile with --no-deps, so without this `import mamba_ssm` (einops,
    # transformers) and `import flash_attn` (einops) fail on the base on its own, which is
    # the image that is published and smoke-tested. Constraining to fm-tune's lock keeps
    # those dependencies at the versions the app's [full] install would choose anyway.
    install = _instructions(_base_dockerfile())
    deps_run = install.split("src/fm-tune/uv.lock /opt/kernel-deps/", 1)[1].split("\nENV ", 1)[0]

    assert "uv export --frozen" in deps_run
    assert "-c " in deps_run
    for kernel in _KERNELS:
        assert f"--only-binary {kernel}" in deps_run, f"{kernel} could be recompiled here"
    assert "--no-deps" not in deps_run
    # A separate layer after the compile, so a lock change does not recompile the kernels.
    assert install.index("--no-deps") < install.index("src/fm-tune/uv.lock /opt/kernel-deps/")
