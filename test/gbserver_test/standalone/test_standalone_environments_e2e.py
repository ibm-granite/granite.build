#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end tests for standalone bash and docker environment verification.

These tests verify that standalone builds complete successfully using
bash and docker environments with real workloads: model inference,
GPU checks, TRL fine-tuning, and unitxt evaluation.

Each test uses the direct Build/BuildRun path (same as `gbserver build run`)
with SQLite storage and thread-based build runners.
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.g4os

TEST_DATA_DIR = (
    Path(__file__).parent.parent.parent.parent / "test-data" / "standalone-environments"
)

# Docker test image name
_DOCKER_TEST_IMAGE = "gbserver-test-trl-unitxt:latest"

# Environment overrides for standalone mode
_STANDALONE_ENV = {
    "GB_ENVIRONMENT": "STANDALONE",
    "GBSERVER_METADATA_STORAGE": "sqlite",
    "GBSERVER_DEFAULT_BUILDRUNNER_TYPE": "thread",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _docker_available() -> bool:
    """Check whether Docker daemon is running and the Python docker package is installed."""
    try:
        import docker as _docker_mod  # noqa: F401
    except ImportError:
        return False
    docker_cmd = shutil.which("docker") or shutil.which("podman")
    if not docker_cmd:
        return False
    try:
        result = subprocess.run(
            [docker_cmd, "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _gpu_available() -> bool:
    """Check whether a CUDA GPU is available via nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _ml_packages_available() -> bool:
    """Check whether torch, transformers, trl, and datasets are importable."""
    try:
        import datasets  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401

        return True
    except ImportError:
        return False


def _build_test_docker_image() -> None:
    """Build the TRL/unitxt test Docker image if it does not already exist."""
    docker_cmd = shutil.which("docker") or shutil.which("podman")
    if not docker_cmd:
        pytest.skip("No docker/podman command found")

    # Check if image already exists
    result = subprocess.run(
        [docker_cmd, "image", "inspect", _DOCKER_TEST_IMAGE],
        capture_output=True,
    )
    if result.returncode == 0:
        return  # Image already present

    dockerfile = TEST_DATA_DIR / "docker" / "Dockerfile.test"
    assert dockerfile.is_file(), f"Dockerfile not found at {dockerfile}"

    result = subprocess.run(
        [docker_cmd, "build", "-t", _DOCKER_TEST_IMAGE, "-f", str(dockerfile), "."],
        cwd=str(TEST_DATA_DIR / "docker"),
        capture_output=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"Docker image build failed:\nstdout: {result.stdout.decode()}\n"
        f"stderr: {result.stderr.decode()}"
    )


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

skipif_no_docker = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker/Podman daemon not available",
)

skipif_no_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason="No CUDA GPU available (nvidia-smi not found or failed)",
)

skipif_no_ml = pytest.mark.skipif(
    not _ml_packages_available(),
    reason="ML packages not available (torch, transformers, trl, datasets)",
)


# ---------------------------------------------------------------------------
# Build runner helper
# ---------------------------------------------------------------------------


def _run_build(build_yaml_name: str, timeout: int = 600) -> None:
    """Run a standalone build using the direct Build/BuildRun path.

    This follows the same pattern as test_standalone_e2e.py: create a Space,
    Build, and BuildRun, then run to completion. The build_yaml_name is the
    filename (without directory) under TEST_DATA_DIR/builds/.

    Because the Build class always reads ``build.yaml`` from the build_dir,
    we create a temporary directory that mirrors the space layout and places
    the requested build YAML as ``build.yaml``.

    Args:
        build_yaml_name: Name of the build YAML file under builds/ directory
            (e.g. "bash-inference.yaml").
        timeout: Maximum seconds to wait for the build to complete.

    Raises:
        AssertionError: If the build does not complete with SUCCESS status.
    """
    from gbserver.build.build import Build
    from gbserver.build.buildrun import BuildRun
    from gbserver.build.space import Space
    from gbserver.types.status import Status

    build_yaml_path = TEST_DATA_DIR / "builds" / build_yaml_name
    assert build_yaml_path.is_file(), f"Build YAML not found: {build_yaml_path}"

    # Build expects a directory containing build.yaml.  Our test data keeps
    # per-scenario build YAMLs under builds/.  Create a temp directory that
    # symlinks the space contents (environments, steps, assetstores, space.yaml)
    # and copies the specific build YAML as build.yaml.
    with tempfile.TemporaryDirectory(prefix="gb_env_e2e_") as tmp:
        tmp_path = Path(tmp)

        # Symlink shared space artefacts into the temp directory
        for name in ("environments", "steps", "assetstores", "space.yaml"):
            src = TEST_DATA_DIR / name
            if src.exists():
                os.symlink(src, tmp_path / name)

        # Place the scenario-specific build YAML as build.yaml
        shutil.copy2(build_yaml_path, tmp_path / "build.yaml")

        # The space URI points at the temp dir so space:// URIs resolve
        space_uri = f"file://{tmp_path}"
        space = Space(uri=space_uri, username="standalone-env-test")

        build = Build(
            build_dir=tmp_path,
            space=space,
            username="standalone-env-test",
        )

        build_run = BuildRun(build=build)

        async def _run_with_timeout():
            await asyncio.wait_for(build_run.run_and_wait(), timeout=timeout)

        asyncio.run(_run_with_timeout())

        assert build_run.status == Status.SUCCESS, (
            f"Build '{build_yaml_name}' did not succeed. " f"Status: {build_run.status}"
        )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestStandaloneEnvironmentsE2E:
    """End-to-end tests for standalone bash and docker environments.

    Each test runs a real build using the test-data/standalone-environments/
    space, which includes bash and docker environment definitions plus steps
    for inference, GPU checks, TRL fine-tuning, and unitxt evaluation.
    """

    # -- Inference tests (model download + run) --

    def test_bash_inference(self):
        """Bash environment: download a small model and run inference."""
        with patch.dict(os.environ, _STANDALONE_ENV):
            _run_build("bash-inference.yaml", timeout=600)

    @skipif_no_docker
    def test_docker_inference(self):
        """Docker environment: download a small model and run inference."""
        with patch.dict(os.environ, _STANDALONE_ENV):
            _run_build("docker-inference.yaml", timeout=600)

    # -- GPU tests --

    @skipif_no_gpu
    @skipif_no_ml
    def test_bash_gpu(self):
        """Bash environment: verify GPU is accessible."""
        with patch.dict(os.environ, _STANDALONE_ENV):
            _run_build("bash-gpu.yaml", timeout=120)

    @skipif_no_docker
    @skipif_no_gpu
    @skipif_no_ml
    def test_docker_gpu_passthrough(self):
        """Docker environment: verify GPU passthrough works."""
        with patch.dict(os.environ, _STANDALONE_ENV):
            _run_build("docker-gpu.yaml", timeout=120)

    # -- TRL fine-tuning tests --

    @skipif_no_ml
    def test_bash_trl_finetune(self):
        """Bash environment: TRL fine-tuning with a small model."""
        with patch.dict(os.environ, _STANDALONE_ENV):
            _run_build("bash-trl.yaml", timeout=900)

    @skipif_no_docker
    @skipif_no_ml
    def test_docker_trl_finetune(self):
        """Docker environment: TRL fine-tuning with a small model."""
        _build_test_docker_image()
        with patch.dict(os.environ, _STANDALONE_ENV):
            _run_build("docker-trl.yaml", timeout=900)

    # -- unitxt evaluation tests --

    @skipif_no_ml
    def test_bash_unitxt_eval(self):
        """Bash environment: unitxt evaluation with a small model."""
        with patch.dict(os.environ, _STANDALONE_ENV):
            _run_build("bash-unitxt.yaml", timeout=900)

    @skipif_no_docker
    @skipif_no_ml
    def test_docker_unitxt_eval(self):
        """Docker environment: unitxt evaluation with a small model."""
        _build_test_docker_image()
        with patch.dict(os.environ, _STANDALONE_ENV):
            _run_build("docker-unitxt.yaml", timeout=900)
