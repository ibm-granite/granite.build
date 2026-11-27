# SPDX-License-Identifier: Apache-2.0

"""The image's venv holds the versions the Dockerfile installs on top of the lock.

The post-`uv sync` installs (clearml and the security pins) are pinned on the
command line, outside upstream's lock, so nothing else checks what they resolved
to. They skip where the image's venv is absent (the repo venv). Run them in the
image, with this step directory mounted, after a build:

    podman run --rm --platform linux/amd64 --entrypoint bash \\
      -v "$PWD:/step:ro" <IMAGE_REF> \\
      -c 'python -m pytest -p no:cacheprovider /step/test/test_image_pins_in_image.py'

OPEN_INSTRUCT_HOME (default /opt/open-instruct) points them at another checkout.
"""

import importlib.metadata
import os
import pathlib
import re
import sys

import pytest

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_OI_HOME = pathlib.Path(os.environ.get("OPEN_INSTRUCT_HOME", "/opt/open-instruct"))
_VENV = _OI_HOME / ".venv"

# Every "name==version" the Dockerfile passes to `uv pip install`.
_PINS = dict(
    re.findall(
        r'"([A-Za-z0-9_.-]+)==([0-9][^"]*)"', (_STEP_DIR / "Dockerfile").read_text()
    )
)

pytestmark = pytest.mark.skipif(
    not _VENV.is_dir(), reason=f"no Open-Instruct venv at {_VENV}"
)


def test_runs_in_the_image_venv():
    assert pathlib.Path(sys.prefix).resolve() == _VENV.resolve()


def test_pins_parsed_from_the_dockerfile():
    assert _PINS.get("wandb") == "0.31.2"
    assert "clearml" in _PINS


@pytest.mark.parametrize("pkg", sorted(_PINS))
def test_pinned_version_installed(pkg):
    assert importlib.metadata.version(pkg) == _PINS[pkg]


def test_vllm_removed():
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        importlib.metadata.version("vllm")
