#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Smoke-check a built lm-engine image from inside it. CPU-safe by design: it imports
# and prints versions, and never touches a GPU, so it runs on a GPU-less build host.
#   podman run --rm <image-ref> bash /app/check-image.sh      (make check-image)
set -euo pipefail

python - <<'PY'
import importlib
import subprocess

# patch 0001 restores this module, patch 0002 routes the FSDP/IBM datasets to it
from lm_engine.training.data.ibm import get_ibm_dataloaders  # noqa: F401
from lm_engine.training.models import GPTBaseConfig  # noqa: F401

# the export step imports this; it lives in lm-engine's tools/ directory, which is NOT
# part of the installed package -- it is reachable only because /app/src is on PYTHONPATH
import tools.model_conversion  # noqa: F401

sha = subprocess.check_output(["git", "-C", "/app/src", "rev-parse", "HEAD"], text=True).strip()
print("lm-engine", sha)
for m in ("torch", "flash_attn", "transformers"):
    print(m, importlib.import_module(m).__version__)
PY

# Only that the binaries are present. ibv_devinfo is NOT run: a build host with no RDMA
# device would fail it, and the image's job is to carry the userspace, not to have a NIC.
for exe in gcc rsync ibv_devinfo; do
  command -v "$exe" >/dev/null || { echo "missing: $exe" >&2; exit 1; }
done

echo "image-check: ok"
