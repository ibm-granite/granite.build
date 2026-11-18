# SPDX-License-Identifier: Apache-2.0

"""Fixtures shared by the fmtune_run.sh unit tests."""

import os
import shutil
import sys

import pytest
from _fmtune_helpers import (
    _STUB_MAIN,
)


@pytest.fixture
def env(tmp_path):
    """A minimal valid environment: stub fm-tune tree, dataset, output, cache."""
    if shutil.which("bash") is None:  # pragma: no cover - bash is present in CI
        pytest.skip("bash not available")
    src = tmp_path / "fm-tune"
    (src / "autotune" / "configs").mkdir(parents=True)
    (src / "autotune" / "configs" / "autotune.yaml").write_text("tune_config: {}\n")
    (src / "main.py").write_text(_STUB_MAIN)
    (src / "pyproject.toml").write_text("[project]\nname='autotune'\n")
    data = tmp_path / "data dir"  # a space checks the quoting
    data.mkdir()
    (data / "d_train.jsonl").write_text("{}\n")
    (data / "d_validation.jsonl").write_text("{}\n")
    return {
        "PATH": os.environ["PATH"],
        "FMTUNE_SKIP_INSTALL": "1",
        "FMTUNE_PYTHON": sys.executable,
        "FMTUNE_SOURCE_PATH": str(src),
        "FMTUNE_DATASET_DIR": str(data),
        "FMTUNE_TRAIN_FILE": "d_train.jsonl",
        "FMTUNE_VALIDATION_FILE": "d_validation.jsonl",
        "FMTUNE_MODEL": "org/tiny",
        "FMTUNE_RUN_NAME": "run1",
        "FMTUNE_TUNING_ALGO": "lora",
        "FMTUNE_OUTPUT_DIR": str(tmp_path / "out"),
        "FMTUNE_CACHE_DIR": str(tmp_path / "cache"),
        "STUB_RECORD": str(tmp_path / "record.json"),
    }
