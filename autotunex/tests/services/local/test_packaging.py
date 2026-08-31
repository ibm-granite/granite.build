"""Tests for the local artifact-packaging helpers."""

from __future__ import annotations

import math
import zipfile
from pathlib import Path

from autotunex.services.local import packaging


def test_parse_result_maps_loss_metric() -> None:
    out = packaging.parse_result(
        {"metric": "loss", "loss": 0.5, "train_loss": 0.4, "time_total_s": 12.0}
    )

    assert out == {"loss": 0.5, "train_loss": 0.4, "total_time": 12.0}


def test_parse_result_nan_becomes_none() -> None:
    out = packaging.parse_result(
        {"metric": "loss", "loss": math.nan, "train_loss": None, "time_total_s": math.nan}
    )

    assert out == {"loss": None, "train_loss": None, "total_time": None}


def test_generate_bash_script_is_executable(tmp_path: Path) -> None:
    packaging.generate_bash_script(tmp_path)

    script = tmp_path / "run_model.sh"
    assert script.exists() and (script.stat().st_mode & 0o111)


def test_write_inference_script_alora_branch(tmp_path: Path) -> None:
    packaging.write_inference_script("base", "model", tmp_path, for_alora=True)

    assert (tmp_path / "inference.py").exists()


def test_zip_folder_produces_readable_archive(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("hi")

    out = packaging.zip_folder(src, "src.zip", tmp_path / "out")

    with zipfile.ZipFile(out) as z:
        assert any(n.endswith("a.txt") for n in z.namelist())
