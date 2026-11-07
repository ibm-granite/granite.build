"""Pins autotune.device.disable_cudnn_sdpa and that every training driver calls it."""

import pathlib
import re

import pytest

from autotune.device import disable_cudnn_sdpa

ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVERS = [
    "autotune/trainers/driver_single.py",
    "autotune/trainers/driver_single_trl.py",
    "autotune/trainers/driver_multi_hf_ds.py",
    "autotune/trainers/driver_multi_hf_fsdp.py",
    "autotune/trainers/driver_multi_trl_ds.py",
    "autotune/trainers/driver_multi_trl_fsdp.py",
]


def test_is_a_noop_without_cuda():
    # Idempotent and safe to call on CPU / MPS boxes (the test host).
    disable_cudnn_sdpa()
    disable_cudnn_sdpa()


@pytest.mark.parametrize("path", DRIVERS)
def test_worker_entry_calls_it(path):
    """The flag is per process, so it has to run where the backward pass runs."""
    src = (ROOT / path).read_text()
    entry = "train_loop_per_worker" if "driver_multi" in path else "train_driver_single_gpu"
    body = src[src.index(f"def {entry}(") :]
    end = re.search(r"\n(def |class )", body[1:])  # None when it is the last top-level def
    body = body if end is None else body[: end.start() + 1]
    assert "disable_cudnn_sdpa()" in body, f"{path}: {entry} does not call disable_cudnn_sdpa()"
