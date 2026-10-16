"""Tests for autotune.trainers._resume.peft_adapter_load_on_cpu.

The context manager patches PEFT's ``infer_device`` (imported by name into three
modules) so a resumed adapter load lands on CPU instead of the bare ``"cuda"``
(== cuda:0), which collides across ranks under ``-gpu mode=exclusive_process``.

These tests are pure: they inject fake ``peft`` modules into ``sys.modules`` so
no real PEFT / Torch / Ray is required.
"""

import sys
import types

import pytest

from autotune.trainers._resume import peft_adapter_load_on_cpu

_PEFT_MODULES = ["peft", "peft.peft_model", "peft.utils", "peft.utils.other", "peft.utils.save_and_load"]
_PATCHED_MODULES = ["peft.peft_model", "peft.utils.save_and_load", "peft.utils.other"]


@pytest.fixture
def fake_peft(monkeypatch):
    """Install fake peft modules whose infer_device returns 'cuda'."""
    mods = {}
    for name in _PEFT_MODULES:
        mod = types.ModuleType(name)
        # Every module that the helper patches exposes infer_device.
        if name in _PATCHED_MODULES:
            mod.infer_device = lambda: "cuda"
        mods[name] = mod
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return mods


def test_forces_cpu_inside_and_restores_after(fake_peft):
    originals = {name: fake_peft[name].infer_device for name in _PATCHED_MODULES}

    with peft_adapter_load_on_cpu():
        for name in _PATCHED_MODULES:
            assert fake_peft[name].infer_device() == "cpu"

    # Restored to the exact original objects (identity).
    for name in _PATCHED_MODULES:
        assert fake_peft[name].infer_device is originals[name]
        assert fake_peft[name].infer_device() == "cuda"


def test_restores_on_exception(fake_peft):
    originals = {name: fake_peft[name].infer_device for name in _PATCHED_MODULES}

    with pytest.raises(RuntimeError):
        with peft_adapter_load_on_cpu():
            assert fake_peft["peft.peft_model"].infer_device() == "cpu"
            raise RuntimeError("boom")

    for name in _PATCHED_MODULES:
        assert fake_peft[name].infer_device is originals[name]


def test_noop_when_peft_absent(monkeypatch):
    # Ensure peft is not importable.
    for name in _PEFT_MODULES:
        monkeypatch.setitem(sys.modules, name, None)  # None -> ImportError on import

    # Must not raise and must yield control.
    entered = False
    with peft_adapter_load_on_cpu():
        entered = True
    assert entered
