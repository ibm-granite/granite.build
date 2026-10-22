"""Pins autotune/trainers/_peft_fsdp.py: a no-grad adapter toggle must not flip requires_grad."""

import pytest
import torch

peft = pytest.importorskip("peft")

from autotune.trainers._peft_fsdp import peft_adapter_toggle_keeps_requires_grad  # noqa: E402


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.lin(x)


def _lora_model():
    torch.manual_seed(0)
    model = peft.get_peft_model(_Toy(), peft.LoraConfig(r=4, target_modules=["lin"], init_lora_weights=False))
    return model


def _lora_flags(model):
    return [p.requires_grad for n, p in model.named_parameters() if "lora_" in n]


def test_stock_peft_flips_requires_grad():
    """The behaviour the patch exists for: disabling under no_grad clears requires_grad."""
    model = _lora_model()
    with torch.no_grad(), model.disable_adapter():
        assert not any(_lora_flags(model))
    assert all(_lora_flags(model))


def test_no_grad_toggle_keeps_requires_grad_and_disables_adapter():
    model = _lora_model()
    x = torch.randn(2, 8)
    with torch.no_grad():
        base_out = model.base_model.model.lin.base_layer(x)
        adapted_out = model(x)
    assert not torch.allclose(base_out, adapted_out)

    with peft_adapter_toggle_keeps_requires_grad():
        with torch.no_grad(), model.disable_adapter():
            assert all(_lora_flags(model))
            assert torch.allclose(model(x), base_out)
        assert all(_lora_flags(model))
        with torch.no_grad():
            assert torch.allclose(model(x), adapted_out)


def test_grad_enabled_toggle_unchanged():
    model = _lora_model()
    with peft_adapter_toggle_keeps_requires_grad():
        with model.disable_adapter():
            assert not any(_lora_flags(model))
        assert all(_lora_flags(model))


def test_original_method_restored():
    from peft.tuners.tuners_utils import BaseTunerLayer

    original = BaseTunerLayer.enable_adapters
    with peft_adapter_toggle_keeps_requires_grad():
        assert BaseTunerLayer.enable_adapters is not original
    assert BaseTunerLayer.enable_adapters is original


def test_driver_wraps_train_in_the_patch():
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "autotune/trainers/driver_multi_trl_fsdp.py"
    assert "peft_adapter_toggle_keeps_requires_grad()" in src.read_text()
