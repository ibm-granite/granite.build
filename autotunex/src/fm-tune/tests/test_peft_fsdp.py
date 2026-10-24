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
    from peft.utils.other import ModulesToSaveWrapper

    originals = [BaseTunerLayer.enable_adapters, ModulesToSaveWrapper.enable_adapters]
    with peft_adapter_toggle_keeps_requires_grad():
        assert BaseTunerLayer.enable_adapters is not originals[0]
        assert ModulesToSaveWrapper.enable_adapters is not originals[1]
    assert [BaseTunerLayer.enable_adapters, ModulesToSaveWrapper.enable_adapters] == originals


class _ToyWithHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)
        self.head = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.head(self.lin(x))


def _modules_to_save_flags(model):
    return [p.requires_grad for n, p in model.named_parameters() if "modules_to_save" in n]


def test_modules_to_save_keeps_requires_grad_under_no_grad():
    """ModulesToSaveWrapper has its own enable_adapters; it must not flip requires_grad either."""
    torch.manual_seed(0)
    model = peft.get_peft_model(_ToyWithHead(), peft.LoraConfig(r=4, target_modules=["lin"], modules_to_save=["head"]))
    assert _modules_to_save_flags(model) and all(_modules_to_save_flags(model))
    with torch.no_grad(), model.disable_adapter():
        assert not any(_modules_to_save_flags(model))  # stock PEFT flips it
    # Make the trained copy differ from the original head so the forward shows which one ran.
    wrapper = model.base_model.model.head
    with torch.no_grad():
        wrapper.modules_to_save["default"].weight.add_(1.0)
        x = torch.randn(2, 8)
        base_out = wrapper.original_module(model.base_model.model.lin.base_layer(x))
    with peft_adapter_toggle_keeps_requires_grad():
        with torch.no_grad(), model.disable_adapter():
            assert all(_modules_to_save_flags(model))
            assert torch.allclose(model(x), base_out)  # reference pass uses the original head
        assert all(_modules_to_save_flags(model))
        with torch.no_grad():
            assert not torch.allclose(model(x), base_out)


def test_driver_wraps_train_in_the_patch():
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "autotune/trainers/driver_multi_trl_fsdp.py"
    assert "peft_adapter_toggle_keeps_requires_grad()" in src.read_text()
