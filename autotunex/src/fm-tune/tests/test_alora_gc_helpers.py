"""Tests for autotune.trainers._alora_gc — pure logic on torch modules."""

import torch
import torch.nn as nn

from autotune.trainers._alora_gc import (
    AloraGradCheckpointDrainCallback,
    drain_alora_gc_hooks,
    install_alora_gc_safety_wrapper,
)

_HOOK_ATTR = "_peft_gradient_checkpointing_forward_hooks"


def _attach_fake_hook(module: nn.Module) -> None:
    """Register a real forward pre-hook and stash its handle in the PEFT list."""
    handle = module.register_forward_pre_hook(lambda m, args: None)
    if not hasattr(module, _HOOK_ATTR):
        setattr(module, _HOOK_ATTR, [])
    getattr(module, _HOOK_ATTR).append(handle)


class TestDrainAloraGcHooks:
    def test_empty_model_returns_zero(self):
        model = nn.Linear(4, 4)
        assert drain_alora_gc_hooks(model) == 0

    def test_single_module_with_hooks(self):
        model = nn.Linear(4, 4)
        _attach_fake_hook(model)
        _attach_fake_hook(model)
        drained = drain_alora_gc_hooks(model)
        assert drained == 2
        assert getattr(model, _HOOK_ATTR) == []

    def test_nested_modules(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        for m in model.modules():
            if isinstance(m, nn.Linear):
                _attach_fake_hook(m)
        drained = drain_alora_gc_hooks(model)
        assert drained == 2

    def test_idempotent_on_second_call(self):
        model = nn.Linear(4, 4)
        _attach_fake_hook(model)
        drain_alora_gc_hooks(model)
        # Second call: nothing left to drain
        assert drain_alora_gc_hooks(model) == 0

    def test_handles_already_removed(self):
        model = nn.Linear(4, 4)
        h = model.register_forward_pre_hook(lambda m, args: None)
        h.remove()
        # Place an already-removed handle in the list — should be tolerated
        setattr(model, _HOOK_ATTR, [h])
        drained = drain_alora_gc_hooks(model)
        assert drained == 1
        assert getattr(model, _HOOK_ATTR) == []


class TestInstallSafetyWrapper:
    def test_sentinel_set(self):
        model = nn.Linear(4, 4)
        install_alora_gc_safety_wrapper(model)
        assert getattr(model, "_alora_gc_safety_installed", False) is True

    def test_idempotent(self):
        model = nn.Linear(4, 4)
        install_alora_gc_safety_wrapper(model)
        # Count pre-hooks before second call
        n_hooks = len(model._forward_pre_hooks)
        install_alora_gc_safety_wrapper(model)
        # No new hooks registered on second call
        assert len(model._forward_pre_hooks) == n_hooks

    def test_none_model_noop(self):
        # Should not raise
        install_alora_gc_safety_wrapper(None)

    def test_drain_called_on_no_grad_forward(self):
        model = nn.Linear(4, 4)
        install_alora_gc_safety_wrapper(model)
        # Attach a fake leaked hook
        _attach_fake_hook(model)
        assert len(getattr(model, _HOOK_ATTR)) == 1
        # Run a no_grad forward → wrapper should drain
        with torch.no_grad():
            model(torch.zeros(1, 4))
        # The leaked hook list is drained (other hook lists may still exist)
        assert getattr(model, _HOOK_ATTR) == []

    def test_does_not_drain_under_grad(self):
        model = nn.Linear(4, 4)
        install_alora_gc_safety_wrapper(model)
        _attach_fake_hook(model)
        # Run a normal grad-enabled forward → should NOT drain
        model(torch.zeros(1, 4))
        # The leaked hook list is preserved
        assert len(getattr(model, _HOOK_ATTR)) == 1


class TestAloraGradCheckpointDrainCallback:
    def test_on_step_begin_drains(self):
        cb = AloraGradCheckpointDrainCallback()
        model = nn.Linear(4, 4)
        _attach_fake_hook(model)
        cb.on_step_begin(args=None, state=None, control=None, model=model)
        assert getattr(model, _HOOK_ATTR) == []

    def test_on_step_end_drains(self):
        cb = AloraGradCheckpointDrainCallback()
        model = nn.Linear(4, 4)
        _attach_fake_hook(model)
        cb.on_step_end(args=None, state=None, control=None, model=model)
        assert getattr(model, _HOOK_ATTR) == []

    def test_on_evaluate_drains(self):
        cb = AloraGradCheckpointDrainCallback()
        model = nn.Linear(4, 4)
        _attach_fake_hook(model)
        cb.on_evaluate(args=None, state=None, control=None, model=model)
        assert getattr(model, _HOOK_ATTR) == []

    def test_callback_handles_none_model(self):
        cb = AloraGradCheckpointDrainCallback()
        # Should not raise
        cb.on_step_begin(args=None, state=None, control=None, model=None)
        cb.on_evaluate(args=None, state=None, control=None, model=None)
