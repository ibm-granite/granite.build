# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Portions of this file are adapted from HuggingFace `peft` (Apache-2.0),
# Copyright The HuggingFace Team — specifically
# ALoraLinearVariant.forward in peft.tuners.lora.variants.
"""aLoRA + gradient checkpointing safety helpers.

PEFT 0.18.0 (PR #2860, fixes huggingface/peft#2826) re-injects `alora_offsets`
during gradient-checkpointing's recomputed forward by registering hooks on
each `GradientCheckpointingLayer`. The hook handles are accumulated in
`layer._peft_gradient_checkpointing_forward_hooks` and drained by a backward
hook. Re-entering `_enable_peft_forward_hooks` while that list is non-empty
raises:

    ValueError: Multiple invocations of PEFT forward hooks before .backward()
                with enabled gradient checkpointing.

That guard fires under HF Trainer's eval loop: eval forwards run under
`torch.no_grad()`, so the backward hook never executes and the list never
drains. The next training forward then trips the check.

A second PEFT 0.18.0 bug surfaces under DPO + gradient checkpointing:
`ALoraLinearVariant.forward` (peft/tuners/lora/variants.py:591) creates a
fallback bool mask via `torch.zeros((B, T), dtype=torch.bool)` without a
`device=` argument when `alora_offsets is None`, producing a CPU tensor while
the rest of the forward runs on cuda. PyTorch's checkpoint recomputation then
trips `CheckpointError: Recomputed values ... different metadata` on backward.
We patch that one line in-place at module import.

This module provides:

  - `drain_alora_gc_hooks(model)` — walk modules and clear leaked hooks.
  - `AloraGradCheckpointDrainCallback` — TrainerCallback that drains at
    train/eval boundaries.
  - `install_alora_gc_safety_wrapper(model)` — wraps the model's forward to
    drain whenever called under no_grad, covering the inside-eval-loop window
    between callback events.

All callable utilities are idempotent and gated by callers on
`peft_type == "ALORA"`. The PEFT monkeypatch is applied at import time and
self-guards against double-application.
"""

from __future__ import annotations

import logging
import types

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


def _patch_peft_alora_zeros_device() -> None:
    """Fix peft.tuners.lora.variants.ALoraLinearVariant.forward to allocate
    its fallback mask on the same device as `result`.

    The upstream code at line 591 does:
        if alora_offsets is None:
            mask = torch.zeros((B, T), dtype=torch.bool)
    which lands on CPU. Under gradient checkpointing the original forward
    produces a cuda mask (via the else branch), and the recomputed forward
    can fall through to this branch (e.g. when alora_offsets is dropped),
    yielding a metadata mismatch that breaks backward. We replace the method
    with a corrected version. Idempotent via a sentinel attribute.
    """
    try:
        from peft.tuners.lora.variants import ALoraLinearVariant
    except Exception:
        return

    if getattr(ALoraLinearVariant, "_fmtune_zeros_device_patched", False):
        return

    @staticmethod
    def forward(module, active_adapter, x, result, **kwargs):
        alora_offsets = kwargs.get("alora_offsets", None)
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]
        x = x.to(lora_A.weight.dtype)
        result_shape = result.shape
        B = result_shape[0]
        T = result_shape[1] if len(result_shape) == 3 else 1
        D = result_shape[-1]
        Dx = x.shape[-1]
        device = result.device
        if alora_offsets is None:
            mask = torch.zeros((B, T), dtype=torch.bool, device=device)
        else:
            offsets = torch.tensor(
                [0 if o is None else min(int(o), T) for o in alora_offsets],
                device=device,
                dtype=torch.long,
            )
            pos = torch.arange(T, device=device).unsqueeze(0)
            mask = pos >= (T - offsets).unsqueeze(1)

        x_flat = x.view(-1, Dx)
        res_flat = result.view(-1, D)
        mask_flat = mask.view(-1)
        res_flat[mask_flat] += lora_B(lora_A(dropout(x_flat[mask_flat]))) * scaling
        return result

    ALoraLinearVariant.forward = forward
    ALoraLinearVariant._fmtune_zeros_device_patched = True
    logger.info(
        "[autotune] Patched peft.tuners.lora.variants.ALoraLinearVariant.forward "
        "to allocate its fallback mask on result.device (workaround for upstream "
        "PEFT 0.18 bug)."
    )


_patch_peft_alora_zeros_device()

_HOOK_ATTR = "_peft_gradient_checkpointing_forward_hooks"
_WRAPPER_SENTINEL = "_alora_gc_safety_installed"


def drain_alora_gc_hooks(model: torch.nn.Module) -> int:
    """Remove and clear any leaked aLoRA gradient-checkpointing forward hooks.

    Walks `model.modules()` and, for each module carrying a non-empty list at
    the attribute name PEFT uses, calls `.remove()` on every handle and
    empties the list. Idempotent.

    Returns the number of handles drained.
    """
    drained = 0
    for module in model.modules():
        handles = getattr(module, _HOOK_ATTR, None)
        if not handles:
            continue
        while handles:
            handle = handles.pop()
            try:
                handle.remove()
            except Exception:
                pass
            drained += 1
    return drained


class AloraGradCheckpointDrainCallback(TrainerCallback):
    """Drain leaked aLoRA gradient-checkpointing hooks at train/eval boundaries.

    `on_step_begin` and `on_step_end` cover the boundary between an eval
    pass and the next training step (eval can fire in either, depending on
    `eval_strategy`). `on_prediction_step` runs after each eval mini-batch's
    forward, draining hooks before the next eval forward. `on_evaluate` and
    `on_predict` cover the post-loop boundary. The forward pre-hook
    installed by `install_alora_gc_safety_wrapper` is the actual safety net
    for inside-eval forwards; this callback is defense in depth.
    """

    def _drain(self, model):
        if model is not None:
            drain_alora_gc_hooks(model)

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        self._drain(model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        self._drain(model)

    def on_prediction_step(self, args, state, control, model=None, **kwargs):
        self._drain(model)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        self._drain(model)

    def on_predict(self, args, state, control, model=None, **kwargs):
        self._drain(model)


def install_alora_gc_safety_wrapper(model: torch.nn.Module) -> None:
    """Drain leaked aLoRA hooks before each no_grad forward.

    Registers a forward pre-hook on the PeftModel that drains stale hooks
    whenever the forward runs under `torch.no_grad()`. Pre-hooks fire via
    `nn.Module.__call__`, so they survive Accelerate's autocast wrapping
    and DDP/FSDP module wrapping (the inner module is still invoked via
    `__call__`).

    Also rebinds `forward` itself with a `MethodType` wrapper as a
    belt-and-suspenders fallback — this catches any code path that calls
    `model.forward(...)` directly (bypassing `__call__` and pre-hooks).

    Both layers gate on `torch.is_grad_enabled()` so they never strip hooks
    that PEFT registered for a real backward pass. Idempotent via sentinel.
    """
    if model is None or getattr(model, _WRAPPER_SENTINEL, False):
        return

    def _pre_hook(module, args, kwargs):
        if not torch.is_grad_enabled():
            drain_alora_gc_hooks(module)
        return None

    model.register_forward_pre_hook(_pre_hook, with_kwargs=True)

    original_forward = model.forward

    def forward(self, *args, **kwargs):
        if not torch.is_grad_enabled():
            drain_alora_gc_hooks(self)
        return original_forward(*args, **kwargs)

    model.forward = types.MethodType(forward, model)
    setattr(model, _WRAPPER_SENTINEL, True)
