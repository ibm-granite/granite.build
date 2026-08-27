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
# LoraModel._enable_peft_forward_hooks in peft.tuners.lora.model.
"""
Monkey-patch for peft==0.18.0 that allows gradient checkpointing to be used
together with aLoRA.

Bug:
    peft.tuners.lora.model.LoraModel._enable_peft_forward_hooks raises
    "Multiple invocations of PEFT forward hooks before .backward() with
    enabled gradient checkpointing" when a second forward call happens
    before a backward has cleaned the per-layer hook bookkeeping list.
    This is triggered by HF Trainer evaluation (forward under no_grad,
    no backward), which is unavoidable with `eval_strategy=epoch`.

Fix:
    Instead of raising, drain any stale handles left on
    `layer._peft_gradient_checkpointing_forward_hooks` before re-registering
    the forward-pre and backward hooks. This preserves the invariant the
    original guard aimed for (no double registration) without forbidding
    back-to-back eval forwards.

Scope:
    Only active on peft==0.18.*. On any other version the patch is a no-op
    and `is_active()` returns False, so callers can fall back to disabling
    gradient checkpointing for aLoRA.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import partial

logger = logging.getLogger(__name__)

_PATCHED: bool = False


def is_active() -> bool:
    """Return True if the aLoRA gradient-checkpointing patch is installed."""
    return _PATCHED


def apply_alora_gc_patch() -> bool:
    """
    Install the patched `_enable_peft_forward_hooks` on peft's LoraModel.

    Idempotent. Safe to call multiple times. Returns True if the patch is
    active after this call, False otherwise (unsupported peft version or
    import failure).
    """
    global _PATCHED
    if _PATCHED:
        return True

    try:
        import peft
    except ImportError:
        logger.warning("[alora_patch] peft not importable; patch not applied")
        return False

    version = getattr(peft, "__version__", "")
    if not version.startswith("0.18."):
        logger.warning(
            "[alora_patch] peft %s is not 0.18.x; aLoRA GC patch will NOT be "
            "applied. Fall back to gradient_checkpointing=False for aLoRA.",
            version,
        )
        return False

    try:
        from peft.tuners.lora import model as lora_model_module
        from peft.tuners.lora.layer import LoraLayer
        from peft.tuners.tuners_utils import AuxiliaryTrainingWrapper

        LoraModel = lora_model_module.LoraModel
        GradientCheckpointingLayer = lora_model_module.GradientCheckpointingLayer
        _alora_offsets_pre_forward_hook = lora_model_module._alora_offsets_pre_forward_hook
        _adapter_names_pre_forward_hook = lora_model_module._adapter_names_pre_forward_hook
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "[alora_patch] could not import peft internals (%s); patch not applied",
            exc,
        )
        return False

    @contextmanager
    def _patched_enable_peft_forward_hooks(self, *args, **kwargs):
        adapter_names = kwargs.pop("adapter_names", None)
        alora_offsets = kwargs.pop("alora_offsets", None)

        if adapter_names is None and alora_offsets is None:
            yield
            return

        hook_handles = []

        if alora_offsets is not None:
            for n, layer in self.named_modules():
                if isinstance(layer, GradientCheckpointingLayer) and layer.gradient_checkpointing:

                    def forward_pre_hook(name, module, inputs, **kwargs):
                        for submodule in module.modules():
                            if isinstance(submodule, LoraLayer):
                                handle = submodule.register_forward_pre_hook(
                                    partial(
                                        _alora_offsets_pre_forward_hook,
                                        alora_offsets=kwargs["alora_offsets"],
                                    ),
                                    with_kwargs=True,
                                )
                                module._peft_gradient_checkpointing_forward_hooks.append(handle)

                    def backward_hook(name, module, *grad_output, **kwargs):
                        while module._peft_gradient_checkpointing_forward_hooks:
                            module._peft_gradient_checkpointing_forward_hooks.pop().remove()

                    # PATCH: drain stale handles from a prior forward that had
                    # no matching backward (e.g. trainer evaluation) instead
                    # of raising. The upstream guard only exists to prevent
                    # double registration; draining achieves the same thing.
                    existing = getattr(layer, "_peft_gradient_checkpointing_forward_hooks", None)
                    if existing:
                        while existing:
                            existing.pop().remove()
                    layer._peft_gradient_checkpointing_forward_hooks = []

                    handle = layer.register_forward_pre_hook(partial(forward_pre_hook, n, alora_offsets=alora_offsets))
                    layer._peft_gradient_checkpointing_forward_hooks.append(handle)
                    handle = layer.register_full_backward_hook(partial(backward_hook, n))
                    layer._peft_gradient_checkpointing_forward_hooks.append(handle)
                if isinstance(layer, LoraLayer):
                    pre_forward = partial(_alora_offsets_pre_forward_hook, alora_offsets=alora_offsets)
                    handle = layer.register_forward_pre_hook(pre_forward, with_kwargs=True)
                    hook_handles.append(handle)

        num_beams = kwargs.get("num_beams", None)
        uses_beam_search = isinstance(num_beams, int) and (num_beams > 1)
        if uses_beam_search:
            if alora_offsets is not None:
                raise ValueError("Beam search not yet supported for aLoRA.")

        if adapter_names is not None:
            if self.training:
                raise ValueError("Cannot pass `adapter_names` when the model is in training mode.")

            expected_adapters = set()
            for layer in self.modules():
                if isinstance(layer, LoraLayer):
                    expected_adapters |= layer.lora_A.keys()
                    expected_adapters |= layer.lora_embedding_A.keys()
            unique_adapters = {name for name in adapter_names if name != "__base__"}
            unexpected_adapters = unique_adapters - expected_adapters
            if unexpected_adapters:
                raise ValueError(
                    f"Trying to infer with non-existing adapter(s): {', '.join(sorted(unexpected_adapters))}"
                )

            original_adapter_names = adapter_names[:]
            if uses_beam_search:
                if not isinstance(adapter_names, (list, tuple)):
                    raise TypeError(f"Got adapter names of type {type(adapter_names)}, expected a list of str.")
                adapter_names = sum(([n] * kwargs["num_beams"] for n in adapter_names), [])

            for module in self.modules():
                if isinstance(module, LoraLayer) or isinstance(module, AuxiliaryTrainingWrapper):
                    pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
                    handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                    hook_handles.append(handle)

            if uses_beam_search and hasattr(self.model, "get_encoder"):
                for module in self.model.get_encoder().modules():
                    if isinstance(module, LoraLayer) or isinstance(module, AuxiliaryTrainingWrapper):
                        pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=original_adapter_names)
                        handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                        hook_handles.append(handle)

        yield

        for handle in hook_handles:
            handle.remove()

    LoraModel._enable_peft_forward_hooks = _patched_enable_peft_forward_hooks
    _PATCHED = True
    logger.info("[alora_patch] installed patched _enable_peft_forward_hooks for peft %s", version)
    print(f"[alora_patch] installed patched _enable_peft_forward_hooks for peft {version}")
    return True
