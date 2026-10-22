# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keep PEFT's adapter toggle from flipping ``requires_grad`` mid-step under FSDP.

TRL's DPO/KTO trainers, given a PEFT model and no ``ref_model``, compute the
reference log-probs *inside* the training step: policy forward (with grad), then
``with torch.no_grad(), use_adapter(model, None):`` a second forward with the
adapter disabled, then ``backward()``. PEFT's ``BaseTunerLayer.enable_adapters``
does more than flip ``_disable_adapters``: disabling sets ``requires_grad=False``
on every adapter weight, and re-enabling sets it back via ``set_adapter``.

Under FSDP that flip lands between the forward that registered FSDP's
post-backward hook (registered only for a FlatParameter that ``requires_grad``)
and the backward that consumes it, so ``_finalize_params`` sees a hook state for a
parameter that no longer requires grad and aborts every run on the first step::

    AssertionError: Invalid: ``_post_backward_hook_state``: (<AccumulateGrad ...>, <RemovableHandle ...>)

Observed identically on torch 2.8/TRL 0.29 and torch 2.11/TRL 1.12, on 2-rank
FULL_SHARD (and, before the single-worker guard, at world size 1). DeepSpeed has
no equivalent bookkeeping, which is why the DeepSpeed TRL driver is unaffected.

Inside ``torch.no_grad()`` the ``requires_grad`` flags are irrelevant to the
forward, so the fix is to make the toggle flag-only there: while
:func:`peft_adapter_toggle_keeps_requires_grad` is active, ``enable_adapters``
called with grad *disabled* only sets ``_disable_adapters``. Calls with grad
enabled (PEFT setup, a deliberate adapter switch) behave exactly as before.

Precomputing the reference log-probs (``precompute_ref_log_probs=True``) would
also avoid the mid-step toggle, but TRL does it in ``__init__``, before the FSDP
wrap, when ``fsdp_cpu_ram_efficient_loading`` leaves ranks > 0 on meta weights.
"""

import contextlib
import logging

import torch

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def peft_adapter_toggle_keeps_requires_grad():
    """Within this context, a no-grad PEFT adapter toggle leaves ``requires_grad`` alone.

    A clean no-op if PEFT is not importable. The original method is restored on exit.
    """
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError:
        yield
        return

    original = BaseTunerLayer.enable_adapters

    def enable_adapters(self, enabled: bool) -> None:
        if torch.is_grad_enabled():
            return original(self, enabled)
        self._disable_adapters = not enabled

    BaseTunerLayer.enable_adapters = enable_adapters
    logger.info("[AutoTune] PEFT adapter toggle under no_grad keeps requires_grad (FSDP reference pass).")
    try:
        yield
    finally:
        BaseTunerLayer.enable_adapters = original
