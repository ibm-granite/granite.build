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

"""Router auxiliary loss for TRL on MoE bases that cannot produce one.

TRL's DPO/KTO trainers treat any model whose config has an
``output_router_logits`` attribute as MoE and, with the default
``router_aux_loss_coef=0.001``, add ``coef * outputs.aux_loss`` and log
``gather_for_metrics(aux_loss).mean()``. transformers 5 collects router logits
through ``_can_record_outputs`` output recorders, and in 5.9 the Granite MoE
classes (``GraniteMoeForCausalLM``, ``GraniteMoeHybridForCausalLM``) register
only ``hidden_states`` / ``attentions`` -- unlike Mixtral, which records
``router_logits``. So ``outputs.router_logits`` is ``None`` and
``load_balancing_loss_func`` returns the Python int ``0``; TRL then dies on the
first step (``'int' object has no attribute 'mean'`` single-GPU,
``'int' object is not iterable`` from the multi-rank gather). It hits Granite
4.0-H dense-MLP bases too (``num_local_experts == 0``): the config still carries
``output_router_logits``.

Because the aux loss is identically 0 for those classes, turning TRL's term off
there changes nothing numerically. :func:`router_aux_loss_coef` returns ``0.0``
only when the model class demonstrably cannot record router logits, and
``None`` (keep TRL's default) otherwise, so a model -- or a future transformers
-- that does compute the loss keeps it.
"""

from typing import Any, Optional


def router_aux_loss_coef(model: Any) -> Optional[float]:
    """``0.0`` if ``model`` is MoE-configured but cannot emit router logits, else ``None``."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    config = getattr(base, "config", None)
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    if getattr(text_config, "output_router_logits", None) is None:
        return None  # TRL does not treat it as MoE
    recorders = getattr(type(base), "_can_record_outputs", None)
    if recorders is None:
        return None  # not a recorder-based model: cannot tell, keep TRL's default
    return None if "router_logits" in recorders else 0.0
