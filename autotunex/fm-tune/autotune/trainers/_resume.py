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

"""Helpers for resuming final training from a checkpoint.

Multi-GPU resume crashes when loading a PEFT/LoRA adapter checkpoint under
``-gpu mode=exclusive_process``. HF Trainer's ``_load_from_checkpoint`` calls
``model.load_adapter(checkpoint, ..., is_trainable=True)`` early in ``train()``
(before Accelerate/FSDP/DeepSpeed places the model). PEFT then resolves the load
device via ``infer_device()``, which returns the bare string ``"cuda"`` (i.e.
``cuda:0``) — NOT rank-aware. Ray Train makes every worker see all assigned GPUs
(``CUDA_VISIBLE_DEVICES`` = all ids), so all ranks try to load the adapter onto
physical GPU 0. Under exclusive-process mode only one process may hold a context
on GPU 0; the rest fail with "CUDA-capable device(s) is/are busy or unavailable".

The fix below temporarily forces PEFT's ``infer_device`` to return ``"cpu"`` for
the duration of the resume call, so the adapter loads on CPU (rank-agnostic, no
GPU-context contention). Accelerate/FSDP/DeepSpeed then shard/move params to the
correct per-rank device afterward — consistent with ``fsdp_cpu_ram_efficient_loading``.

``infer_device`` is imported BY NAME into both ``peft.peft_model`` (used by
``load_adapter``) and ``peft.utils.save_and_load`` (used by ``load_peft_weights``),
so each module holds its own bound reference and both must be patched.
"""

import contextlib
import logging

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def peft_adapter_load_on_cpu():
    """Force PEFT adapter weights to load on CPU for the duration of the block.

    Scope this around ``trainer.train(resume_from_checkpoint=...)`` ONLY when
    actually resuming (a truthy checkpoint path). It is a no-op for non-PEFT
    models (``load_adapter`` is never called) and a clean no-op if PEFT is not
    importable. All patched references are restored on exit, including on error.
    """
    patched = []  # list of (module, original_infer_device)
    try:
        try:
            import peft.peft_model as _peft_model
            import peft.utils.other as _peft_other
            import peft.utils.save_and_load as _peft_sal

            modules = [_peft_model, _peft_sal, _peft_other]
        except ImportError:
            # PEFT absent or relayout — nothing to patch.
            modules = []

        for mod in modules:
            if hasattr(mod, "infer_device"):
                patched.append((mod, mod.infer_device))
                mod.infer_device = lambda: "cpu"

        if patched:
            logger.info(
                "[AutoTune] Resume: forcing PEFT adapter load onto CPU "
                "(avoids exclusive-process GPU-0 contention across ranks)."
            )
        yield
    finally:
        for mod, original in patched:
            mod.infer_device = original
