# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import atexit
import json
import shutil

from collections.abc import Callable
from contextlib import nullcontext
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from transformers import AutoTokenizer, GenerationConfig

# Workaround: In transformers>=4.57.0, _is_package_available() returns a tuple (bool, version).
# TRL stores the raw tuple for some packages, making guards truthy even when packages are absent.
# Patch affected values before any other TRL imports to prevent spurious ModuleNotFoundErrors.
import importlib
_trl_import_utils = importlib.import_module("trl.import_utils")
for _attr in ("_mergekit_available", "_vllm_ascend_available"):
    _val = getattr(_trl_import_utils, _attr, None)
    if isinstance(_val, tuple):
        setattr(_trl_import_utils, _attr, _val[0])
del _trl_import_utils, _attr, _val

# Workaround: DeepSpeed ZeRO optimizer wrappers don't expose `defaults`
# from the underlying optimizer. transformers' cosine_with_min_lr scheduler needs it.
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3 as _DS3Opt
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer as _DS12Opt
for _cls in (_DS3Opt, _DS12Opt):
    if not hasattr(_cls, "defaults"):
        _cls.defaults = property(lambda self: self.optimizer.defaults)
del _DS3Opt, _DS12Opt, _cls

# FA3 dispatch workaround for the vendored GraniteSWA model. In an earlier exploratory
# checkout this was an unconditional `import _fa3_preamble`, which raises on any machine whose HF
# cache has no kernels-community/vllm-flash-attn3 snapshot -- i.e. on this account,
# before a single batch, confirmed directly on the gold path (identical import here). The
# granite 4.1/4.2 pairing this recipe trains uses no sliding-window attention, so the
# import is gated on the SWA arm actually being requested. Same helper and same reason
# as gold.py -- see _swa_arm.py for why the preamble's refusal is correct for the model
# it was written for and must not be softened.
from gb_steps_post_training.distillation._swa_arm import activate_swa_arm
activate_swa_arm()

# `granite_swa` Auto*-API registration, so TRL's `create_model_from_path` can
# instantiate the vendored classes. Unconditional module-scope imports of the vendored
# package in that earlier checkout; gated now. activate_swa_arm() was already called above and
# is idempotent -- the second call keeps this block readable on its own and keeps the
# "must run before any `from trl...` import" ordering visible.
#
# Unlike gold.py there is NO activate_swa_vllm_model() call here, and that is not an
# omission: the vLLM ModelRegistry entry exists for `vllm_mode: colocate`, which builds a
# vllm.LLM inside the trainer process. This step never generates -- no sampling, no server,
# no colocate path -- so registering a vLLM model class would import vllm into a process
# that has no use for it.
activate_swa_arm()

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from gb_steps_post_training.distillation.custom_sft_config import CustomSFTConfig

from dataclasses import dataclass, field
import hashlib

from pprint import pprint

import os
import glob
# `sys` under an underscore alias for the same reason gold.py:111 does it: this module's
# __main__ block reads sys.argv to find `--config`, and a bare `sys` at module scope in a
# file this long is easy to shadow.
import sys as _sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from accelerate import PartialState
from transformers.utils import is_liger_kernel_available

from gb_steps_post_training.distillation.utils import (
    verify_fast_tokenizer,
    verify_tokenizer_consistency,
)
# Top-level, unlike `tracking` (imported inside main at the point where report_to must be
# final): train_manifest is stdlib-only, so importing it here costs nothing and a typo in it
# fails before the allocation does any work rather than after a corpus has been tokenized.
from gb_steps_post_training.distillation import train_manifest


def verify_optimization_stack(model, *, expect_liger_swiglu: bool) -> None:
    """Per-rank fail-fast check that the attention backend and (for SWA
    models) the Liger tiled-MLP patch are active on every rank before
    `trainer.train()`.

    Architecture-aware:
      - `GraniteSWA*` (vendored) requires FA3 dispatch and the FA3 kernel
        alias installed by `_fa3_preamble.py`. Falling back to eager
        materializes a [B,H,T,T] score tensor (~1100 GiB at 128K) and
        OOMs, confirmed by direct measurement.
      - Standard `GraniteForCausalLM` / other non-SWA Granite variants
        do not support the `flash_attention_3` string in stock
        transformers. They must run with `flash_attention_2` instead.
        transformers verifies that backend itself at model-load time,
        by one of TWO routes: the `flash_attn` package, or -- when that
        package is absent -- the kernels hub, in which case it rewrites
        `config._attn_implementation` to the kernel repo id. Both count
        as FA2 active here; see the comment on `accepted` below.

    Aborts symmetrically: each rank gathers its local failure flag via
    `all_reduce(SUM)`, every rank logs its own failures, then all ranks
    `barrier()` and `raise RuntimeError`.
    """
    import sys
    import torch
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    failures = []

    cfg = getattr(model, "config", None)
    attn_impl = getattr(cfg, "_attn_implementation", None)
    model_type = getattr(cfg, "model_type", None)
    is_swa = (model_type == "granite_swa") or type(model).__name__.startswith("GraniteSWA")

    if is_swa:
        # NOT given the kernel-fallback treatment the non-SWA branch gets below, on purpose:
        # the vendored GraniteSWAAttention path calls flash_attn_func through
        # sys.modules["flash_attn_interface"], which _fa3_preamble.py installs and which the
        # `kernels` route does not set. Accepting kernels-community/vllm-flash-attn3 here would
        # green-light a configuration that still ImportErrors inside the forward pass. The arm
        # is unreachable in this collection anyway (see _swa_arm.py).
        if attn_impl != "flash_attention_3":
            failures.append(
                f"FA3 inactive: model.config._attn_implementation={attn_impl!r} "
                "(expected 'flash_attention_3'). GraniteSWAAttention will fall through to "
                "the eager path and materialize a [B,H,T,T] score tensor (~1100 GiB at 128K)."
            )

        if "flash_attn_interface" not in sys.modules:
            failures.append(
                "FA3 kernel alias missing: sys.modules['flash_attn_interface'] not set. "
                "_fa3_preamble.py preload did not run on this rank; "
                "GraniteSWAAttention._flash_attention_3_forward will ImportError."
            )
        else:
            _fa3_mod = sys.modules["flash_attn_interface"]
            if not hasattr(_fa3_mod, "flash_attn_func"):
                failures.append(
                    "FA3 kernel malformed: sys.modules['flash_attn_interface'] has no "
                    "flash_attn_func attribute."
                )

        if expect_liger_swiglu:
            from _liger_granite_swa_patch import verify_liger_swiglu_active
            failures.extend(verify_liger_swiglu_active(model))
    else:
        # WHAT COUNTS AS FA2 IS NOT ONE STRING, and this is the correction the upstream copy of
        # this function needs. transformers 5.x, when the `flash_attn` PACKAGE is absent, serves
        # flash_attention_2 out of the kernels hub and REWRITES config._attn_implementation to
        # the kernel repo id: modeling_utils.py:1898 assigns
        # FLASH_ATTN_KERNEL_FALLBACK["flash_attention_2"] == "kernels-community/flash-attn2",
        # and _check_and_adjust_attn_implementation's return lands on
        # config._attn_implementation_internal (modeling_utils.py:1322).
        #
        # FA2 is genuinely active in that case -- the same kernel, dispatched through `kernels`
        # instead of through the package -- so comparing against the bare literal aborts every
        # rank on a configuration that is correct. That is worse than not checking: it fails
        # only on the environments where the fallback is the sole FA2 available, which is
        # exactly this account's (no flash_attn in the venv, and it cannot be built here --
        # CUDA 13.1 against torch's 12.8).
        #
        # Read from transformers rather than hard-coded, so an upstream rename makes this check
        # FOLLOW instead of silently rejecting the working path again. The literal is kept as a
        # fallback for the case where the private module moves, since an ImportError here would
        # otherwise abort a run over a check.
        accepted = {"flash_attention_2"}
        try:
            from transformers.modeling_flash_attention_utils import FLASH_ATTN_KERNEL_FALLBACK
            accepted.add(FLASH_ATTN_KERNEL_FALLBACK["flash_attention_2"])
        except Exception:
            accepted.add("kernels-community/flash-attn2")
        # `paged|flash_attention_2` is still FA2: split_attention_implementation puts the
        # KV-cache mode in front of the backend, and only the backend is what this asserts.
        if (attn_impl or "").split("|")[-1] not in accepted:
            failures.append(
                f"FA2 inactive: model.config._attn_implementation={attn_impl!r} on non-SWA "
                f"model (expected one of {sorted(accepted)}); use "
                "`attn_implementation: flash_attention_2` in the yaml. Note that transformers "
                "rewrites that value to the kernels-hub repo id when the flash_attn package is "
                "absent, and that form is accepted here -- so this failure means the backend "
                "really did fall back to sdpa or eager."
            )

        if expect_liger_swiglu:
            failures.append(
                f"use_liger_swiglu_mlp=True is not supported on non-SWA student "
                f"(model_type={model_type!r}, class={type(model).__name__}); the "
                "Liger tiled SwiGLU patch targets GraniteSWAMLP only. Unset "
                "`use_liger_swiglu_mlp` in the yaml."
            )

    local_failed = 1 if failures else 0
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

    total_mem_gib = None
    if torch.cuda.is_available():
        try:
            total_mem_gib = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        except Exception:
            total_mem_gib = None

    if dist.is_initialized():
        flag = torch.tensor([local_failed], dtype=torch.long, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.SUM)
        any_failed = int(flag.item())
    else:
        any_failed = local_failed

    if local_failed:
        hdr = f"[opt-stack check] rank {rank}/{world_size}"
        if total_mem_gib is not None:
            hdr += f" (GPU total_memory={total_mem_gib:.1f} GiB)"
        msg = hdr + " FAILED:\n  " + "\n  ".join(failures)
        print(msg, flush=True)

    if any_failed > 0:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
        raise RuntimeError(
            f"Optimization stack check failed on {any_failed}/{world_size} rank(s). "
            "Aborting to prevent silent OOM / hung NCCL collective. See per-rank "
            "FAILED log lines above for the specific broken optimization."
        )

    if rank == 0:
        if is_swa:
            print("[opt-stack check] [OK] FA3 active: config._attn_implementation=flash_attention_3, "
                  "flash_attn_interface kernel loaded")
            if expect_liger_swiglu:
                print("[opt-stack check] [OK] Liger tiled SwiGLU MLP active on all GraniteSWAMLP layers")
        else:
            # Print the value we ACTUALLY saw, not the literal: on this account it is the
            # kernels-hub repo id, and a log line asserting otherwise is how the next reader
            # concludes the package route is in use when it is not.
            print(f"[opt-stack check] [OK] FA2 active: config._attn_implementation={attn_impl!r} "
                  f"(non-SWA model_type={model_type!r})")


@dataclass
class CustomArguments:
    """
    Custom arguments for SFT training script.
    """
    max_dataset_size: int = field(
        default=-1,
        metadata={"help": "Reduce dataset size for quick debugging"}
    )
    response_template: str = field(
        default="<|start_of_role|>assistant<|end_of_role|>",
        metadata={"help": "Response template string to identify assistant turns for loss masking (e.g. '<|start_of_role|>assistant<|end_of_role|>')"}
    )
    use_liger_memory_opt: bool = field(
        default=False,
        metadata={"help": "Enable Liger-style memory optimization without patching model "
                  "internals (safe for GraniteMoEHybrid/Mamba models). In the CE-only path, "
                  "uses LigerFusedLinearCrossEntropyLoss (hidden capture + fused linear+CE "
                  "kernel). In the KD path, uses only the hidden-capture trick (mask-first, "
                  "project survivors); KD math runs on the resulting [N, V] tensor."}
    )
    use_liger_swiglu_mlp: bool = field(
        default=False,
        metadata={"help": "Monkey-patch GraniteSWAMLP.forward to use Liger's "
                  "tiled SwiGLU MLP (apply_tiled_mlp + LigerSiLUMulFunction). "
                  "Shards the MLP along the sequence dim (auto "
                  "num_shards = ceil(seqlen / hidden_size)) so per-shard "
                  "activation peak is intermediate_size * seqlen/num_shards "
                  "instead of intermediate_size * seqlen. Recomputes forward "
                  "per shard in backward, so MLP forward cost grows ~2-3x. "
                  "Required for long-context (>=64K) GraniteSWA training "
                  "to avoid backward-pass OOM. Distinct from "
                  "use_liger_memory_opt, which only touches the loss path."}
    )

    # ---- Off-policy KD with precomputed top-K teacher logits ----
    precomputed_logits_dir: str = field(
        default="",
        metadata={"help": "Path to a directory produced by data/precompute_logits.py. "
                  "If set, the trainer reads precomputed teacher top-K logits "
                  "from this directory and uses a forward-KL distillation loss."}
    )
    kd_top_k: int = field(default=256, metadata={"help": "Sanity-check top-K against meta.json."})
    kd_weight: float = field(default=1.0, metadata={"help": "Weight on the KL term."})
    ce_weight: float = field(default=0.0, metadata={"help": "Weight on the cross-entropy term."})
    kd_temperature: float = field(default=1.0, metadata={"help": "Softmax temperature for KD."})

    # Index-side slicing/filter flags (applied to non-skipped index.jsonl rows)
    dataset_subsample_start: float = field(default=0.0)
    dataset_subsample_end: float = field(default=1.0)
    dataset_shuffle: bool = field(default=False)
    dataset_subsample_seed: int = field(default=0)
    filter_no_tools: bool = field(default=False)
    filter_no_rag: bool = field(default=False)
    filter_tools_only: bool = field(default=False)
    filter_rag_only: bool = field(default=False)
    filter_strict: bool = field(
        default=False,
        metadata={"help": "Kept for compatibility with subsample_sft_4.1.py CLI; "
                  "filtering on the merged index is already per-row strict."}
    )

    # ---- Added for the granite.build step. Everything above this line is the earlier checkout's.
    #
    # A DECLARED INVARIANT, not a knob. Nothing downstream reads effective_batch_size; setting
    # it changes no behaviour. What it does is let a config state the batch shape it was tuned
    # for, so that a node shape which silently produces a different one is a refusal instead of
    # a run. It matters MORE here than on the gold path, not less: this step is the control, and
    # a control whose global batch quietly differs from the treatment's is not a comparison.
    #
    # WHY IT IS A TYPED FIELD AND NOT A COMMENT, learned the expensive way on the gold path:
    # TrlParser hands the WHOLE yaml to the argument parser with fail_with_unknown_args=True, so
    # a key nobody reads is not ignored, it raises. A direct measurement took a 4-node allocation, passed
    # every preflight, and died 42 s in with `Unknown arguments from config file:
    # ['--effective_batch_size', '192']`.
    effective_batch_size: int | None = field(
        default=None,
        metadata={"help": "Optional. The per_device x grad_accum x world_size product this "
                          "config was tuned for. When set, a run whose actual product differs "
                          "aborts before the model is loaded. Nothing reads it as a setting."},
    )

    # ---- Experiment tracking. ClearML is the DEFAULT backend; W&B is optional. The rules and
    # the (measured) list of what each backend actually reads live in tracking.py -- read its
    # docstring before changing any of this. In short: a backend needs a COMPLETE config to be
    # enabled, a PARTIAL config aborts the launch, and a complete config with no credentials
    # logs nothing and says so at full volume.
    #
    # These are typed fields rather than a `clearml:` block in the yaml for the same
    # fail_with_unknown_args reason as above.
    #
    # ALL FIVE LIVE HERE, which is the one place this file's dataclass surface differs from
    # gold.py's on purpose. There, `wandb_entity` and `wandb_project` had to be OMITTED because
    # upstream GOLDConfig already declares them (trl/experimental/gold/gold_config.py:339,343)
    # and re-declaring raised `argparse.ArgumentError: conflicting option strings` before a
    # single argument was parsed. SFTConfig declares NEITHER -- measured against
    # dataclasses.fields(SFTConfig), which contains no wandb_* name at all -- so here they must
    # be declared or the rendered yaml key would be an unknown argument. Do not "make the two
    # files consistent" by copying gold.py's omission across; the asymmetry is upstream's.
    clearml_project: str | None = field(
        default=None,
        metadata={"help": "ClearML project name (-> CLEARML_PROJECT). Required to enable ClearML. "
                          "ClearML has no `entity`: which workspace receives the task is carried "
                          "by the credentials (clearml.conf / CLEARML_API_*), not by this config."},
    )
    clearml_run_name: str | None = field(
        default=None,
        metadata={"help": "ClearML task name (-> CLEARML_TASK). Optional: derived from this "
                          "config's filename when omitted, plus the LSF job id when there is one. "
                          "Note ClearML ignores TrainingArguments.run_name entirely."},
    )
    wandb_entity: str | None = field(
        default=None,
        metadata={"help": "W&B entity (-> WANDB_ENTITY). Required together with wandb_project to "
                          "enable W&B; a partial config aborts rather than logging to a default."},
    )
    wandb_project: str | None = field(
        default=None,
        metadata={"help": "W&B project (-> WANDB_PROJECT). Required together with wandb_entity."},
    )
    wandb_run_name: str | None = field(
        default=None,
        metadata={"help": "W&B run name (-> TrainingArguments.run_name, which is what transformers "
                          "passes to wandb.init as `name`). Optional; auto-derived when omitted."},
    )


class KDDataCollator:
    """Wraps a base collator and attaches precomputed teacher top-K logits.

    The dataset rows carry `shard_id`, `shard_offset`, `assistant_logits_used`
    integers that are popped before the base collator runs. After collation,
    the corresponding rows are mmap-loaded from
    `<shards_dir>/indices_<id>.bin` and `logits_<id>.bin`, concatenated across
    the batch in example order, and added to the batch dict as
    `teacher_topk_indices` (long, [ΣN, K]) and `teacher_topk_logits` (float, [ΣN, K]).

    The concatenation order matches `(shift_labels.view(-1) != -100)` ordering
    in `_compute_loss_kd`, so the per-token alignment is 1-to-1.
    """

    KD_KEYS = ("shard_id", "shard_offset", "assistant_logits_used")

    def __init__(self, base_collator, shards_dir, top_k):
        self.base = base_collator
        self.shards_dir = shards_dir
        self.top_k = int(top_k)
        self._idx_mmaps = {}
        self._lg_mmaps = {}

    def _get_mmap(self, shard_id):
        if shard_id in self._idx_mmaps:
            return self._idx_mmaps[shard_id], self._lg_mmaps[shard_id]
        idx_path = os.path.join(self.shards_dir, f"indices_{shard_id:06d}.bin")
        lg_path = os.path.join(self.shards_dir, f"logits_{shard_id:06d}.bin")
        idx_bytes = os.path.getsize(idx_path)
        lg_bytes = os.path.getsize(lg_path)
        idx_rows = idx_bytes // (self.top_k * 4)   # int32
        lg_rows = lg_bytes // (self.top_k * 2)     # float16
        if idx_rows != lg_rows:
            raise RuntimeError(
                f"Shard {shard_id}: indices rows={idx_rows} != logits rows={lg_rows}"
            )
        idx_mm = np.memmap(idx_path, dtype=np.int32, mode="r", shape=(idx_rows, self.top_k))
        lg_mm = np.memmap(lg_path, dtype=np.float16, mode="r", shape=(lg_rows, self.top_k))
        self._idx_mmaps[shard_id] = idx_mm
        self._lg_mmaps[shard_id] = lg_mm
        return idx_mm, lg_mm

    def __call__(self, examples):
        kd_meta = []
        cleaned = []
        for ex in examples:
            ex = dict(ex)  # shallow copy to avoid mutating the caller's dict
            meta = {k: ex.pop(k) for k in self.KD_KEYS}
            kd_meta.append(meta)
            cleaned.append(ex)

        batch = self.base(cleaned)

        idx_chunks = []
        lg_chunks = []
        for meta in kd_meta:
            shard_id = int(meta["shard_id"])
            offset = int(meta["shard_offset"])
            n = int(meta["assistant_logits_used"])
            if n <= 0:
                continue
            idx_mm, lg_mm = self._get_mmap(shard_id)
            idx_chunks.append(torch.from_numpy(np.asarray(idx_mm[offset:offset + n])).long())
            lg_chunks.append(torch.from_numpy(np.asarray(lg_mm[offset:offset + n])).float())

        if idx_chunks:
            batch["teacher_topk_indices"] = torch.cat(idx_chunks, dim=0)
            batch["teacher_topk_logits"] = torch.cat(lg_chunks, dim=0)
        else:
            batch["teacher_topk_indices"] = torch.zeros((0, self.top_k), dtype=torch.long)
            batch["teacher_topk_logits"] = torch.zeros((0, self.top_k), dtype=torch.float32)
        return batch


class CustomSFTTrainer(SFTTrainer):
    """SFTTrainer that handles tools stored as JSON strings in the dataset."""

    def __init__(
        self,
        response_template=None,
        use_liger_memory_opt=False,
        precomputed_logits_dir="",
        kd_top_k=256,
        kd_weight=1.0,
        ce_weight=0.0,
        kd_temperature=1.0,
        sft_cache_dir=None,
        sft_cache_key_blob=None,
        **kwargs,
    ):
        self.response_template = response_template
        self.precomputed_logits_dir = precomputed_logits_dir
        self.kd_top_k = kd_top_k
        self.kd_weight = kd_weight
        self.ce_weight = ce_weight
        self.kd_temperature = kd_temperature
        self.sft_cache_dir = sft_cache_dir
        self.sft_cache_key_blob = sft_cache_key_blob

        if precomputed_logits_dir and kwargs.get("args").use_liger_kernel:
            raise ValueError(
                "precomputed_logits_dir (KD) is mutually exclusive with use_liger_kernel."
            )

        super().__init__(**kwargs)

        if use_liger_memory_opt or self.args.use_liger_kernel:
            if not is_liger_kernel_available():
                raise ImportError("use_liger_memory_opt=True but liger-kernel is not installed.")
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            self.liger_ce_loss = LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100,
                reduction="mean",
                return_token_accuracy=True,
            )

        if self.precomputed_logits_dir:
            shards_dir = os.path.join(self.precomputed_logits_dir, "shards")
            self.data_collator = KDDataCollator(
                base_collator=self.data_collator,
                shards_dir=shards_dir,
                top_k=self.kd_top_k,
            )

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if self.precomputed_logits_dir and self._signature_columns is not None:
            for col in KDDataCollator.KD_KEYS:
                if col not in self._signature_columns:
                    self._signature_columns.append(col)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.precomputed_logits_dir:
            return self._compute_loss_kd(model, inputs, return_outputs, num_items_in_batch)
        if hasattr(self, "liger_ce_loss"):
            return self._compute_loss_liger(model, inputs, return_outputs, num_items_in_batch)
        return self._compute_loss_default(model, inputs, return_outputs, num_items_in_batch)

    def _mark_lm_head_persistent(self):
        """On ZeRO-3, mark lm_head.weight (and bias) with is_external_param=True
        so DeepSpeed keeps them gathered across the full forward+backward of
        the step. Required by the hidden-capture path (_forward_capture_hidden
        + _project_masked_hidden / _compute_loss_liger), which uses lm_head.weight
        via F.linear *after* the model's forward returns. Without this flag,
        lm_head's post-forward release hook partitions the weight before our
        F.linear runs (or before its backward), causing a `vec (0)` size
        mismatch in deepspeed/runtime/zero/linear.py.

        Idempotent; the param keeps the flag once set (DeepSpeed only resets it
        during one-time Init._convert_to_zero_parameters, which happens before
        the first compute_loss call).
        """
        if not self.is_deepspeed_enabled:
            return
        unwrapped = self.accelerator.unwrap_model(self.model)
        lm_head = unwrapped.get_output_embeddings()
        for p in (lm_head.weight, getattr(lm_head, "bias", None)):
            if p is not None and not getattr(p, "is_external_param", False):
                p.is_external_param = True

    def _forward_capture_hidden(self, model, inputs):
        self._mark_lm_head_persistent()
        unwrapped = self.accelerator.unwrap_model(model)
        lm_head = unwrapped.get_output_embeddings()
        captured_hidden = {}

        def _capture_hook(module, args, kwargs):
            captured_hidden["value"] = args[0]
            return args, kwargs

        hook = lm_head.register_forward_pre_hook(_capture_hook, with_kwargs=True)
        original_forward = lm_head.forward
        lm_head.forward = lambda x: x
        try:
            outputs = model(**inputs)
        finally:
            hook.remove()
            lm_head.forward = original_forward

        hidden = captured_hidden["value"]
        lm_head_bias = getattr(lm_head, "bias", None)
        logits_scaling = getattr(unwrapped.config, "logits_scaling", 1.0)
        return outputs, hidden, lm_head, lm_head_bias, logits_scaling

    def _project_masked_hidden(self, hidden_masked, lm_head, lm_head_bias):
        return F.linear(hidden_masked, lm_head.weight, lm_head_bias)

    def _compute_loss_liger(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._mark_lm_head_persistent()
        labels = inputs.pop("labels")
        inputs["use_cache"] = False

        # Capture hidden states via hook and replace lm_head with identity to avoid
        # materializing full [B*T, V] logits. This goes through the DeepSpeed wrapper
        # so ZeRO-3 parameter gathering works correctly.
        unwrapped = self.accelerator.unwrap_model(model)
        lm_head = unwrapped.get_output_embeddings()
        captured_hidden = {}

        def _capture_hook(module, args, kwargs):
            captured_hidden["value"] = args[0]
            return args, kwargs

        hook = lm_head.register_forward_pre_hook(_capture_hook, with_kwargs=True)
        original_forward = lm_head.forward
        lm_head.forward = lambda x: x  # identity — skip lm_head matmul
        try:
            outputs = model(**inputs)
        finally:
            hook.remove()
            lm_head.forward = original_forward

        # Shift for next-token prediction
        hidden = captured_hidden["value"][:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Apply model-specific logits scaling (e.g. Granite divides logits by config.logits_scaling)
        logits_scaling = getattr(unwrapped.config, "logits_scaling", 1.0)
        if logits_scaling != 1.0:
            hidden = hidden / logits_scaling

        # Flatten to 2D [B*T, H] / 1D [B*T]
        hidden = hidden.view(-1, hidden.size(-1))
        shift_labels = shift_labels.view(-1)

        # Fused linear CE: never materializes full [B*T, V] logits.
        # With ZeRO-3, lm_head.weight is sharded — gather it first.
        lm_head_bias = getattr(lm_head, "bias", None)
        gather_params = [lm_head.weight]
        if lm_head_bias is not None:
            gather_params.append(lm_head_bias)

        if self.is_deepspeed_enabled:
            import deepspeed
            ctx = deepspeed.zero.GatheredParameters(gather_params, modifier_rank=None)
        else:
            ctx = nullcontext()

        with ctx:
            ce_output = self.liger_ce_loss(
                lm_head.weight, hidden, shift_labels, bias=lm_head_bias
            )
        loss = ce_output.loss
        token_accuracy = ce_output.token_accuracy

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["mean_token_accuracy"].append(token_accuracy.item())

        if mode == "train":
            if "attention_mask" in inputs:
                num_tokens = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' in inputs.")
            self._total_train_tokens += num_tokens
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        return (loss, outputs) if return_outputs else loss

    def _compute_loss_default(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Pop labels to prevent model from computing loss internally on full logits.
        # Model-internal CE stores full log_softmax [B*T, V] for backward → OOM.
        # Instead, mask first then CE on the small subset (same as GOLD's cross_entropy_loss).
        labels = inputs.pop("labels")
        inputs["use_cache"] = False

        outputs = model(**inputs)

        # Shift for next-token prediction
        logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        # Mask before CE to avoid storing full log_softmax during backward
        mask = shift_labels != -100
        flat_logits = logits.reshape(-1, logits.size(-1))[mask.view(-1)]
        flat_labels = shift_labels.reshape(-1)[mask.view(-1)]
        loss = F.cross_entropy(flat_logits, flat_labels)

        # Token accuracy (reuse masked tensors, no extra memory)
        mode = "train" if self.model.training else "eval"
        with torch.no_grad():
            predictions = flat_logits.argmax(dim=-1)
            correct = (predictions == flat_labels).sum()
            total = torch.tensor(flat_labels.numel(), device=correct.device)
            correct = self.accelerator.gather_for_metrics(correct)
            total = self.accelerator.gather_for_metrics(total)
            accuracy = (correct.sum() / total.sum()).item() if total.sum() > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        if mode == "train":
            if "attention_mask" in inputs:
                num_tokens = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' in inputs.")
            self._total_train_tokens += num_tokens
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        return (loss, outputs) if return_outputs else loss

    def _compute_loss_kd(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forward KL on the renormalized top-K teacher distribution, plus
        # optional CE blend. Mirrors `_compute_loss_default`'s mask-first
        # idiom to avoid materializing full [B*T, V] log_softmax for backward.
        labels = inputs.pop("labels")
        t_idx = inputs.pop("teacher_topk_indices").to(self.accelerator.device, non_blocking=True)
        t_lg = inputs.pop("teacher_topk_logits").to(self.accelerator.device, non_blocking=True)
        inputs["use_cache"] = False

        if hasattr(self, "liger_ce_loss"):
            outputs, hidden, lm_head, lm_head_bias, logits_scaling = self._forward_capture_hidden(model, inputs)
            hidden = hidden[:, :-1].contiguous()
            if logits_scaling != 1.0:
                hidden = hidden / logits_scaling
            shift_labels = labels[:, 1:].contiguous()
            mask = (shift_labels != -100).view(-1)
            hidden_masked = hidden.reshape(-1, hidden.size(-1))[mask]
            flat_labels = shift_labels.view(-1)[mask]
            if flat_labels.numel() == 0:
                loss = hidden.sum() * 0.0
                return (loss, outputs) if return_outputs else loss
            flat_logits = self._project_masked_hidden(hidden_masked, lm_head, lm_head_bias)
        else:
            outputs = model(**inputs)
            logits = outputs.logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            mask = (shift_labels != -100).view(-1)
            flat_logits = logits.reshape(-1, logits.size(-1))[mask]   # [N, V]
            flat_labels = shift_labels.reshape(-1)[mask]              # [N]

        if flat_logits.size(0) != t_idx.size(0):
            raise RuntimeError(
                f"KD alignment mismatch: student tokens={flat_logits.size(0)} "
                f"teacher tokens={t_idx.size(0)}. This usually means the chat "
                f"template / tokenizer changed between precompute and training, "
                f"or assistant_logits_used was computed incorrectly."
            )

        T = self.kd_temperature
        t_log_probs = F.log_softmax(t_lg.float() / T, dim=-1)              # [N, K]
        t_probs = t_log_probs.exp()
        s_log_probs_topk = torch.gather(
            F.log_softmax(flat_logits / T, dim=-1),
            dim=-1, index=t_idx.long(),
        )                                                                   # [N, K]
        kl = (t_probs * (t_log_probs - s_log_probs_topk)).sum(-1).mean()
        loss = self.kd_weight * (T * T) * kl
        if self.ce_weight > 0.0:
            loss = loss + self.ce_weight * F.cross_entropy(flat_logits, flat_labels)

        mode = "train" if self.model.training else "eval"
        with torch.no_grad():
            predictions = flat_logits.argmax(dim=-1)
            correct = (predictions == flat_labels).sum()
            total = torch.tensor(flat_labels.numel(), device=correct.device)
            correct = self.accelerator.gather_for_metrics(correct)
            total = self.accelerator.gather_for_metrics(total)
            accuracy = (correct.sum() / total.sum()).item() if total.sum() > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        if mode == "train":
            if "attention_mask" not in inputs:
                raise ValueError("Expected 'attention_mask' in inputs.")
            # Defer the cross-rank gather and host sync to logging boundaries.
            # The per-step `.item()` here was a CPU sync inside the training
            # step that amplified first-step NCCL watchdog hangs on 192-rank
            # ZeRO-3, confirmed by direct measurement, and added measurable step-0 latency.
            local_count = inputs["attention_mask"].sum().detach()
            if not hasattr(self, "_pending_local_tokens") \
                    or self._pending_local_tokens.device != local_count.device:
                self._pending_local_tokens = torch.zeros(
                    (), dtype=torch.long, device=local_count.device
                )
                self._pending_step_count = 0
            self._pending_local_tokens += local_count.long()
            self._pending_step_count += 1
            flush_every = max(int(getattr(self.args, "logging_steps", 1) or 1), 1)
            if self._pending_step_count >= flush_every:
                synced = self.accelerator.gather_for_metrics(self._pending_local_tokens).sum().item()
                self._total_train_tokens += synced
                self._pending_local_tokens.zero_()
                self._pending_step_count = 0
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        return (loss, outputs) if return_outputs else loss

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class,
        args: CustomSFTConfig,
        packing: bool,
        formatting_func: Callable[[dict], str] | None,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        state = PartialState()

        # Cached path: rank 0 runs the map chain on cache miss and saves under
        # `data/preprocessed_sft_cache/<hash>/<split>/`; non-main ranks wait
        # inside `main_process_first`, then all ranks `load_from_disk`.
        cache_dir = getattr(self, "sft_cache_dir", None)
        if cache_dir and isinstance(dataset, Dataset):
            from datasets import load_from_disk

            cache_split_dir = os.path.join(cache_dir, dataset_name)
            info_path = os.path.join(cache_split_dir, "dataset_info.json")

            # HIT or MISS, said out loud on rank 0. Without this the cache is invisible: a run
            # that silently re-tokenizes the whole corpus and a run that reused it look the same
            # in the log, differing only in a wall-clock number nobody has the baseline for. It
            # also makes the ONE failure mode legible -- a key that changes when it should not
            # (an mtime in the dataset fingerprint, say) shows up as MISS on every run.
            if state.is_main_process:
                print(f"[sft-cache] {'HIT ' if os.path.exists(info_path) else 'MISS'} "
                      f"{cache_split_dir}", flush=True)
            with state.main_process_first():
                if not os.path.exists(info_path):
                    if state.is_main_process:
                        prepared = self._run_map_chain(
                            dataset, processing_class, args, packing, formatting_func, dataset_name
                        )
                        os.makedirs(cache_dir, exist_ok=True)
                        prepared.save_to_disk(cache_split_dir)
                        key_path = os.path.join(cache_dir, "key.json")
                        if not os.path.exists(key_path) and self.sft_cache_key_blob is not None:
                            with open(key_path, "w") as f:
                                json.dump(self.sft_cache_key_blob, f, indent=2, sort_keys=True)

            loaded = load_from_disk(cache_split_dir)
            if args.shuffle_dataset:
                loaded = loaded.shuffle(seed=args.seed)
            return loaded

        # No caching (e.g. streaming / IterableDataset): keep the historical
        # inline map-chain path so behaviour is unchanged for non-cached callers.
        with state.main_process_first():
            prepared = self._run_map_chain(
                dataset, processing_class, args, packing, formatting_func, dataset_name
            )
        if args.shuffle_dataset:
            prepared = prepared.shuffle(seed=args.seed)
        return prepared

    def _run_map_chain(
        self,
        dataset: Dataset | IterableDataset,
        processing_class,
        args: CustomSFTConfig,
        packing: bool,
        formatting_func: Callable[[dict], str] | None,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        """The formatting / tokenize / truncate / kd-count map chain,
        extracted from `_prepare_dataset` so that it can be short-circuited
        by an on-disk cache. Callers are responsible for the enclosing
        `main_process_first` block and the trailing `shuffle_dataset` step.
        """
        from trl.data_utils import (
            is_conversational,
            is_conversational_from_value,
            maybe_convert_to_chatml,
            pack_dataset,
            truncate_dataset,
        )
        from trl.trainer.sft_trainer import get_dataset_column_names
        from trl.trainer.utils import remove_none_values

        if isinstance(dataset, Dataset):
            dataset = dataset.with_transform(remove_none_values)

        column_names = get_dataset_column_names(dataset)
        is_processed = "input_ids" in column_names

        map_kwargs = {}
        if isinstance(dataset, Dataset):
            map_kwargs["num_proc"] = args.dataset_num_proc

        if formatting_func is not None and not is_processed:
            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

            def _func(example):
                return {"text": formatting_func(example)}

            dataset = dataset.map(_func, batched=False, **map_kwargs)

        if not is_processed:
            first_example = next(iter(dataset))
            if is_conversational_from_value(first_example):
                if isinstance(dataset, Dataset):
                    map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
                column_names = get_dataset_column_names(dataset)
                dataset = dataset.map(
                    maybe_convert_to_chatml,
                    remove_columns="conversations" if "conversations" in column_names else None,
                    **map_kwargs,
                )

            first_example = next(iter(dataset))
            if not is_conversational(first_example):
                if isinstance(dataset, Dataset):
                    map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                def add_eos(example, eos_token):
                    if "text" in example and not example["text"].endswith(eos_token):
                        example["text"] = example["text"] + eos_token
                    elif "completion" in example and not example["completion"].endswith(eos_token):
                        example["completion"] = example["completion"] + eos_token
                    return example

                eos_token = processing_class.tokenizer.eos_token if self._is_vlm else processing_class.eos_token
                dataset = dataset.map(
                    add_eos,
                    fn_kwargs={"eos_token": eos_token},
                    remove_columns="messages" if "messages" in column_names else None,
                    **map_kwargs,
                )

            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            response_template_ids = None
            if self.response_template:
                response_template_ids = processing_class.encode(self.response_template, add_special_tokens=False)

            kd_mode = bool(self.precomputed_logits_dir)

            def _tokenize_one_row(example, processing_class, dataset_text_field, assistant_only_loss, response_template_ids):
                # Parse tools from JSON string before passing to apply_chat_template
                tools = example.get("tools")
                if tools and isinstance(tools, str):
                    tools = json.loads(tools)
                elif not tools:
                    tools = None

                # Build chat_template_kwargs from documents column
                chat_template_kwargs = example.get("chat_template_kwargs", {})
                if isinstance(chat_template_kwargs, str):
                    chat_template_kwargs = json.loads(chat_template_kwargs)
                documents = example.get("documents")
                if documents:
                    chat_template_kwargs["documents"] = documents

                if "prompt" in example:
                    output = {}
                    if is_conversational(example):
                        prompt = example["prompt"]
                        completion = example["completion"]
                        prompt_ids = processing_class.apply_chat_template(
                            prompt,
                            tokenize=True,
                            add_generation_prompt=True,
                            tools=tools,
                            **chat_template_kwargs,
                        )
                        prompt_ids = prompt_ids[0] if isinstance(prompt_ids[0], list) else prompt_ids
                        prompt_completion_processed = processing_class.apply_chat_template(
                            prompt + completion,
                            return_dict=True,
                            tokenize=True,
                            # `or kd_mode`: assistant_only_loss decides what the LOSS covers, but KD needs
                            # this mask to ALIGN each stored teacher row with the token it was computed for.
                            # Gating it on assistant_only_loss silently switched the mask MECHANISM to the
                            # response-template fallback below, whose span ends at <|im_end|> while
                            # {% generation %} also covers the newline after it -- one token of disagreement
                            # per assistant turn, and every KD run died on its first row, confirmed directly.
                            return_assistant_tokens_mask=(assistant_only_loss or kd_mode),
                            tools=tools,
                            **chat_template_kwargs,
                        )
                        prompt_completion_processed = {
                            k: v[0] if isinstance(v[0], list) else v
                            for k, v in prompt_completion_processed.items()
                        }
                        prompt_completion_ids = prompt_completion_processed["input_ids"]
                        if "assistant_masks" in prompt_completion_processed:
                            output["assistant_masks"] = prompt_completion_processed["assistant_masks"]
                    else:
                        prompt_ids = processing_class(text=example["prompt"])["input_ids"]
                        prompt_completion_ids = processing_class(text=example["prompt"] + example["completion"])[
                            "input_ids"
                        ]
                        prompt_ids = prompt_ids[0] if isinstance(prompt_ids[0], list) else prompt_ids
                        prompt_completion_ids = (
                            prompt_completion_ids[0]
                            if isinstance(prompt_completion_ids[0], list)
                            else prompt_completion_ids
                        )

                    completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
                    output["input_ids"] = prompt_completion_ids
                    output["completion_mask"] = completion_mask

                else:  # language modeling case
                    if is_conversational(example):
                        messages = example["messages"]
                        processed = processing_class.apply_chat_template(
                            messages,
                            return_dict=True,
                            tokenize=True,
                            # `or kd_mode` for the same reason as the prompt/completion branch above: the
                            # mask is KD's alignment key, independent of what the loss covers.
                            return_assistant_tokens_mask=(assistant_only_loss or kd_mode),
                            tools=tools,
                            **chat_template_kwargs,
                        )
                        processed = {k: v[0] if isinstance(v[0], list) else v for k, v in processed.items()}
                        output = {k: processed[k] for k in ("input_ids", "assistant_masks") if k in processed}
                    else:
                        output = {"input_ids": processing_class(text=example[dataset_text_field])["input_ids"]}

                needs_fallback = (
                    "assistant_masks" not in output
                    or 1 not in output["assistant_masks"]
                )
                if response_template_ids is not None and needs_fallback:
                    input_ids = output["input_ids"]
                    assistant_masks = [0] * len(input_ids)
                    template_len = len(response_template_ids)
                    i = 0
                    while i <= len(input_ids) - template_len:
                        if input_ids[i:i + template_len] == response_template_ids:
                            start_idx = i + template_len
                            end_idx = len(input_ids)
                            eos_id = processing_class.eos_token_id
                            if eos_id is not None:
                                for j in range(start_idx, len(input_ids)):
                                    if input_ids[j] == eos_id:
                                        end_idx = j + 1
                                        break
                            for k in range(start_idx, end_idx):
                                assistant_masks[k] = 1
                            i = end_idx
                        else:
                            i += 1
                    output["assistant_masks"] = assistant_masks

                if "assistant_masks" in output and 1 not in output["assistant_masks"]:
                    # Upstream words this as "You're using assistant_only_loss=True", which was
                    # true when that was the only caller. KD reaches it too, with
                    # assistant_only_loss False, so naming that flag would send the reader to
                    # the wrong config key.
                    why = "KD mode" if kd_mode else "`assistant_only_loss=True`"
                    raise RuntimeError(
                        f"{why} needs an assistant mask, but at least one example came back with no "
                        "assistant tokens, and the response-template fallback did not fill it in "
                        "either. Two things produce that: a chat template with no `{% generation %}` "
                        "keyword, or a `response_template` that does not occur in the rendered text "
                        "(check its trailing newline). Neither is silent-safe -- training would "
                        "otherwise proceed over a corpus it scores nothing in."
                    )

                if kd_mode and "assistant_masks" not in output:
                    raise RuntimeError(
                        "KD mode requires `assistant_masks` -- not to mask the loss, but to "
                        "align each stored teacher row with the token it was computed for. It is "
                        "therefore requested regardless of `assistant_only_loss`, and neither the "
                        "chat template's `{% generation %}` markers nor the `response_template` "
                        "fallback produced one here. Check both."
                    )

                return output

            def tokenize_fn(examples, processing_class, dataset_text_field, assistant_only_loss, response_template_ids=None):
                keys = list(examples.keys())
                batch_size = len(examples[keys[0]])
                output_batch: dict = {}
                for i in range(batch_size):
                    row = {k: examples[k][i] for k in keys}
                    row_out = _tokenize_one_row(
                        row,
                        processing_class=processing_class,
                        dataset_text_field=dataset_text_field,
                        assistant_only_loss=assistant_only_loss,
                        response_template_ids=response_template_ids,
                    )
                    for k, v in row_out.items():
                        output_batch.setdefault(k, []).append(v)

                if kd_mode:
                    masks_batch = output_batch["assistant_masks"]
                    expected_batch = examples["num_assistant_tokens"]
                    source_idx_batch = examples.get("source_idx", [None] * batch_size)
                    for i in range(batch_size):
                        masks = masks_batch[i]
                        n_here = sum(masks[1:]) if len(masks) > 1 else 0
                        expected = int(expected_batch[i])
                        if n_here != expected:
                            raise RuntimeError(
                                f"KD: assistant token count from chat template ({n_here}) "
                                f"does not match num_assistant_tokens ({expected}) from index.jsonl "
                                f"for source_idx={source_idx_batch[i]}. Two causes look alike here. "
                                f"Either the chat template or tokenizer changed between precompute "
                                f"and training, or the two sides derived the mask by different "
                                f"MECHANISMS: the template's generation markers cover the newline "
                                f"after <|im_end|>, the response-template scan stops at it, and they "
                                f"differ by exactly one token per assistant turn. A difference of "
                                f"{expected - n_here} on a {expected}-token span points at the "
                                f"mechanism; a large or erratic difference points at the template."
                            )
                    output_batch["shard_id"] = [int(x) for x in examples["shard_id"]]
                    output_batch["shard_offset"] = [int(x) for x in examples["shard_offset"]]
                    output_batch["num_assistant_tokens"] = [int(x) for x in expected_batch]

                return output_batch

            dataset = dataset.map(
                tokenize_fn,
                batched=True,
                fn_kwargs={
                    "processing_class": processing_class,
                    "dataset_text_field": args.dataset_text_field,
                    "assistant_only_loss": args.assistant_only_loss,
                    "response_template_ids": response_template_ids,
                },
                **map_kwargs,
            )

        if packing:
            if args.max_length is None:
                raise ValueError("When packing is enabled, `max_length` can't be `None`.")
            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Packing {dataset_name} dataset"

            columns = ["input_ids"]
            if "completion_mask" in get_dataset_column_names(dataset):
                columns.append("completion_mask")
            if "assistant_masks" in get_dataset_column_names(dataset):
                columns.append("assistant_masks")

            dataset = dataset.select_columns(columns)

            if args.shuffle_dataset:
                dataset = dataset.shuffle(seed=args.seed)

            dataset = pack_dataset(dataset, args.max_length, args.packing_strategy, map_kwargs)
        elif args.max_length is not None:
            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Truncating {dataset_name} dataset"
            dataset = truncate_dataset(dataset, args.max_length, map_kwargs)

        kd_mode = bool(self.precomputed_logits_dir)
        if kd_mode:
            # After possible truncation, recompute the count of assistant
            # tokens whose predicting logit position is `>= 0` (i.e. positions
            # i >= 1 in assistant_masks). This matches the precompute-side
            # convention and gives the number of teacher top-K rows we need
            # to slice from the memmap. Truncation drops trailing assistant
            # tokens together with their precomputed logits.
            def _assistant_logits_used(examples):
                masks_batch = examples["assistant_masks"]
                return {
                    "assistant_logits_used": [
                        (sum(m[1:]) if len(m) > 1 else 0) for m in masks_batch
                    ]
                }

            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Computing assistant_logits_used for {dataset_name} dataset"
            dataset = dataset.map(_assistant_logits_used, batched=True, **map_kwargs)

        if args.use_liger_kernel:
            collator_expected_keys = {"input_ids", "seq_lengths", "completion_mask", "assistant_masks"}
            column_names = get_dataset_column_names(dataset)
            dataset = dataset.select_columns(collator_expected_keys.intersection(column_names))
        elif kd_mode:
            collator_expected_keys = {
                "input_ids", "seq_lengths", "completion_mask", "assistant_masks",
                "shard_id", "shard_offset", "assistant_logits_used",
            }
            column_names = get_dataset_column_names(dataset)
            dataset = dataset.select_columns(collator_expected_keys.intersection(column_names))

        return dataset


class DebugSFTTrainer(CustomSFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # --- Debug: inspect tokenization and loss masking ---
        print(inputs["input_ids"].shape)
        for i in range(inputs["input_ids"].shape[0]):
            print('-' * 60)
            print(self.processing_class.decode(inputs["input_ids"][i]))
        print('-' * 60)
        valid_mask = inputs["labels"] != -100
        decoded_labels = [
            self.processing_class.decode(ids[mask])
            for ids, mask in zip(inputs["input_ids"], valid_mask)
        ]
        print(decoded_labels)
        import pdb; pdb.set_trace()
        # --- End debug ---
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


def _compute_sft_cache_dir(script_args, training_args, custom_args, tokenizer):
    """Compute a stable on-disk cache dir for the preprocessed SFT dataset.

    Returns `(cache_dir, key_blob)` where `cache_dir` is
    `<repo>/data/preprocessed_sft_cache/<sha256[:32]>` and `key_blob` is the
    pre-hash JSON dict (dumped alongside the cached data as `key.json`).

    The key fingerprints every input the map chain in `_prepare_dataset`
    depends on: tokenizer, chat_template, max_length / packing / assistant-only
    settings, response_template, dataset identity (with mtime+size for
    on-disk sources), and KD / subsample / filter flags. Any change forces a
    fresh preprocessed dataset; identical inputs reuse the cache.
    """
    tok_json_sha = None
    try:
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is not None:
            tok_json_sha = hashlib.sha256(backend.to_str().encode()).hexdigest()
    except Exception:
        pass

    dataset_id = {"name": script_args.dataset_name}
    if custom_args.precomputed_logits_dir:
        index_path = os.path.join(custom_args.precomputed_logits_dir, "index.jsonl")
        dataset_id["index_path"] = index_path
        try:
            st = os.stat(index_path)
            dataset_id["index_mtime"] = st.st_mtime
            dataset_id["index_size"] = st.st_size
        except OSError:
            pass
    elif script_args.dataset_name.endswith(".json") or script_args.dataset_name.endswith(".jsonl"):
        try:
            st = os.stat(script_args.dataset_name)
            dataset_id["mtime"] = st.st_mtime
            dataset_id["size"] = st.st_size
        except OSError:
            pass

    key_blob = {
        "tokenizer_json_sha": tok_json_sha,
        "chat_template": getattr(tokenizer, "chat_template", None) or "",
        "max_length": training_args.max_length,
        "dataset_text_field": training_args.dataset_text_field,
        "assistant_only_loss": getattr(training_args, "assistant_only_loss", False),
        "packing": getattr(training_args, "packing", False),
        "packing_strategy": getattr(training_args, "packing_strategy", None),
        "response_template": custom_args.response_template,
        "dataset_id": dataset_id,
        "precomputed_logits_dir": custom_args.precomputed_logits_dir or "",
        "kd_top_k": custom_args.kd_top_k,
        "kd_mode": bool(custom_args.precomputed_logits_dir),
        "dataset_subsample_start": custom_args.dataset_subsample_start,
        "dataset_subsample_end": custom_args.dataset_subsample_end,
        "dataset_subsample_seed": custom_args.dataset_subsample_seed,
        "dataset_shuffle": custom_args.dataset_shuffle,
        "filter_no_tools": custom_args.filter_no_tools,
        "filter_no_rag": custom_args.filter_no_rag,
        "filter_tools_only": custom_args.filter_tools_only,
        "filter_rag_only": custom_args.filter_rag_only,
        "filter_strict": custom_args.filter_strict,
        "max_dataset_size": custom_args.max_dataset_size,
    }

    blob_json = json.dumps(key_blob, sort_keys=True, default=str)
    hash_hex = hashlib.sha256(blob_json.encode()).hexdigest()[:32]

    # WHERE THE CACHE GOES, and the inherited answer is wrong outside that earlier checkout's layout.
    # `dirname(dirname(__file__))` was the repo root when this file sat two levels below it; in
    # this collection sft.py lives at src/gb_steps_post_training/distillation/sft.py, so the
    # same expression lands on src/gb_steps_post_training and the cache is written INSIDE the
    # source tree (harmless-looking: `.gitignore`'s `data/` rule covers it, and a direct
    # measurement duly wrote 1.5 MiB there). In an IMAGE the same expression resolves to
    # /opt/<step>/vendor/gb_steps_post_training, which is a container layer: writable only by
    # accident, discarded at exit, and not shared between the ranks of a multi-node job that do
    # not share a filesystem there. The cache is the one artifact whose whole value is being
    # found again by a LATER process.
    #
    # So the location is now the caller's to state, and run-sft.sh states it: the launcher knows
    # both the checkout root and the output directory, and this function knows neither. Falling
    # back to the historical path rather than refusing, because a bare `python sft.py` with no
    # launcher is how this file gets debugged and a cache miss is a slowdown, not a wrong number.
    cache_root = os.environ.get("SFT_PREPROCESS_CACHE_ROOT", "").strip()
    if not cache_root:
        cache_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "preprocessed_sft_cache",
        )
    cache_dir = os.path.join(cache_root, hash_hex)
    return cache_dir, key_blob


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, CustomSFTConfig, ModelConfig, CustomArguments))
    script_args, training_args, model_args, custom_args = parser.parse_args_and_config()

    # WHICH CODE IS THIS. Printed before anything else that can fail, because a run that dies
    # during setup is exactly the one whose code state you later want to know. It matters here
    # for a reason specific to a control arm: the whole point of this step is to be compared
    # against a gold run that dispatched hours or days apart, and the comparison is only sound
    # if both were the same code. Compare the DIGEST, not the commit -- see code_provenance.py.
    # Cannot end a run: every failure path prints "unavailable".
    from gb_steps_post_training.distillation import code_provenance as _code_provenance
    _code_provenance.report()

    print("Script args:")
    pprint(vars(script_args))

    print("\nTraining args:")
    pprint(vars(training_args))

    print("\nModel args:")
    pprint(vars(model_args))

    print("\nCustom args:")
    pprint(vars(custom_args))

    ################
    # Experiment tracking (ClearML default, W&B optional).
    #
    # HERE, and not later, for two reasons: `training_args.report_to` has to be final before the
    # trainer is constructed, and a tracking config that cannot be honoured should end the run
    # while it has cost nothing. run-sft.sh runs the same resolution in its preflight, before the
    # allocation -- this pass is the authoritative one because it sees the PARSED dataclass rather
    # than the yaml, so a `--clearml_project` on the command line counts.
    ################
    from gb_steps_post_training.distillation.tracking import (
        ALL_CONFIG_FIELDS as _TRACK_FIELDS,
        resolve as _resolve_tracking,
    )

    _cfg_path = None
    if "--config" in _sys.argv:
        _cfg_path = _sys.argv[_sys.argv.index("--config") + 1]
    # Same two-object fallthrough as gold.py even though all five fields are on CustomArguments
    # here: `custom_args` is asked FIRST and only falls through when it has no such attribute, so
    # a field that ever moves onto SFTConfig (upstream could add wandb_project at any release)
    # keeps working rather than silently resolving to None -- which for tracking means "off", the
    # failure mode that is invisible.
    def _track_field(name):
        for obj in (custom_args, training_args):
            if hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(
            f"tracking field {name!r} is on neither CustomArguments nor the SFTConfig. "
            "tracking.py's ALL_CONFIG_FIELDS and sft.py's dataclasses have diverged; "
            "a companion config check asserts they agree and would have said so.")

    # The teacher hint is the PRECOMPUTE DIRECTORY, and only when there is one. That is what makes
    # the auto-generated run name tell the truth about which of this step's two modes ran: with
    # `precomputed_logits_dir` set this is forward-KL distillation against a teacher that was
    # sampled offline, and without it there is no teacher at all. tracking.auto_run_name() reads
    # the absence of both this and `lmbda` as "plain SFT" and names it accordingly.
    _track = _resolve_tracking(
        {f: _track_field(f) for f in _TRACK_FIELDS},
        hints={
            "model_name_or_path": model_args.model_name_or_path,
            "teacher_model_name_or_path": custom_args.precomputed_logits_dir or None,
            "dataset_name": script_args.dataset_name,
        },
        config_path=_cfg_path,
    )
    # RANK rather than PartialState().is_main_process: PartialState() initialises the distributed
    # state, and this file deliberately does that later (at the `state = PartialState()` below,
    # after the tokenizer). Asking a question about logging must not move that.
    if int(os.environ.get("RANK", "0")) == 0:
        _track.report()
    if _track.fatal:
        # Every rank resolves the same config against the same env, so every rank aborts.
        raise SystemExit(1)
    _track.apply(training_args)

    # ---- Declared batch shape. See CustomArguments.effective_batch_size for why this is a field.
    # WORLD_SIZE from the environment rather than training_args.world_size, for the same reason the
    # rank guard above reads RANK: touching training_args.world_size constructs the distributed
    # state. torchrun sets WORLD_SIZE, and on this step every rank is a trainer rank -- there are
    # no server processes carved off the allocation, so the count is simply nodes x gpus_per_node.
    if custom_args.effective_batch_size:
        _ws = int(os.environ.get("WORLD_SIZE", "1"))
        _actual = (training_args.per_device_train_batch_size
                   * training_args.gradient_accumulation_steps * _ws)
        if _actual != custom_args.effective_batch_size:
            raise SystemExit(
                f"FATAL: this config declares effective_batch_size "
                f"{custom_args.effective_batch_size}, but this invocation gives {_actual} = "
                f"{training_args.per_device_train_batch_size} (per-device) x "
                f"{training_args.gradient_accumulation_steps} (grad-accum) x {_ws} (ranks).\n"
                "       The learning rate in this config was chosen for the declared batch, so a "
                "silent change of shape is a silent change of experiment. Fix the node shape, or "
                "change effective_batch_size AND re-justify the LR.")
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"batch shape: {_actual} = {training_args.per_device_train_batch_size} x "
                  f"{training_args.gradient_accumulation_steps} x {_ws} ranks, as declared")

    if custom_args.use_liger_swiglu_mlp:
        from _liger_granite_swa_patch import apply_liger_swiglu_to_granite_swa
        apply_liger_swiglu_to_granite_swa()

    ################
    # Model & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
    )
    verify_fast_tokenizer(
        tokenizer,
        model_args.model_name_or_path,
        source_label="student",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    state = PartialState()

    dataset_tmpdir = [None]

    def _load_dataset():
        if custom_args.precomputed_logits_dir:
            from datasets import load_from_disk

            meta_path = os.path.join(custom_args.precomputed_logits_dir, "meta.json")
            with open(meta_path) as f:
                meta = json.load(f)
            if meta["top_k"] != custom_args.kd_top_k:
                raise ValueError(
                    f"meta.json top_k={meta['top_k']} does not match kd_top_k={custom_args.kd_top_k}."
                )

            precompute_tok_path = meta["tokenizer_name_or_path"]
            precompute_tok = AutoTokenizer.from_pretrained(
                precompute_tok_path,
                trust_remote_code=model_args.trust_remote_code,
            )
            verify_fast_tokenizer(
                precompute_tok,
                precompute_tok_path,
                source_label="precompute-teacher",
            )
            verify_tokenizer_consistency(
                tokenizer,
                precompute_tok,
                train_source=model_args.model_name_or_path,
                ref_source=precompute_tok_path,
                context="KD (precomputed logits)",
            )

            dataset_tmpdir[0] = os.path.join(training_args.output_dir, ".tmp_dataset")

            if state.is_main_process:
                os.makedirs(dataset_tmpdir[0], exist_ok=True)
                index_path = os.path.join(custom_args.precomputed_logits_dir, "index.jsonl")
                rows = []
                with open(index_path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if row.get("skipped"):
                            continue
                        rows.append(row)

                # Per-row tools/rag filters (semantics mirror data/subsample_sft_4.1.py)
                def _has_tools(row):
                    tools = row.get("tools")
                    if not tools:
                        return False
                    if isinstance(tools, str):
                        try:
                            return bool(json.loads(tools))
                        except json.JSONDecodeError:
                            return False
                    if isinstance(tools, list):
                        return bool(tools)
                    return False

                def _has_rag(row):
                    docs = row.get("documents")
                    return bool(docs)

                def _keep(row):
                    has_tools = _has_tools(row)
                    has_rag = _has_rag(row)
                    if custom_args.filter_no_tools and has_tools:
                        return False
                    if custom_args.filter_no_rag and has_rag:
                        return False
                    if custom_args.filter_tools_only and not has_tools:
                        return False
                    if custom_args.filter_rag_only and not has_rag:
                        return False
                    return True

                rows = [r for r in rows if _keep(r)]

                if custom_args.dataset_shuffle:
                    import random
                    rng = random.Random(custom_args.dataset_subsample_seed)
                    rng.shuffle(rows)

                n_rows = len(rows)
                start = int(custom_args.dataset_subsample_start * n_rows)
                end = int(custom_args.dataset_subsample_end * n_rows)
                rows = rows[start:end]

                ds = Dataset.from_list(rows)
                ds.save_to_disk(dataset_tmpdir[0])

            if dist.is_initialized():
                dist.barrier()

            return DatasetDict({"train": load_from_disk(dataset_tmpdir[0])})
        elif script_args.dataset_name.endswith(".json") or script_args.dataset_name.endswith(".jsonl"):
            import pandas as pd
            from datasets import load_from_disk

            dataset_tmpdir[0] = os.path.join(training_args.output_dir, ".tmp_dataset")

            if state.is_main_process:
                os.makedirs(dataset_tmpdir[0], exist_ok=True)
                df = pd.read_json(script_args.dataset_name, lines=script_args.dataset_name.endswith(".jsonl"))
                dataset = Dataset.from_pandas(df)
                dataset.save_to_disk(dataset_tmpdir[0])

            if dist.is_initialized():
                dist.barrier()

            return DatasetDict({"train": load_from_disk(dataset_tmpdir[0])})
        elif script_args.dataset_name == "HuggingFaceTB/Countdown-Task-GOLD":
            ds = load_dataset(
                "HuggingFaceTB/Countdown-Task-GOLD",
                "verified_Qwen3-4B-Instruct-2507",
            )
            assert training_args.eval_strategy == "no", "Countdown dataset does not contain eval split"
            return ds
        else:
            return load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    def _cleanup_dataset_tmpdir():
        if dataset_tmpdir[0] and os.path.exists(dataset_tmpdir[0]):
            shutil.rmtree(dataset_tmpdir[0], ignore_errors=True)

    atexit.register(_cleanup_dataset_tmpdir)

    dataset = _load_dataset()

    if custom_args.max_dataset_size > 0:
        dataset[script_args.dataset_train_split] = dataset[script_args.dataset_train_split].select(
            range(min(custom_args.max_dataset_size, len(dataset[script_args.dataset_train_split])))
        )
    print(f"{script_args.dataset_name}: {len(dataset[script_args.dataset_train_split])}")

    sft_cache_dir, sft_cache_key_blob = _compute_sft_cache_dir(
        script_args, training_args, custom_args, tokenizer
    )
    # Printed with its PROVENANCE. A cache path is the kind of line a reader skims, and the two
    # cases mean different things: an env-set root is a deliberate, persistent location, while
    # the fallback is inside the source tree (or, in an image, inside a discarded layer).
    _cache_src = ("SFT_PREPROCESS_CACHE_ROOT" if os.environ.get("SFT_PREPROCESS_CACHE_ROOT", "").strip()
                  else "fallback beside sft.py -- NOT persistent in a container")
    print(f"SFT preprocessing cache dir: {sft_cache_dir}  [{_cache_src}]")

    ################
    # Training
    ################
    # Pre-trainer NCCL sanity broadcast: splits fabric/NCCL faults from
    # DeepSpeed ZeRO-3 large-buffer stalls. A direct measurement hung 30 min inside
    # zero.Init on a 411 M-elem tied-embed BROADCAST (SeqNum=2) on every
    # one of 64 ranks. A small broadcast here (1-elem, ~ms cost) that
    # succeeds isolates the fault to the large-buffer path; if it also
    # hangs, the fabric is at fault and the LSF allocation should be
    # rejected before the full 30-min watchdog wait.
    if dist.is_initialized():
        sanity_tensor = torch.zeros(1, device="cuda")
        dist.broadcast(sanity_tensor, src=0)
        torch.cuda.synchronize()
        print(f"SFT sanity broadcast OK on rank {dist.get_rank()}", flush=True)

    # SFTTrainer automatically tokenizes the dataset (messages -> input_ids) and uses
    # its own DataCollatorForLanguageModeling, so we don't need to pass a custom collator
    trainer_class = DebugSFTTrainer if os.environ.get("DEBUG_SFT") else CustomSFTTrainer
    trainer = trainer_class(
        response_template=custom_args.response_template,
        use_liger_memory_opt=custom_args.use_liger_memory_opt,
        precomputed_logits_dir=custom_args.precomputed_logits_dir,
        kd_top_k=custom_args.kd_top_k,
        kd_weight=custom_args.kd_weight,
        ce_weight=custom_args.ce_weight,
        kd_temperature=custom_args.kd_temperature,
        sft_cache_dir=sft_cache_dir,
        sft_cache_key_blob=sft_cache_key_blob,
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_length, do_sample=True, temperature=0.7
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    verify_optimization_stack(
        trainer.model,
        expect_liger_swiglu=custom_args.use_liger_swiglu_mlp,
    )

    # WHICH ROWS THIS RUN TRAINS ON, recorded before the first step.
    #
    # Placed here and not earlier because `trainer.train_dataset` is the dataset the trainer
    # actually built, after its own tokenization and any row filtering. Placed here and not after
    # train() because the cross-checks inside can REFUSE the run: a trainer/prep disagreement
    # means the manifest would describe a corpus other than the one being trained on, and that is
    # worth catching while the only cost is a model load. Writes on rank 0 only and never fails
    # for a merely absent id column.
    #
    # The SAME function as the gold path, deliberately -- it reads `lmbda`, `max_length` and
    # `max_completion_length` with getattr defaults, so on SFTConfig (which has no lmbda) it
    # records the off-policy arm, which is the truth about a run that never samples. Two arms
    # whose manifests were built by two different implementations could not be compared row for
    # row, which is the one thing this manifest exists to allow.
    _tm_manifest = train_manifest.build_from_trainer(
        trainer, dataset_path=script_args.dataset_name)
    if _tm_manifest is not None:
        _tm_rows = _tm_manifest.get("rows_file", {}).get("entries")
        print(f"provenance : {_tm_manifest['arm']} arm, "
              f"{_tm_manifest['rows_available']} rows available"
              + (f", {_tm_rows} ids -> {train_manifest.CONSUMED_NAME}"
                 if _tm_rows else f" (no row ids: {_tm_manifest['ids_unavailable_reason']})")
              + f"; {train_manifest.MANIFEST_NAME} written")
        _tm_frac = (_tm_manifest.get("schedule") or {}).get("epoch_fraction_scheduled")
        if _tm_frac is not None and _tm_frac < 1.0:
            print(f"provenance : NOTE this schedule covers {_tm_frac:.1%} of an epoch, so the "
                  "rows-available count above is an upper bound on rows actually seen")

    # Split ClearML train scalars across per-quantity plots. AFTER the Trainer exists (it adds the
    # ClearMLCallback itself from `report_to`) and BEFORE train() (the callback's setup runs lazily
    # on the first log).
    from gb_steps_post_training.distillation.clearml_scalar_groups import install as _install_scalar_groups
    _install_scalar_groups(trainer)

    if glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")):
        print(f"Resuming training from checkpoint under {training_args.output_dir}")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
