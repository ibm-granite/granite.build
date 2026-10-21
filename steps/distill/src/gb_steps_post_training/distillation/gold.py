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

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "peft",
#     "trackio",
# ]
# ///

import atexit
import shutil

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

# Workaround: vLLM 0.11.0 accesses config.rope_theta directly, but transformers 5.x
# stores it inside config.rope_parameters dict instead. Inject the attribute on-the-fly.
# Remove this patch once vLLM is updated to handle transformers 5.x configs natively.
_vllm_gmh = importlib.import_module("vllm.model_executor.models.granitemoehybrid")
_orig_gmh_attn_init = _vllm_gmh.GraniteMoeHybridAttention.__init__

def _patched_gmh_attn_init(self, config, *args, _orig=_orig_gmh_attn_init, **kwargs):
    if not hasattr(config, "rope_theta"):
        rp = getattr(config, "rope_parameters", None)
        if isinstance(rp, dict) and "rope_theta" in rp:
            config.rope_theta = rp["rope_theta"]
        else:
            config.rope_theta = 10000
    _orig(self, config, *args, **kwargs)

_vllm_gmh.GraniteMoeHybridAttention.__init__ = _patched_gmh_attn_init
del _vllm_gmh, _orig_gmh_attn_init

# Same issue for regular GraniteForCausalLM (not MoeHybrid).
# GraniteDecoderLayer.__init__ reads config.rope_theta with a fallback to 10000,
# which is wrong for models like granite-4.1-20b that have rope_theta=50000000.
_vllm_granite = importlib.import_module("vllm.model_executor.models.granite")
_orig_granite_layer_init = _vllm_granite.GraniteDecoderLayer.__init__

def _patched_granite_layer_init(self, config, *args, _orig=_orig_granite_layer_init, **kwargs):
    if not hasattr(config, "rope_theta"):
        rp = getattr(config, "rope_parameters", None)
        if isinstance(rp, dict) and "rope_theta" in rp:
            config.rope_theta = rp["rope_theta"]
        else:
            config.rope_theta = 10000
    _orig(self, config, *args, **kwargs)

_vllm_granite.GraniteDecoderLayer.__init__ = _patched_granite_layer_init
del _vllm_granite, _orig_granite_layer_init

# Workaround: DeepSpeed ZeRO optimizer wrappers don't expose `defaults`
# from the underlying optimizer. transformers' cosine_with_min_lr scheduler needs it.
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3 as _DS3Opt
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer as _DS12Opt
for _cls in (_DS3Opt, _DS12Opt):
    if not hasattr(_cls, "defaults"):
        _cls.defaults = property(lambda self: self.optimizer.defaults)
del _DS3Opt, _DS12Opt, _cls

# Workaround: The kernels-community/mamba-ssm HF Hub kernel imports
# GreedySearchDecoderOnlyOutput and SampleDecoderOnlyOutput which were removed in
# transformers 5.x. Inject them as aliases for GenerateDecoderOnlyOutput.
import transformers.generation as _tf_gen
if not hasattr(_tf_gen, "GreedySearchDecoderOnlyOutput"):
    from transformers.generation import GenerateDecoderOnlyOutput as _GenOut
    _tf_gen.GreedySearchDecoderOnlyOutput = _GenOut
    _tf_gen.SampleDecoderOnlyOutput = _GenOut
    del _GenOut
del _tf_gen

# FA3 dispatch workaround for the vendored GraniteSWA model. In an earlier
# exploratory checkout this was an unconditional `import _fa3_preamble` -- which raises on any machine
# whose HF cache has no kernels-community/vllm-flash-attn3 snapshot, i.e. on this
# account, before a single batch, confirmed by direct measurement. Nothing in the granite 4.1/4.2
# pairing uses sliding-window attention, so the import is now gated on the SWA arm
# actually being requested. See _swa_arm.py for why the preamble's refusal is
# correct for the model it was written for and must not be softened.
from gb_steps_post_training.distillation._swa_arm import (
    activate_swa_arm,
    activate_swa_vllm_model,
)
activate_swa_arm()

# Workaround: The kernels-community/mamba-ssm HF Hub kernel internally does
# `from causal_conv1d.causal_conv1d_interface import causal_conv1d_cuda`.
# Two problems: (1) the `kernels` package registers modules with hash-based names
# so `causal_conv1d` is never directly importable, and (2) the HF Hub version of
# causal-conv1d doesn't expose the legacy `causal_conv1d_cuda` object.
# Fix: Pre-load the kernel, register it under the expected name in sys.modules,
# and provide a shim for causal_conv1d_cuda wrapping the new cpp_functions API.
# This requires transformers 5.x hub_kernels module; skip on older versions.
import sys as _sys
try:
    from transformers.integrations.hub_kernels import lazy_load_kernel as _lazy_load_kernel
except (ImportError, ModuleNotFoundError):
    _lazy_load_kernel = None

# Pre-populate transformers' kernel module mapping from the warm local cache so that
# `lazy_load_kernel` short-circuits without ever calling `HfApi().list_repo_refs`
# (which would happen for `version=`-style entries in `_HUB_KERNEL_MAPPING`) or
# `snapshot_download` (which races across ranks on NFS). This is the supported
# short-circuit at transformers/integrations/hub_kernels.py:368-369. Pure cache
# reads via `install_kernel(..., revision=<sha>, local_files_only=True)` only
# perform `os.readlink` on existing symlinks and are rank-safe.
try:
    from pathlib import Path as _Path
    from huggingface_hub.constants import HF_HUB_CACHE as _HF_HUB_CACHE
    from kernels.utils import install_kernel as _install_kernel, _import_from_path
    from transformers.integrations.hub_kernels import (
        _KERNEL_MODULE_MAPPING,
        _HUB_KERNEL_MAPPING,
    )

    def _read_cached_sha(_repo_id):
        # The current `kernels` package writes the standard HF hub layout
        # (`models--<repo>/...`) via `snapshot_download`, not the legacy
        # `kernels--<repo>/...` prefix. See kernels/utils.py:194-203.
        _ref_path = (
            _Path(_HF_HUB_CACHE)
            / f"models--{_repo_id.replace('/', '--')}"
            / "refs"
            / "main"
        )
        return _ref_path.read_text().strip()

    def _validated_import_from_path(_package_name, _variant_path):
        # `_import_from_path` will raise on a structurally-broken metadata.json
        # only when the loader actually parses it; pre-validate so a corrupt
        # cache aborts the preamble loudly with a JSONDecodeError instead of
        # later showing up as a hard-to-diagnose ImportError under the FA3
        # dispatcher.
        import json as _json
        _meta_path = _Path(_variant_path) / "metadata.json"
        if _meta_path.exists():
            _json.loads(_meta_path.read_text())
        return _import_from_path(_package_name, _variant_path)

    def _preload_kernel_into_mapping(_kernel_name, _repo_id):
        from types import ModuleType as _ModuleType
        _existing = _KERNEL_MODULE_MAPPING.get(_kernel_name)
        if isinstance(_existing, _ModuleType):
            return
        _sha = _read_cached_sha(_repo_id)
        _package_name, _variant_path = _install_kernel(
            _repo_id, revision=_sha, local_files_only=True
        )
        _module = _validated_import_from_path(_package_name, _variant_path)
        _KERNEL_MODULE_MAPPING[_kernel_name] = _module
        return _module

    for _kname, _kspec in (
        ("causal-conv1d", "kernels-community/causal-conv1d"),
        ("mamba-ssm", "kernels-community/mamba-ssm"),
    ):
        if _kname in _HUB_KERNEL_MAPPING:
            _preload_kernel_into_mapping(_kname, _kspec)

    del _kname, _kspec, _Path, _HF_HUB_CACHE, _install_kernel, _import_from_path
    del _KERNEL_MODULE_MAPPING, _HUB_KERNEL_MAPPING
    del _read_cached_sha, _preload_kernel_into_mapping, _validated_import_from_path
except (FileNotFoundError, OSError, ImportError, ModuleNotFoundError) as _kernel_preload_err:
    # Fresh node / kernels API change / missing cache. Fall through to the
    # online `lazy_load_kernel` path below; preflight will repopulate the cache.
    # Note: ValueError (incl. JSONDecodeError) is intentionally NOT caught here.
    # A corrupt cache (e.g. doubled-JSON metadata.json from a parallel-write race)
    # must surface as a hard failure so the launcher's pre-download cleanup runs,
    # rather than fall through to the racing kernel-hub fallback path that would
    # produce a confusing downstream `ImportError: FlashAttention3 has been toggled on`.
    print(f"[gold preamble] kernel pre-population skipped: {type(_kernel_preload_err).__name__}: {_kernel_preload_err}")
    del _kernel_preload_err

if _lazy_load_kernel is not None:
    _cc1d = _lazy_load_kernel("causal-conv1d")
    if _cc1d is not None and "causal_conv1d" not in _sys.modules:
        _sys.modules["causal_conv1d"] = _cc1d

        _cc1d_iface = getattr(_cc1d, "causal_conv1d_interface", None)
        if _cc1d_iface is not None:
            _fwd = getattr(_cc1d_iface, "causal_conv1d_fwd_function", None)
            _bwd = getattr(_cc1d_iface, "causal_conv1d_bwd_function", None)

            if _fwd is not None and _bwd is not None:
                class _CausalConv1dCudaShim:
                    causal_conv1d_fwd = staticmethod(_fwd)
                    causal_conv1d_bwd = staticmethod(_bwd)

                _cc1d_iface.causal_conv1d_cuda = _CausalConv1dCudaShim()

            _sys.modules["causal_conv1d.causal_conv1d_interface"] = _cc1d_iface
    del _cc1d, _lazy_load_kernel

# `granite_swa` Auto*-API registration, and the matching vLLM ModelRegistry entry
# that `vllm_mode: colocate` needs (the colocate path instantiates vllm.LLM inside
# this process and never imports run_vllm_serve.py). Both were unconditional
# module-scope imports of the vendored package in that earlier checkout; both are now
# gated on the SWA arm. activate_swa_arm() was already called above and is
# idempotent -- the second call here keeps this block readable on its own and
# keeps the "must run before any `from trl...` import" ordering visible.
activate_swa_arm()
activate_swa_vllm_model()

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold.gold_config import GOLDConfig
from trl.experimental.gold.gold_trainer import GOLDTrainer
from trl.trainer.utils import DataCollatorForChatML

from gb_steps_post_training.distillation.custom_gold_trainer import CustomGOLDTrainer
from gb_steps_post_training.distillation.custom_gold_config import CustomGOLDConfig
from gb_steps_post_training.distillation.utils import (
    CustomDataCollatorForChatML,
    verify_fast_tokenizer,
    verify_tokenizer_consistency,
)
# Top-level, unlike `tracking` (imported inside main at the point where report_to must be
# final): train_manifest is stdlib-only, so importing it here costs nothing and a typo in it
# fails before the allocation does any work, rather than after a 30B teacher has been loaded.
from gb_steps_post_training.distillation import train_manifest

from dataclasses import dataclass, field

from pprint import pprint

import os
import glob
import torch.distributed as dist
from accelerate import PartialState

@dataclass
class CustomArguments:
    """
    Custom arguments for your training script.
    """
    max_dataset_size: int = field(
        default=-1,
        metadata={"help": "Reduce dataset size for quick debugging"}
    )
    teacher_attn_implementation: str | None = field(
        default=None,
        metadata={"help": "Override attn_implementation for the teacher model. "
                          "If None, falls back to model_args.attn_implementation."},
    )
    # A DECLARED INVARIANT, not a knob. Nothing downstream reads it; setting it changes no
    # behaviour. What it does is let a config state the batch shape it was tuned for, so that a
    # node shape which silently produces a different one is a refusal instead of a run.
    #
    # WHY IT IS A TYPED FIELD AND NOT JUST A COMMENT, learned the expensive way. It began as a key
    # that only gold-submit.sh read, on the reasoning -- written into that script -- that "the
    # trainer never reads it". That reasoning was wrong about the mechanism: TrlParser hands the
    # WHOLE yaml to the argument parser with fail_with_unknown_args=True, so a key nobody reads is
    # not ignored, it raises. A direct measurement took a 4-node allocation, passed every preflight, and
    # died 42 s in with `Unknown arguments from config file: ['--effective_batch_size', '192']` --
    # and all five of the deliverable configs carried the key, so none of them could have launched.
    # The rule was already written 15 lines below for the clearml fields; this is the same rule.
    #
    # ASSERTED AT TRAIN TIME rather than only at submit time, and the train-time check is the
    # stronger one: gold-submit.sh multiplies by a GPU count it infers from the config filename's
    # `_node{N}` suffix, while the block after parsing multiplies by the WORLD_SIZE the launcher
    # actually produced. The two disagree exactly when the naming contract has drifted, which is
    # the case worth catching.
    effective_batch_size: int | None = field(
        default=None,
        metadata={"help": "Optional. The per_device x grad_accum x world_size product this "
                          "config was tuned for. When set, a run whose actual product differs "
                          "aborts before the model is loaded. Nothing reads it as a setting."},
    )
    use_liger_swiglu_mlp: bool = field(
        default=False,
        metadata={"help": "Monkey-patch GraniteSWAMLP.forward to use Liger's "
                          "tiled SwiGLU MLP (see gold/_liger_granite_swa_patch.py). "
                          "Required for long-context GraniteSWA training to avoid "
                          "backward-pass OOM on the MLP intermediate activations."},
    )

    # ---- Experiment tracking. ClearML is the DEFAULT backend; W&B is optional. The rules and
    # the (measured) list of what each backend actually reads live in tracking.py -- read its
    # docstring before changing any of this. In short: a backend needs a COMPLETE config to be
    # enabled, a PARTIAL config aborts the launch, and a complete config with no credentials
    # logs nothing and says so at full volume.
    #
    # These live on CustomArguments rather than in a `clearml:` block in the yaml because
    # TrlParser.parse_args_and_config defaults to fail_with_unknown_args=True -- an unrecognised
    # top-level key does not get ignored, it raises. A typed field is the only place config can
    # go, and it gets validated by the same parser as everything else.
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
    # `wandb_entity` and `wandb_project` are DELIBERATELY ABSENT here. They are already fields of
    # upstream `GOLDConfig` (trl/experimental/gold/gold_config.py:339,343), which CustomGOLDConfig
    # inherits, so declaring them again made TrlParser raise before parsing a single argument:
    #
    #     argparse.ArgumentError: argument --wandb_entity/--wandb-entity: conflicting option strings
    #
    # i.e. gold.py could not start AT ALL. Measured: trl declares those three (`wandb_run_group`
    # too) and READS none of them anywhere in the package -- inert config surface. So the right
    # move is to use them rather than shadow them: the names are the ones a trl user expects, the
    # rendered yaml key is unchanged, and the resolution below now reads them off `training_args`.
    # W&B's run name is NOT among them -- upstream has `wandb_run_group`, a different coordinate --
    # so `wandb_run_name` stays here.
    wandb_run_name: str | None = field(
        default=None,
        metadata={"help": "W&B run name (-> TrainingArguments.run_name, which is what transformers "
                          "passes to wandb.init as `name`). Optional; auto-derived when omitted."},
    )

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, CustomGOLDConfig, ModelConfig, CustomArguments))
    script_args, training_args, model_args, custom_args = parser.parse_args_and_config()

    # WHICH CODE IS THIS. Printed before anything else that can fail, because a run that dies during
    # setup is exactly the one whose code state you later want to know. The lmbda sweep needs it: its
    # arms dispatch hours apart from a repo still being committed to, and a companion check
    # asserts the arms' CONFIGS match while nothing asserted their code did. Compare the DIGEST, not
    # the commit -- see code_provenance.py. Cannot end a run: every failure path prints "unavailable".
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
    # while it has cost nothing. gold-train-onpolicy.sh already ran the same resolution in its
    # preflight, before the allocation -- this pass is the authoritative one because it sees the
    # PARSED dataclass rather than the yaml, so a `--clearml_project` on the command line counts.
    ################
    from gb_steps_post_training.distillation.tracking import (
        ALL_CONFIG_FIELDS as _TRACK_FIELDS,
        resolve as _resolve_tracking,
    )

    _cfg_path = None
    if "--config" in _sys.argv:
        _cfg_path = _sys.argv[_sys.argv.index("--config") + 1]
    # Two objects, because the five fields have two owners: `clearml_*` and `wandb_run_name` are
    # ours (CustomArguments), `wandb_entity`/`wandb_project` are upstream GOLDConfig's and arrive on
    # training_args. `custom_args` is asked FIRST and only falls through when it has no such
    # attribute, so a field that ever moves between the two keeps working rather than silently
    # resolving to None -- which for tracking means "off", the failure mode that is invisible.
    def _track_field(name):
        for obj in (custom_args, training_args):
            if hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(
            f"tracking field {name!r} is on neither CustomArguments nor the GOLDConfig. "
            "tracking.py's ALL_CONFIG_FIELDS and gold.py's dataclasses have diverged; "
            "a companion config check asserts they agree and would have said so.")

    _track = _resolve_tracking(
        {f: _track_field(f) for f in _TRACK_FIELDS},
        hints={
            "model_name_or_path": model_args.model_name_or_path,
            "teacher_model_name_or_path": training_args.teacher_model_name_or_path,
            "dataset_name": script_args.dataset_name,
            "lmbda": training_args.lmbda,
        },
        config_path=_cfg_path,
    )
    # RANK rather than PartialState().is_main_process: PartialState() initialises the distributed
    # state, and gold.py deliberately does that later (line ~368, just before the trainer). Asking
    # a question about logging must not move that.
    if int(os.environ.get("RANK", "0")) == 0:
        _track.report()
    if _track.fatal:
        # Every rank resolves the same config against the same env, so every rank aborts.
        raise SystemExit(1)
    _track.apply(training_args)

    # ---- Declared batch shape. See CustomArguments.effective_batch_size for why this is a field.
    # WORLD_SIZE from the environment rather than training_args.world_size, for the same reason the
    # rank guard above reads RANK: touching training_args.world_size constructs the distributed
    # state, and gold.py deliberately initialises that later. torchrun sets WORLD_SIZE, and on the
    # on-policy arm it counts trainer ranks only -- the vLLM servers are separate processes on
    # hosts carved off the tail of the allocation, not members of this group.
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
        revision=training_args.student_model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    if training_args.teacher_tokenizer_name_or_path is None and training_args.use_uld_loss:
        training_args.teacher_tokenizer_name_or_path = training_args.teacher_model_name_or_path
    teacher_model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=custom_args.teacher_attn_implementation or model_args.attn_implementation,
        torch_dtype=model_args.dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.teacher_model_init_kwargs = teacher_model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    verify_fast_tokenizer(
        tokenizer,
        model_args.model_name_or_path,
        source_label="student",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if student and teacher use the same tokenizer
    assert training_args.teacher_model_name_or_path is not None, "--teacher_model_name_or_path missing"
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        training_args.teacher_model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    verify_fast_tokenizer(
        teacher_tokenizer,
        training_args.teacher_model_name_or_path,
        source_label="teacher",
    )
    student_vocab_size = len(tokenizer)
    teacher_vocab_size = len(teacher_tokenizer)
    if student_vocab_size != teacher_vocab_size:
        print("WARNING: student and teacher have different vocab size")

    # ULD (cross-tokenizer) distillation INTENDS the student and teacher tokenizers to differ,
    # and ULDLoss aligns them via byte offsets. The consistency check only applies to the
    # shared-vocab (JSD) path, where a mismatch would silently corrupt token-span alignment.
    # So skip it when use_uld_loss is set.
    if not training_args.use_uld_loss:
        # The context string is the FIRST thing a reader sees in the traceback, so it must name
        # the arm that actually failed. It was hardcoded "on-policy GOLD" while this branch is
        # reached by ANY use_uld_loss=false run -- so the b9 smoke (lmbda 0.0, off-policy by
        # config) died reporting "on-policy GOLD", and the first hour of that diagnosis went
        # looking for a bug in the on-policy path that was never involved. lmbda is the same
        # switch the trainer itself uses to decide whether to sample from the student.
        mode = "on-policy" if (training_args.lmbda or 0) > 0 else "off-policy"
        verify_tokenizer_consistency(
            tokenizer,
            teacher_tokenizer,
            train_source=model_args.model_name_or_path,
            ref_source=training_args.teacher_model_name_or_path,
            context=f"{mode} GOLD (lmbda={training_args.lmbda})",
        )
    else:
        print("ULD loss enabled: skipping verify_tokenizer_consistency (cross-tokenizer distillation expects differing tokenizers)")

    ################
    # Dataset
    ################
    # Synchronize dataset loading across distributed processes to avoid cache race conditions
    state = PartialState()

    dataset_tmpdir = [None]

    def _load_dataset():
        if script_args.dataset_name.endswith(".json") or script_args.dataset_name.endswith(".jsonl"):
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

    ################
    # Training
    ################
    print("Using CustomGOLDTrainer and CustomDataCollatorForChatML")
    trainer_class = CustomGOLDTrainer
    data_collator = CustomDataCollatorForChatML(
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        instruction_template=training_args.instruction_template,
        response_template=training_args.response_template,
        last_message_only=training_args.last_message_only,
    )

    trainer = trainer_class(
        model=model_args.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        data_collator=data_collator,
    )

    if getattr(training_args, "vllm_mode", None) == "server":
        _probe_failure = None
        if trainer.accelerator.is_main_process and getattr(trainer, "vllm_client", None) is not None:
            for _probe in ("a\n\nb", "public.roads2", "1200"):
                _local_ids = tokenizer.encode(_probe, add_special_tokens=False)
                _vllm_resp = trainer.vllm_client.generate(
                    prompts=[_probe],
                    n=1,
                    max_tokens=1,
                    temperature=0.0,
                )
                _vllm_prompt_ids = _vllm_resp["prompt_ids"][0]
                if _vllm_prompt_ids != _local_ids:
                    _probe_failure = (
                        f"[trainer→vllm] tokenization drift on probe {_probe!r}:\n"
                        f"  trainer-local ids: {_local_ids}\n"
                        f"  vllm prompt_ids:   {_vllm_prompt_ids}\n"
                        f"  student dir:       {model_args.model_name_or_path}\n"
                        "See docs/tokenizer_mismatch.md → 'Which tokenizer was "
                        "the teacher trained with?' for root cause."
                    )
                    break
        trainer.accelerator.wait_for_everyone()
        from accelerate.utils import broadcast_object_list as _bcast
        _probe_list = _bcast([_probe_failure], from_process=0)
        if _probe_list[0] is not None:
            raise RuntimeError(_probe_list[0])

    # Log student/teacher model configs for debugging
    if state.is_main_process:
        from transformers import AutoConfig

        _CONFIG_FIELDS = [
            "architectures", "model_type",
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "vocab_size",
            "max_position_embeddings",
            "rope_theta", "rope_parameters", "rope_scaling",
            "torch_dtype", "tie_word_embeddings", "use_cache",
            "num_local_experts", "num_experts_per_tok",
        ]

        def _log_config(label, model_path):
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=model_args.trust_remote_code)
            print(f"\n{'='*60}")
            print(f"  {label} config: {model_path}")
            print(f"{'='*60}")
            for field in _CONFIG_FIELDS:
                val = getattr(cfg, field, None)
                if val is not None:
                    print(f"  {field}: {val}")
            print()

        _log_config("Student", model_args.model_name_or_path)
        _log_config("Teacher", training_args.teacher_model_name_or_path)

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_new_tokens, do_sample=True, temperature=training_args.temperature
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    if custom_args.use_liger_swiglu_mlp:
        from _liger_granite_swa_patch import verify_liger_swiglu_active
        failures = verify_liger_swiglu_active(trainer.model)
        if failures:
            raise RuntimeError("Liger tiled-MLP patch not active: " + "; ".join(failures))

    # WHICH ROWS THIS RUN TRAINS ON, recorded before the first step.
    #
    # Placed here and not earlier because `trainer.train_dataset` is the POST-FILTER dataset:
    # the trainer applies its own row filter at custom_gold_trainer.py:1824, and an
    # arm-dependent one -- on-policy additionally drops prompts at or above
    # `max_length - max_completion_length`, off-policy keeps them. Asking prep to predict that
    # would mean a second implementation of the trainer's prompt render, so the question is put
    # to the dataset the trainer actually built.
    #
    # Placed here and not after train() because the cross-checks inside can REFUSE the run: a
    # trainer/prep disagreement means the manifest would describe a corpus other than the one
    # being trained on, and that is worth catching while the only cost is a model load. It
    # writes on rank 0 only and never fails for a merely absent id column.
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
            # Said out loud because the rows-available count is the number a reader will quote,
            # and for a short run it overstates what was seen by roughly this factor.
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

    # Reset tokenizer truncation state before saving to avoid polluting the saved config
    if hasattr(tokenizer, 'backend_tokenizer') and tokenizer.backend_tokenizer.truncation is not None:
        tokenizer.backend_tokenizer.no_truncation()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
