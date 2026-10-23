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

"""Single source of truth for the optional Shadow Residual (SR) tuning algorithm.

Shadow Residual is a dual-stream Granite decoder: a frozen base stream that stays
bit-exact with the unadapted base, plus a trainable adapter stream fed by stock
LoRA deltas and a per-layer rank-R ``cross_stream`` injection from the base. Only
``norm(h_adapt)`` reaches the LM head.

The adapter itself is **100% stock PEFT LoRA** -- there is no custom ``LoraLayer``
and no ``PeftType`` registration. ``CrossStream`` is a real frozen zero-init
``nn.Linear(H, H)`` that stock ``peft.tuners.lora.layer.Linear`` wraps, yielding a
pure ``B*A*scaling`` rank-R injection. So SR reduces to "a different way to
*construct* the model, then a normal ``LoraConfig`` on top", which is why the
drivers only need two changes: build the model through :func:`build_sr_peft_model`
instead of ``AutoModelForCausalLM.from_pretrained``, and skip ``get_peft_model``
(SR's factory returns an already-wrapped ``PeftModel``).

All ``shadow_residual`` imports are lazy (inside functions), mirroring how
mlx_backend.py imports mlx, so this module imports cleanly whether or not the SR
package is installed. SR is not a declared fm-tune dependency: install it with

    git clone https://github.com/generative-computing/shadow-residual
    pip install -e shadow-residual --no-deps

``--no-deps`` is deliberate. fm-tune's transformers 5.9.0 / peft 0.19.1 pins now
satisfy shadow-residual's ``transformers>=5.5.1,<5.10`` / ``peft==0.19.1``, but it
also asks for ``trl<1.7`` while fm-tune pins trl 1.12.0 (verl's PPO critic needs
``trl.experimental.ppo``); SR's own code does not import TRL, so the remaining
conflict is metadata-only.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

# String sentinel rather than a PeftType member, following the "alora": "ALORA"
# precedent in constants.py. SR produces a plain LoraConfig, so PEFT itself never
# sees this value -- only fm-tune's own dispatch does.
SR_PEFT_TYPE = "SR"

# The tuning-algorithm name as it appears on the CLI and in training_config.
SR_TUNING_ALGO = "sr"

# The decoder layer class accelerate's TRANSFORMER_BASED_WRAP policy wraps. Not
# passed to TrainingArguments: _build_fsdp_config deliberately omits
# fsdp_transformer_layer_cls_to_wrap and relies on model._no_split_modules, which
# resolves through the PEFT wrapper (PeftModel.__getattr__ -> base_model,
# BaseTuner.__getattr__ -> self.model). Kept here for log messages and tests.
SR_WRAP_CLASS = "ShadowResidualDecoderLayer"

# SR-safe LoRA target set, mirroring shadow_residual.config.adapters._SR_SCALAR_TARGETS.
#
# Excludes k_proj/v_proj: SR uses shared base-only K/V (K and V are computed once
# from the base stream, one KV cache), so an adapter-side K/V delta has nowhere to
# land. SR's factory rejects them, and rejects target_modules="all-linear" too,
# because that would expand to the forbidden projections (SR's k_proj/v_proj are
# _StreamGatedLinear, an nn.Linear subclass, so "all-linear" *would* match them).
#
# Includes cross_stream: its presence is what engages the dual-stream forward. A
# config without it silently trains a plain single-stream LoRA.
#
# Never build this list from utils.get_adapter_compatible_modules() -- that helper
# returns the bare string "all-linear" unconditionally, which SR rejects outright.
SR_TARGET_MODULES = [
    "q_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "cross_stream",
]

_SR_INSTALL_HINT = (
    "The 'sr' (Shadow Residual) tuning algorithm requires the shadow-residual "
    "package, which is not a declared fm-tune dependency. Install it with:  "
    "git clone https://github.com/generative-computing/shadow-residual && "
    "pip install -e shadow-residual --no-deps  "
    "(--no-deps is required: shadow-residual's metadata asks for trl<1.7, which "
    "conflicts with fm-tune's trl 1.12.0 pin; SR's code does not use TRL, so the "
    "conflict is metadata-only)."
)


def require_sr() -> None:
    """Import the shadow_residual package, raising a clear install hint if absent."""
    try:
        from shadow_residual.training.factory import (  # noqa: F401
            get_shadow_residual_peft_model,
        )
    except ImportError as e:
        raise ImportError(f"{_SR_INSTALL_HINT} Original import error: {e}") from e


def is_sr(training_config: Optional[Dict[str, Any]]) -> bool:
    """True when this run's tuning algorithm is Shadow Residual."""
    return (training_config or {}).get("tuning_algorithm") == SR_TUNING_ALGO


def build_sr_lora_config(base_kwargs: Dict[str, Any]) -> Any:
    """Build the stock ``LoraConfig`` that drives a Shadow Residual model.

    Reproduces shadow_residual.config.adapters.to_peft_config: SR carries both
    the module selection and the per-module rank in one place, so the projection
    into PEFT's three-field surface is (r, target_modules, rank_pattern) with the
    *minimum* rank as the scalar default and **every** target named explicitly in
    ``rank_pattern``.

    Naming every target in ``rank_pattern`` is load-bearing, not tidiness: when a
    trial samples ``cross_stream_r < r``, ``r=min(...)`` drops the scalar default,
    and without a ``rank_pattern`` entry the other five sites would silently train
    at ``cross_stream_r`` instead of the ``r`` the trial thinks it is evaluating.
    PEFT matches ``rank_pattern`` keys as ``re.match(rf".*\\.{key}$", key_path)``,
    so the bare name ``"cross_stream"`` correctly hits ``model.layers.N.cross_stream``.

    Args:
        base_kwargs: Dict[str, Any]
            The sampled tuner hyperparams (the ``for_tuner: true`` bucket). Reads
            ``r``, ``cross_stream_r``, ``alpha_ratio``, ``lora_dropout``, ``bias``.

    Returns:
        A ``peft.LoraConfig``. ``peft_type`` is LORA -- SR needs no custom type.
    """
    from peft import LoraConfig, TaskType

    r = int(base_kwargs.get("r", 32))
    # The SR-specific knob: the rank of the per-layer cross-stream injection from
    # the frozen base into the adapter stream. Defaults to r when unset so a
    # config that omits it still builds a valid (uniform-rank) SR model.
    cross_stream_r = int(base_kwargs.get("cross_stream_r", r))

    ranks = {name: r for name in SR_TARGET_MODULES}
    ranks["cross_stream"] = cross_stream_r

    # alpha_ratio reaches us through peft_kwargs (for_tuner: true) rather than
    # train_kwargs, unlike every other fm-tune tuner. The drivers would otherwise
    # compute lora_alpha = alpha_ratio * peft_kwargs["r"], but SR's `r` is the
    # *minimum* rank (PEFT's scalar default, with per-name overrides in
    # rank_pattern), which would give alpha from min instead of max. SR's own
    # default is alpha = 2 * max_rank (AdapterConfig._default_alpha), so compute
    # it from max(ranks) here. This means alpha is pinned to the largest rank and
    # each site's effective scaling is alpha / r_site -- do not "simplify" it back
    # into the drivers, which would silently change every SR run's scaling.
    alpha_ratio = base_kwargs.get("alpha_ratio")
    if alpha_ratio is None:
        alpha_ratio = 2.0
    lora_alpha = int(float(alpha_ratio) * max(ranks.values()))

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        # Explicit list, never "all-linear": SR's factory rejects that string
        # because it expands to the forbidden K/V projections.
        target_modules=list(SR_TARGET_MODULES),
        r=min(ranks.values()),
        rank_pattern=dict(ranks),
        lora_alpha=lora_alpha,
        lora_dropout=base_kwargs.get("lora_dropout", 0.05),
        bias=base_kwargs.get("bias", "none"),
    )
    logger.info(
        "[AutoTune] Shadow Residual LoRA config: r=%s cross_stream_r=%s "
        "lora_alpha=%s (alpha_ratio=%s x max_rank=%s) dropout=%s targets=%s",
        config.r,
        cross_stream_r,
        lora_alpha,
        alpha_ratio,
        max(ranks.values()),
        config.lora_dropout,
        config.target_modules,
    )
    return config


def build_sr_peft_model(
    model_name_or_path: str,
    lora_config: Any,
    *,
    torch_dtype: Any = None,
    attn_implementation: Optional[str] = None,
    share_moe_routing: bool = True,
    fsdp_active: Optional[bool] = None,
) -> Any:
    """Build an already-PEFT-wrapped, training-ready Shadow Residual model.

    Thin wrapper over ``shadow_residual.training.factory.get_shadow_residual_peft_model``,
    which owns the whole FSDP meta-init pipeline: every rank constructs the SR +
    PEFT module tree under ``accelerate.init_empty_weights()`` with identical
    traversal order, and only rank 0 materializes real CPU tensors (transferring
    upstream base weights, re-initializing LoRA, re-zeroing the frozen
    ``CrossStream`` linears, recomputing the non-persistent RoPE ``inv_freq``
    buffers) when ``ACCELERATE_USE_FSDP == "true"``. FSDP's
    ``sync_module_states`` then broadcasts rank 0's values. **None of that is
    reimplemented here** -- it is subtle, and the failure modes are silent NaNs
    and all-gather hangs rather than clean errors.

    Two environment variables the factory reads:

    * ``LOCAL_RANK`` -- Ray Train sets it on every worker via
      ``_set_torch_distributed_env_vars`` (ray/train/torch/config.py), called
      from ``_TorchBackend.on_training_start`` before the train loop body runs.
      Note it is the *local* rank, so on a multi-node job one process per node
      materializes (each node needs its own host copy for CPU-RAM-efficient
      loading) -- host RAM scales with nodes, not with world size.
    * ``ACCELERATE_USE_FSDP`` -- ``TrainingArguments.__post_init__`` sets it when
      ``len(self.fsdp) > 0``. The FSDP driver therefore **must** construct its
      ``TrainingArguments`` before calling this function. Nothing ever *clears*
      the variable, so pass ``fsdp_active`` explicitly and we set it here rather
      than trusting whatever a previous trial in a reused process left behind.

    Beyond the factory call this applies the three fixes that the HF Trainer
    would otherwise do for us on the ``from_pretrained`` path we are bypassing:
    ``enable_input_require_grads()``, ``use_cache=False``, and a rank-gated
    trainable-parameter log.

    Returns:
        A ``peft.PeftModel``. Callers must NOT call ``get_peft_model`` on it.
    """
    require_sr()
    from shadow_residual.training.factory import get_shadow_residual_peft_model

    if fsdp_active is not None:
        # Make the rank gate depend on this driver's own strategy rather than on
        # a stale value: a worker process that previously ran an FSDP trial would
        # otherwise leave ACCELERATE_USE_FSDP="true" behind, and ranks 1..N-1 of a
        # non-FSDP SR run would skip materialization and die on the Trainer's
        # .to(device) with "Cannot copy out of meta tensor".
        os.environ["ACCELERATE_USE_FSDP"] = "true" if fsdp_active else "false"

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    logger.info(
        "[AutoTune] Building Shadow Residual model from %s (dtype=%s, attn=%s, "
        "share_moe_routing=%s, LOCAL_RANK=%s, ACCELERATE_USE_FSDP=%s -> this rank "
        "%s materialize base weights on CPU)",
        model_name_or_path,
        torch_dtype,
        attn_implementation,
        share_moe_routing,
        local_rank,
        os.environ.get("ACCELERATE_USE_FSDP", ""),
        "WILL" if local_rank == 0 or os.environ.get("ACCELERATE_USE_FSDP", "").lower() != "true" else "will NOT",
    )
    model = get_shadow_residual_peft_model(
        model_name_or_path,
        lora_config,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        share_moe_routing=share_moe_routing,
    )

    # Required, and NOT automatic on this path. transformers only auto-calls
    # enable_input_require_grads() when `_hf_peft_config_loaded` is set, which
    # its own native `add_adapter` sets but `peft.get_peft_model` (what SR's
    # factory uses) does not -- which is why the LoRA/QLoRA paths in the drivers
    # call it explicitly too. Without it, non-reentrant gradient checkpointing
    # over a fully frozen embedding leaves the checkpointed segment with no
    # grad-requiring input: zero grads, or "none of the output has requires_grad".
    # Safe on still-meta ranks -- it only registers a forward hook on embed_tokens.
    model.enable_input_require_grads()

    # The drivers pass use_cache=False in model_kwargs on the from_pretrained
    # path; SR builds its config from the upstream config instead, where use_cache
    # is normally True. SR self-heals during *training* (it disables the cache when
    # gradient_checkpointing and self.training), but during **evaluation**
    # self.training is False, so a DynamicCache would be built and grown on every
    # eval forward -- wasted memory and an easy end-of-epoch OOM that looks like a
    # batch-size problem. Set it on both the proxy and the inner config so it
    # survives a peft refactor of PeftModel.config.
    model.config.use_cache = False
    try:
        model.base_model.model.config.use_cache = False
    except AttributeError:  # pragma: no cover - defensive
        pass

    if local_rank == 0:
        # Gate on rank 0 for log volume. Counts are pre-shard and identical to
        # what the LoRA path logs; on non-zero ranks the params may still be on
        # meta (numel() works there, so this would not crash -- it is just noise).
        model.print_trainable_parameters()

    return model


def share_moe_routing_flag(training_config: Optional[Dict[str, Any]]) -> bool:
    """Read the ``sr_share_moe_routing`` factory kwarg from the training config.

    Lives in ``training_config`` rather than the tuner ``hyperparams`` block
    because it is neither a ``LoraConfig`` field nor a ``TrainingArguments``
    kwarg -- a sampled hyperparam would be splatted into one of those and raise
    ``TypeError``. No-op on dense Granite 4.x bases; it only matters for MoE.
    """
    return bool((training_config or {}).get("sr_share_moe_routing", True))


def resolve_fsdp_strategy(strategy: str) -> str:
    """Resolve the FSDP sharding strategy for an SR run.

    Only ``auto`` is overridden; every explicit strategy passes through. The one
    underlying fact is that fm-tune's sizing heuristic was written for a
    single-stream LoRA, so it cannot be trusted to pick for SR:

    * ``auto`` -> ``full_shard``. ``estimate_fsdp_strategy`` ->
      ``_estimate_memory_components`` derives the trainable-parameter count from
      ``peft_config["r"]`` assuming a plain LoRA over a single stream: it ignores
      ``rank_pattern`` (so a large ``cross_stream_r`` is costed at ``r``), counts
      seven all-linear LoRA sites including the K/V that SR forbids, omits SR's
      per-layer frozen ``nn.Linear(H, H)`` ``CrossStream``, and estimates
      activations for one stream where SR stacks ``[2, B, S, H]``. It is a
      first-fit search, so on a 3B it would answer ``no_shard``.
    * ``no_shard`` (plain DDP) passes through, with a warning. fm-tune maps it to
      *no FSDP at all* (``_build_fsdp_config`` returns ``("", {})``), so this is
      Accelerate's DDP wrap, not torch's ``ShardingStrategy.NO_SHARD`` -- the
      latter is what breaks against ``fsdp_use_orig_params=True`` and fm-tune
      never emits it. With ``ACCELERATE_USE_FSDP`` unset SR's factory
      materializes on *every* rank, so the Trainer's ``.to(device)`` is valid and
      each rank ends up with real weights; it is upstream shadow-residual's own
      default multi-GPU recipe. The cost is host RAM: one full materialize per
      **rank** rather than per node (~2x the base's bf16 size at peak, since the
      transient upstream ``from_pretrained`` copy overlaps the SR copy), so a
      3B base runs ~12-14 GB x ranks-per-node. That is a capacity limit, not a
      defect -- ``shard_grad_op`` is the cheaper way to keep replicated weights.

      The ranks do *not* drift: each one draws its own ``reset_lora_parameters``
      values (the driver's ``set_seed`` runs in the trial process, and the worker
      seeds itself before building the model -- see ``train_loop_per_worker`` --
      but nothing guarantees a third-party factory leaves the RNG untouched), and
      DDP's constructor then broadcasts every ``named_parameters()`` entry from
      rank 0 unconditionally (``torch.distributed.utils._sync_module_states``,
      gated only on ``init_sync``, which Accelerate cannot disable). SR's
      non-persistent RoPE ``inv_freq`` buffers are recomputed identically per
      rank, so ``ddp_broadcast_buffers=False`` is safe too.
    """
    if strategy == "auto":
        logger.warning(
            "[AutoTune] Shadow Residual: overriding fsdp_strategy 'auto' -> "
            "'full_shard'. estimate_fsdp_strategy sizes trainable params from a "
            "single-stream LoRA assumption and under-counts SR's second stream "
            "plus its per-layer frozen CrossStream linears."
        )
        return "full_shard"
    if strategy == "":
        raise ValueError(
            "[AutoTune] Shadow Residual got an empty fsdp_strategy. Set one of "
            "'full_shard' (recommended), 'shard_grad_op', 'hybrid_shard' or "
            "'no_shard' in the training_config."
        )
    if strategy == "no_shard":
        logger.warning(
            "[AutoTune] Shadow Residual on fsdp_strategy 'no_shard' (plain DDP, "
            "no FSDP wrap): every rank materializes its own full host-RAM copy "
            "of the base, ~2x the base's bf16 size at peak (~12-14 GB on a 3B) "
            "x ranks-per-node, because there is no fsdp_cpu_ram_efficient_loading "
            "to keep it to one copy per node. Correctness is fine -- DDP's "
            "constructor broadcasts rank 0's parameters to every rank -- but "
            "prefer 'shard_grad_op' if the node cannot hold that, and "
            "'full_shard' for 8B and above."
        )
    return strategy


def assert_sr_supported(
    training_config: Dict[str, Any],
    *,
    accel: Any = None,
    backend: str = "hf",
    train_implementation: Optional[str] = None,
    model_name_or_path: Optional[str] = None,
    rl_algo: Optional[str] = None,
) -> None:
    """Reject Shadow Residual combinations that cannot work.

    Called from main.py (so a bad combination fails in a second, before Ray
    starts and before an HPO sweep burns its whole budget producing N identical
    trial errors) and again from each driver, which is reachable directly via
    ``--resume_from_checkpoint`` and therefore bypasses the main.py guard.

    Args:
        training_config: Dict[str, Any]
            The resolved training config section.
        accel: Optional[Accelerator]
            The detected accelerator; SR requires CUDA.
        backend: str
            The training backend; SR is HF/PEFT-only (no MLX equivalent).
        train_implementation: Optional[str]
            When given, must not select DeepSpeed.
        model_name_or_path: Optional[str]
            The base model. Pass explicitly from main.py, where the CLI value has
            not been folded into ``training_config`` yet (the optimizer does that
            at trial-launch time); the drivers can let it default.
        rl_algo: Optional[str]
            The RL algorithm. Same reason as ``model_name_or_path``.
    """
    training_config = training_config or {}

    if (backend or "hf").lower() == "mlx":
        raise ValueError(
            "[AutoTune] Shadow Residual is not supported by the MLX backend "
            "(it is a PEFT/transformers architecture with no mlx-lm equivalent). "
            "Use --backend hf on CUDA."
        )

    if accel is not None and getattr(accel, "kind", None) != "cuda":
        raise ValueError(
            f"[AutoTune] Shadow Residual requires CUDA and cannot run on "
            f"{getattr(accel, 'kind', 'unknown')}. Its FSDP meta-init path and "
            "bf16 dual-stream forward are only validated on NVIDIA GPUs."
        )

    impl = str(train_implementation or training_config.get("train_implementation") or "")
    if "ds" in impl.lower() or "deepspeed" in impl.lower():
        raise ValueError(
            f"[AutoTune] Shadow Residual requires FSDP, got train_implementation="
            f"{impl!r}. Set train_implementation: FSDP in the training_config -- "
            "DeepSpeed ZeRO-3 sharded init is incompatible with SR's rank-0 "
            "meta-materialize path."
        )

    rl_algo = rl_algo or training_config.get("rl_algo", "none")
    if rl_algo and rl_algo != "none":
        raise ValueError(f"[AutoTune] Shadow Residual supports SFT only; got rl_algo={rl_algo!r}. Use --rl_algo none.")

    assert_sr_base_model(model_name_or_path or training_config.get("model_name_or_path"))


def assert_sr_base_model(model_name_or_path: Optional[str]) -> None:
    """Reject base models whose architecture SR cannot adapt.

    All three checks exist because the failure mode is otherwise *silent* rather
    than loud, thanks to PEFT's partial-match behaviour: ``inject_adapter`` raises
    only when **nothing** in ``target_modules`` matched, so a base where some of
    the six SR targets are absent trains a smaller adapter than the config
    describes, with a plausible-looking ``trainable params`` line and no warning.

    * **Not Granite.** SR's config/decoder subclass Granite's and
      ``transfer_base_weights`` slices Granite's fused projections into SR's
      unfused ones; anything else dies deep in the transfer with a
      missing-tensor error.
    * **MoE** (``num_local_experts > 0``). ``ShadowResidualMLP`` builds
      ``gate_proj``/``up_proj``/``down_proj`` **only** on the dense branch -- a MoE
      base gets ``router``/``input_linear``/``output_linear`` instead. Three of the
      six SR targets would silently not exist, so ``r`` and ``alpha_ratio`` would
      mean something different on a MoE base than on a dense one, which quietly
      corrupts an HPO comparison.
    * **Hybrid / non-attention layers.** ``build_sr_config`` normalizes
      ``layer_types`` to all-``"attention"``; if the checkpoint really does carry
      Mamba/SSM weights the transfer then fails on missing q/k/v.
    """
    if not model_name_or_path:
        return
    try:
        from transformers import AutoConfig

        from autotune.utils import resolve_trust_remote_code

        base_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=resolve_trust_remote_code())
    except Exception as e:  # pragma: no cover - network/cache failures
        logger.warning(
            "[AutoTune] Could not read the base model config to validate the "
            "Shadow Residual architecture (%s); continuing.",
            e,
        )
        return

    model_type = str(getattr(base_config, "model_type", "") or "")
    if "granite" not in model_type.lower():
        raise ValueError(
            f"[AutoTune] Shadow Residual requires a Granite base model; "
            f"{model_name_or_path} has model_type={model_type!r}. The SR decoder "
            "is built from the Granite config and transfers Granite's fused "
            "projections into its unfused ones."
        )

    num_local_experts = int(getattr(base_config, "num_local_experts", 0) or 0)
    if num_local_experts > 0:
        raise ValueError(
            f"[AutoTune] Shadow Residual on a MoE base ({model_name_or_path}, "
            f"num_local_experts={num_local_experts}) is not supported by fm-tune. "
            "SR's MoE MLP exposes router/input_linear/output_linear instead of "
            "gate_proj/up_proj/down_proj, so three of the six SR target modules "
            "would silently not match and the sampled r/alpha_ratio would not "
            "mean what the search space says. Use a dense Granite base."
        )

    layer_types = getattr(base_config, "layer_types", None) or []
    non_attention = sorted({str(t) for t in layer_types if str(t) != "attention"})
    if non_attention:
        raise ValueError(
            f"[AutoTune] Shadow Residual requires an attention-only base; "
            f"{model_name_or_path} declares layer_types {non_attention}. SR is an "
            "attention-only architecture -- a checkpoint carrying Mamba/SSM "
            "weights fails in transfer_base_weights on missing q/k/v tensors."
        )


def assert_no_vocab_resize(num_new_tokens: int) -> None:
    """Reject tokenizer customization that would grow the vocabulary under SR.

    ``transfer_base_weights`` slices the upstream checkpoint at its original
    vocab size, and resizing the embeddings of a PEFT-wrapped (and, on non-rank-0
    under FSDP, still-meta) model is not a supported operation -- it would give
    either a crash or a silently-garbage new embedding row that
    ``sync_module_states`` then broadcasts to every rank. A no-op when the
    tokenizer added nothing, which is the common case.
    """
    if num_new_tokens:
        raise ValueError(
            f"[AutoTune] Shadow Residual cannot resize token embeddings "
            f"({num_new_tokens} new token(s) requested). SR transfers base weights "
            "sliced at the checkpoint's original vocab size, and the model may "
            "still be on meta on non-zero ranks. Drop the tokenizer_config "
            "additions, or pre-extend the base checkpoint's vocabulary."
        )


def patch_sr_adapter_config(save_dir: Union[str, "os.PathLike[str]"], *, share_moe_routing: bool = True) -> None:
    """Mark a saved SR adapter as Shadow Residual so it is self-describing.

    ``trainer.save_model()`` writes a stock ``adapter_config.json``
    (``peft_type: LORA``, ``base_model_name_or_path: <upstream granite>``), which
    reloads the *obvious* way -- ``AutoModelForCausalLM.from_pretrained(base)`` +
    ``PeftModel.from_pretrained`` -- as a plain LoRA on plain Granite, where
    ``cross_stream`` silently matches nothing and the dual-stream forward never
    engages. The adapter weights would load, the model would run, and it would be
    the wrong architecture with no error anywhere.

    So record two things the weights cannot carry: a ``shadow_residual`` marker
    (reload through ``shadow_residual.shadow_residual.build.build_sr_base`` +
    ``PeftModel.from_pretrained``, not ``AutoModelForCausalLM``) and
    ``share_moe_routing`` (a forward-path choice, not a weight -- the served value
    must match what the adapter was trained with). shadow-residual's own train.py
    patches the same file for the same reason, and its serving path reads
    ``share_moe_routing`` back from it.

    Unknown keys are safe: PEFT's ``from_peft_type`` catches the resulting
    ``TypeError``, drops them with a warning, and retries.
    """
    path = os.path.join(str(save_dir), "adapter_config.json")
    if not os.path.exists(path):
        logger.warning(
            "[AutoTune] No adapter_config.json at %s -- cannot mark the saved adapter as Shadow Residual.",
            path,
        )
        return
    try:
        with open(path, "r") as f:
            adapter_cfg = json.load(f)
        adapter_cfg["shadow_residual"] = True
        adapter_cfg["share_moe_routing"] = bool(share_moe_routing)
        with open(path, "w") as f:
            json.dump(adapter_cfg, f, indent=2)
    except (OSError, ValueError) as e:
        logger.warning("[AutoTune] Could not patch %s: %s", path, e)
        return
    logger.info(
        "[AutoTune] Marked %s as a Shadow Residual adapter "
        "(shadow_residual=True, share_moe_routing=%s). Reload it with "
        "shadow_residual build_sr_base() + PeftModel.from_pretrained, NOT with "
        "AutoModelForCausalLM.from_pretrained (which would silently drop the "
        "cross_stream sites and give a plain single-stream LoRA).",
        path,
        bool(share_moe_routing),
    )


__all__ = [
    "SR_PEFT_TYPE",
    "SR_TARGET_MODULES",
    "SR_TUNING_ALGO",
    "SR_WRAP_CLASS",
    "assert_no_vocab_resize",
    "assert_sr_base_model",
    "assert_sr_supported",
    "build_sr_lora_config",
    "build_sr_peft_model",
    "is_sr",
    "patch_sr_adapter_config",
    "require_sr",
    "resolve_fsdp_strategy",
    "share_moe_routing_flag",
]
