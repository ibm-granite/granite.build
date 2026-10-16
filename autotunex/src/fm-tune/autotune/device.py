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

"""Single source of truth for every platform (accelerator) decision.

fm-tune historically assumed CUDA. This module centralises accelerator
detection and the derived choices (precision, attention impl, Ray resource
bundle, guard rails, runtime env) so the rest of the codebase never calls
``torch.cuda.is_available()`` directly.

Invariant: every helper returns exactly the pre-MPS value when
``accel.kind == "cuda"``. That is what keeps the CUDA cluster path unchanged;
the unit tests assert it by equality.

torch is imported lazily inside functions so this module imports cleanly
during CLI argument parsing (before any heavy stack is loaded).
"""

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

_VALID_OVERRIDES = {"cuda", "mps", "cpu"}


@dataclass(frozen=True)
class Accelerator:
    """A resolved training accelerator and its capabilities.

    kind: "cuda" | "mps" | "cpu"
    count: number of devices (cuda: device count; mps: 1; cpu: 0)
    supports_distributed: DeepSpeed / FSDP / NCCL (cuda only)
    supports_4bit: bitsandbytes NF4 kernels (cuda only)
    supports_flash_attn: flash-attention kernels (cuda only)
    supports_bf16: bf16 compute (cuda: is_bf16_supported; mps: macOS>=14; cpu: no)
    """

    kind: str
    count: int
    supports_distributed: bool
    supports_4bit: bool
    supports_flash_attn: bool
    supports_bf16: bool


def _cuda_accelerator() -> "Accelerator":
    import torch

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible:
        count = len([d for d in cuda_visible.split(",") if d])
    else:
        count = torch.cuda.device_count()
    try:
        bf16 = torch.cuda.is_bf16_supported()
    except Exception:
        bf16 = True
    return Accelerator(
        kind="cuda",
        count=count,
        supports_distributed=True,
        supports_4bit=True,
        supports_flash_attn=True,
        supports_bf16=bf16,
    )


def _mps_accelerator() -> "Accelerator":
    import torch

    bf16 = torch.backends.mps.is_available() and torch.backends.mps.is_macos_or_newer(14, 0)
    return Accelerator(
        kind="mps",
        count=1,
        supports_distributed=False,
        supports_4bit=False,
        supports_flash_attn=False,
        supports_bf16=bool(bf16),
    )


def _cpu_accelerator() -> "Accelerator":
    return Accelerator(
        kind="cpu",
        count=0,
        supports_distributed=False,
        supports_4bit=False,
        supports_flash_attn=False,
        supports_bf16=False,
    )


def detect_accelerator() -> "Accelerator":
    """Detect the training accelerator: CUDA, then MPS, then CPU.

    ``FMTUNE_DEVICE=cuda|mps|cpu`` overrides detection entirely (useful for
    debugging MPS op gaps without editing code).
    """
    override = os.environ.get("FMTUNE_DEVICE", "").strip().lower()
    if override:
        if override not in _VALID_OVERRIDES:
            raise ValueError(f"FMTUNE_DEVICE must be one of {sorted(_VALID_OVERRIDES)}; got {override!r}.")
        if override == "cuda":
            return _cuda_accelerator()
        if override == "mps":
            return _mps_accelerator()
        return _cpu_accelerator()

    try:
        import torch

        if torch.cuda.is_available():
            return _cuda_accelerator()
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available() and mps.is_built():
            return _mps_accelerator()
    except Exception as e:  # torch missing or probe failed
        logger.debug(f"Accelerator probe failed, defaulting to CPU: {e}")
    return _cpu_accelerator()


@lru_cache(maxsize=1)
def supports_autocast_bf16() -> bool:
    """Probe whether torch implements MPS autocast to bf16 on this machine.

    transformers sanctioning bf16 on macOS >= 14 does not guarantee torch
    actually implements MPS autocast, so we measure rather than assume: a
    tiny matmul inside autocast, checking the output dtype. ~1 ms, cached.
    """
    try:
        import torch

        a = torch.ones((2, 2), device="mps", dtype=torch.float32)
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            out = a @ a
        return out.dtype == torch.bfloat16
    except Exception as e:
        logger.debug(f"MPS bf16 autocast probe failed: {e}")
        return False


def resolve_precision(requested: str, accel: "Accelerator", probe_autocast: bool = True) -> str:
    """Resolve the requested precision to what the accelerator can actually run.

    Returns "bf16" or "fp32". On CUDA, "bf16" is preserved (today's behaviour).
    On MPS, "bf16" requires both macOS>=14 (accel.supports_bf16) and, when
    ``probe_autocast`` is True, a passing autocast probe; otherwise it degrades
    to "fp32" with a warning. ``probe_autocast=False`` lets the parent process
    (main.py guards) resolve without allocating an MPS tensor.
    """
    requested = (requested or "bf16").lower()
    if requested not in ("bf16", "fp32"):
        raise ValueError(f"precision must be 'bf16' or 'fp32'; got {requested!r}.")

    if requested == "fp32":
        return "fp32"

    # requested == "bf16"
    if accel.kind == "cuda":
        return "bf16"  # unchanged from the pre-MPS hardcoded bf16=True
    if accel.kind == "mps":
        if not accel.supports_bf16:
            logger.warning("[AutoTune] bf16 requested but macOS < 14; using fp32.")
            return "fp32"
        if probe_autocast and not supports_autocast_bf16():
            logger.warning("[AutoTune] bf16 requested but MPS autocast unavailable; using fp32.")
            return "fp32"
        return "bf16"
    return "fp32"  # cpu


_FLASH_ATTN_NAMES = {"flash_attention_2", "flash_attention", "flash_attn_2"}


def resolve_attn_implementation(requested: str, accel: "Accelerator") -> str:
    """Keep the requested attention impl on CUDA; downgrade flash-attn to eager
    on accelerators without flash-attention kernels (MPS, CPU)."""
    requested = requested or "eager"
    if accel.supports_flash_attn:
        return requested
    if requested in _FLASH_ATTN_NAMES:
        return "eager"
    return requested


def model_load_kwargs(accel: "Accelerator", precision: str) -> dict:
    """from_pretrained kwargs for the given accelerator and resolved precision.

    On CUDA this equals the literal driver_single.py used pre-MPS
    (device_map="auto", dtype=bf16, low_cpu_mem_usage=True). On MPS/CPU
    device_map is None so HF Trainer moves the model to args.device itself;
    accelerate's "auto" placement can CPU/disk-offload shards on Metal.
    """
    from autotune.utils import get_torch_dtype

    dtype = get_torch_dtype(precision)
    device_map = "auto" if accel.kind == "cuda" else None
    return {"device_map": device_map, "dtype": dtype, "low_cpu_mem_usage": True}


def ray_num_gpus(accel: "Accelerator", num_gpus_per_trial: int) -> int:
    """Ray GPU reservation per trial: the requested count on CUDA, 0 otherwise.

    Ray cannot schedule Metal as a resource, so an MPS trial reserves 0 GPUs
    and uses the device inside the trial; correctness comes from clamping
    max_concurrent_trials to 1 (see apply_platform_guards)."""
    return int(num_gpus_per_trial) if accel.kind == "cuda" else 0


def apply_platform_guards(
    training_config: dict,
    tune_config: dict,
    tuning_algo: str,
    rl_algo: str,
    accel: "Accelerator",
    backend: str = "torch",
) -> None:
    """Enforce accelerator + backend capabilities on the config, mutating in place.

    On CUDA with the default torch backend this is a strict no-op. On a non-CUDA
    accelerator it raises for configurations that cannot run and rewrites benign
    mismatches with a WARNING. When ``backend == "mlx"`` the MLX capability rules
    apply instead of the torch/MPS QLoRA rejection.
    """
    if accel.kind == "cuda" and backend == "torch":
        return

    from autotune.constants import MLX_SUPPORTED_TUNING_ALGO

    if backend == "mlx":
        if accel.kind != "mps":
            raise ValueError(
                f"The MLX backend requires Apple Silicon (MPS); detected accelerator "
                f"'{accel.kind}'. Use --backend torch, or run on an Apple Silicon Mac."
            )
        if tuning_algo not in MLX_SUPPORTED_TUNING_ALGO:
            raise ValueError(
                f"The MLX backend supports {sorted(MLX_SUPPORTED_TUNING_ALGO)}; "
                f"'{tuning_algo}' has no mlx-lm equivalent. Use --backend torch for it."
            )
        if rl_algo and rl_algo != "none":
            raise ValueError(f"RL fine-tuning ('{rl_algo}') is not supported on the MLX backend.")
        n_gpus = int(training_config.get("num_gpus_per_trial", 1) or 1)
        if n_gpus > 1:
            raise ValueError(
                f"num_gpus_per_trial={n_gpus} is not supported on the MLX backend "
                "(single-device only). Set num_gpus_per_trial to 1."
            )
        # qlora is allowed here (MLX 4-bit); skip the torch/MPS bitsandbytes checks.
        mct = int(tune_config.get("max_concurrent_trials", 1) or 1)
        if mct > 1:
            logger.warning(
                f"[AutoTune] max_concurrent_trials={mct} would oversubscribe the single MPS device; clamping to 1."
            )
            tune_config["max_concurrent_trials"] = 1
        return

    # --- Impossible on non-CUDA: fail fast with a clear message -------------
    if tuning_algo == "qlora":
        raise ValueError(
            f"QLoRA (4-bit bitsandbytes) requires CUDA and is not supported on {accel.kind}. "
            "Use --tuning_algo lora for a bf16/fp32 LoRA adapter instead."
        )
    if rl_algo and rl_algo != "none":
        raise ValueError(
            f"RL fine-tuning ('{rl_algo}') is not supported on {accel.kind}: online RL needs "
            "vLLM (CUDA-only) and offline RL (DPO/KTO) is out of scope for local MPS runs. "
            "Run RL on a CUDA host."
        )
    n_gpus = int(training_config.get("num_gpus_per_trial", 1) or 1)
    if n_gpus > 1:
        raise ValueError(
            f"num_gpus_per_trial={n_gpus} is not supported on {accel.kind}: there is no "
            "distributed backend (NCCL) on Metal. Set num_gpus_per_trial to 1."
        )

    # --- Benign: auto-fix with a WARNING ------------------------------------
    attn = training_config.get("use_flash_attention", "eager")
    fixed_attn = resolve_attn_implementation(attn, accel)
    if fixed_attn != attn:
        logger.warning(f"[AutoTune] use_flash_attention='{attn}' unavailable on {accel.kind}; using '{fixed_attn}'.")
        training_config["use_flash_attention"] = fixed_attn

    train_impl = training_config.get("train_implementation")
    if train_impl in ("FSDP", "DeepSpeed"):
        logger.warning(
            f"[AutoTune] train_implementation='{train_impl}' has no effect on {accel.kind}; "
            "the single-device driver is used."
        )

    mct = int(tune_config.get("max_concurrent_trials", 1) or 1)
    if mct > 1:
        logger.warning(
            f"[AutoTune] max_concurrent_trials={mct} would oversubscribe the single {accel.kind} device; clamping to 1."
        )
        tune_config["max_concurrent_trials"] = 1

    prec = training_config.get("precision", "bf16")
    fixed_prec = resolve_precision(prec, accel, probe_autocast=False)
    if fixed_prec != prec:
        logger.warning(f"[AutoTune] precision='{prec}' unavailable on {accel.kind}; using '{fixed_prec}'.")
        training_config["precision"] = fixed_prec


_DEFAULT_OBJECT_STORE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


def object_store_bytes() -> int:
    """Ray object-store size (bytes) for a local non-CUDA cluster.

    The single-device driver never uses ray.data, so the object store only
    carries the trial config and result dict. 2 GiB by default (vs the 0.5
    RAM proportion used on GPU clusters, which would reserve 8 GiB on a 16 GB
    Mac and starve Metal). Override with FMTUNE_OBJECT_STORE_BYTES.
    """
    val = os.environ.get("FMTUNE_OBJECT_STORE_BYTES", "").strip()
    if val:
        try:
            return int(val)
        except ValueError:
            logger.warning(f"[AutoTune] FMTUNE_OBJECT_STORE_BYTES={val!r} is not an int; using default.")
    return _DEFAULT_OBJECT_STORE_BYTES


def configure_runtime_env(accel: "Accelerator") -> None:
    """Set process env vars appropriate to the accelerator.

    Shared vars are always set. CUDA-only vars (NCCL, vLLM, verl, the 0.5
    object-store proportion) are set only on CUDA — identical to the pre-MPS
    main.py env block. On non-CUDA, the MPS CPU-fallback var is enabled so an
    unimplemented op runs on CPU (with a warning) instead of crashing.
    """
    # Shared across platforms
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
    os.environ["RAY_DEDUP_LOGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["TUNE_WARN_SLOW_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "900"
    os.environ["TUNE_GLOBAL_CHECKPOINT_S"] = "600"
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "8"

    if accel.kind == "cuda":
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["NCCL_ALGO"] = "Ring"
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        os.environ["VERL_REWARD_DEBUG"] = "1"
        os.environ["HYDRA_FULL_ERROR"] = "1"
        os.environ["VLLM_USE_V1"] = "1"
        if "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION" not in os.environ:
            os.environ["RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION"] = "0.5"
    else:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            logger.warning(
                "[AutoTune] PYTORCH_ENABLE_MPS_FALLBACK=1: unimplemented MPS ops run on CPU "
                "(slower). Set it to 0 to locate op gaps."
            )
