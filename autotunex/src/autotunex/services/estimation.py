# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Resource estimation for the start-tuning wizard's Step 3.

Estimates GPU/CPU memory and GPU count for a model against either a saved
configuration (owner-scoped lookup by id) or an unsaved one supplied inline
(``config_data``) — the inline path is what makes estimation possible
mid-wizard, before a configuration is persisted. The heuristic itself
(:func:`parse_model_parameters`, :func:`estimate_memory_usage`) is a planning-time
approximation ported from the tuning pipeline's own estimator
(``autotune.utils``), not a measured profile; the optional ``autotune`` import
is dropped and the two functions are inlined instead.
"""

from __future__ import annotations

import math
import re
from typing import Any

from autotunex.core.exceptions import ConfigurationNotFoundError, DomainValidationError
from autotunex.db.repositories.protocols import ConfigurationRepository
from autotunex.models.auth import Principal
from autotunex.models.estimation import EstimateUsagesRequest, EstimateUsagesResponse
from autotunex.models.job import ONLINE_RL_TUNER_TYPES

_PARAM_COUNT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)([bBmM])")

_GRANITE_4_MODEL_PARAMS: dict[str, float] = {
    "ibm-granite/granite-4.0-micro": 3.0,
    "ibm-granite/granite-4.0-h-micro": 3.0,
    "ibm-granite/granite-4.0-h-tiny": 7.0,
    "ibm-granite/granite-4.0-tiny": 7.0,
    "ibm-granite/granite-4.0-h-small": 32.0,
}
"""Parameter counts (billions) for Granite 4.0 hybrid models.

These names carry no ``<number><b|m>`` suffix :func:`parse_model_parameters`
can regex out, so they need an explicit fallback.
"""

_PRECISION_BYTES: dict[str, float] = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
}

_DEFAULT_BATCH_SIZE = 4
_DEFAULT_SEQUENCE_LENGTH = 512
_DEFAULT_PRECISION = "bf16"
_BYTES_PER_GB = 1024**3

# Fixed assumptions the 2025 estimator also made, named here rather than left
# as magic literals at each call site.
_USE_LORA = True
_ZERO_STAGE = 3
_USE_GRADIENT_CHECKPOINTING = True


def parse_model_parameters(model_name: str) -> float | None:
    """Return a model's parameter count in billions, parsed from its name.

    Reads a ``<number><b|m>`` suffix (case-insensitive) out of ``model_name``,
    e.g. ``"Llama-2-7b"`` -> ``7.0``, ``"some-500m-model"`` -> ``0.5``. Falls
    back to :data:`_GRANITE_4_MODEL_PARAMS` for Granite 4.0 hybrid models,
    whose names carry no parseable size suffix. Returns ``None`` when neither
    yields a size — the caller treats that as an unparseable model name.
    """
    match = _PARAM_COUNT_PATTERN.search(model_name)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        return value if unit == "b" else value / 1000

    if model_name.startswith("ibm-granite/granite-4.0"):
        matched_key = next((key for key in _GRANITE_4_MODEL_PARAMS if key in model_name), None)
        if matched_key is not None:
            return _GRANITE_4_MODEL_PARAMS[matched_key]

    return None


def estimate_memory_usage(
    *,
    model_size_billion_params: float,
    precision: str = _DEFAULT_PRECISION,
    batch_size: int = 1,
    sequence_length: int = 128,
    gpu_size_gb: int = 75,
    use_lora: bool = _USE_LORA,
    zero_stage: int = _ZERO_STAGE,
    use_gradient_checkpointing: bool = _USE_GRADIENT_CHECKPOINTING,
) -> dict[str, float | int]:
    """Heuristically estimate GPU/CPU memory and GPU count for a training run.

    Ported from the tuning pipeline's own estimator: weights/optimizer/gradients
    scale off ``precision`` and, when LoRA is used, off a 10% trainable-parameter
    fraction; activations scale off ``batch_size`` and ``sequence_length``.
    Gradient checkpointing and each successive ZeRO stage move a portion of GPU
    memory to CPU rather than eliminating it. This is a planning-time estimate
    the wizard renders before a run starts, not a measured profile.

    Returns:
        A dict with ``model_size_billion_params``, ``gpu_memory_gb``,
        ``cpu_memory_gb``, ``num_gpus``, ``weights_memory``,
        ``optimizer_memory``, ``gradients_memory`` and ``activations_memory``
        (all memory figures in GB) — matching :class:`EstimateUsagesResponse`
        field-for-field.

    Raises:
        ValueError: ``precision`` is not one of the supported dtypes.
    """
    if precision not in _PRECISION_BYTES:
        raise ValueError(f"Unsupported precision: {precision}")

    overhead = 1.5
    cpu_memory = 0.0
    bytes_per_param = _PRECISION_BYTES[precision]

    num_params = model_size_billion_params * 1e9
    trainable_params = num_params * 0.1 if use_lora else num_params

    weights_memory = num_params * bytes_per_param * overhead
    optimizer_memory = 2 * trainable_params * bytes_per_param * overhead
    gradients_memory = trainable_params * bytes_per_param * overhead
    activations_memory = (
        batch_size * sequence_length * bytes_per_param * model_size_billion_params * 1e9 / 1024
    )

    if use_gradient_checkpointing:
        halved = activations_memory / 2
        cpu_memory += activations_memory - halved
        activations_memory = halved

    if zero_stage >= 1:
        reduced = optimizer_memory * 0.25
        cpu_memory += optimizer_memory - reduced
        optimizer_memory = reduced

    if zero_stage >= 2:
        reduced = gradients_memory * 0.25
        cpu_memory += gradients_memory - reduced
        gradients_memory = reduced

    if zero_stage == 3:
        reduced = weights_memory * 0.25
        cpu_memory += weights_memory - reduced
        weights_memory = reduced

    total_memory_bytes = weights_memory + optimizer_memory + gradients_memory + activations_memory
    gpu_memory_gb = total_memory_bytes / _BYTES_PER_GB
    cpu_memory_gb = cpu_memory / _BYTES_PER_GB
    num_gpus = max(1, math.ceil(gpu_memory_gb / gpu_size_gb))

    return {
        "model_size_billion_params": model_size_billion_params,
        "gpu_memory_gb": gpu_memory_gb,
        "cpu_memory_gb": cpu_memory_gb,
        "num_gpus": num_gpus,
        "weights_memory": weights_memory / _BYTES_PER_GB,
        "optimizer_memory": optimizer_memory / _BYTES_PER_GB,
        "gradients_memory": gradients_memory / _BYTES_PER_GB,
        "activations_memory": activations_memory / _BYTES_PER_GB,
    }


def _dig(d: dict[str, Any], *keys: str) -> Any:  # noqa: ANN401 — genuinely-arbitrary JSON
    """Return the nested value at ``keys`` in ``d``, or ``None`` if any level is missing.

    ``d`` is the schema-less ``config_data`` blob, where any level may be
    absent — not just the leaf — so a chain of plain ``.get()`` calls would
    need the same ``or {}`` fallback repeated at every level.
    """
    current: Any = d
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _largest_batch(
    config_data: dict[str, Any], tuner_type: str | None, rl_tuner_type: str | None
) -> int:
    """Return the largest configured per-device train batch size, or a default of 4.

    Reads ``tuners_rl_config[rl_tuner_type]`` when an RL tuner is set,
    otherwise ``tuners_config[tuner_type]``. The 2025 estimator indexed the
    SFT path's ``values[-1]`` unguarded, which raised an uncaught ``IndexError``
    (surfacing as a 500) whenever ``values`` was missing or empty; this guards
    both cases and falls back to :data:`_DEFAULT_BATCH_SIZE`.
    """
    section = "tuners_rl_config" if rl_tuner_type else "tuners_config"
    key = rl_tuner_type or tuner_type
    if key is None:
        return _DEFAULT_BATCH_SIZE

    values = _dig(config_data, section, key, "hyperparams", "per_device_train_batch_size", "values")
    if not isinstance(values, list) or not values:
        return _DEFAULT_BATCH_SIZE

    return int(max(values))


class EstimationService:
    """Resource estimation for the start-tuning wizard's Step 3.

    Estimates GPU/CPU memory and GPU count for a model against either a saved
    configuration (owner-scoped lookup by id via ``configuration_repository``)
    or an unsaved one supplied inline on the request. The inline path never
    touches the database, so ``configuration_repository`` may be ``None`` for
    a caller that only ever exercises that path (see the service's tests).
    """

    def __init__(
        self, configuration_repository: ConfigurationRepository | None, principal: Principal
    ) -> None:
        """Store the collaborators used to resolve a saved configuration's data."""
        self._configuration_repository = configuration_repository
        self._principal = principal

    async def estimate(self, request: EstimateUsagesRequest) -> EstimateUsagesResponse:
        """Return the memory/GPU estimate for ``request``.

        Raises:
            DomainValidationError: ``request.model_name`` has no parseable
                parameter count.
            ConfigurationNotFoundError: ``request.config_id`` does not resolve
                to a configuration visible to the caller.
        """
        model_size = parse_model_parameters(request.model_name)
        if model_size is None:
            raise DomainValidationError(
                f"Unable to parse model parameters from model name: {request.model_name}"
            )

        config_data, tuner_type, rl_tuner_type = await self._resolve_config_source(request)
        config_data = config_data or {}

        precision = _dig(config_data, "training_config", "precision", "default")
        # A schema-less config may carry an unexpected dtype; the estimate is
        # best-effort, so fall back to the default rather than 500 on an
        # unsupported precision (the wizard hides a failed estimate entirely).
        if precision not in _PRECISION_BYTES:
            precision = _DEFAULT_PRECISION
        sequence_length = _dig(config_data, "training_config", "max_length", "default")
        sequence_length = sequence_length or _DEFAULT_SEQUENCE_LENGTH
        batch_size = _largest_batch(config_data, tuner_type, rl_tuner_type)

        result = estimate_memory_usage(
            model_size_billion_params=model_size,
            precision=precision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            gpu_size_gb=request.gpu_memory,
        )

        normalized_rl_tuner_type = (rl_tuner_type or "").lower()
        if normalized_rl_tuner_type in ONLINE_RL_TUNER_TYPES:
            self._add_online_rl_memory(result, normalized_rl_tuner_type, request.gpu_memory)

        return EstimateUsagesResponse(**result)

    async def _resolve_config_source(
        self, request: EstimateUsagesRequest
    ) -> tuple[dict[str, Any] | None, str | None, str | None]:
        """Return ``(config_data, tuner_type, rl_tuner_type)`` for ``request``.

        Reads a saved configuration, owner-scoped, when ``config_id`` is set
        (fixing the 2025 unscoped lookup); otherwise passes through the
        inline fields, which :class:`EstimateUsagesRequest` already validated
        to be present exactly when ``config_id`` is not.
        """
        if request.config_id is not None:
            repository = self._configuration_repository
            if repository is None:
                raise ConfigurationNotFoundError(request.config_id)

            config = await repository.get(request.config_id, owner_id=self._principal.user_id)
            if config is None:
                raise ConfigurationNotFoundError(request.config_id)

            return config.config_data, config.tuner_type, config.rl_tuner_type

        return request.config_data, request.tuner_type, request.rl_tuner_type

    @staticmethod
    def _add_online_rl_memory(
        result: dict[str, float | int], rl_tuner_type: str, gpu_memory: int
    ) -> None:
        """Add the online-RL extra-memory heuristic to ``result``, in place.

        Online-RL tuners keep more than one model resident: PPO trains a
        policy alongside a value model, a frozen reference model and a reward
        model (``weights * 3 + optimizer + gradients``); GRPO/DAPO keep only a
        policy and a frozen reference copy (``weights``). This is a
        documented, tunable approximation (spec §5.3 step 5), not a measured
        figure. Mutates ``gpu_memory_gb``/``cpu_memory_gb`` and recomputes
        ``num_gpus`` from the updated total.
        """
        weights_memory = result["weights_memory"]
        optimizer_memory = result["optimizer_memory"]
        gradients_memory = result["gradients_memory"]

        if rl_tuner_type == "ppo":
            additional_memory = weights_memory * 3 + optimizer_memory + gradients_memory
        else:
            additional_memory = weights_memory

        result["gpu_memory_gb"] = result["gpu_memory_gb"] + additional_memory
        result["cpu_memory_gb"] = result["cpu_memory_gb"] + additional_memory
        result["num_gpus"] = max(1, math.ceil(result["gpu_memory_gb"] / gpu_memory))
