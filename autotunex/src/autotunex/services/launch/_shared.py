# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Assembly helpers shared by the remote granite.build spec builders.

The custom_code builder (``spec.py``) and the LSF/SkyPilot builder
(``lsf_spec.py``) emit different ``build.yaml`` trees but share four concerns:
the ``|``-literal YAML dumper, the ``num_gpus_per_node`` derivation, reward-
function injection for online RL, and the ``python main.py`` start command.
They live here so there is exactly one implementation of each, rather than a
copy per builder. The local-bash builder (``bash_spec.py``) deliberately shares
none of these — it is a different execution shape.
"""

from __future__ import annotations

from typing import Any

import yaml

from autotunex.services.launch.protocols import LaunchContext

REWARD_FUNCTION_PATH = "/tmp/reward_function.py"
"""In-container path the injected reward function is written to, and the value
``training_rl_config.reward_function_path`` is rewritten to for online RL."""

_DATASET_BINDING = "{{ bindings.dataset_files.binding.path }}"
"""Granite-build runtime binding that resolves to the mounted dataset directory."""


class BlockStringDumper(yaml.SafeDumper):
    """A ``SafeDumper`` that renders multi-line strings as ``|`` literal blocks.

    The embedded per-trial config and any injected reward function are multi-line;
    literal-block style keeps them readable in the generated ``build.yaml`` instead
    of collapsing to a single escaped line. Subclassed rather than mutating the
    global representer registry, so nothing else in the process is affected.
    """


def _represent_str(dumper: BlockStringDumper, data: str) -> yaml.nodes.ScalarNode:
    """Represent multi-line strings with block style; everything else as plain."""
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


BlockStringDumper.add_representer(str, _represent_str)


def num_gpus_per_node(config_data: dict[str, Any]) -> int:
    """Return ``max_concurrent_trials * num_gpus_per_trial``, each defaulting to 1.

    Mirrors ``gb_runner.update_compute_config`` exactly, including the ``.default``
    lookups into ``tune_config`` / ``training_config``.
    """
    tune = config_data.get("tune_config", {}).get("max_concurrent_trials", {})
    train = config_data.get("training_config", {}).get("num_gpus_per_trial", {})
    return int(tune.get("default", 1)) * int(train.get("default", 1))


def inject_reward_function(config_data: dict[str, Any], ctx: LaunchContext) -> dict[str, str]:
    """Rewrite the RL reward paths in ``config_data`` and return the file to embed.

    For online RL (``ctx.reward_function_code`` present) the config's
    ``training_rl_config.reward_function_path`` — and ``reward_function_name`` when
    ``ctx.reward_function_name`` is set — are rewritten to point at
    :data:`REWARD_FUNCTION_PATH`, and ``{REWARD_FUNCTION_PATH: code}`` is returned
    to merge into the build's ``additional_files``. Mutates ``config_data`` in
    place, so callers pass a deep copy. Returns ``{}`` when there is no reward code.
    """
    if not ctx.reward_function_code:
        return {}
    rl_config = config_data.get("training_rl_config")
    if isinstance(rl_config, dict):
        reward_path = rl_config.get("reward_function_path")
        if isinstance(reward_path, dict):
            reward_path["default"] = REWARD_FUNCTION_PATH
        if ctx.reward_function_name:
            reward_name = rl_config.get("reward_function_name")
            if isinstance(reward_name, dict):
                reward_name["default"] = ctx.reward_function_name
    return {REWARD_FUNCTION_PATH: ctx.reward_function_code}


def start_command(
    ctx: LaunchContext, *, config_file: str, callback_url: str | None, cuda_home: str
) -> str:
    """Build the container's ``main.py`` invocation, a faithful port of build_start_cmd.

    Flag order and the ``export … && export … && python main.py`` prefix match the
    2025 command; ``--tuning_algo``, ``--rl_algo``, ``--no_autotune`` and
    ``--autotunex_server_url`` are conditional. ``cuda_home`` is the only value that
    differs between the custom_code (``/usr/local/cuda-13.0``) and LSF
    (``/opt/share/cuda-12.9``) builders — every other flag is identical.
    """
    flags = [
        "python main.py",
        f"--config_file {config_file}",
        f"--train_file {_DATASET_BINDING}/{ctx.dataset_name}_train.{ctx.data_format}",
        f"--validation_file {_DATASET_BINDING}/{ctx.dataset_name}_validation.{ctx.data_format}",
        f"--model_name_or_path {ctx.model}",
    ]
    if ctx.tuning_type:
        flags.append(f"--tuning_algo {ctx.tuning_type}")
    if ctx.rl_tuner_type:
        flags.append(f"--rl_algo {ctx.rl_tuner_type}")
    flags.append(f"--run_name {ctx.experiment_name}")
    flags.append("--output_dir $OUTPUT_PATH")
    flags.append(f"--output_model_name {ctx.experiment_name}")
    flags.append("--cleanup --save_history")
    if not ctx.autotune:
        flags.append("--no_autotune")
    flags.append(f"--job_id {ctx.job_id}")
    if callback_url:
        flags.append(f"--autotunex_server_url {callback_url}")
    python_cmd = " ".join(flags)
    return f"export CUDA_HOME={cuda_home} && export LOG_PATH=$OUTPUT_PATH && {python_cmd}"
