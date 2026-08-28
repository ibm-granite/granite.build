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

"""Algorithm-aware validation of an AutotuneConfig.

Load-time validation (autotune.config) only guarantees the always-consumed
sections exist. This module checks that the sections actually needed by the
chosen tuning_algo / rl_algo are present, so users can omit sections their
algorithm does not use without hitting a cryptic downstream failure.
"""

from autotune.config import AutotuneConfig
from autotune.constants import AUTOTUNE_OFFLINE_RL, AUTOTUNE_ONLINE_RL


def _require_tuner_entry(section: dict, section_name: str, algo: str) -> None:
    """Assert `section[algo]` exists and carries a `hyperparams` block."""
    if not section:
        raise ValueError(
            f"Algorithm '{algo}' requires a '{algo}' entry under '{section_name}', "
            f"but '{section_name}' is missing or empty."
        )
    if algo not in section:
        found = sorted(section.keys())
        raise ValueError(
            f"Algorithm '{algo}' requires a '{algo}' entry under '{section_name}'; found entries: {found}."
        )
    if "hyperparams" not in section[algo]:
        raise ValueError(f"Entry '{algo}' under '{section_name}' is missing a 'hyperparams' block.")


def validate_config_for_pipeline(config: AutotuneConfig, tuning_algo: str, rl_algo: str) -> None:
    """Validate that `config` carries the sections the algorithms require.

    Args:
        config: the loaded AutotuneConfig.
        tuning_algo: pipeline-resolved tuning algorithm (e.g. "lora", "sft",
            or "none"). Use AutotunePipeline.get_tuning_algo(), which already
            normalizes online RL to "none".
        rl_algo: pipeline-resolved RL algorithm (e.g. "dpo", "ppo", or "none").

    Raises:
        ValueError: if a required section or algorithm entry is missing.
    """
    tuners_config = config.get_tuners_config_dict()
    tuners_rl_config = config.get_tuners_rl_config_dict()
    training_rl_config = config.get_training_rl_config_dict()

    # SFT/PEFT tuner: any non-none tuning algo needs its tuners_config entry.
    if tuning_algo != "none":
        _require_tuner_entry(tuners_config, "tuners_config", tuning_algo)

    # Any RL (offline or online) needs training_rl_config populated and a
    # tuners_rl_config entry for the chosen RL algorithm.
    if rl_algo in AUTOTUNE_OFFLINE_RL or rl_algo in AUTOTUNE_ONLINE_RL:
        _require_tuner_entry(tuners_rl_config, "tuners_rl_config", rl_algo)
        if not training_rl_config:
            raise ValueError(
                f"RL algorithm '{rl_algo}' requires a non-empty 'training_rl_config' "
                f"section, but it is missing or empty."
            )
