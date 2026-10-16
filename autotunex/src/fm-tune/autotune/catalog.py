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
"""Torch-free access to AutoTune's static UI catalog data.

Import-light by contract: only the standard library and PyYAML — never torch /
ray / transformers / peft. AutoTuneX imports this to serve the tuning wizard's
configuration template and dataset-type catalog on a slim install that has not
installed the heavy training stack. Do NOT import ``autotune.utils`` or
``autotune.constants`` here — both pull in the heavy stack.
"""

from pathlib import Path
from typing import Any, Dict, List, Union  # noqa: F401  (List/Union used by AutotuneDatasetTypes)


def get_autotune_config() -> Dict[str, Any]:
    """Return the main AutoTune config from ``configs/autotune.yaml``."""
    import yaml

    d = Path(__file__).resolve().parent
    filename = Path.joinpath(d, "configs", "autotune.yaml")
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    return config


# Dataset types supported by AutoTune
AutotuneDatasetTypes = {
    "dataset_type_a": {
        "desc": "Dataset type used by the SFT/LoRA tuning algorithms",
        "columns": {
            "input_col": {
                "name": "input",
                "desc": "Input sequence",
                "type": Union[str, List[Dict[str, str]]],
                "required": True,
            },
            "output_col": {"name": "output", "desc": "Output sequence", "type": str, "required": True},
            "documents_col": {
                "name": "documents",
                "desc": "Retrieved documents associated with the input",
                "type": List[Dict[str, str]],
                "required": False,
            },
            "tools_col": {
                "name": "tools",
                "desc": "Tool calls associated with the input",
                "type": List[Dict[str, str]],
                "required": False,
            },
        },
    },
    "dataset_type_b": {
        "desc": "Dataset type used by the DPO/ORPO preference alignment algorithms",
        "columns": {
            "prompt_col": {"name": "prompt", "desc": "Input prompt", "type": str, "required": True},
            "chosen_col": {"name": "chosen", "desc": "Accepted generated sequence", "type": str, "required": True},
            "rejected_col": {"name": "rejected", "desc": "Rejected generated sequence", "type": str, "required": True},
        },
    },
    "dataset_type_c": {
        "desc": "Dataset type used by the KTO preference alignment algorithm",
        "columns": {
            "prompt": {"name": "prompt", "desc": "Input prompt", "type": str, "required": True},
            "completion": {"name": "completion", "desc": "Generated completion", "type": str, "required": True},
            "label": {
                "name": "label",
                "desc": "Label of the completion (e.g., positive/negative)",
                "type": str,
                "required": True,
            },
        },
    },
    "dataset_type_d": {
        "desc": "Dataset type used by the PPO, GRPO and DAPO reinforcement learning algorithms",
        "columns": {
            "data_source_col": {
                "name": "data_source",
                "desc": "Source of the dataset (e.g., openai/gsm8k)",
                "type": str,
                "required": True,
            },
            "prompt_col": {
                "name": "prompt",
                "desc": "Input prompt messages",
                "type": List[Dict[str, str]],
                "required_keys": ["role", "content"],
                "required": True,
            },
            "ability_col": {"name": "ability", "desc": "Ability of the dataset", "type": str, "required": True},
            "reward_model_col": {
                "name": "reward_model",
                "desc": "Reward model",
                "type": Dict[str, str],
                "required_keys": ["style", "ground_truth"],
                "required": True,
            },
            "extra_info_col": {
                "name": "extra_info",
                "desc": "Extra information associated with the dataset",
                "type": Dict[str, str],
                "required_keys": ["split", "index"],
                "required": True,
            },
        },
    },
}


def get_autotune_dataset_types() -> Dict[str, Any]:
    """Return the dataset types supported by AutoTune."""
    return AutotuneDatasetTypes
