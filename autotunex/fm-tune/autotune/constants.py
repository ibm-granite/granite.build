# coding=utf-8
# Copyright 2023-present the International Business Machines.
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

from enum import Enum
from typing import Dict, List, Union

from peft import PeftType

AUTOTUNE_DEFAULT_METRIC = "loss"
AUTOTUNE_DEFAULT_MODE = "min"


class AutotunePrecision(str, Enum):
    FP32 = "fp32"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


AUTOTUNE_CONFIG_SECTIONS = [
    "training_config",
    "tune_config",
]

AUTOTUNE_OPTIONAL_CONFIG_SECTIONS = [
    "tuners_config",
    "tuners_rl_config",
    "training_rl_config",
    "tokenizer_config",
]

# List of supported tuning methods including both PEFT and SFT
AUTOTUNE_TUNING_ALGO = [
    "prompt_tuning",
    "prefix_tuning",
    "p_tuning",
    "lora",
    "qlora",
    "loha",
    "lokr",
    "vera",
    "sft",
    "alora",
    "none",
]

# List of supported RL methods (both online and offline)
AUTOTUNE_RL_ALGO = ["dpo", "kto", "ppo", "grpo", "dapo", "none"]

# List of supported offline RL methods
AUTOTUNE_OFFLINE_RL = ["dpo", "kto"]

# List of supported online RL methods
AUTOTUNE_ONLINE_RL = ["ppo", "grpo", "dapo"]

# List of supported evaluation metrics
AUTOTUNE_METRICS = ["accuracy", "f1", "rouge1", "rouge2", "rougeL", "exact_match", "precision", "recall"]

# Mapping from tuning types to PEFT types
AUTOTUNE_TUNING_TO_PEFT_TYPE = {
    "prompt_tuning": PeftType.PROMPT_TUNING,
    "prefix_tuning": PeftType.PREFIX_TUNING,
    "p_tuning": PeftType.P_TUNING,
    "lora": PeftType.LORA,
    # QLoRA is LoRA on a 4-bit (NF4) bitsandbytes-quantized base; PEFT has no
    # dedicated QLoRA type, so it maps to LoRA. The quantized base load is
    # triggered by the "qlora" tuning-algorithm name inside the drivers.
    "qlora": PeftType.LORA,
    "loha": PeftType.LOHA,
    "lokr": PeftType.LOKR,
    "vera": PeftType.VERA,
    "sft": None,
    "alora": "ALORA",
    "none": None,
}

##################

# Tuning types supported by AutoTune
AutotuneTuningTypes = {
    "sft": {"description": "Supervised Fine-Tuning", "peft_type": None, "tuner_name": "tuner.sft"},
    "lora": {
        "description": "Low Rank Adaptor Fine-Tuning",
        "peft_type": PeftType.LORA,
        "tuner_name": "tuner.lora",
    },
    "qlora": {
        "description": "Quantized (4-bit NF4) Low Rank Adaptor Fine-Tuning",
        "peft_type": PeftType.LORA,
        "tuner_name": "tuner.qlora",
    },
    "loha": {
        "description": "Low Rank Fine-Tuning",
        "peft_type": PeftType.LOHA,
        "tuner_name": "tuner.loha",
    },
    "lokr": {
        "description": "Low Rank Fine-Tuning",
        "peft_type": PeftType.LOKR,
        "tuner_name": "tuner.lokr",
    },
    "prompt_tuning": {
        "description": "Prompt Tuning",
        "peft_type": PeftType.PROMPT_TUNING,
        "tuner_name": "tuner.prompt_tuning",
    },
    "prefix_tuning": {
        "description": "Prefix Tuning",
        "peft_type": PeftType.PREFIX_TUNING,
        "tuner_name": "tuner.prefix_tuning",
    },
    "p_tuning": {"description": "P-Tuning", "peft_type": PeftType.P_TUNING, "tuner_name": "tuner.p_tuning"},
}

# Metrics supported by AutoTune
AutotuneMetrics = {
    "accuracy": {
        "description": "The accuracy metric for classification tasks",
    },
    "f1": {
        "description": "The F1 metric for classification tasks",
    },
    "precision": {
        "description": "The precision metric for classification tasks",
    },
    "recall": {
        "description": "The recall metric for classification tasks",
    },
    "exact_match": {
        "description": "The exact match metric for generative tasks",
    },
    "rouge1": {
        "description": "The rouge1 metric for generative tasks",
    },
    "rouge2": {
        "description": "The rouge2 metric for generative tasks",
    },
    "rougeL": {
        "description": "The rougeL metric for generative tasks",
    },
}

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
