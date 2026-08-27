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

from enum import Enum

from peft import PeftType

from autotune.catalog import AutotuneDatasetTypes  # noqa: F401  (re-export for back-compat)

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

# Tuning algorithms the MLX backend can run (sft = full fine-tune, lora, qlora
# = 4-bit quantized base + LoRA). PEFT-specific methods (alora/loha/lokr/vera)
# have no mlx-lm equivalent.
MLX_SUPPORTED_TUNING_ALGO = {"sft", "lora", "qlora"}

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
