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

# Local
from typing import Any, Dict

from autotune.constants import (
    AUTOTUNE_OFFLINE_RL,
    AUTOTUNE_ONLINE_RL,
    AUTOTUNE_RL_ALGO,
    AUTOTUNE_TUNING_ALGO,
    AUTOTUNE_TUNING_TO_PEFT_TYPE,
    AutotunePrecision,
)
from autotune.utils import resolve_model_path


class AutotunePipeline:
    """
    The pipeline structure used by AutoTune for finetuning. The AutotunePipeline
    supports supervised finetuning (full SFT and PEFT), offline RL (e.g., DPO)
    and online RL (e.g., PPO, GRPO). The pipeline is designed to be flexible and
    extensible to support new finetuning methods and RL algorithms.

    """

    def __init__(
        self,
        tuning_algo: str,
        rl_algo: str,
        model_name_or_path: str,
    ):
        """
        Create an Autotune pipeline.

        Args:
            tuning_algo: str
                The finetuning algorithm (SFT or PEFT).
                The allowed values are (default is `sft`):
                    - alora for PeftType.ALORA
                    - lora for PeftType.LORA
                    - loha for PeftType.LOHA
                    - lokr for PeftType.LOKR
                    - vera for PeftType.VERA
                    - sft
                    - none for online RL post-training
            rl_algo: str
                The RL based post training.
                    - dpo for DPO (offline)
                    - orpo for ORPO (offline)
                    - grpo for GRPO (online)
                    - ppo for PPO (online)
                    - none for no RL post-training (default)
            model_name_or_path: str
                The model or path to the model to be tuned.

        """

        super().__init__()

        # Set the internal members
        self.name = "autotunex"
        self.tuning_algo = "none" if tuning_algo is None or str(tuning_algo).lower() == "none" else tuning_algo
        self.rl_algo = "none" if rl_algo is None or str(rl_algo).lower() == "none" else rl_algo
        self.model_name_or_path = resolve_model_path(model_name_or_path)
        self.precision = AutotunePrecision.BF16  # default is bf16
        self.multi_gpu = (
            False  # default to single GPU for better compatibility, can be set to True for multi-GPU training
        )

        # Ensure that the tuning type is supported
        assert self.tuning_algo in AUTOTUNE_TUNING_ALGO, f"Tuning algorithm `{tuning_algo}` is not supported."
        assert rl_algo in AUTOTUNE_RL_ALGO, f"RL algorithm `{rl_algo}` is not supported."

        # The RL and tuning algorithms cannot be `none` at the same time
        if self.tuning_algo == "none" and self.rl_algo == "none":
            raise ValueError("Tuning and RL algorithms cannot be `none` at the same time.")

        # Online RL doesn't require a tuning algorithm such as SFT/LoRA
        if self.rl_algo in AUTOTUNE_ONLINE_RL:
            self.tuning_algo = "none"

        # Offline RL requires a tuning algorithm such as SFT/LoRA
        if self.rl_algo in AUTOTUNE_OFFLINE_RL and self.tuning_algo == "none":
            raise ValueError("Offline RL requires a tuning algorithm e.g., SFT or LoRA.")

        # Set the peft type from the input tuning type
        self.peft_type = AUTOTUNE_TUNING_TO_PEFT_TYPE[self.tuning_algo]

    def get_tuning_algo(self):
        return self.tuning_algo

    def get_rl_algo(self):
        return self.rl_algo  # can be None for non-RL tuning methods

    def get_model_name_or_path(self):
        return self.model_name_or_path

    def get_peft_type(self):
        return self.peft_type

    def get_precision(self):
        return self.precision

    def get_multi_gpu(self):
        return self.multi_gpu

    def set_multi_gpu(self, multi_gpu: bool):
        self.multi_gpu = multi_gpu

    def set_precision(self, precision):
        self.precision = precision

    def make_config(self) -> Dict[str, Any]:
        """
        Prepare the pipeline configs.
        """

        self.config = {}
        self.config["pipeline.name"] = self.name
        self.config["pipeline.tuning_algo"] = self.tuning_algo
        self.config["pipeline.rl_algo"] = self.rl_algo
        self.config["pipeline.peft_type"] = self.peft_type
        self.config["pipeline.model_name_or_path"] = self.model_name_or_path
        self.config["pipeline.precision"] = self.precision
        self.config["pipeline.multi_gpu"] = self.multi_gpu

        return self.config
