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

from copy import deepcopy
from typing import Any, Dict

import yaml

# Local
from autotune.constants import (
    AUTOTUNE_CONFIG_SECTIONS,
    AUTOTUNE_OPTIONAL_CONFIG_SECTIONS,
)


class AutotuneConfig:
    """
    Main AutoTune config containing the following sections:
        - training_config
        - tune_config
        - tuners_config
    """

    def __init__(self):
        """
        Create an empty AutoTune configuration.
        """

        self.config = {}
        self.training_config = {}
        self.training_rl_config = {}
        self.tune_config = {}
        self.tuners_config = {}
        self.tuners_rl_config = {}
        self.tokenizer_config = {}

    def from_dict(self, config: Dict[str, Any]):
        """
        Initialize the configuration from a dict.

        Args:
            config: Dict[str, Any]
                A dict containing the config
        """

        self.config = deepcopy(config)

        # Safety checks
        for sec in AUTOTUNE_CONFIG_SECTIONS:
            if sec not in self.config.keys():
                raise ValueError(f"Missing {sec} from AutoTune configuration.")

        # Init the config sections
        for sec in AUTOTUNE_CONFIG_SECTIONS:
            self._init_section(sec)

        for sec in AUTOTUNE_OPTIONAL_CONFIG_SECTIONS:
            if sec in self.config:
                self._init_section(sec)

    def load(self, filename: str):
        """
        Load the configuration from a YAML file (e.g., autotune.yaml).

        Args:
            filename: str
                The path to the YAML file containing the configuration.
        """

        # Load the config file
        with open(filename, "r") as f:
            self.config = yaml.safe_load(f)
            f.close()

        # Init the config sections
        for section in AUTOTUNE_CONFIG_SECTIONS:
            if section not in self.config.keys():
                raise ValueError(f"Missing {section} from AutoTune configuration.")
            else:
                self._init_section(section)

        for section in AUTOTUNE_OPTIONAL_CONFIG_SECTIONS:
            if section in self.config:
                self._init_section(section)

    def _init_section(self, section: str):
        """
        Initialize a config section.

        Args:
            section: str
                The name of the section.
        """

        if section.lower() == "training_config":
            cfg = self.config.get("training_config")
            self.training_config = {}
            for key in cfg.keys():
                self.training_config[key] = cfg[key].get("default")
        elif section.lower() == "training_rl_config":
            cfg = self.config.get("training_rl_config")
            self.training_rl_config = {}
            for key in cfg.keys():
                self.training_rl_config[key] = cfg[key].get("default")
        elif section.lower() == "tune_config":
            cfg = self.config.get("tune_config")
            self.tune_config = {}
            for key in cfg.keys():
                self.tune_config[key] = cfg[key].get("default")
        elif section.lower() == "tuners_config":
            cfg = self.config.get("tuners_config")
            self.tuners_config = deepcopy(cfg)
        elif section.lower() == "tuners_rl_config":
            cfg = self.config.get("tuners_rl_config")
            self.tuners_rl_config = deepcopy(cfg)
        elif section.lower() == "tokenizer_config":
            cfg = self.config.get("tokenizer_config")
            self.tokenizer_config = {}
            if cfg:
                for key in cfg.keys():
                    self.tokenizer_config[key] = cfg[key].get("default")
        else:
            raise ValueError(f"Unknown config section: {section}")

    def get_default_config_dict(self, tuning_algo: str) -> Dict[str, Any]:
        """Returns the default configuration of the hyperparameters"""
        if tuning_algo == "none":
            return {
                "training_config": self.get_training_config_dict(),
                "tune_config": self.get_tune_config_dict(),
                "tuner_flags": {},
            }

        temp = self.tuners_config.get(tuning_algo)
        tuner_config = {}
        tuner_flags = {}
        hyperparams = temp.get("hyperparams")
        for key in hyperparams.keys():
            tuner_config[key] = hyperparams[key].get("default")
            tuner_flags[key] = hyperparams[key].get("for_tuner")

        result = {
            "training_config": self.get_training_config_dict(),
            "tune_config": self.get_tune_config_dict(),
            "tuner_flags": tuner_flags,
            **tuner_config,
        }

        return result

    def get_default_rl_config_dict(self, rl_algo: str) -> Dict[str, Any]:
        """Returns the default configuration of the hyperparameters for RL"""
        if rl_algo == "none":
            return {
                "training_rl_config": self.get_training_rl_config_dict(),
                "tune_config": self.get_tune_config_dict(),
                "tuner_rl_flags": {},
            }

        temp = self.tuners_rl_config.get(rl_algo)
        tuner_rl_config = {}
        tuner_rl_flags = {}
        hyperparams = temp.get("hyperparams")
        for key in hyperparams.keys():
            tuner_rl_config[key] = hyperparams[key].get("default")
            tuner_rl_flags[key] = hyperparams[key].get("for_tuner")

        result = {
            "training_rl_config": self.get_training_rl_config_dict(),
            "tune_config": self.get_tune_config_dict(),
            "tuner_rl_flags": tuner_rl_flags,
            **tuner_rl_config,
        }

        return result

    def get_training_config_dict(self) -> Dict[str, Any]:
        return self.training_config

    def get_training_rl_config_dict(self) -> Dict[str, Any]:
        return self.training_rl_config

    def get_tune_config_dict(self) -> Dict[str, Any]:
        return self.tune_config

    def get_tuners_config_dict(self) -> Dict[str, Any]:
        return self.tuners_config

    def get_tuners_rl_config_dict(self) -> Dict[str, Any]:
        return self.tuners_rl_config

    def get_tokenizer_config_dict(self) -> Dict[str, Any]:
        return self.tokenizer_config

    def get_tuner_config_dict(self, tuning_algo: str) -> Dict[str, Any]:
        """
        Get the tuner config for a given tuning algorithm.

        Args:
            tuning_type: str
                The tuning type: lora, loha, lokr, vera, prompt_tuning,
                prefix_tuning, p_tuning, sft.
        """

        # tuners_config is optional (e.g. omitted for online RL, where
        # tuning_algo is "none"). Return {} rather than asserting.
        if not self.tuners_config:
            return {}
        return self.tuners_config.get(tuning_algo, {})

    def get_tuner_rl_config_dict(self, rl_algo: str) -> Dict[str, Any]:
        """
        Get the RL tuner config for a given algorithm.

        Args:
            rl_algorithm: str
                The RL algorithm: dpo, orpo, kto, ppo, grpo.
        """

        # tuners_rl_config is optional (e.g. omitted for SFT/PEFT, where
        # rl_algo is "none"). Return {} rather than asserting.
        if not self.tuners_rl_config:
            return {}
        return self.tuners_rl_config.get(rl_algo, {})

    def get_autotune_config(self) -> Dict[str, Any]:
        return self.config

    def get_metric(self) -> str:
        """
        Return the metric associated with the AutoTune config.
        """

        return self.tune_config.get("metric", None)

    def get_mode(self) -> str:
        """
        Return the mode associated with the AutoTune config.
        """

        return self.tune_config.get("mode", None)
