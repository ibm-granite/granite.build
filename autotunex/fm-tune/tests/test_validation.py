"""Tests for autotune.validation.validate_config_for_pipeline."""

import pytest

from autotune.config import AutotuneConfig
from autotune.validation import validate_config_for_pipeline


def _cfg(sample_config_dict):
    cfg = AutotuneConfig()
    cfg.from_dict(sample_config_dict)
    return cfg


class TestSFTPeft:
    def test_lora_no_rl_sections_passes(self, sample_config_dict):
        del sample_config_dict["training_rl_config"]
        del sample_config_dict["tuners_rl_config"]
        cfg = _cfg(sample_config_dict)
        # Should not raise.
        validate_config_for_pipeline(cfg, tuning_algo="lora", rl_algo="none")

    def test_missing_tuners_config_entry_raises(self, sample_config_dict):
        # tuners_config has lora + sft, but not loha.
        cfg = _cfg(sample_config_dict)
        with pytest.raises(ValueError, match="loha"):
            validate_config_for_pipeline(cfg, tuning_algo="loha", rl_algo="none")

    def test_missing_tuners_config_section_raises(self, sample_config_dict):
        del sample_config_dict["tuners_config"]
        cfg = _cfg(sample_config_dict)
        with pytest.raises(ValueError, match="tuners_config"):
            validate_config_for_pipeline(cfg, tuning_algo="lora", rl_algo="none")

    def test_entry_missing_hyperparams_raises(self, sample_config_dict):
        # tuners_config['lora'] exists but its hyperparams block is removed.
        del sample_config_dict["tuners_config"]["lora"]["hyperparams"]
        cfg = _cfg(sample_config_dict)
        with pytest.raises(ValueError, match="hyperparams"):
            validate_config_for_pipeline(cfg, tuning_algo="lora", rl_algo="none")


class TestOfflineRL:
    def test_dpo_complete_passes(self, sample_config_dict):
        cfg = _cfg(sample_config_dict)
        validate_config_for_pipeline(cfg, tuning_algo="lora", rl_algo="dpo")

    def test_dpo_missing_rl_entry_raises(self, sample_config_dict):
        # tuners_rl_config has dpo + ppo; remove dpo to trigger.
        del sample_config_dict["tuners_rl_config"]["dpo"]
        cfg = _cfg(sample_config_dict)
        with pytest.raises(ValueError, match="dpo"):
            validate_config_for_pipeline(cfg, tuning_algo="lora", rl_algo="dpo")

    def test_dpo_missing_training_rl_config_raises(self, sample_config_dict):
        del sample_config_dict["training_rl_config"]
        cfg = _cfg(sample_config_dict)
        with pytest.raises(ValueError, match="training_rl_config"):
            validate_config_for_pipeline(cfg, tuning_algo="lora", rl_algo="dpo")


class TestOnlineRL:
    def test_grpo_without_tuners_config_passes(self, sample_config_dict):
        # online RL forces tuning_algo == "none"; tuners_config not required.
        del sample_config_dict["tuners_config"]
        # fixture has ppo but not grpo; add a grpo entry.
        sample_config_dict["tuners_rl_config"]["grpo"] = {
            "hyperparams": {"kl_coef": {"default": 0.001, "for_tuner": True, "type": "float"}}
        }
        cfg = _cfg(sample_config_dict)
        validate_config_for_pipeline(cfg, tuning_algo="none", rl_algo="grpo")

    def test_ppo_missing_tuners_rl_config_raises(self, sample_config_dict):
        del sample_config_dict["tuners_rl_config"]
        cfg = _cfg(sample_config_dict)
        with pytest.raises(ValueError, match="tuners_rl_config"):
            validate_config_for_pipeline(cfg, tuning_algo="none", rl_algo="ppo")

    def test_ppo_missing_training_rl_config_raises(self, sample_config_dict):
        del sample_config_dict["training_rl_config"]
        cfg = _cfg(sample_config_dict)
        with pytest.raises(ValueError, match="training_rl_config"):
            validate_config_for_pipeline(cfg, tuning_algo="none", rl_algo="ppo")
