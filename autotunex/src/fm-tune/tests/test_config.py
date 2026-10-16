"""Tests for autotune.config.AutotuneConfig."""

import pytest

from autotune.config import AutotuneConfig


class TestFromDict:
    def test_round_trip(self, sample_config_dict):
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        # Original dict preserved on `.config`
        assert cfg.get_autotune_config() is not None

    def test_missing_required_section_raises(self, sample_config_dict):
        del sample_config_dict["training_config"]
        cfg = AutotuneConfig()
        with pytest.raises(ValueError, match="training_config"):
            cfg.from_dict(sample_config_dict)

    def test_optional_tokenizer_config(self, sample_config_dict):
        # Should work with or without tokenizer_config
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        assert cfg.get_tokenizer_config_dict() == {"pad_token": None}

    def test_no_tokenizer_section_ok(self, sample_config_dict):
        del sample_config_dict["tokenizer_config"]
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        assert cfg.get_tokenizer_config_dict() == {}

    def test_rl_sections_optional(self, sample_config_dict):
        # SFT/LoRA users may omit both RL sections without error.
        del sample_config_dict["training_rl_config"]
        del sample_config_dict["tuners_rl_config"]
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        assert cfg.get_training_rl_config_dict() == {}
        assert cfg.get_tuners_rl_config_dict() == {}

    def test_tuners_config_optional_at_load(self, sample_config_dict):
        # Online-RL users may omit tuners_config without a load error.
        del sample_config_dict["tuners_config"]
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        assert cfg.get_tuners_config_dict() == {}

    def test_training_config_still_required(self, sample_config_dict):
        del sample_config_dict["training_config"]
        cfg = AutotuneConfig()
        with pytest.raises(ValueError, match="training_config"):
            cfg.from_dict(sample_config_dict)

    def test_tune_config_still_required(self, sample_config_dict):
        del sample_config_dict["tune_config"]
        cfg = AutotuneConfig()
        with pytest.raises(ValueError, match="tune_config"):
            cfg.from_dict(sample_config_dict)


class TestLoadFromYaml:
    def test_load_yaml(self, sample_config_yaml_path):
        cfg = AutotuneConfig()
        cfg.load(sample_config_yaml_path)
        assert cfg.get_training_config_dict()["num_train_epochs"] == 1
        assert cfg.get_tune_config_dict()["search_alg"] == "lds"


class TestSectionExtraction:
    def test_training_config_extracts_defaults(self, sample_config_dict):
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        tc = cfg.get_training_config_dict()
        assert tc["num_train_epochs"] == 1
        assert tc["max_length"] == 128
        assert tc["precision"] == "fp32"

    def test_tune_config_extracts_defaults(self, sample_config_dict):
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        tune_cfg = cfg.get_tune_config_dict()
        assert tune_cfg["search_alg"] == "lds"
        assert tune_cfg["num_samples"] == 4
        assert tune_cfg["metric"] == "loss"
        assert tune_cfg["mode"] == "min"

    def test_training_rl_config(self, sample_config_dict):
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        rl = cfg.get_training_rl_config_dict()
        assert rl["rollout_temperature"] == 1.0

    def test_tuners_config_preserves_structure(self, sample_config_dict):
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        tuners = cfg.get_tuners_config_dict()
        assert "lora" in tuners
        assert "hyperparams" in tuners["lora"]


class TestDefaultConfigDict:
    def test_none_tuning_algo(self, sample_config_dict):
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        d = cfg.get_default_config_dict("none")
        assert d["tuner_flags"] == {}
        assert "training_config" in d
        assert "tune_config" in d

    def test_lora(self, sample_config_dict):
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        d = cfg.get_default_config_dict("lora")
        # LoRA hyperparams from fixture: r, lora_alpha, learning_rate
        assert d["r"] == 8
        assert d["lora_alpha"] == 16
        assert d["learning_rate"] == 1e-4
        assert d["tuner_flags"] == {"r": True, "lora_alpha": True, "learning_rate": False}

    def test_unknown_algo_raises(self, sample_config_dict):
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        with pytest.raises(AttributeError):
            # Underlying impl: temp.get("hyperparams") on None
            cfg.get_default_config_dict("nonexistent_algo")


class TestDefaultRLConfigDict:
    def test_none_rl(self, sample_config_dict):
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        d = cfg.get_default_rl_config_dict("none")
        assert d["tuner_rl_flags"] == {}
        assert "training_rl_config" in d
        assert "tune_config" in d

    def test_dpo(self, sample_config_dict):
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        d = cfg.get_default_rl_config_dict("dpo")
        assert d["beta"] == 1.0
        assert d["tuner_rl_flags"] == {"beta": True}

    def test_ppo(self, sample_config_dict):
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        d = cfg.get_default_rl_config_dict("ppo")
        assert d["kl_coef"] == 0.001


class TestMetricAndMode:
    def test_get_metric(self, sample_config_dict):
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        assert cfg.get_metric() == "loss"

    def test_get_mode(self, sample_config_dict):
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        assert cfg.get_mode() == "min"


class TestTunerLookup:
    def test_get_tuner_config(self, sample_config_dict):
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        out = cfg.get_tuner_config_dict("lora")
        assert "hyperparams" in out

    def test_get_tuner_config_unknown_returns_empty(self, sample_config_dict):
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        # Falls back to {} via dict.get default
        assert cfg.get_tuner_config_dict("nonexistent") == {}

    def test_get_tuner_rl_config(self, sample_config_dict):
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        out = cfg.get_tuner_rl_config_dict("dpo")
        assert "hyperparams" in out


class TestTunerAccessorsTolerateEmpty:
    def test_tuner_rl_config_empty_returns_empty(self, sample_config_dict):
        del sample_config_dict["tuners_rl_config"]
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        # "none" rl_algo on an absent section must not assert.
        assert cfg.get_tuner_rl_config_dict("none") == {}

    def test_tuner_config_empty_returns_empty(self, sample_config_dict):
        del sample_config_dict["tuners_config"]
        cfg = AutotuneConfig()
        cfg.from_dict(sample_config_dict)
        # online RL forces tuning_algo == "none"; absent section must not assert.
        assert cfg.get_tuner_config_dict("none") == {}
