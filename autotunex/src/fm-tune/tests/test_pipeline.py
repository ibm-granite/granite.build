"""Tests for autotune.pipeline.AutotunePipeline.

These tests use a HuggingFace identifier as the model so resolve_model_path
returns it untouched (no filesystem access).
"""

import pytest

from autotune.constants import AutotunePrecision
from autotune.pipeline import AutotunePipeline

MODEL = "facebook/opt-125m"  # HF identifier — passes through resolve_model_path


class TestValidCombinations:
    def test_sft_only(self):
        p = AutotunePipeline(tuning_algo="sft", rl_algo="none", model_name_or_path=MODEL)
        assert p.tuning_algo == "sft"
        assert p.rl_algo == "none"
        assert p.peft_type is None

    def test_lora_only(self):
        p = AutotunePipeline(tuning_algo="lora", rl_algo="none", model_name_or_path=MODEL)
        assert p.tuning_algo == "lora"
        assert p.peft_type is not None

    def test_qlora_only(self):
        # QLoRA maps to the same PeftType as LoRA (LoRA on a quantized base).
        from peft import PeftType

        p = AutotunePipeline(tuning_algo="qlora", rl_algo="none", model_name_or_path=MODEL)
        assert p.tuning_algo == "qlora"
        assert p.peft_type == PeftType.LORA

    def test_qlora_with_dpo(self):
        p = AutotunePipeline(tuning_algo="qlora", rl_algo="dpo", model_name_or_path=MODEL)
        assert p.tuning_algo == "qlora"
        assert p.rl_algo == "dpo"

    def test_lora_with_dpo(self):
        # Offline RL on top of LoRA
        p = AutotunePipeline(tuning_algo="lora", rl_algo="dpo", model_name_or_path=MODEL)
        assert p.tuning_algo == "lora"
        assert p.rl_algo == "dpo"

    def test_lora_with_kto(self):
        p = AutotunePipeline(tuning_algo="lora", rl_algo="kto", model_name_or_path=MODEL)
        assert p.rl_algo == "kto"

    def test_online_rl_resets_tuning_to_none(self):
        # Online RL doesn't need separate tuning
        p = AutotunePipeline(tuning_algo="lora", rl_algo="ppo", model_name_or_path=MODEL)
        assert p.tuning_algo == "none"
        assert p.rl_algo == "ppo"

    def test_grpo_alone(self):
        p = AutotunePipeline(tuning_algo="none", rl_algo="grpo", model_name_or_path=MODEL)
        assert p.rl_algo == "grpo"

    def test_dapo_alone(self):
        p = AutotunePipeline(tuning_algo="none", rl_algo="dapo", model_name_or_path=MODEL)
        assert p.rl_algo == "dapo"

    def test_none_string_treated_as_none(self):
        # Both lowercase string "None" and None object are accepted
        p = AutotunePipeline(tuning_algo="None", rl_algo="ppo", model_name_or_path=MODEL)
        assert p.tuning_algo == "none"

    def test_default_precision_is_bf16(self):
        p = AutotunePipeline(tuning_algo="sft", rl_algo="none", model_name_or_path=MODEL)
        assert p.precision == AutotunePrecision.BF16

    def test_default_multi_gpu_false(self):
        p = AutotunePipeline(tuning_algo="sft", rl_algo="none", model_name_or_path=MODEL)
        assert p.multi_gpu is False


class TestInvalidCombinations:
    def test_both_none_raises(self):
        with pytest.raises(ValueError, match="cannot be `none`"):
            AutotunePipeline(tuning_algo="none", rl_algo="none", model_name_or_path=MODEL)

    def test_offline_rl_without_tuning_raises(self):
        with pytest.raises(ValueError, match="Offline RL"):
            AutotunePipeline(tuning_algo="none", rl_algo="dpo", model_name_or_path=MODEL)

    def test_unknown_tuning_algo_raises(self):
        with pytest.raises(AssertionError):
            AutotunePipeline(tuning_algo="garbage", rl_algo="none", model_name_or_path=MODEL)

    def test_unknown_rl_algo_raises(self):
        with pytest.raises(AssertionError):
            AutotunePipeline(tuning_algo="sft", rl_algo="garbage", model_name_or_path=MODEL)


class TestAccessors:
    def test_getters(self):
        p = AutotunePipeline(tuning_algo="lora", rl_algo="none", model_name_or_path=MODEL)
        assert p.get_tuning_algo() == "lora"
        assert p.get_rl_algo() == "none"
        assert p.get_model_name_or_path() == MODEL
        assert p.get_peft_type() is not None
        assert p.get_precision() == AutotunePrecision.BF16
        assert p.get_multi_gpu() is False

    def test_set_multi_gpu(self):
        p = AutotunePipeline(tuning_algo="lora", rl_algo="none", model_name_or_path=MODEL)
        p.set_multi_gpu(True)
        assert p.get_multi_gpu() is True


class TestMakeConfig:
    """make_config currently has a bug: references self.rl_type instead of self.rl_algo.

    This test will fail until pipeline.py:131 is fixed. It locks in the
    expected behavior so the regression cannot return.
    """

    def test_make_config_returns_dict(self):
        p = AutotunePipeline(tuning_algo="lora", rl_algo="none", model_name_or_path=MODEL)
        cfg = p.make_config()
        assert isinstance(cfg, dict)
        assert cfg["pipeline.tuning_algo"] == "lora"
        assert cfg["pipeline.rl_algo"] == "none"
        assert cfg["pipeline.model_name_or_path"] == MODEL
        assert cfg["pipeline.multi_gpu"] is False

    def test_make_config_with_rl(self):
        p = AutotunePipeline(tuning_algo="none", rl_algo="ppo", model_name_or_path=MODEL)
        cfg = p.make_config()
        assert cfg["pipeline.rl_algo"] == "ppo"


def test_pipeline_set_precision():
    from autotune.constants import AutotunePrecision
    from autotune.pipeline import AutotunePipeline

    p = AutotunePipeline(tuning_algo="lora", rl_algo="none", model_name_or_path="facebook/opt-125m")
    assert p.get_precision() == AutotunePrecision.BF16
    p.set_precision(AutotunePrecision.FP32)
    assert p.get_precision() == AutotunePrecision.FP32
