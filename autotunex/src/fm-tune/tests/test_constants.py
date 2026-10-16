"""Tests for autotune.constants — algorithm registries and enum integrity."""

from autotune.constants import (
    AUTOTUNE_OFFLINE_RL,
    AUTOTUNE_ONLINE_RL,
    AUTOTUNE_RL_ALGO,
    AUTOTUNE_TUNING_ALGO,
    AUTOTUNE_TUNING_TO_PEFT_TYPE,
    AutotuneDatasetTypes,
    AutotuneMetrics,
    AutotunePrecision,
    AutotuneTuningTypes,
)


class TestAutotunePrecision:
    def test_members(self):
        assert AutotunePrecision.FP32.value == "fp32"
        assert AutotunePrecision.BF16.value == "bf16"
        assert AutotunePrecision.INT8.value == "int8"
        assert AutotunePrecision.INT4.value == "int4"

    def test_str_enum(self):
        # AutotunePrecision inherits from str
        assert AutotunePrecision.FP32 == "fp32"


class TestTuningAlgoRegistry:
    def test_no_duplicates(self):
        assert len(AUTOTUNE_TUNING_ALGO) == len(set(AUTOTUNE_TUNING_ALGO))

    def test_includes_none_and_sft(self):
        assert "none" in AUTOTUNE_TUNING_ALGO
        assert "sft" in AUTOTUNE_TUNING_ALGO

    def test_peft_type_mapping_covers_all_algos(self):
        for algo in AUTOTUNE_TUNING_ALGO:
            assert algo in AUTOTUNE_TUNING_TO_PEFT_TYPE, (
                f"Tuning algo {algo!r} missing from AUTOTUNE_TUNING_TO_PEFT_TYPE"
            )

    def test_sft_and_none_map_to_none(self):
        assert AUTOTUNE_TUNING_TO_PEFT_TYPE["sft"] is None
        assert AUTOTUNE_TUNING_TO_PEFT_TYPE["none"] is None


class TestRLAlgoRegistry:
    def test_no_duplicates(self):
        assert len(AUTOTUNE_RL_ALGO) == len(set(AUTOTUNE_RL_ALGO))

    def test_includes_none(self):
        assert "none" in AUTOTUNE_RL_ALGO

    def test_offline_subset_of_rl_algo(self):
        assert set(AUTOTUNE_OFFLINE_RL).issubset(set(AUTOTUNE_RL_ALGO))

    def test_online_subset_of_rl_algo(self):
        assert set(AUTOTUNE_ONLINE_RL).issubset(set(AUTOTUNE_RL_ALGO))

    def test_offline_and_online_disjoint(self):
        assert set(AUTOTUNE_OFFLINE_RL).isdisjoint(set(AUTOTUNE_ONLINE_RL))

    def test_offline_contains_dpo_kto(self):
        assert "dpo" in AUTOTUNE_OFFLINE_RL
        assert "kto" in AUTOTUNE_OFFLINE_RL

    def test_online_contains_ppo_grpo_dapo(self):
        assert "ppo" in AUTOTUNE_ONLINE_RL
        assert "grpo" in AUTOTUNE_ONLINE_RL
        assert "dapo" in AUTOTUNE_ONLINE_RL

    def test_none_in_neither_offline_nor_online(self):
        assert "none" not in AUTOTUNE_OFFLINE_RL
        assert "none" not in AUTOTUNE_ONLINE_RL


class TestDescriptiveRegistries:
    def test_tuning_types_keys_subset_of_algo_list(self):
        # AutotuneTuningTypes is a curated subset; every key should be valid
        for key in AutotuneTuningTypes:
            assert key in AUTOTUNE_TUNING_ALGO

    def test_metrics_have_descriptions(self):
        for name, body in AutotuneMetrics.items():
            assert "description" in body, f"Metric {name} missing description"

    def test_dataset_types_have_columns(self):
        for name, body in AutotuneDatasetTypes.items():
            assert "columns" in body
            assert "desc" in body
