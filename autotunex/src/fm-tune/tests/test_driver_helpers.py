"""Tests for pure helper functions inside autotune.trainers.

These avoid running real training. They cover:
  - dataset extension dispatch
  - chat-template mapping (with mocked tokenizer)
  - DeepSpeed config construction per ZeRO strategy
  - log-history metrics extraction
  - verl tensor-parallel resolution and resource pool construction
  - verl _InMemoryMetricsLogger aggregation
  - verl _select_best_checkpoint priority logic
"""

import json
import math
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ---------------- driver_single helpers ----------------


class TestLoadDatasetFromFile:
    def test_unknown_extension_raises(self, tmp_path):
        from autotune.trainers.driver_single import _load_dataset_from_file

        p = tmp_path / "data.xyz"
        p.write_text("foo")
        with pytest.raises(ValueError, match="Unsupported"):
            _load_dataset_from_file(str(p))

    def test_jsonl(self, sample_train_jsonl):
        from autotune.trainers.driver_single import _load_dataset_from_file

        ds = _load_dataset_from_file(str(sample_train_jsonl))
        assert len(ds) == 5
        assert "input" in ds.column_names

    def test_csv(self, tmp_path):
        from autotune.trainers.driver_single import _load_dataset_from_file

        p = tmp_path / "data.csv"
        p.write_text("input,output\nfoo,bar\nbaz,qux\n")
        ds = _load_dataset_from_file(str(p))
        assert len(ds) == 2


class TestMakeChatTemplateMapper:
    def test_passthrough_for_non_list_input(self):
        from autotune.trainers.driver_single import _make_chat_template_mapper

        tok = MagicMock()
        fn = _make_chat_template_mapper(tok, input_col="input")
        out = fn({"input": "plain string"})
        assert out["input"] == "plain string"
        tok.apply_chat_template.assert_not_called()

    def test_applies_template_for_message_list(self):
        from autotune.trainers.driver_single import _make_chat_template_mapper

        tok = MagicMock()
        tok.apply_chat_template.return_value = "TEMPLATED"
        fn = _make_chat_template_mapper(tok, input_col="input")
        msgs = [{"role": "user", "content": "hi"}]
        out = fn({"input": msgs})
        assert out["input"] == "TEMPLATED"
        # Called with tokenize=False, add_generation_prompt=True
        kwargs = tok.apply_chat_template.call_args.kwargs
        assert kwargs["tokenize"] is False
        assert kwargs["add_generation_prompt"] is True

    def test_forwards_documents(self):
        from autotune.trainers.driver_single import _make_chat_template_mapper

        tok = MagicMock()
        tok.apply_chat_template.return_value = "X"
        fn = _make_chat_template_mapper(tok, input_col="input")
        fn({"input": [{"role": "u", "content": "x"}], "documents": [{"text": "doc"}]})
        assert "documents" in tok.apply_chat_template.call_args.kwargs

    def test_forwards_tools(self):
        from autotune.trainers.driver_single import _make_chat_template_mapper

        tok = MagicMock()
        tok.apply_chat_template.return_value = "X"
        fn = _make_chat_template_mapper(tok, input_col="input")
        fn({"input": [{"role": "u", "content": "x"}], "tools": [{"name": "t"}]})
        assert "tools" in tok.apply_chat_template.call_args.kwargs

    def test_skips_empty_optional_columns(self):
        from autotune.trainers.driver_single import _make_chat_template_mapper

        tok = MagicMock()
        tok.apply_chat_template.return_value = "X"
        fn = _make_chat_template_mapper(tok, input_col="input")
        fn({"input": [{"role": "u", "content": "x"}], "documents": [], "tools": None})
        kwargs = tok.apply_chat_template.call_args.kwargs
        assert "documents" not in kwargs
        assert "tools" not in kwargs


# ---------------- driver_multi_hf_ds helpers ----------------


class TestBuildDeepspeedConfig:
    @pytest.mark.parametrize(
        "strategy,expected_stage",
        [
            ("zero1_gpu", 1),
            ("zero2_gpu", 2),
            ("zero2_cpu", 2),
            ("zero3_gpu", 3),
            ("zero3_cpu", 3),
        ],
    )
    def test_zero_stage_set(self, strategy, expected_stage):
        from autotune.trainers.driver_multi_hf_ds import _build_deepspeed_config

        cfg = _build_deepspeed_config(strategy)
        assert cfg["zero_optimization"]["stage"] == expected_stage

    def test_unknown_strategy_raises(self):
        from autotune.trainers.driver_multi_hf_ds import _build_deepspeed_config

        with pytest.raises(ValueError, match="Unknown"):
            _build_deepspeed_config("zero99_quantum")

    def test_cpu_offload_present_for_cpu_strategies(self):
        from autotune.trainers.driver_multi_hf_ds import _build_deepspeed_config

        z2_cpu = _build_deepspeed_config("zero2_cpu")
        assert "offload_optimizer" in z2_cpu["zero_optimization"]

        z3_cpu = _build_deepspeed_config("zero3_cpu")
        assert "offload_optimizer" in z3_cpu["zero_optimization"]
        assert "offload_param" in z3_cpu["zero_optimization"]

    def test_zero3_gpu_gathers_weights_on_save(self):
        from autotune.trainers.driver_multi_hf_ds import _build_deepspeed_config

        cfg = _build_deepspeed_config("zero3_gpu")
        assert cfg["zero_optimization"]["gather_16bit_weights_on_model_save"] is True

    def test_bf16_enabled_default(self):
        from autotune.trainers.driver_multi_hf_ds import _build_deepspeed_config

        cfg = _build_deepspeed_config("zero2_gpu")
        assert cfg["bf16"]["enabled"] is True


class TestApplyChatTemplateToDf:
    def test_empty_df_passes_through(self):
        from autotune.trainers.driver_multi_hf_ds import _apply_chat_template_to_df

        df = pd.DataFrame({"input": []})
        out = _apply_chat_template_to_df(df, MagicMock(), "input")
        assert len(out) == 0

    def test_no_message_lists_passes_through(self):
        from autotune.trainers.driver_multi_hf_ds import _apply_chat_template_to_df

        df = pd.DataFrame({"input": ["plain1", "plain2"]})
        tok = MagicMock()
        out = _apply_chat_template_to_df(df, tok, "input")
        # First row is a string → no template applied, df returned as-is
        assert list(out["input"]) == ["plain1", "plain2"]
        tok.apply_chat_template.assert_not_called()

    def test_applies_to_message_lists(self):
        from autotune.trainers.driver_multi_hf_ds import _apply_chat_template_to_df

        df = pd.DataFrame({"input": [[{"role": "user", "content": "hi"}], [{"role": "user", "content": "ho"}]]})
        tok = MagicMock()
        tok.apply_chat_template.side_effect = ["A", "B"]
        out = _apply_chat_template_to_df(df, tok, "input")
        assert list(out["input"]) == ["A", "B"]


class TestExtractMetricsFromLogHistory:
    def test_empty_log_returns_nans(self):
        from autotune.trainers.driver_multi_hf_ds import _extract_metrics_from_log_history

        out = _extract_metrics_from_log_history([])
        assert math.isnan(out["train_loss"])
        assert math.isnan(out["eval_loss"])

    def test_typical_log_history(self):
        from autotune.trainers.driver_multi_hf_ds import _extract_metrics_from_log_history

        log = [
            {"loss": 1.0, "epoch": 0.5, "learning_rate": 1e-4},
            {"loss": 0.9, "epoch": 1.0, "learning_rate": 5e-5},
            {"eval_loss": 0.85, "epoch": 1.0},
            {"loss": 0.7, "epoch": 1.5, "learning_rate": 2.5e-5},
        ]
        out = _extract_metrics_from_log_history(log)
        assert out["train_loss"] == 0.7  # last train loss
        assert out["eval_loss"] == 0.85
        assert out["train_loss_history"] == [1.0, 0.9, 0.7]
        assert out["eval_loss_history"] == [0.85]

    def test_only_eval(self):
        from autotune.trainers.driver_multi_hf_ds import _extract_metrics_from_log_history

        log = [{"eval_loss": 0.5}, {"eval_loss": 0.4}]
        out = _extract_metrics_from_log_history(log)
        assert out["eval_loss"] == 0.4
        assert math.isnan(out["train_loss"])


# ---------------- driver_multi_verl helpers ----------------


class TestResolveTensorParallelSize:
    def test_user_override_valid(self, tmp_path):
        from autotune.trainers.driver_multi_verl import _resolve_tensor_parallel_size

        # Explicit user TP=2 with 4 workers
        assert _resolve_tensor_parallel_size(str(tmp_path), num_workers=4, user_tp=2) == 2

    def test_user_override_not_power_of_2_raises(self, tmp_path):
        from autotune.trainers.driver_multi_verl import _resolve_tensor_parallel_size

        with pytest.raises(ValueError, match="power of 2"):
            _resolve_tensor_parallel_size(str(tmp_path), num_workers=4, user_tp=3)

    def test_user_override_doesnt_divide_workers_raises(self, tmp_path):
        from autotune.trainers.driver_multi_verl import _resolve_tensor_parallel_size

        with pytest.raises(ValueError, match="divide num_workers"):
            _resolve_tensor_parallel_size(str(tmp_path), num_workers=4, user_tp=8)

    def test_auto_detect_small_model_tp1(self, tmp_path):
        from autotune.trainers.driver_multi_verl import _resolve_tensor_parallel_size

        # ~1B: hidden_size=2048, layers=24
        cfg = {"hidden_size": 2048, "num_hidden_layers": 24, "vocab_size": 32000}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        assert _resolve_tensor_parallel_size(str(tmp_path), num_workers=4) == 1

    def test_auto_detect_medium_model_tp2(self, tmp_path):
        from autotune.trainers.driver_multi_verl import _resolve_tensor_parallel_size

        # ~13B: hidden_size=5120, layers=40 → params ≈ (12·5120²·40 + V·5120) / 1e9 ≈ 12.6B
        cfg = {"hidden_size": 5120, "num_hidden_layers": 40, "vocab_size": 32000}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        assert _resolve_tensor_parallel_size(str(tmp_path), num_workers=4) == 2

    def test_auto_detect_large_model_tp4(self, tmp_path):
        from autotune.trainers.driver_multi_verl import _resolve_tensor_parallel_size

        # ~40B: hidden_size=8192, layers=48
        cfg = {"hidden_size": 8192, "num_hidden_layers": 48, "vocab_size": 32000}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        assert _resolve_tensor_parallel_size(str(tmp_path), num_workers=4) == 4

    def test_clamp_to_workers(self, tmp_path):
        from autotune.trainers.driver_multi_verl import _resolve_tensor_parallel_size

        # Large model, 2 workers → TP must be clamped to 2 (not 4)
        cfg = {"hidden_size": 8192, "num_hidden_layers": 48, "vocab_size": 32000}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        assert _resolve_tensor_parallel_size(str(tmp_path), num_workers=2) == 2

    def test_missing_config_falls_back_to_tp1(self, tmp_path):
        from autotune.trainers.driver_multi_verl import _resolve_tensor_parallel_size

        # No config.json
        assert _resolve_tensor_parallel_size(str(tmp_path), num_workers=4) == 1


class TestBuildResourcePoolManager:
    def test_grpo_no_critic(self):
        from autotune.trainers.driver_multi_verl import build_resource_pool_manager

        rpm = build_resource_pool_manager(num_workers=4, rl_algorithm="grpo")
        # Inspect role_mapping: GRPO must NOT include Critic
        from verl.trainer.ppo.ray_trainer import Role

        assert Role.Critic not in rpm.mapping

    def test_ppo_has_critic(self):
        from autotune.trainers.driver_multi_verl import build_resource_pool_manager

        rpm = build_resource_pool_manager(num_workers=4, rl_algorithm="ppo")
        from verl.trainer.ppo.ray_trainer import Role

        assert Role.Critic in rpm.mapping

    def test_dapo_no_critic(self):
        from autotune.trainers.driver_multi_verl import build_resource_pool_manager

        rpm = build_resource_pool_manager(num_workers=4, rl_algorithm="dapo")
        from verl.trainer.ppo.ray_trainer import Role

        assert Role.Critic not in rpm.mapping

    def test_use_reward_model(self):
        from autotune.trainers.driver_multi_verl import build_resource_pool_manager

        rpm = build_resource_pool_manager(num_workers=4, rl_algorithm="ppo", use_reward_model=True)
        from verl.trainer.ppo.ray_trainer import Role

        assert Role.RewardModel in rpm.mapping

    def test_unknown_algo_raises(self):
        from autotune.trainers.driver_multi_verl import build_resource_pool_manager

        with pytest.raises(ValueError, match="Unsupported"):
            build_resource_pool_manager(num_workers=4, rl_algorithm="reinforce")


class TestInMemoryMetricsLogger:
    def test_empty_steps(self):
        from autotune.trainers.driver_multi_verl import _InMemoryMetricsLogger

        _InMemoryMetricsLogger._all_steps = []
        out = _InMemoryMetricsLogger.collect("ppo")
        assert out == {}

    def test_collect_aggregates(self):
        from autotune.trainers.driver_multi_verl import _InMemoryMetricsLogger

        _InMemoryMetricsLogger._all_steps = [
            {
                "_step": 0,
                "actor/ppo_loss": 1.0,
                "actor/pg_loss": 0.8,
                "actor/entropy": 0.5,
                "actor/kl_loss": 0.05,
                "critic/score/mean": 0.1,
                "critic/score/max": 0.2,
                "critic/score/min": 0.0,
                "training/global_step": 0,
            },
            {
                "_step": 1,
                "actor/ppo_loss": 0.5,
                "actor/pg_loss": 0.4,
                "actor/entropy": 0.4,
                "actor/kl_loss": 0.03,
                "critic/score/mean": 0.4,
                "critic/score/max": 0.5,
                "critic/score/min": 0.3,
                "training/global_step": 1,
            },
        ]
        out = _InMemoryMetricsLogger.collect("grpo")
        assert out["actor_loss"] == pytest.approx(0.75)
        assert out["pg_loss"] == pytest.approx(0.6)
        assert out["actor_entropy"] == pytest.approx(0.45)
        assert out["reward_mean"] == 0.4  # last step
        assert out["kl_divergence"] == pytest.approx(0.04)
        assert out["global_steps"] == 1
        # Not PPO → no critic_loss
        assert "critic_loss" not in out

    def test_ppo_includes_critic_loss(self):
        from autotune.trainers.driver_multi_verl import _InMemoryMetricsLogger

        _InMemoryMetricsLogger._all_steps = [
            {"_step": 0, "critic/loss": 0.6, "actor/ppo_loss": 0.5},
            {"_step": 1, "critic/loss": 0.4, "actor/ppo_loss": 0.3},
        ]
        out = _InMemoryMetricsLogger.collect("ppo")
        assert out["critic_loss"] == pytest.approx(0.5)

    def test_missing_keys_fall_back_to_nan(self):
        from autotune.trainers.driver_multi_verl import _InMemoryMetricsLogger

        _InMemoryMetricsLogger._all_steps = [{"_step": 0}]
        out = _InMemoryMetricsLogger.collect("grpo")
        assert math.isnan(out["actor_loss"])
        assert math.isnan(out["reward_mean"])


class TestSelectBestCheckpoint:
    def test_no_checkpoints(self):
        from autotune.trainers.driver_multi_verl import _select_best_checkpoint

        assert _select_best_checkpoint([], []) is None

    def test_no_metrics_returns_last(self):
        from autotune.trainers.driver_multi_verl import _select_best_checkpoint

        ckpts = ["/x/global_step_10", "/x/global_step_20"]
        assert _select_best_checkpoint(ckpts, []) == "/x/global_step_20"

    def test_picks_best_by_reward(self):
        from autotune.trainers.driver_multi_verl import _select_best_checkpoint

        ckpts = ["/x/global_step_10", "/x/global_step_20", "/x/global_step_30"]
        steps = [
            {"_step": 10, "critic/score/mean": 0.2},
            {"_step": 20, "critic/score/mean": 0.5},  # best
            {"_step": 30, "critic/score/mean": 0.3},
        ]
        # Best step is 20 → matching checkpoint is global_step_20
        out = _select_best_checkpoint(ckpts, steps)
        assert out == "/x/global_step_20"

    def test_falls_back_to_actor_loss_when_no_reward(self):
        from autotune.trainers.driver_multi_verl import _select_best_checkpoint

        ckpts = ["/x/global_step_10", "/x/global_step_20"]
        steps = [
            {"_step": 10, "actor/ppo_loss": 1.0},
            {"_step": 20, "actor/ppo_loss": 0.4},  # best (lower)
        ]
        assert _select_best_checkpoint(ckpts, steps) == "/x/global_step_20"

    def test_skips_nan_reward(self):
        from autotune.trainers.driver_multi_verl import _select_best_checkpoint

        ckpts = ["/x/global_step_10", "/x/global_step_20"]
        steps = [
            {"_step": 10, "critic/score/mean": float("nan")},
            {"_step": 20, "critic/score/mean": 0.5},
        ]
        assert _select_best_checkpoint(ckpts, steps) == "/x/global_step_20"


# Template stand-ins: 4.2 branches on enable_thinking, 4.1 does not.
GRANITE42_TEMPLATE = "{%- if enable_thinking %}<|im_start|>assistant<think>{%- endif %}"
GRANITE41_TEMPLATE = "{%- if documents %}{%- set x = (document | tojson) %}{%- endif %}"


def _thinking_tok(template):
    tok = MagicMock()
    tok.chat_template = template
    tok.apply_chat_template.return_value = "RENDERED"
    return tok


class TestThinkingKwargs:
    """Granite 4.2 defaults thinking on; the drivers must turn it off."""

    def test_forwarded_when_template_supports_it(self):
        from autotune.trainers.driver_single import _thinking_kwargs

        assert _thinking_kwargs(_thinking_tok(GRANITE42_TEMPLATE)) == {"enable_thinking": False}

    def test_absent_when_template_ignores_it(self):
        from autotune.trainers.driver_single import _thinking_kwargs

        assert _thinking_kwargs(_thinking_tok(GRANITE41_TEMPLATE)) == {}

    def test_absent_when_no_template(self):
        from autotune.trainers.driver_single import _thinking_kwargs

        assert _thinking_kwargs(_thinking_tok(None)) == {}

    def test_override_enables_thinking(self):
        from autotune.trainers.driver_single import _thinking_kwargs

        assert _thinking_kwargs(_thinking_tok(GRANITE42_TEMPLATE), thinking=True) == {"enable_thinking": True}

    def test_all_three_drivers_share_the_helper(self):
        from autotune.trainers.driver_multi_hf_ds import _thinking_kwargs as ds
        from autotune.trainers.driver_multi_hf_fsdp import _thinking_kwargs as fsdp
        from autotune.trainers.driver_single import _thinking_kwargs as single

        tok = _thinking_tok(GRANITE42_TEMPLATE)
        assert single(tok) == fsdp(tok) == ds(tok) == {"enable_thinking": False}


class TestChatTemplateThinking:
    """The thinking flag reaches apply_chat_template in every driver."""

    ROW = {"input": [{"role": "user", "content": "Q"}]}

    def test_single_driver_disables_thinking(self):
        from autotune.trainers.driver_single import _make_chat_template_mapper

        tok = _thinking_tok(GRANITE42_TEMPLATE)
        _make_chat_template_mapper(tok, "input")(dict(self.ROW))
        assert tok.apply_chat_template.call_args.kwargs["enable_thinking"] is False

    def test_single_driver_override(self):
        from autotune.trainers.driver_single import _make_chat_template_mapper

        tok = _thinking_tok(GRANITE42_TEMPLATE)
        _make_chat_template_mapper(tok, "input", thinking=True)(dict(self.ROW))
        assert tok.apply_chat_template.call_args.kwargs["enable_thinking"] is True

    def test_single_driver_skips_unsupported_template(self):
        from autotune.trainers.driver_single import _make_chat_template_mapper

        tok = _thinking_tok(GRANITE41_TEMPLATE)
        _make_chat_template_mapper(tok, "input")(dict(self.ROW))
        assert "enable_thinking" not in tok.apply_chat_template.call_args.kwargs

    def test_plain_string_rows_are_untouched(self):
        from autotune.trainers.driver_single import _make_chat_template_mapper

        tok = _thinking_tok(GRANITE42_TEMPLATE)
        out = _make_chat_template_mapper(tok, "input")({"input": "already a string"})
        assert out["input"] == "already a string"
        tok.apply_chat_template.assert_not_called()

    @pytest.mark.parametrize("driver", ["driver_multi_hf_fsdp", "driver_multi_hf_ds"])
    def test_multi_gpu_drivers_disable_thinking(self, driver):
        import importlib

        mod = importlib.import_module(f"autotune.trainers.{driver}")
        tok = _thinking_tok(GRANITE42_TEMPLATE)
        mod._apply_chat_template_to_df(pd.DataFrame([dict(self.ROW)]), tok, "input")
        assert tok.apply_chat_template.call_args.kwargs["enable_thinking"] is False

    @pytest.mark.parametrize("driver", ["driver_multi_hf_fsdp", "driver_multi_hf_ds"])
    def test_multi_gpu_drivers_override(self, driver):
        import importlib

        mod = importlib.import_module(f"autotune.trainers.{driver}")
        tok = _thinking_tok(GRANITE42_TEMPLATE)
        mod._apply_chat_template_to_df(pd.DataFrame([dict(self.ROW)]), tok, "input", thinking=True)
        assert tok.apply_chat_template.call_args.kwargs["enable_thinking"] is True

    @pytest.mark.parametrize("driver", ["driver_multi_hf_fsdp", "driver_multi_hf_ds"])
    @pytest.mark.parametrize("has_documents", [False, True])
    @pytest.mark.parametrize("has_tools", [False, True])
    def test_documents_and_tools_still_forwarded(self, driver, has_documents, has_tools):
        # The four documents x tools branches were collapsed to one call site;
        # this pins that the collapse is behaviour-preserving.
        import importlib

        mod = importlib.import_module(f"autotune.trainers.{driver}")
        tok = _thinking_tok(GRANITE42_TEMPLATE)
        row = dict(self.ROW)
        if has_documents:
            row["documents"] = [{"text": "D"}]
        if has_tools:
            row["tools"] = [{"type": "function"}]

        mod._apply_chat_template_to_df(pd.DataFrame([row]), tok, "input")
        kwargs = tok.apply_chat_template.call_args.kwargs
        assert ("documents" in kwargs) is has_documents
        assert ("tools" in kwargs) is has_tools
        assert kwargs["enable_thinking"] is False

    @pytest.mark.parametrize("driver", ["driver_multi_hf_fsdp", "driver_multi_hf_ds"])
    def test_non_message_column_untouched(self, driver):
        import importlib

        mod = importlib.import_module(f"autotune.trainers.{driver}")
        tok = _thinking_tok(GRANITE42_TEMPLATE)
        df = mod._apply_chat_template_to_df(pd.DataFrame([{"input": "plain"}]), tok, "input")
        assert df["input"].iloc[0] == "plain"
        tok.apply_chat_template.assert_not_called()
