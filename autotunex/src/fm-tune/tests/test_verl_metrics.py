import json
import logging
import math
import types

import numpy as np
import pytest

from autotune.callbacks.logging_service import RecordType
from autotune.callbacks.metrics_emitter import METRIC_MARKER, close_metrics_handler
from autotune.callbacks.verl_metrics import (
    VerlStepReporter,
    _InMemoryMetricsLogger,
    _last_hf_model_dir,
    build_verl_row,
    format_step_summary,
    set_active_reporter,
)


class _FakeTensor:
    """Duck-types the slice of torch.Tensor that _InMemoryMetricsLogger.log uses."""

    def __init__(self, values):
        self._values = values if isinstance(values, list) else [values]

    def numel(self):
        return len(self._values)

    def item(self):
        return self._values[0]

    def tolist(self):
        return list(self._values)


class TestInMemoryMetricsLogger:
    def test_empty_steps(self):
        _InMemoryMetricsLogger._all_steps = []
        out = _InMemoryMetricsLogger.collect("ppo")
        assert out == {}

    def test_collect_aggregates(self):
        _InMemoryMetricsLogger._all_steps = [
            {
                "_step": 0,
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
        # verl 0.7.1 emits no actor/ppo_loss: actor_loss is the mean policy-gradient loss.
        assert out["actor_loss"] == pytest.approx(0.6)
        assert out["pg_loss"] == pytest.approx(0.6)
        assert out["actor_entropy"] == pytest.approx(0.45)
        assert out["reward_mean"] == 0.4  # last step
        assert out["kl_divergence"] == pytest.approx(0.04)
        assert out["global_steps"] == 1
        # Not PPO → no critic_loss
        assert "critic_loss" not in out

    def test_ppo_includes_critic_loss(self):
        _InMemoryMetricsLogger._all_steps = [
            {"_step": 0, "critic/loss": 0.6, "actor/pg_loss": 0.5},
            {"_step": 1, "critic/loss": 0.4, "actor/pg_loss": 0.3},
        ]
        out = _InMemoryMetricsLogger.collect("ppo")
        assert out["critic_loss"] == pytest.approx(0.5)

    def test_missing_keys_fall_back_to_nan(self):
        _InMemoryMetricsLogger._all_steps = [{"_step": 0}]
        out = _InMemoryMetricsLogger.collect("grpo")
        assert math.isnan(out["actor_loss"])
        assert math.isnan(out["reward_mean"])

    def test_log_converts_tensor_like_values_to_python_scalars(self):
        mem = _InMemoryMetricsLogger()

        mem.log(
            {
                "scalar": _FakeTensor(0.5),
                "vector": _FakeTensor([1.0, 2.0]),
                "numpy_scalar": types.SimpleNamespace(item=lambda: 4),
                "plain": 3.0,
            },
            step=7,
        )

        assert _InMemoryMetricsLogger._all_steps == [
            {"_step": 7, "scalar": 0.5, "vector": [1.0, 2.0], "numpy_scalar": 4, "plain": 3.0}
        ]

    def test_log_keeps_a_multi_element_numpy_array_as_a_list(self):
        mem = _InMemoryMetricsLogger()

        mem.log({"per_sample": np.array([1.0, 2.0]), "scalar": np.float32(4.0)}, step=1)

        assert _InMemoryMetricsLogger._all_steps == [{"_step": 1, "per_sample": [1.0, 2.0], "scalar": 4.0}]

    def test_collect_reads_throughput_as_tokens_per_second_per_gpu(self):
        _InMemoryMetricsLogger._all_steps = [
            {"_step": 0, "perf/throughput": 800.0},
            {"_step": 1, "perf/throughput": 812.5},
        ]
        out = _InMemoryMetricsLogger.collect("grpo")
        assert out["tokens_per_second_per_gpu"] == 812.5  # last step
        assert "tokens_per_second" not in out


class TestLastHfModelDir:
    @staticmethod
    def _make(tmp_path, step, with_model=True):
        d = tmp_path / f"global_step_{step}"
        (d / "actor" / "huggingface" if with_model else d).mkdir(parents=True)
        return str(d)

    def test_no_checkpoints(self):
        assert _last_hf_model_dir([]) is None

    def test_picks_highest_step_numerically(self, tmp_path):
        # sorted(glob) puts global_step_99 after global_step_100.
        ckpts = sorted([self._make(tmp_path, 100), self._make(tmp_path, 99)])
        assert _last_hf_model_dir(ckpts) == str(tmp_path / "global_step_100" / "actor" / "huggingface")

    def test_ignores_rotated_older_checkpoints(self, tmp_path):
        # verl's max_actor_ckpt_to_keep rotation removes the oldest
        # global_step_N/actor but leaves global_step_N/ itself behind.
        ckpts = [self._make(tmp_path, s, with_model=False) for s in (108, 216)]
        ckpts += [self._make(tmp_path, s) for s in (324, 432, 540)]
        assert _last_hf_model_dir(ckpts) == str(tmp_path / "global_step_540" / "actor" / "huggingface")

    def test_skips_checkpoint_without_model(self, tmp_path):
        ckpts = [self._make(tmp_path, 486), self._make(tmp_path, 540, with_model=False)]
        assert _last_hf_model_dir(ckpts) == str(tmp_path / "global_step_486" / "actor" / "huggingface")

    def test_none_when_no_checkpoint_has_model(self, tmp_path):
        assert _last_hf_model_dir([self._make(tmp_path, 378, with_model=False)]) is None


# One step as verl 0.7.1 reports it (after _InMemoryMetricsLogger.log added "_step").
_REAL_STEP = {
    "_step": 37,
    "training/global_step": 37,
    "training/epoch": 0,
    "actor/pg_loss": -0.036,
    "actor/grad_norm": 0.41,
    "actor/lr": 1e-06,
    "actor/kl_loss": 0.0,
    "actor/entropy": 0.283,
    "critic/score/mean": -0.951,
    "critic/score/max": -0.951,
    "critic/score/min": -0.951,
    "response_length/mean": 2.0,
    "perf/throughput": 812.5,
}


class TestBuildVerlRow:
    def test_maps_the_columns(self):
        row = build_verl_row(_REAL_STEP, 37, trial_id="t1", job_id="job-1")

        assert row["job_id"] == "job-1"
        assert row["trial_id"] == "t1"
        assert row["global_step"] == 37
        assert row["epoch"] == 0.0
        assert row["loss"] == -0.036
        assert row["grad_norm"] == 0.41
        assert row["learning_rate"] == 1e-06
        assert row["split"] == "train"

    def test_epoch_is_fractional_progress_when_steps_per_epoch_is_known(self):
        # Matches HF's state.epoch: 1.0 at the end of the first epoch.
        assert build_verl_row(_REAL_STEP, 25, trial_id="t1", job_id="j", steps_per_epoch=50)["epoch"] == 0.5
        assert build_verl_row(_REAL_STEP, 50, trial_id="t1", job_id="j", steps_per_epoch=50)["epoch"] == 1.0
        assert build_verl_row(_REAL_STEP, 75, trial_id="t1", job_id="j", steps_per_epoch=50)["epoch"] == 1.5

    def test_puts_every_other_numeric_key_in_extra(self):
        row = build_verl_row(_REAL_STEP, 37, trial_id="t1", job_id="job-1")

        assert row["extra"] == {
            "actor/kl_loss": 0.0,
            "actor/entropy": 0.283,
            "critic/score/mean": -0.951,
            "critic/score/max": -0.951,
            "critic/score/min": -0.951,
            "response_length/mean": 2.0,
            "perf/throughput": 812.5,
        }

    def test_has_the_same_keys_as_an_hf_row(self):
        row = build_verl_row(_REAL_STEP, 37, trial_id="t1", job_id="job-1")

        assert set(row) == {
            "job_id",
            "trial_id",
            "global_step",
            "epoch",
            "loss",
            "grad_norm",
            "learning_rate",
            "split",
            "extra",
        }

    def test_leaves_missing_columns_none(self):
        row = build_verl_row({"_step": 1}, 1, trial_id=None, job_id=None)

        assert row["epoch"] is None
        assert row["loss"] is None
        assert row["grad_norm"] is None
        assert row["learning_rate"] is None
        assert row["split"] == "train"
        assert row["extra"] == {}

    def test_replaces_non_finite_values_with_none(self):
        row = build_verl_row(
            {"actor/pg_loss": float("nan"), "actor/entropy": float("inf")}, 2, trial_id="t1", job_id="j"
        )

        assert row["loss"] is None
        assert row["extra"] == {"actor/entropy": None}

    def test_drops_strings_lists_and_bools_from_extra(self):
        row = build_verl_row(
            {"note": "x", "hist": [1.0, 2.0], "flag": True, "actor/entropy": 0.2}, 3, trial_id="t1", job_id="j"
        )

        assert row["extra"] == {"actor/entropy": 0.2}

    def test_marks_a_validation_only_step_as_eval(self):
        row = build_verl_row({"_step": 0, "val-core/gsm8k/reward/mean@1": 0.4}, 0, trial_id="t1", job_id="j")

        assert row["split"] == "eval"
        assert row["extra"] == {"val-core/gsm8k/reward/mean@1": 0.4}

    def test_is_train_when_validation_keys_ride_along_with_training_keys(self):
        row = build_verl_row({"actor/pg_loss": 0.1, "val-core/x": 0.4}, 5, trial_id="t1", job_id="j")

        assert row["split"] == "train"


class TestFormatStepSummary:
    def test_renders_the_fields_in_a_fixed_order(self):
        assert format_step_summary(_REAL_STEP, 37) == (
            "[AutoTune] verl step 37: reward=-0.951 (min -0.951, max -0.951) pg_loss=-0.036 "
            "kl=0.000 entropy=0.283 resp_len=2.0 grad_norm=0.41 lr=1.0e-06"
        )

    def test_omits_missing_and_non_finite_fields(self):
        assert format_step_summary({"actor/pg_loss": float("nan"), "actor/entropy": 0.5}, 2) == (
            "[AutoTune] verl step 2: entropy=0.500"
        )

    def test_falls_back_to_the_reward_kl_penalty(self):
        assert format_step_summary({"actor/reward_kl_penalty": 0.012}, 4) == "[AutoTune] verl step 4: kl=0.012"

    def test_is_just_the_step_when_no_known_field_is_present(self):
        assert format_step_summary({"_step": 9}, 9) == "[AutoTune] verl step 9:"


_STEP = {"_step": 5, "actor/pg_loss": -0.036, "critic/score/mean": -0.951}


class _RecordingReporter:
    def __init__(self):
        self.calls = []

    def report(self, data, step):
        self.calls.append((data, step))


class TestVerlStepReporter:
    def test_prints_one_marker_line_without_an_endpoint(self, monkeypatch, capsys):
        monkeypatch.delenv("AUTOTUNE_ENDPOINT_URL", raising=False)
        monkeypatch.setenv("AUTOTUNE_JOB_ID", "job-1")
        reporter = VerlStepReporter(trial_id="t1")

        reporter.report(_STEP, 5)

        lines = [ln for ln in capsys.readouterr().out.splitlines() if METRIC_MARKER in ln]
        assert len(lines) == 1
        payload = json.loads(lines[0].split(METRIC_MARKER, 1)[1])
        assert payload["job_id"] == "job-1"
        assert payload["trial_id"] == "t1"
        assert payload["global_step"] == 5
        assert payload["loss"] == -0.036

    def test_stamps_fractional_epoch_from_steps_per_epoch(self, monkeypatch, capsys):
        monkeypatch.delenv("AUTOTUNE_ENDPOINT_URL", raising=False)
        reporter = VerlStepReporter(trial_id="t1", steps_per_epoch=10)

        reporter.report(_STEP, 5)

        line = next(ln for ln in capsys.readouterr().out.splitlines() if METRIC_MARKER in ln)
        assert json.loads(line.split(METRIC_MARKER, 1)[1])["epoch"] == 0.5

    def test_posts_through_the_handler_when_an_endpoint_is_set(self, monkeypatch):
        monkeypatch.setenv("AUTOTUNE_ENDPOINT_URL", "http://x/fmtune/api")
        monkeypatch.setenv("AUTOTUNE_JOB_ID", "job-1")
        reporter = VerlStepReporter(trial_id="t1")
        close_metrics_handler(reporter._handler)  # release the real handler before swapping in a fake
        calls = []
        reporter._handler = types.SimpleNamespace(
            record_data=lambda data, record_type: calls.append((data, record_type))
        )

        reporter.report(_STEP, 5)

        assert len(calls) == 1
        data, record_type = calls[0]
        assert record_type is RecordType.RECORD_METRICS
        assert data["job_id"] == "job-1"
        assert data["trial_id"] == "t1"
        assert data["loss"] == -0.036

    def test_logs_the_step_summary_line(self, monkeypatch, caplog):
        monkeypatch.delenv("AUTOTUNE_ENDPOINT_URL", raising=False)
        caplog.set_level(logging.INFO, logger="autotune.callbacks.verl_metrics")

        VerlStepReporter(trial_id="t1").report(_STEP, 5)

        assert "[AutoTune] verl step 5: reward=-0.951 pg_loss=-0.036" in caplog.messages

    def test_swallows_a_failing_handler(self, monkeypatch):
        monkeypatch.delenv("AUTOTUNE_ENDPOINT_URL", raising=False)
        reporter = VerlStepReporter(trial_id="t1")

        def _boom(data, record_type):
            raise RuntimeError("bridge down")

        reporter._handler = types.SimpleNamespace(record_data=_boom)

        reporter.report(_STEP, 5)  # must not raise into verl's training loop

    def test_close_is_idempotent_and_survives_a_failing_close(self, monkeypatch):
        monkeypatch.delenv("AUTOTUNE_ENDPOINT_URL", raising=False)
        reporter = VerlStepReporter(trial_id="t1")

        def _boom():
            raise RuntimeError("endpoint gone")

        reporter._handler = types.SimpleNamespace(close=_boom)

        reporter.close()
        reporter.close()

        assert reporter._handler is None


class TestActiveReporter:
    def test_log_forwards_each_converted_step_to_the_active_reporter(self):
        reporter = _RecordingReporter()
        set_active_reporter(reporter)
        try:
            _InMemoryMetricsLogger().log({"actor/pg_loss": _FakeTensor(0.5)}, step=3)
        finally:
            set_active_reporter(None)

        assert reporter.calls == [({"_step": 3, "actor/pg_loss": 0.5}, 3)]

    def test_log_records_to_memory_only_when_no_reporter_is_active(self, capsys):
        set_active_reporter(None)
        mem = _InMemoryMetricsLogger()

        mem.log({"actor/pg_loss": 0.5}, step=1)

        assert _InMemoryMetricsLogger._all_steps == [{"_step": 1, "actor/pg_loss": 0.5}]
        assert METRIC_MARKER not in capsys.readouterr().out
