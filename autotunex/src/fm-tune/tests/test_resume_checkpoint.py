"""Tests for the --resume_from_checkpoint feature.

Covers three GPU-free units:
  1. ``_resolve_resume_checkpoint`` — the pure filesystem logic that the
     multi-GPU drivers use to pick the checkpoint to resume from.
  2. ``AutotuneOptimizer`` plumbing — the flag is stored.
  3. ``has_resumable_final_checkpoint`` / ``save_final_config`` /
     ``load_final_config`` — the head-node "can we resume?" check and the
     final-config round-trip that lets a resume skip HPO.
"""

from unittest.mock import MagicMock

import pytest

from autotune.constants import AutotunePrecision
from autotune.utils import (
    has_resumable_final_checkpoint,
    load_final_config,
    save_final_config,
)


def _make_checkpoint(tmp_path, step: int):
    """Create a ``checkpoint-<step>`` dir under tmp_path and return its path."""
    d = tmp_path / f"checkpoint-{step}"
    d.mkdir()
    return str(d)


class TestResolveResumeCheckpoint:
    """Both FSDP drivers expose an identical _resolve_resume_checkpoint."""

    @pytest.fixture(
        params=[
            "autotune.trainers.driver_multi_hf_fsdp",
            "autotune.trainers.driver_multi_trl_fsdp",
            "autotune.trainers.driver_multi_hf_ds",
            "autotune.trainers.driver_multi_trl_ds",
        ]
    )
    def resolver(self, request):
        import importlib

        mod = importlib.import_module(request.param)
        return mod._resolve_resume_checkpoint

    def test_flag_off_returns_false(self, resolver, tmp_path):
        _make_checkpoint(tmp_path, 10)
        assert resolver(str(tmp_path), False) is False

    def test_missing_dir_returns_false(self, resolver, tmp_path):
        missing = tmp_path / "does-not-exist"
        assert resolver(str(missing), True) is False

    def test_empty_dir_returns_false(self, resolver, tmp_path):
        # Flag on but no checkpoint-* dirs -> warn-and-train-fresh (False).
        assert resolver(str(tmp_path), True) is False

    def test_picks_latest_checkpoint_by_step(self, resolver, tmp_path):
        _make_checkpoint(tmp_path, 10)
        _make_checkpoint(tmp_path, 200)  # highest step = most recent
        _make_checkpoint(tmp_path, 50)
        assert resolver(str(tmp_path), True) == str(tmp_path / "checkpoint-200")

    def test_ignores_non_numeric_and_files(self, resolver, tmp_path):
        _make_checkpoint(tmp_path, 5)
        (tmp_path / "checkpoint-best").mkdir()  # non-numeric suffix, ignored
        (tmp_path / "checkpoint-7").write_text("not a dir")  # a file, ignored
        assert resolver(str(tmp_path), True) == str(tmp_path / "checkpoint-5")


def _make_optimizer(*, resume_from_checkpoint: bool):
    """Construct an AutotuneOptimizer with a stubbed pipeline (no GPU/IO)."""
    from autotune.optimizer import AutotuneOptimizer

    pipeline = MagicMock()
    pipeline.get_model_name_or_path.return_value = "stub/model"
    pipeline.get_tuning_algo.return_value = "lora"
    pipeline.get_rl_algo.return_value = "none"
    pipeline.get_precision.return_value = AutotunePrecision.BF16

    return AutotuneOptimizer(
        pipeline=pipeline,
        config=MagicMock(),
        train_file="train.jsonl",
        validation_file="val.jsonl",
        output_dir="/tmp/out",
        output_model_name="m",
        resume_from_checkpoint=resume_from_checkpoint,
    )


class TestOptimizerPlumbing:
    def test_defaults_to_false(self):
        opt = _make_optimizer(resume_from_checkpoint=False)
        assert opt.resume_from_checkpoint is False

    def test_stores_flag(self):
        opt = _make_optimizer(resume_from_checkpoint=True)
        assert opt.resume_from_checkpoint is True


class TestResumableHelper:
    """has_resumable_final_checkpoint + save/load round-trip (GPU-free)."""

    def _final_ckpt_dir(self, tmp_path):
        # save_final_config may have already created final_checkpoints/, so
        # this must be idempotent (exist_ok=True) rather than assuming a fresh dir.
        d = tmp_path / "final_checkpoints"
        d.mkdir(exist_ok=True)
        return d

    def test_no_dir_returns_false(self, tmp_path):
        assert has_resumable_final_checkpoint(str(tmp_path)) is False

    def test_config_only_returns_false(self, tmp_path):
        save_final_config(str(tmp_path), {"training_config": {"a": 1}})
        # config present but no checkpoint dir yet
        assert has_resumable_final_checkpoint(str(tmp_path)) is False

    def test_checkpoint_only_returns_false(self, tmp_path):
        (self._final_ckpt_dir(tmp_path) / "checkpoint-10").mkdir()
        assert has_resumable_final_checkpoint(str(tmp_path)) is False

    def test_both_present_returns_true(self, tmp_path):
        save_final_config(str(tmp_path), {"training_config": {"a": 1}})
        (self._final_ckpt_dir(tmp_path) / "checkpoint-10").mkdir()
        assert has_resumable_final_checkpoint(str(tmp_path)) is True

    def test_non_numeric_checkpoint_ignored(self, tmp_path):
        save_final_config(str(tmp_path), {"training_config": {"a": 1}})
        (self._final_ckpt_dir(tmp_path) / "checkpoint-best").mkdir()
        assert has_resumable_final_checkpoint(str(tmp_path)) is False

    def test_save_load_round_trip(self, tmp_path):
        cfg = {"training_config": {"lr": 0.001}, "tuner_flags": {"r": True}, "r": 16}
        save_final_config(str(tmp_path), cfg)
        assert load_final_config(str(tmp_path)) == cfg

    def test_save_handles_non_jsonable_under_tune_config(self, tmp_path):
        # Mirrors the HPO best config: tune_config carries live Ray
        # search-alg/scheduler objects (non-JSON-serializable). save_final_config
        # must sanitize them (stringify) while leaving the values resume actually
        # consumes untouched, and must produce valid JSON.
        class _Dummy:
            pass

        cfg = {
            "r": 16,
            "lora_alpha": 32,
            "training_config": {"lr": 0.001, "num_gpus_per_trial": 1},
            "training_rl_config": {},
            "tuner_flags": {"r": True},
            "tune_config": {
                "search_alg": _Dummy(),  # non-jsonable (mimics a Ray searcher object)
                "scheduler": (lambda: None),  # non-jsonable
                "max_concurrent_trials": 2,  # jsonable scalar resume reads
                "metric": "eval_loss",  # jsonable scalar resume reads
            },
        }
        path = save_final_config(str(tmp_path), cfg)
        assert path  # truthy path returned (save succeeded)

        loaded = load_final_config(str(tmp_path))  # valid JSON — no exception
        # Values resume consumes survive unchanged.
        assert loaded["r"] == 16
        assert loaded["lora_alpha"] == 32
        assert loaded["training_config"] == {"lr": 0.001, "num_gpus_per_trial": 1}
        assert loaded["tuner_flags"] == {"r": True}
        assert loaded["tune_config"]["max_concurrent_trials"] == 2
        assert loaded["tune_config"]["metric"] == "eval_loss"
        # Structure preserved (keys not dropped); offending values stringified.
        assert isinstance(loaded["tune_config"]["search_alg"], str)
        assert isinstance(loaded["tune_config"]["scheduler"], str)

    def test_save_never_raises_on_failure(self, tmp_path, monkeypatch):
        # A serialization/IO failure must be swallowed (returns None), never
        # abort the training run.
        import autotune.utils as utils

        def _boom(*a, **k):
            raise RuntimeError("disk full")

        monkeypatch.setattr(utils.json, "dump", _boom)
        assert save_final_config(str(tmp_path), {"r": 16}) is None
