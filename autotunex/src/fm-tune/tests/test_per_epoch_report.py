# Copyright Granite.Build Authors
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

"""Tests for PerEpochTuneReportCallback.

The callback bridges single-GPU drivers' per-epoch eval signal into
Ray Tune's tune.report() so ASHA can act on it. Tests mock tune.report
so they don't need a real Ray runtime.
"""

import math
from unittest.mock import patch

from autotune.callbacks.per_epoch_report import PerEpochTuneReportCallback


class FakeState:
    """Minimal stand-in for transformers.TrainerState."""

    def __init__(self, epoch, global_step):
        self.epoch = epoch
        self.global_step = global_step


def test_emits_loss_and_eval_loss():
    cb = PerEpochTuneReportCallback()
    with patch("autotune.callbacks.per_epoch_report.tune.report") as mock_report:
        cb.on_evaluate(
            args=None,
            state=FakeState(epoch=2.0, global_step=200),
            control=None,
            metrics={"eval_loss": 0.5, "loss": 0.4},
        )
    mock_report.assert_called_once()
    payload = mock_report.call_args[0][0]
    assert payload["loss"] == 0.5
    assert payload["eval_loss"] == 0.5
    assert payload["train_loss"] == 0.4
    assert payload["epoch"] == 2.0
    assert payload["global_step"] == 200
    assert payload["done"] is False


def test_no_emit_when_metrics_none():
    cb = PerEpochTuneReportCallback()
    with patch("autotune.callbacks.per_epoch_report.tune.report") as mock_report:
        cb.on_evaluate(args=None, state=FakeState(0.0, 0), control=None, metrics=None)
    mock_report.assert_not_called()


def test_handles_missing_train_loss():
    cb = PerEpochTuneReportCallback()
    with patch("autotune.callbacks.per_epoch_report.tune.report") as mock_report:
        cb.on_evaluate(
            args=None,
            state=FakeState(1.0, 100),
            control=None,
            metrics={"eval_loss": 0.7},  # no "loss" or "train_loss" key
        )
    payload = mock_report.call_args[0][0]
    assert payload["eval_loss"] == 0.7
    assert math.isnan(payload["train_loss"])


def test_handles_explicit_train_loss_key():
    """Some HF metrics dicts use 'train_loss' instead of 'loss'."""
    cb = PerEpochTuneReportCallback()
    with patch("autotune.callbacks.per_epoch_report.tune.report") as mock_report:
        cb.on_evaluate(
            args=None,
            state=FakeState(1.0, 100),
            control=None,
            metrics={"eval_loss": 0.7, "train_loss": 0.6},
        )
    payload = mock_report.call_args[0][0]
    assert payload["train_loss"] == 0.6


def test_done_flag_is_false():
    """Critical: per-epoch reports must NOT carry done=True, otherwise BLDS's
    on_trial_complete would treat them as final results at non-top rungs."""
    cb = PerEpochTuneReportCallback()
    with patch("autotune.callbacks.per_epoch_report.tune.report") as mock_report:
        cb.on_evaluate(
            args=None,
            state=FakeState(1.0, 50),
            control=None,
            metrics={"eval_loss": 0.3},
        )
    payload = mock_report.call_args[0][0]
    assert payload["done"] is False


def test_handles_none_epoch():
    """state.epoch can be None during the very first evaluation; coerce to 0.0."""
    cb = PerEpochTuneReportCallback()
    with patch("autotune.callbacks.per_epoch_report.tune.report") as mock_report:
        cb.on_evaluate(
            args=None,
            state=FakeState(epoch=None, global_step=10),
            control=None,
            metrics={"eval_loss": 0.9},
        )
    payload = mock_report.call_args[0][0]
    assert payload["epoch"] == 0.0


def test_is_a_trainer_callback():
    """Make sure the class is a real HF TrainerCallback so it integrates
    with Trainer.add_callback(...) without surprises."""
    from transformers import TrainerCallback

    assert isinstance(PerEpochTuneReportCallback(), TrainerCallback)
