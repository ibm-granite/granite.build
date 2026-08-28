"""Tests for autotune.callbacks.tuner_callback.CustomLoggerCallback."""

from unittest.mock import MagicMock

from autotune.callbacks.logging_service import RecordType
from autotune.callbacks.tuner_callback import CustomLoggerCallback


class _DummySearchAlg:
    """A stand-in for a Ray Tune search algorithm object."""


class TestSanitizedResult:
    def test_replaces_search_alg_with_class_name(self):
        cb = CustomLoggerCallback()
        data = {
            "config": {
                "tune_config": {"search_alg": _DummySearchAlg()},
                "lr": 0.001,
            }
        }
        out = cb.sanitized_result(data)
        assert out["tune_config"]["search_alg"] == "_DummySearchAlg"
        assert out["lr"] == 0.001

    def test_no_search_alg_returns_config_unchanged(self):
        # sanitized_result always returns the (copied) config. When tune_config
        # has no non-serializable entries, it is returned as-is (a string
        # scheduler like "fifo" is JSON-safe and left untouched).
        cb = CustomLoggerCallback()
        data = {"config": {"tune_config": {"scheduler": "fifo"}}}
        assert cb.sanitized_result(data) == {"tune_config": {"scheduler": "fifo"}}

    def test_non_dict_config_returns_empty_dict(self):
        # A non-dict config cannot be sanitized field-by-field, so an empty
        # dict is returned rather than None.
        cb = CustomLoggerCallback()
        data = {"config": "not a dict"}
        assert cb.sanitized_result(data) == {}

    def test_returns_sanitized_dict(self):
        cb = CustomLoggerCallback()
        data = {"config": {"tune_config": {"search_alg": _DummySearchAlg()}, "lr": 0.1}}
        out = cb.sanitized_result(data)
        # Output replaces search_alg with class name string
        assert isinstance(out["tune_config"]["search_alg"], str)
        assert out["lr"] == 0.1


class TestTrialCallbacks:
    def _build(self):
        handler = MagicMock()
        handler.get_job_id.return_value = "job-1"
        cb = CustomLoggerCallback(job_id="job-1", handler=handler)
        return cb, handler

    def _make_trial(self):
        trial = MagicMock()
        trial.trial_id = "trial-0001"
        trial.config = {"lr": 0.001}
        trial.trainable_name = "trainer"
        trial.status = "RUNNING"
        return trial

    def test_on_trial_start_records_trial(self):
        cb, handler = self._build()
        trial = self._make_trial()
        cb.on_trial_start(iteration=0, trials=[trial], trial=trial)
        # set_trial_id called with the trial id
        handler.set_trial_id.assert_any_call("trial-0001")
        # record_data called with RECORD_TRIAL type
        record_calls = [c for c in handler.record_data.call_args_list]
        assert any(c.kwargs.get("record_type") == RecordType.RECORD_TRIAL for c in record_calls)
        handler.flush.assert_called()

    def test_on_trial_complete_flushes(self):
        cb, handler = self._build()
        trial = self._make_trial()
        cb.on_trial_complete(iteration=1, trials=[trial], trial=trial)
        handler.flush.assert_called()

    def test_on_trial_error_records_status(self):
        cb, handler = self._build()
        trial = self._make_trial()
        cb.on_trial_error(iteration=1, trials=[trial], trial=trial)
        # Records UPDATE_STATUS with ERROR
        calls = handler.record_data.call_args_list
        update_calls = [c for c in calls if c.kwargs.get("record_type") == RecordType.UPDATE_STATUS]
        assert len(update_calls) >= 1
        payload = update_calls[0].args[0]
        assert payload["status"] == "ERROR"

    def test_on_trial_result_records_result(self):
        cb, handler = self._build()
        trial = self._make_trial()
        result = {"loss": 0.5, "config": {"tune_config": {"search_alg": _DummySearchAlg()}}}
        cb.on_trial_result(iteration=2, trials=[trial], trial=trial, result=result)
        # Records both RECORD_RESULT and UPDATE_STATUS
        kinds = [c.kwargs.get("record_type") for c in handler.record_data.call_args_list]
        assert RecordType.RECORD_RESULT in kinds
        assert RecordType.UPDATE_STATUS in kinds
