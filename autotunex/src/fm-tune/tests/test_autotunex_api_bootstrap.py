"""Tests for AutoTuneXAPI.bootstrap()."""

from unittest.mock import MagicMock, patch

import pytest

from autotune.callbacks.autotunex_api import AutoTuneXAPI, AutoTuneXAPIError


def _make_response(status_code=200, json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {}
    resp.text = ""
    return resp


def test_bootstrap_posts_and_returns_ids():
    api = AutoTuneXAPI(base_url="http://bridge", email="u@example.com")
    payload = {
        "job_id": "J1",
        "build_id": "J1",
        "config": {"name": "c", "tuner_type": "lora", "rl_tuner_type": None, "config_data": {}},
        "dataset": {"name": "d", "artifact_uri": "lh://x"},
        "job": {"model": "m", "experiment_name": "e", "tuning_type": "lora", "seed": 42},
    }
    expected = {"config_id": "C1", "dataset_id": "D1", "job_id": "J1", "created": False}

    with patch.object(api.session, "request", return_value=_make_response(200, expected)) as req:
        result = api.bootstrap(payload)

    assert result == expected
    args, kwargs = req.call_args
    assert args[0] == "POST"
    assert args[1].endswith("/fmtune/api/job/bootstrap")
    assert kwargs["json"] == payload


def test_bootstrap_raises_on_4xx():
    api = AutoTuneXAPI(base_url="http://bridge", email="u@example.com")
    with patch.object(api.session, "request", return_value=_make_response(400, {})):
        with pytest.raises(AutoTuneXAPIError) as excinfo:
            api.bootstrap({"job_id": "J"})

    # The error must come from the 4xx path (_raise_for_4xx), which embeds the
    # "bootstrap" context string and the 400 status. An AttributeError from a
    # missing method would NOT be an AutoTuneXAPIError, so the vacuous pass is gone.
    message = str(excinfo.value)
    assert "bootstrap" in message
    assert "400" in message
