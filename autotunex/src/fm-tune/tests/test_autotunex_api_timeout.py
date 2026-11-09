"""AutoTuneXAPI requests must carry a timeout, and a stalled bridge must be retried
and then raised like any other connection failure (main.py degrades it to a warning).

Without a timeout, ``requests`` waits forever on a bridge that accepts the TCP
connection but never answers, hanging fm-tune's startup at ``bootstrap()``.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from autotune.callbacks import autotunex_api
from autotune.callbacks.autotunex_api import AutoTuneXAPI


def _ok_response(json_body=None):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = json_body or {}
    resp.text = ""
    return resp


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(autotunex_api.time, "sleep", lambda _s: None)


def test_request_passes_default_timeout():
    api = AutoTuneXAPI(base_url="http://bridge", email="u@example.com")
    with patch.object(api.session, "request", return_value=_ok_response()) as req:
        api.bootstrap({"job_id": "J"})
    _, kwargs = req.call_args
    assert kwargs.get("timeout") == autotunex_api._DEFAULT_TIMEOUT
    assert kwargs["timeout"] > 0


def test_explicit_timeout_is_not_overridden():
    api = AutoTuneXAPI(base_url="http://bridge", email="u@example.com")
    with patch.object(api.session, "request", return_value=_ok_response()) as req:
        api._get("/api/job/J", timeout=5)
    _, kwargs = req.call_args
    assert kwargs["timeout"] == 5


@pytest.mark.parametrize("exc_cls", [requests.ReadTimeout, requests.ConnectTimeout, requests.ConnectionError])
def test_timeout_is_retried_then_raised(exc_cls):
    # ReadTimeout is NOT a ConnectionError subclass, so it used to escape the
    # retry loop on the first attempt.
    api = AutoTuneXAPI(base_url="http://bridge", email="u@example.com")
    with patch.object(api.session, "request", side_effect=exc_cls("stalled")) as req:
        with pytest.raises(exc_cls):
            api.bootstrap({"job_id": "J"})
    assert req.call_count == autotunex_api._DEFAULT_RETRIES


def test_timeout_then_success_recovers():
    api = AutoTuneXAPI(base_url="http://bridge", email="u@example.com")
    expected = {"config_id": "C1", "dataset_id": "D1", "job_id": "J1"}
    with patch.object(api.session, "request", side_effect=[requests.ReadTimeout("stalled"), _ok_response(expected)]):
        assert api.bootstrap({"job_id": "J1"}) == expected
