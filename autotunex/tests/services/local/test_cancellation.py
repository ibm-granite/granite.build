# tests/services/local/test_cancellation.py
from __future__ import annotations

from uuid import uuid4

from autotunex.services.local import cancellation


def test_request_cancel_on_unregistered_job_returns_false() -> None:
    assert cancellation.request_cancel(uuid4()) is False


def test_registered_job_is_active_and_can_be_cancelled() -> None:
    job_id = uuid4()
    cancellation.register(job_id)
    try:
        assert cancellation.is_active(job_id) is True
        assert cancellation.is_cancelled(job_id) is False
        assert cancellation.request_cancel(job_id) is True
        assert cancellation.is_cancelled(job_id) is True
        assert cancellation.cancel_event(job_id) is not None
    finally:
        cancellation.unregister(job_id)


def test_unregister_removes_the_job() -> None:
    job_id = uuid4()
    cancellation.register(job_id)
    cancellation.unregister(job_id)
    assert cancellation.is_active(job_id) is False
    assert cancellation.cancel_event(job_id) is None
