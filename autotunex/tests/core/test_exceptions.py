"""Status codes and titles for the dataset exceptions.

These are pure value assertions — the exceptions carry the HTTP status that
``api/errors.handle_domain_error`` reads, so pinning them here catches a wrong
base class (a 413 that silently became a 500) without an end-to-end request.
"""

from __future__ import annotations

from http import HTTPStatus

from autotunex.core.exceptions import (
    AdminRequiredError,
    AutotuneCoreUnavailableError,
    BuildCancelUpstreamError,
    BuildReconcileUnavailableError,
    BuildReconcileUpstreamError,
    CannotChangeOwnRoleError,
    DatasetInUseError,
    DatasetNameConflictError,
    DatasetNotFoundError,
    DatasetNotReadyError,
    DatasetProcessingTimeoutError,
    DatasetPushFailedError,
    DatasetPushTimeoutError,
    DatasetTooLargeError,
    EmptyDatasetError,
    EmptySplitError,
    GbLogsUnavailableError,
    GbLogsUpstreamError,
    InsufficientStorageError,
    InvalidDatasetFormatError,
    InvalidSampleError,
    JobCancellationInProgressError,
    JobNotCancellableError,
    JobNotReconcilableError,
    LastAdminError,
    LlmNotConfiguredError,
    LlmUnavailableError,
    UnknownTrainingFormatError,
    UnsupportedDatasetFormatError,
    UploadProcessingError,
    UserNotFoundError,
)


def test_not_found_is_404() -> None:
    assert DatasetNotFoundError("x").status_code == HTTPStatus.NOT_FOUND


def test_name_conflict_and_in_use_and_not_ready_are_409() -> None:
    assert DatasetNameConflictError("x").status_code == HTTPStatus.CONFLICT
    assert DatasetInUseError("x").status_code == HTTPStatus.CONFLICT
    assert DatasetNotReadyError("x").status_code == HTTPStatus.CONFLICT


def test_invalid_and_empty_and_empty_split_are_422() -> None:
    assert InvalidDatasetFormatError("x").status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert EmptyDatasetError("x").status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert EmptySplitError("x").status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_too_large_is_413() -> None:
    assert DatasetTooLargeError("x").status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE


def test_unsupported_format_is_415() -> None:
    assert UnsupportedDatasetFormatError("x").status_code == HTTPStatus.UNSUPPORTED_MEDIA_TYPE


def test_llm_not_configured_is_503() -> None:
    assert LlmNotConfiguredError().status_code == HTTPStatus.SERVICE_UNAVAILABLE


def test_autotune_core_unavailable_is_503() -> None:
    assert AutotuneCoreUnavailableError().status_code == HTTPStatus.SERVICE_UNAVAILABLE


def test_llm_unavailable_is_502() -> None:
    assert LlmUnavailableError().status_code == HTTPStatus.BAD_GATEWAY


def test_llm_unavailable_message_is_generic_and_safe() -> None:
    assert "raw" not in LlmUnavailableError().detail.lower()
    assert LlmUnavailableError().detail  # non-empty, no upstream text


def test_invalid_sample_and_unknown_format_are_422() -> None:
    assert InvalidSampleError("x").status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert UnknownTrainingFormatError("x").status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_job_not_cancellable_and_cancellation_in_progress_are_409() -> None:
    assert JobNotCancellableError("x", "completed").status_code == HTTPStatus.CONFLICT
    assert JobCancellationInProgressError("x").status_code == HTTPStatus.CONFLICT


def test_build_cancel_upstream_is_502() -> None:
    assert BuildCancelUpstreamError("boom").status_code == HTTPStatus.BAD_GATEWAY


def test_gb_logs_unavailable_is_a_503() -> None:
    assert GbLogsUnavailableError().status_code == HTTPStatus.SERVICE_UNAVAILABLE


def test_gb_logs_upstream_is_a_502() -> None:
    assert GbLogsUpstreamError().status_code == HTTPStatus.BAD_GATEWAY


def test_admin_required_error_is_a_403() -> None:
    error = AdminRequiredError()

    assert error.status_code == HTTPStatus.FORBIDDEN
    assert error.title == "Forbidden"


def test_user_not_found_error_is_a_404_and_names_the_id() -> None:
    error = UserNotFoundError("abc-123")

    assert error.status_code == HTTPStatus.NOT_FOUND
    assert "abc-123" in error.detail


def test_cannot_change_own_role_error_is_a_409() -> None:
    error = CannotChangeOwnRoleError()

    assert error.status_code == HTTPStatus.CONFLICT


def test_last_admin_error_is_a_409() -> None:
    error = LastAdminError()

    assert error.status_code == HTTPStatus.CONFLICT


def test_job_not_reconcilable_is_a_409() -> None:
    exc = JobNotReconcilableError("abc")

    assert exc.status_code == HTTPStatus.CONFLICT
    assert "abc" in exc.detail


def test_build_reconcile_unavailable_is_a_503() -> None:
    assert BuildReconcileUnavailableError().status_code == HTTPStatus.SERVICE_UNAVAILABLE


def test_build_reconcile_upstream_is_a_502() -> None:
    exc = BuildReconcileUpstreamError("gbserver down")

    assert exc.status_code == HTTPStatus.BAD_GATEWAY
    assert "gbserver down" in exc.detail


def test_upload_processing_errors_carry_safe_authored_details() -> None:
    assert "3600" in DatasetProcessingTimeoutError(3600).detail
    assert "1800" in DatasetPushTimeoutError(1800).detail
    assert DatasetPushFailedError().detail
    assert "disk" in InsufficientStorageError().detail.lower()


def test_upload_processing_errors_share_a_common_base() -> None:
    for exc in (
        DatasetProcessingTimeoutError(1),
        DatasetPushTimeoutError(1),
        DatasetPushFailedError(),
        InsufficientStorageError(),
    ):
        assert isinstance(exc, UploadProcessingError)
