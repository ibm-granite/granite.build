# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The artifact-listing domain exceptions carry the right HTTP shapes."""

from __future__ import annotations

from http import HTTPStatus

from autotunex.core.exceptions import (
    ArtifactSourceUnavailableError,
    JobArtifactsNotFoundError,
    JobArtifactsNotReadyError,
)


def test_not_ready_is_a_409_naming_the_status() -> None:
    error = JobArtifactsNotReadyError("job-1", "running")

    assert error.status_code == HTTPStatus.CONFLICT
    assert "running" in error.detail


def test_not_found_is_a_404_with_a_fixed_message() -> None:
    error = JobArtifactsNotFoundError()

    assert error.status_code == HTTPStatus.NOT_FOUND
    assert error.detail


def test_source_unavailable_is_a_502() -> None:
    error = ArtifactSourceUnavailableError()

    assert error.status_code == HTTPStatus.BAD_GATEWAY
