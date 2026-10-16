# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Exception handlers.

The one place where exceptions become HTTP responses. Every error the API emits
is an RFC 9457 problem detail, served as ``application/problem+json`` — including
request validation failures, so clients only ever parse one error shape.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from autotunex.core.exceptions import AutoTuneXError
from autotunex.core.logging import get_logger
from autotunex.models.common import ProblemDetail

logger = get_logger(__name__)
PROBLEM_JSON = "application/problem+json"


def _problem(
    status: int,
    title: str,
    detail: str,
    errors: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ProblemDetail(title=title, status=status, detail=detail, errors=errors)
    return JSONResponse(
        status_code=status,
        content=body.model_dump(exclude_none=True),
        media_type=PROBLEM_JSON,
        headers=headers,
    )


async def handle_domain_error(request: Request, exc: Exception) -> JSONResponse:
    """Map an :class:`AutoTuneXError` to its declared status code."""
    if not isinstance(exc, AutoTuneXError):  # pragma: no cover - defensive
        return await handle_unexpected_error(request, exc)
    return _problem(exc.status_code, exc.title, exc.detail, headers=exc.headers)


async def handle_request_validation_error(request: Request, exc: Exception) -> JSONResponse:
    """Convert FastAPI's validation error into a problem detail."""
    if not isinstance(exc, RequestValidationError):  # pragma: no cover - defensive
        return await handle_unexpected_error(request, exc)
    return _problem(
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "Unprocessable Entity",
        "Request validation failed.",
        errors=jsonable_encoder(exc.errors()),
    )


async def handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
    """Map any unhandled exception to a 500 without leaking internals."""
    logger.exception("Unhandled error", exc_info=exc)
    return _problem(
        HTTPStatus.INTERNAL_SERVER_ERROR,
        "Internal Server Error",
        "An unexpected error occurred.",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the handlers above to ``app``."""
    app.add_exception_handler(AutoTuneXError, handle_domain_error)
    app.add_exception_handler(RequestValidationError, handle_request_validation_error)
    app.add_exception_handler(Exception, handle_unexpected_error)
