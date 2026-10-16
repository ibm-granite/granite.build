# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""HTTP request-logging middleware for the api-bridge.

Logs one line per request once it completes — method, path, status code,
duration, and the caller's email — and logs unhandled exceptions with the same
context before letting them propagate (so FastAPI still returns its 500).

Bodies are intentionally NOT logged: requests carry configs, datasets, and
tokens that we don't want in operational logs.
"""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api-bridge.access")


def _caller_email(request: Request) -> str:
    """Best-effort caller identity, mirroring server._email_from_request."""
    return request.headers.get("X-User-Email") or request.cookies.get("email") or "-"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = uuid.uuid4().hex[:8]
        request.state.request_id = request_id
        start = time.perf_counter()

        # Shared context for whichever log line we emit below.
        context = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "caller": _caller_email(request),
        }

        try:
            response: Response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            logger.exception(
                "%s %s -> unhandled exception after %sms",
                context["method"],
                context["path"],
                duration_ms,
                extra={**context, "status_code": 500, "duration_ms": duration_ms},
            )
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        extra = {
            **context,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        }
        # 5xx is our problem, 4xx is the caller's; everything else is routine.
        if response.status_code >= 500:
            log = logger.error
        elif response.status_code >= 400:
            log = logger.warning
        else:
            log = logger.info
        log(
            "%s %s -> %s (%sms) caller=%s",
            context["method"],
            context["path"],
            response.status_code,
            duration_ms,
            context["caller"],
            extra=extra,
        )

        response.headers["X-Request-ID"] = request_id
        return response
