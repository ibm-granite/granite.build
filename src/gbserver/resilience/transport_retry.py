#!/usr/bin/env python3

# Copyright LLM.build Authors
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

"""
Low-level transport retries for aiohttp and kubernetes_asyncio.

Build runs in our clusters must survive transient connection and DNS blips
without scattering retry logic across the app. Historically this was achieved
by patching the upstream ``aiohttp`` and ``kubernetes_asyncio`` sources at
container-build time (``connector.py.patch`` / ``api_client.py.patch``). Those
patches were fragile (line-offset based, broke on version bumps), applied only
in the image, and used a blocking ``time.sleep`` inside coroutines (stalling the
event loop).

This module replaces them with runtime monkeypatches that wrap the exact same
upstream seams using ``tenacity`` (already the codebase's retry mechanism). The
wrappers retry with async exponential backoff, so they no longer block the event
loop, and they take effect in every process (local, tests, container) once
:func:`install_transport_retries` is called.

The two wrapped seams are:

* ``aiohttp.connector.TCPConnector._resolve_host`` -- DNS resolution. Every
  direct connection funnels through this coroutine, so retrying here covers the
  same path the old ``_create_direct_connection`` patch did, at the true source.
* ``kubernetes_asyncio.client.api_client.ApiClient.request`` -- the public
  request method that ``__call_api`` awaits. Wrapping it is equivalent to the
  old ``__call_api`` patch without touching name-mangled internals.

:func:`install_transport_retries` is idempotent and is invoked once from the
``gbserver()`` CLI root callback (see ``cli.py``), which runs before every
subcommand.
"""

import asyncio
import functools
from typing import Callable

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from gbserver.types.constants import (
    TRANSPORT_RETRY_BASE_DELAY,
    TRANSPORT_RETRY_MAX_ATTEMPTS,
    TRANSPORT_RETRY_MAX_DELAY,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# Marker attribute stamped on wrapped methods so re-installation is a no-op.
_WRAPPED_MARKER = "_gbserver_transport_retry_wrapped"

# Module-level guard: install_transport_retries() may be called more than once
# (the CLI root callback runs per-subcommand, tests may call it repeatedly).
_INSTALLED = False


def _make_before_sleep(label: str) -> Callable[["RetryCallState"], None]:
    """Build a tenacity before_sleep hook that names the seam being retried.

    The default ``before_sleep_log`` logs the wrapped callable's name, which is
    ``<unknown>`` for the ``AsyncRetrying`` iterator form used here; naming the
    seam explicitly keeps the retry logs greppable.
    """

    def _log(retry_state: "RetryCallState") -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        sleep = getattr(retry_state.next_action, "sleep", 0.0)
        logger.warning(
            "Retrying %s in %.1fs (attempt %d/%d) after %r",
            label,
            sleep,
            retry_state.attempt_number,
            TRANSPORT_RETRY_MAX_ATTEMPTS,
            exc,
        )

    return _log


def _make_retrying(
    predicate: Callable[[BaseException], bool], label: str
) -> AsyncRetrying:
    """Build an AsyncRetrying with the shared transport retry policy.

    Mirrors the tenacity structure used elsewhere (see ``utils/git_retry.py``):
    capped exponential backoff with jitter, retrying only when ``predicate``
    returns True, and re-raising the original exception once attempts are
    exhausted. ``label`` names the seam in the retry logs.
    """
    return AsyncRetrying(
        stop=stop_after_attempt(TRANSPORT_RETRY_MAX_ATTEMPTS),
        wait=wait_random_exponential(
            multiplier=TRANSPORT_RETRY_BASE_DELAY,
            max=TRANSPORT_RETRY_MAX_DELAY,
        ),
        retry=retry_if_exception(predicate),
        before_sleep=_make_before_sleep(label),
        reraise=True,
    )


def _is_retryable_dns_error(exc: BaseException) -> bool:
    """Retry transient DNS/resolution failures, but not the cancellation timeout.

    Mirrors the original connector patch, whose guard was
    ``exc.errno is None and isinstance(exc, asyncio.TimeoutError)`` -- i.e. it
    re-raised only the errno-less ``asyncio.TimeoutError`` (the lookup
    cancellation surfaced by ``async_timeout``) and treated every other
    ``OSError`` -- including a ``TimeoutError`` that carries an ``errno`` -- as
    transient. Preserving the ``errno is None`` condition keeps that fidelity.
    """
    if isinstance(exc, asyncio.TimeoutError) and getattr(exc, "errno", None) is None:
        return False
    return isinstance(exc, OSError)


def _install_aiohttp_dns_retry() -> None:
    """Wrap ``TCPConnector._resolve_host`` with the transport retry policy."""
    # Imported lazily so module import stays cheap and the wrap happens only
    # when the installer runs.
    # pylint: disable-next=import-outside-toplevel
    from aiohttp.connector import TCPConnector

    # Wrapping a protected upstream method is the whole point of this module.
    original = TCPConnector._resolve_host  # pylint: disable=protected-access
    if getattr(original, _WRAPPED_MARKER, False):
        return

    @functools.wraps(original)
    async def _resolve_host_with_retry(self, host, port, traces=None):
        async for attempt in _make_retrying(
            _is_retryable_dns_error, "aiohttp DNS resolution"
        ):
            with attempt:
                return await original(self, host, port, traces=traces)
        # Unreachable: tenacity either returns a value or re-raises.
        raise RuntimeError("transport DNS retry exhausted without result")

    setattr(_resolve_host_with_retry, _WRAPPED_MARKER, True)
    TCPConnector._resolve_host = _resolve_host_with_retry  # type: ignore[method-assign]  # pylint: disable=protected-access
    logger.info(
        "Installed transport retry around aiohttp TCPConnector._resolve_host "
        "(max_attempts=%d)",
        TRANSPORT_RETRY_MAX_ATTEMPTS,
    )


def _is_retryable_connector_error(exc: BaseException) -> bool:
    """Retry aiohttp connection failures; let ApiException propagate.

    Retries ``ClientConnectorError`` (mirroring the original api_client patch,
    which retried only that and re-raised ``ApiException`` so callers still see
    real API errors with the decoded body).

    Excludes ``ClientConnectorDNSError``: it is a ``ClientConnectorError``
    subclass that ``_create_direct_connection`` raises after the ``OSError``
    from ``_resolve_host``, which the aiohttp DNS seam has *already* retried to
    exhaustion. Retrying it again here would nest the two seams (up to
    ``MAX_ATTEMPTS`` request retries x ``MAX_ATTEMPTS`` DNS retries each),
    blowing past the documented backoff budget. Letting it propagate keeps DNS
    retried exactly once, at its true source.
    """
    # pylint: disable-next=import-outside-toplevel
    from aiohttp.client_exceptions import ClientConnectorError

    try:
        # ClientConnectorDNSError only exists in aiohttp >= 3.10; on older pins
        # (the floor is >=3.8) there is no separate DNS subclass, so nothing to
        # exclude and DNS errors simply retry here as before.
        # pylint: disable-next=import-outside-toplevel
        from aiohttp.client_exceptions import ClientConnectorDNSError

        if isinstance(exc, ClientConnectorDNSError):
            return False
    except ImportError:
        pass
    return isinstance(exc, ClientConnectorError)


def _install_k8s_request_retry() -> None:
    """Wrap ``ApiClient.request`` with the transport retry policy."""
    # pylint: disable-next=import-outside-toplevel
    from kubernetes_asyncio.client.api_client import ApiClient

    original = ApiClient.request
    if getattr(original, _WRAPPED_MARKER, False):
        return

    @functools.wraps(original)
    async def _request_with_retry(self, *args, **kwargs):
        async for attempt in _make_retrying(
            _is_retryable_connector_error, "kubernetes_asyncio request"
        ):
            with attempt:
                # Upstream ApiClient.request is a sync def that returns a
                # coroutine (it dispatches to async RESTClient methods), so the
                # original call must be awaited.
                return await original(self, *args, **kwargs)
        # Unreachable: tenacity either returns a value or re-raises.
        raise RuntimeError("transport k8s request retry exhausted without result")

    setattr(_request_with_retry, _WRAPPED_MARKER, True)
    ApiClient.request = _request_with_retry  # type: ignore[method-assign]
    logger.info(
        "Installed transport retry around kubernetes_asyncio ApiClient.request "
        "(max_attempts=%d)",
        TRANSPORT_RETRY_MAX_ATTEMPTS,
    )


def install_transport_retries() -> None:
    """Install low-level transport retries (idempotent).

    Monkeypatches the upstream aiohttp and kubernetes_asyncio seams in place so
    that all current and future ``TCPConnector`` / ``ApiClient`` instances pick
    up the retry behavior. Safe to call multiple times; only the first call has
    an effect. Set ``GBSERVER_TRANSPORT_RETRY_MAX_ATTEMPTS=1`` to effectively
    disable the retries.

    Each seam is wrapped independently and skipped if its library is not
    installed. ``kubernetes_asyncio`` in particular lives in the optional
    ``ibm`` extra, so it is absent in lightweight environments (e.g. the
    quick-test CI matrix); a missing library is logged and ignored rather than
    raised, since there is nothing to wrap.
    """
    global _INSTALLED  # pylint: disable=global-statement
    if _INSTALLED:
        return
    for installer in (_install_aiohttp_dns_retry, _install_k8s_request_retry):
        try:
            installer()
        except ImportError as exc:
            logger.info(
                "Skipping %s: dependency not installed (%s)",
                installer.__name__,
                exc,
            )
    _INSTALLED = True
