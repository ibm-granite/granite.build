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
Tests for the low-level transport retry installer.

Covers idempotent installation of the aiohttp / kubernetes_asyncio monkeypatches
and the retry predicates that decide which transport errors are transient.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Self

import pytest
from aiohttp.client_exceptions import ClientConnectorError

try:
    from aiohttp.client_exceptions import ClientConnectorDNSError

    HAS_DNS_ERROR = True
except ImportError:  # aiohttp < 3.10
    ClientConnectorDNSError = None  # type: ignore[assignment,misc]
    HAS_DNS_ERROR = False
from aiohttp.connector import TCPConnector

# kubernetes_asyncio lives in the optional ``ibm`` extra and is absent in
# lightweight environments (e.g. the quick-test CI matrix). HAS_K8S / requires_k8s
# are shared via libgbtest.constants; import the client symbols this module uses
# directly, guarded by HAS_K8S so it still collects when the extra is absent.
from libgbtest.constants import HAS_K8S, requires_k8s

import gbserver.resilience.transport_retry as tr
from gbserver.resilience.transport_retry import (
    _WRAPPED_MARKER,
    _is_retryable_api_status,
    _is_retryable_connector_error,
    _is_retryable_dns_error,
    _make_retrying,
    _retry_after_seconds,
    _WaitRetryAfterOrExponential,
    install_transport_retries,
)

if HAS_K8S:
    from kubernetes_asyncio.client.api_client import ApiClient
    from kubernetes_asyncio.client.exceptions import ApiException
else:
    ApiClient = None  # type: ignore[assignment,misc]
    ApiException = None  # type: ignore[assignment,misc]


@pytest.fixture
def fresh_install(monkeypatch: pytest.MonkeyPatch):
    """Install the patches against fast (no-wait) retries and restore after.

    Resets the module-level ``_INSTALLED`` guard and snapshots the original
    upstream methods so the global monkeypatch does not leak into other tests.
    """
    # Fast, deterministic retries: a few attempts, no backoff wait.
    monkeypatch.setattr(tr, "TRANSPORT_RETRY_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(tr, "TRANSPORT_RETRY_BASE_DELAY", 0.0)
    monkeypatch.setattr(tr, "TRANSPORT_RETRY_MAX_DELAY", 0.0)

    orig_resolve = TCPConnector._resolve_host
    orig_request = ApiClient.request if HAS_K8S else None
    monkeypatch.setattr(tr, "_INSTALLED", False)

    install_transport_retries()
    try:
        yield
    finally:
        TCPConnector._resolve_host = orig_resolve  # type: ignore[method-assign]
        if HAS_K8S:
            ApiClient.request = orig_request  # type: ignore[method-assign]
        tr._INSTALLED = False


class TestInstall:
    """Installation is idempotent and stamps both seams."""

    def test_wraps_aiohttp_seam(self: Self, fresh_install) -> None:
        assert getattr(TCPConnector._resolve_host, _WRAPPED_MARKER, False)

    @requires_k8s
    def test_wraps_k8s_seam(self: Self, fresh_install) -> None:
        assert getattr(ApiClient.request, _WRAPPED_MARKER, False)

    def test_idempotent(self: Self, fresh_install) -> None:
        # The fixture already installed once. Re-running (even after clearing
        # the _INSTALLED guard) must not re-wrap: the per-method marker check
        # short-circuits.
        wrapped_resolve = TCPConnector._resolve_host
        wrapped_request = ApiClient.request if HAS_K8S else None
        tr._INSTALLED = False
        install_transport_retries()
        assert TCPConnector._resolve_host is wrapped_resolve
        if HAS_K8S:
            assert ApiClient.request is wrapped_request

    def test_skips_seam_with_missing_dependency(
        self: Self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A seam whose library is not installed is skipped, not fatal.

        kubernetes_asyncio lives in the optional ``ibm`` extra and is absent in
        lightweight environments (e.g. the quick-test CI matrix). The installer
        must still wrap the aiohttp seam and not raise.
        """
        monkeypatch.setattr(tr, "TRANSPORT_RETRY_MAX_ATTEMPTS", 3)
        monkeypatch.setattr(tr, "TRANSPORT_RETRY_BASE_DELAY", 0.0)
        monkeypatch.setattr(tr, "TRANSPORT_RETRY_MAX_DELAY", 0.0)
        orig_resolve = TCPConnector._resolve_host
        monkeypatch.setattr(tr, "_INSTALLED", False)

        def boom() -> None:
            raise ModuleNotFoundError("No module named 'kubernetes_asyncio'")

        monkeypatch.setattr(tr, "_install_k8s_request_retry", boom)

        try:
            # Must not raise despite the missing dependency.
            install_transport_retries()
            assert getattr(TCPConnector._resolve_host, _WRAPPED_MARKER, False)
        finally:
            TCPConnector._resolve_host = orig_resolve  # type: ignore[method-assign]
            tr._INSTALLED = False


class TestPredicates:
    """Retry predicates mirror the original patches."""

    def test_dns_retries_oserror_not_timeout(self: Self) -> None:
        assert _is_retryable_dns_error(OSError("dns down")) is True
        assert _is_retryable_dns_error(asyncio.TimeoutError()) is False
        assert _is_retryable_dns_error(ValueError("nope")) is False

    def test_dns_retries_timeout_with_errno(self: Self) -> None:
        # The original patch only re-raised the errno-less cancellation
        # TimeoutError; a TimeoutError carrying an errno is a real network error
        # and must still be retried (mirrors ``exc.errno is None`` guard).
        errno_timeout = asyncio.TimeoutError()
        errno_timeout.errno = 110  # ETIMEDOUT
        assert _is_retryable_dns_error(errno_timeout) is True

    def test_connector_retries_only_client_connector_error(self: Self) -> None:
        # Build a minimal ClientConnectorError instance without a real socket.
        err = ClientConnectorError(connection_key=_FakeKey(), os_error=OSError("x"))
        assert _is_retryable_connector_error(err) is True
        assert _is_retryable_connector_error(OSError("x")) is False

    @pytest.mark.skipif(
        not HAS_DNS_ERROR, reason="ClientConnectorDNSError needs aiohttp >= 3.10"
    )
    def test_connector_does_not_retry_dns_error(self: Self) -> None:
        # ClientConnectorDNSError is handled at the DNS seam (_resolve_host);
        # retrying it again at the request seam would nest the two budgets.
        dns_err = ClientConnectorDNSError(
            connection_key=_FakeKey(), os_error=OSError("dns")
        )
        assert isinstance(dns_err, ClientConnectorError)
        assert _is_retryable_connector_error(dns_err) is False

    @requires_k8s
    def test_connector_does_not_retry_api_exception(self: Self) -> None:
        # The connector predicate never matches ApiException; retryable HTTP
        # statuses are handled by _is_retryable_api_status instead.
        assert _is_retryable_connector_error(ApiException(status=500)) is False


class TestApiStatusPredicate:
    """The apiserver-status predicate retries transient HTTP codes only."""

    @requires_k8s
    def test_retries_transient_statuses_when_verb_unknown(self: Self) -> None:
        # method=None permits the full transient set (classification without a verb).
        for status in (429, 500, 502, 503, 504):
            assert _is_retryable_api_status(ApiException(status=status)) is True

    @requires_k8s
    def test_does_not_retry_non_transient_statuses(self: Self) -> None:
        for status in (400, 401, 403, 404, 409, 422):
            assert _is_retryable_api_status(ApiException(status=status)) is False

    @requires_k8s
    def test_429_retries_for_any_verb(self: Self) -> None:
        for method in ("GET", "POST", "PATCH", "DELETE"):
            assert _is_retryable_api_status(ApiException(status=429), method) is True

    @requires_k8s
    def test_5xx_retries_only_for_read_verbs(self: Self) -> None:
        # 5xx on a write may have mutated state before the response was lost, so
        # only idempotent reads are retried.
        for status in (500, 502, 503, 504):
            for read in ("GET", "HEAD", "OPTIONS", "get"):
                assert (
                    _is_retryable_api_status(ApiException(status=status), read) is True
                )
            for write in ("POST", "PUT", "PATCH", "DELETE"):
                assert (
                    _is_retryable_api_status(ApiException(status=status), write)
                    is False
                )

    def test_non_api_exception_is_not_retryable(self: Self) -> None:
        # Non-ApiException (and the missing-library case) must not match.
        assert _is_retryable_api_status(ValueError("nope")) is False
        assert _is_retryable_api_status(OSError("x")) is False


class TestRetryAfterWait:
    """The k8s wait honors Retry-After hints, capped, else falls back."""

    def test_retry_after_header_seconds(self: Self) -> None:
        exc = _FakeApiExc(status=429, headers={"Retry-After": "2"})
        assert _retry_after_seconds(exc) == 2.0

    def test_retry_after_body_details(self: Self) -> None:
        exc = _FakeApiExc(status=429, body='{"details": {"retryAfterSeconds": 3}}')
        assert _retry_after_seconds(exc) == 3.0

    def test_no_hint_returns_none(self: Self) -> None:
        assert _retry_after_seconds(_FakeApiExc(status=500)) is None
        assert _retry_after_seconds(ValueError("x")) is None

    def test_malformed_header_ignored(self: Self) -> None:
        exc = _FakeApiExc(status=429, headers={"Retry-After": "soon"})
        assert _retry_after_seconds(exc) is None

    def test_retry_after_http_date_future(self: Self) -> None:
        # RFC 7231 allows an HTTP-date; parse it to a positive delay.
        future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30))
        exc = _FakeApiExc(status=429, headers={"Retry-After": future})
        delay = _retry_after_seconds(exc)
        assert delay is not None and 20.0 <= delay <= 31.0

    def test_retry_after_http_date_past_clamps_to_zero(self: Self) -> None:
        past = format_datetime(datetime.now(timezone.utc) - timedelta(seconds=30))
        exc = _FakeApiExc(status=429, headers={"Retry-After": past})
        assert _retry_after_seconds(exc) == 0.0

    def test_wait_honors_hint_capped(
        self: Self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tr, "TRANSPORT_RETRY_MAX_DELAY", 15.0)
        wait = _WaitRetryAfterOrExponential()
        # Server asks for 2s -> honored.
        assert (
            wait(_FakeRetryState(_FakeApiExc(status=429, headers={"Retry-After": "2"})))
            == 2.0
        )
        # Server asks for a huge delay -> capped at MAX_DELAY.
        assert (
            wait(
                _FakeRetryState(
                    _FakeApiExc(status=429, headers={"Retry-After": "9999"})
                )
            )
            == 15.0
        )

    def test_wait_falls_back_to_backoff(
        self: Self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No hint -> exponential backoff (bounded by MAX_DELAY, non-negative).
        monkeypatch.setattr(tr, "TRANSPORT_RETRY_BASE_DELAY", 1.0)
        monkeypatch.setattr(tr, "TRANSPORT_RETRY_MAX_DELAY", 15.0)
        wait = _WaitRetryAfterOrExponential()
        delay = wait(_FakeRetryState(OSError("x")))
        assert 0.0 <= delay <= 15.0


class _FakeApiExc(Exception):
    """Stand-in for kubernetes_asyncio ApiException (status/headers/body only)."""

    def __init__(self, status: int, headers=None, body=None) -> None:
        super().__init__(f"({status})")
        self.status = status
        self.headers = headers
        self.body = body


class _FakeOutcome:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def exception(self) -> BaseException:
        return self._exc


class _FakeRetryState:
    """Minimal tenacity RetryCallState carrying a failed outcome."""

    def __init__(self, exc: BaseException) -> None:
        self.outcome = _FakeOutcome(exc)
        self.attempt_number = 1


class _FakeKey:
    """Minimal stand-in for aiohttp ConnectionKey used to construct errors."""

    host = "example.com"
    port = 443
    is_ssl = True
    ssl = None
    proxy = None
    proxy_auth = None
    proxy_headers_hash = None


class TestRetryDriver:
    """The shared AsyncRetrying retries transient errors and gives up cleanly."""

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(
        self: Self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tr, "TRANSPORT_RETRY_MAX_ATTEMPTS", 5)
        monkeypatch.setattr(tr, "TRANSPORT_RETRY_BASE_DELAY", 0.0)
        monkeypatch.setattr(tr, "TRANSPORT_RETRY_MAX_DELAY", 0.0)

        calls = {"n": 0}

        async def flaky() -> str:
            async for attempt in _make_retrying(_is_retryable_dns_error, "test"):
                with attempt:
                    calls["n"] += 1
                    if calls["n"] < 3:
                        raise OSError("transient")
                    return "ok"
            raise AssertionError("unreachable")

        assert await flaky() == "ok"
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_does_not_retry_non_transient(
        self: Self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tr, "TRANSPORT_RETRY_MAX_ATTEMPTS", 5)
        monkeypatch.setattr(tr, "TRANSPORT_RETRY_BASE_DELAY", 0.0)
        monkeypatch.setattr(tr, "TRANSPORT_RETRY_MAX_DELAY", 0.0)

        calls = {"n": 0}

        async def boom() -> None:
            async for attempt in _make_retrying(_is_retryable_dns_error, "test"):
                with attempt:
                    calls["n"] += 1
                    raise asyncio.TimeoutError()

        with pytest.raises(asyncio.TimeoutError):
            await boom()
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_reraises_after_exhaustion(
        self: Self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tr, "TRANSPORT_RETRY_MAX_ATTEMPTS", 3)
        monkeypatch.setattr(tr, "TRANSPORT_RETRY_BASE_DELAY", 0.0)
        monkeypatch.setattr(tr, "TRANSPORT_RETRY_MAX_DELAY", 0.0)

        calls = {"n": 0}

        async def always_fail() -> None:
            async for attempt in _make_retrying(_is_retryable_dns_error, "test"):
                with attempt:
                    calls["n"] += 1
                    raise OSError("still down")

        with pytest.raises(OSError):
            await always_fail()
        assert calls["n"] == 3


class TestK8sSeamRetries:
    """The installed ApiClient.request wrapper retries transient HTTP statuses."""

    @staticmethod
    def _install_fake_base(monkeypatch: pytest.MonkeyPatch, fake_request) -> None:
        # fresh_install already wrapped ApiClient.request. Swap the *original*
        # base for our fake (marker-less, so re-install wraps it) and re-install.
        monkeypatch.setattr(ApiClient, "request", fake_request, raising=True)
        tr._INSTALLED = False
        install_transport_retries()

    @requires_k8s
    @pytest.mark.asyncio
    async def test_installed_wrapper_retries_429_then_succeeds(
        self: Self, fresh_install, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        async def flaky_request(self, method, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ApiException(status=429, reason="Too Many Requests")
            return "ok"

        self._install_fake_base(monkeypatch, flaky_request)
        client_obj = ApiClient.__new__(ApiClient)
        # 429 retries even on a write verb.
        result = await ApiClient.request(client_obj, "POST")
        assert result == "ok"
        assert calls["n"] == 3

    @requires_k8s
    @pytest.mark.asyncio
    async def test_installed_wrapper_does_not_retry_404(
        self: Self, fresh_install, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        async def not_found_request(self, method, *args, **kwargs):
            calls["n"] += 1
            raise ApiException(status=404, reason="Not Found")

        self._install_fake_base(monkeypatch, not_found_request)
        client_obj = ApiClient.__new__(ApiClient)
        with pytest.raises(ApiException):
            await ApiClient.request(client_obj, "GET")
        assert calls["n"] == 1

    @requires_k8s
    @pytest.mark.asyncio
    async def test_installed_wrapper_retries_5xx_on_read_verb(
        self: Self, fresh_install, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        async def flaky_request(self, method, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ApiException(status=503, reason="Service Unavailable")
            return "ok"

        self._install_fake_base(monkeypatch, flaky_request)
        client_obj = ApiClient.__new__(ApiClient)
        result = await ApiClient.request(client_obj, "GET")
        assert result == "ok"
        assert calls["n"] == 3

    @requires_k8s
    @pytest.mark.asyncio
    async def test_installed_wrapper_does_not_retry_5xx_on_write_verb(
        self: Self, fresh_install, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A 5xx on a write may have mutated state; must not be blindly retried.
        calls = {"n": 0}

        async def failing_write(self, method, *args, **kwargs):
            calls["n"] += 1
            raise ApiException(status=503, reason="Service Unavailable")

        self._install_fake_base(monkeypatch, failing_write)
        client_obj = ApiClient.__new__(ApiClient)
        with pytest.raises(ApiException):
            await ApiClient.request(client_obj, "POST")
        assert calls["n"] == 1
