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
    _is_retryable_connector_error,
    _is_retryable_dns_error,
    _make_retrying,
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
        # ApiException must propagate so callers see real API errors.
        assert _is_retryable_connector_error(ApiException(status=500)) is False


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
