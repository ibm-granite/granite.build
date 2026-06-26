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
from aiohttp.connector import TCPConnector
from kubernetes_asyncio.client.api_client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

import gbserver.resilience.transport_retry as tr
from gbserver.resilience.transport_retry import (
    _WRAPPED_MARKER,
    _is_retryable_connector_error,
    _is_retryable_dns_error,
    _make_retrying,
    install_transport_retries,
)


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
    orig_request = ApiClient.request
    monkeypatch.setattr(tr, "_INSTALLED", False)

    install_transport_retries()
    try:
        yield
    finally:
        TCPConnector._resolve_host = orig_resolve  # type: ignore[method-assign]
        ApiClient.request = orig_request  # type: ignore[method-assign]
        tr._INSTALLED = False


class TestInstall:
    """Installation is idempotent and stamps both seams."""

    def test_wraps_both_seams(self: Self, fresh_install) -> None:
        assert getattr(TCPConnector._resolve_host, _WRAPPED_MARKER, False)
        assert getattr(ApiClient.request, _WRAPPED_MARKER, False)

    def test_idempotent(self: Self, fresh_install) -> None:
        wrapped_resolve = TCPConnector._resolve_host
        wrapped_request = ApiClient.request
        # _INSTALLED guard short-circuits, but even without it the marker check
        # prevents double-wrapping.
        tr._INSTALLED = False
        install_transport_retries()
        assert TCPConnector._resolve_host is wrapped_resolve
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

    def test_connector_retries_only_client_connector_error(self: Self) -> None:
        # Build a minimal ClientConnectorError instance without a real socket.
        err = ClientConnectorError(connection_key=_FakeKey(), os_error=OSError("x"))
        assert _is_retryable_connector_error(err) is True
        assert _is_retryable_connector_error(ApiException(status=500)) is False
        assert _is_retryable_connector_error(OSError("x")) is False


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
            async for attempt in _make_retrying(_is_retryable_dns_error):
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
            async for attempt in _make_retrying(_is_retryable_dns_error):
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
            async for attempt in _make_retrying(_is_retryable_dns_error):
                with attempt:
                    calls["n"] += 1
                    raise OSError("still down")

        with pytest.raises(OSError):
            await always_fail()
        assert calls["n"] == 3
