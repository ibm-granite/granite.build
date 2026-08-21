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

"""ToolLoopBackend._sessions is a plain dict keyed on whatever session_id a
caller supplies — it has no per-user access control of its own. These tests
cover api/chat.py's _scoped_session_id(), which namespaces session_id by the
caller's trusted identity before it ever reaches the backend, so two
different authenticated users can never collide on (or hijack) the same
backend session key even if they end up holding the same raw session_id.

Mirrors test_analytics_authz.py's identity precedence: gbserver's
AuthMiddleware-trusted user wins over the X-User-Email header, which only
matters when running standalone outside gbserver.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from gb_ui_backend.api import chat as chat_module
from gb_ui_backend.api.chat import (
    _rate_limit_chat_stream,
    _resolve_identity,
    _scoped_session_id,
)


def _fake_request(
    user_email: str | None = None, header_email: str | None = None
) -> SimpleNamespace:
    state = SimpleNamespace(
        data={"user": SimpleNamespace(email=user_email)} if user_email else {}
    )
    headers = {"x-user-email": header_email} if header_email else {}
    return SimpleNamespace(state=state, headers=headers)


class TestResolveIdentity:
    def test_trusted_authmiddleware_user_wins_over_header(self):
        request = _fake_request(
            user_email="alice@example.com", header_email="bob@example.com"
        )
        assert _resolve_identity(request) == "alice@example.com"

    def test_header_is_a_fallback_when_no_authmiddleware_user(self):
        request = _fake_request(header_email="alice@example.com")
        assert _resolve_identity(request) == "alice@example.com"

    def test_standalone_sentinel_when_neither_present(self):
        request = _fake_request()
        assert _resolve_identity(request) == "standalone"


class TestScopedSessionId:
    def test_same_raw_session_id_scopes_differently_per_identity(self):
        """The actual security property: if alice and bob's browsers ever
        ended up holding the same raw session_id (or bob simply guessed
        alice's), they must not resolve to the same backend session key."""
        raw_id = "11111111-1111-1111-1111-111111111111"
        alice_request = _fake_request(user_email="alice@example.com")
        bob_request = _fake_request(user_email="bob@example.com")

        assert _scoped_session_id(alice_request, raw_id) != _scoped_session_id(
            bob_request, raw_id
        )

    def test_same_identity_and_raw_id_scopes_identically(self):
        raw_id = "11111111-1111-1111-1111-111111111111"
        request_a = _fake_request(user_email="alice@example.com")
        request_b = _fake_request(user_email="alice@example.com")

        assert _scoped_session_id(request_a, raw_id) == _scoped_session_id(
            request_b, raw_id
        )

    def test_standalone_mode_still_scopes_by_the_shared_sentinel(self):
        """Standalone/apikey mode has no per-user identity at all — this is
        the one case where scoping provides no cross-user isolation, matching
        the rest of the app's security model (localhost is trusted wholesale
        in that mode)."""
        raw_id = "abc123"
        assert _scoped_session_id(_fake_request(), raw_id) == "standalone:abc123"


class TestRateLimitChatStream:
    """Code-review finding: /chat/stream had no rate limit or cap on session
    creation at all — each distinct session_id spawns a gbmcp subprocess, so
    an unbounded caller could exhaust host resources. Mirrors ai.py's
    _rate_limit_analyze_logs (same sliding-window shape, this endpoint's own
    call-times dict)."""

    def setup_method(self):
        chat_module._chat_stream_call_times.clear()

    def test_allows_calls_under_the_limit(self):
        request = _fake_request(user_email="alice@example.com")
        for _ in range(chat_module._CHAT_STREAM_RATE_LIMIT_MAX_CALLS):
            _rate_limit_chat_stream(request)  # must not raise

    def test_blocks_once_the_limit_is_exceeded(self):
        request = _fake_request(user_email="alice@example.com")
        for _ in range(chat_module._CHAT_STREAM_RATE_LIMIT_MAX_CALLS):
            _rate_limit_chat_stream(request)

        with pytest.raises(HTTPException) as exc_info:
            _rate_limit_chat_stream(request)
        assert exc_info.value.status_code == 429

    def test_limit_is_tracked_independently_per_identity(self):
        alice = _fake_request(user_email="alice@example.com")
        bob = _fake_request(user_email="bob@example.com")
        for _ in range(chat_module._CHAT_STREAM_RATE_LIMIT_MAX_CALLS):
            _rate_limit_chat_stream(alice)

        _rate_limit_chat_stream(bob)  # must not raise — separate identity, fresh window

    def test_old_calls_outside_the_window_are_not_counted(self, monkeypatch):
        request = _fake_request(user_email="alice@example.com")
        fake_now = [1000.0]
        monkeypatch.setattr(chat_module.time, "monotonic", lambda: fake_now[0])

        for _ in range(chat_module._CHAT_STREAM_RATE_LIMIT_MAX_CALLS):
            _rate_limit_chat_stream(request)

        fake_now[0] += chat_module._CHAT_STREAM_RATE_LIMIT_WINDOW_SECONDS + 1
        _rate_limit_chat_stream(request)  # must not raise — the old window has expired
